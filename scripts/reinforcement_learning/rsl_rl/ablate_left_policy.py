#!/usr/bin/env python3
"""Native first-episode test of a mirrored canonical-left policy inference.

The real robot still executes the physical LEFT maneuver, with the unchanged
production LiftLegEnv joint controller, contacts, limits and termination logic.
Native mass/inertia asymmetry prevents an exact reflected-dynamics claim.
This diagnostic cannot train, promote a policy or unlock interactive controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import traceback


VARIANTS=("baseline","canonical_left")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fresh_json(path,value):
    with Path(path).open('x') as output:
        json.dump(value,output,indent=2,allow_nan=False)
        output.write('\n')


def validate_contract(saved,expected):
    keys=('robot','schema','urdf_sha256','gait_sha256','control_sha256','control_profile',
          'motor','joint_names','action_size','observation_size')
    if (expected.get('robot')!='q1_leg_lift_v5' or expected.get('action_size')!=23
            or expected.get('observation_size')!=110 or len(set(expected.get('joint_names',[])))!=23
            or expected.get('proxy') is not False or saved.get('proxy') is not False
            or any(saved.get(key)!=expected.get(key) for key in keys)):
        raise ValueError('Diagnostic checkpoint must match the current native v5 geometry, gait and full control contract')


class CanonicalLeftPolicy:
    """Reflect only feasible, first-episode, explicitly requested left rows.

    ``augment_policy`` supplies the exact production mirror-feasibility rules,
    including the shoulder reference's 0.35-rad residual margin. The private
    cache avoids altering the production environment or its training callback.
    """
    def __init__(self,policy,env,canonical_rows):
        import torch
        self.policy=policy
        self.env=env
        self.canonical_rows=canonical_rows.bool()
        self.active=torch.ones_like(self.canonical_rows)
        self.cache=SimpleNamespace(unwrapped=env)
        self.requested_samples=torch.zeros_like(self.canonical_rows,dtype=torch.long)
        self.mirrored_samples=self.requested_samples.clone()
        self.last_mask=torch.zeros_like(self.canonical_rows)
        self.last_requested=self.last_mask.clone()
        self.last_feasible=self.last_mask.clone()

    def __call__(self,obs):
        import torch
        from leg_humanoid_lab.lift_symmetry import augment_policy
        # The evaluator updates _manual_mode before inference; env.mode and
        # reference-conditioned observations update in _pre_physics_step.
        # Thus a mode boundary retains the same one-frame observation delay as
        # baseline, while this selection follows the newly requested mode.
        requested=self.canonical_rows & self.active & (self.env._manual_mode==2)
        self.last_requested=requested.clone()
        self.last_mask=torch.zeros_like(requested)
        self.last_feasible=torch.zeros_like(requested)
        if bool(requested.any()):
            doubled,_=augment_policy(self.cache,obs=obs,actions=None)
            feasible=self.cache._q1_mirror_feasible
            selected=requested & feasible
            transformed=obs.clone()
            for key,value in obs.items():
                transformed[key]=torch.where(selected[:,None],doubled[key][len(requested):],value)
            actions=self.policy(transformed)
            mirrored_actions=self.cache._q1_native_mirror.actions(actions)
            result=torch.where(selected[:,None],mirrored_actions,actions)
            self.last_feasible=feasible.clone()
            self.last_mask=selected.clone()
        else:
            result=self.policy(obs)
        self.requested_samples+=requested
        self.mirrored_samples+=self.last_mask
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=4)
    parser.add_argument('--seed',type=int,default=481131)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.replicates<1:parser.error('--replicates must be positive')
    args.output=args.output.resolve()
    error_path=args.output.with_name(args.output.stem+'_error.json')
    if args.output.exists() or error_path.exists():
        parser.error('Refusing to overwrite an existing diagnostic or error report')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    app=env=None
    try:
        app=AppLauncher(args).app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from leg_humanoid_lab.lift_env import LiftEnvCfg,LiftLegEnv,lift_contract
        from leg_humanoid_lab.lift_evaluation import STAGE_CASES,check_lift_gates,evaluate_lift
        from leg_humanoid_lab.robot import ROOT
        from runtime_compat import load_checkpoint,runner_cfg
        from ablate_lift_control import summarize

        cases=STAGE_CASES[1]
        block=len(cases)*args.replicates
        count=len(VARIANTS)*block
        cfg=LiftEnvCfg();cfg.scene.num_envs=count;cfg.training_mode=False
        cfg.curriculum_stage=1;cfg.episode_length_s=20.5;cfg.seed=args.seed;cfg.sim.device=args.device
        env=LiftLegEnv(cfg)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=args.device)
        contract=lift_contract()
        contract_path=args.checkpoint.parent/'contract.json'
        validate_contract(json.loads(contract_path.read_text()),contract)
        checkpoint_sha=digest(args.checkpoint);contract_sha=digest(contract_path)
        sources=[Path(__file__),Path(__file__).with_name('ablate_lift_control.py'),ROOT/'config/lift_control_v5.json',ROOT/contract['gait_file']]
        sources+=[ROOT/'source/leg_humanoid_lab/leg_humanoid_lab'/name for name in
                  ('lift_env.py','lift_objective.py','lift_symmetry.py','lift_evaluation.py',
                   'interactive_env.py','interactive_control.py','isaac_env.py','robot.py','agent_cfg.py')]
        sources+=[Path(__file__).with_name('runtime_compat.py')]
        manifest={str(path.resolve().relative_to(ROOT.resolve())):digest(path) for path in sources}
        load_checkpoint(runner,args.checkpoint,load_optimizer=False,load_iteration=True)
        indices=torch.arange(count,device=env.device)
        policy=CanonicalLeftPolicy(runner.get_inference_policy(device=env.device),env,indices>=block)

        class InferenceRunner:
            def get_inference_policy(self,device=None):return policy

        class FirstEpisodeTrace:
            def __init__(self,transport):self.transport=transport;self.masks=[]
            def __getattr__(self,name):return getattr(self.transport,name)
            def reset(self):
                result=self.transport.reset()
                self.active=torch.ones(count,device=env.device,dtype=torch.bool)
                policy.active=self.active.clone()
                self.length=env.episode_length_buf.clone()
                self.unloaded_peak=torch.zeros((count,2),device=env.device)
                self.side_peak=torch.zeros((count,2,2),device=env.device)
                self.side_trials=self.side_peak.clone();self.side_unloaded=self.side_peak.clone()
                return result
            def step(self,actions):
                result=self.transport.step(actions)
                done=result[2].reshape(-1).bool()
                reset=(self.length>0)&(env.episode_length_buf<=self.length)
                self.length=env.episode_length_buf.clone()
                height,speed,loaded,_=env.foot_step_metrics();data=env.robot.data
                finite=(torch.isfinite(height).all(-1)&torch.isfinite(speed).all(-1)
                        &torch.isfinite(data.root_pos_w).all(-1)&torch.isfinite(data.root_lin_vel_b).all(-1)
                        &torch.isfinite(data.projected_gravity_b).all(-1)&torch.isfinite(actions).all(-1))
                self.active &= ~done & ~reset & finite
                policy.active=self.active.clone()
                moving=self.active&((env.mode==2)|(env.mode==3))
                unloaded=torch.where(moving[:,None]&~loaded,height.clamp_min(0.),0.)
                self.unloaded_peak=torch.maximum(self.unloaded_peak,unloaded)
                for side,mode in enumerate((2,3)):
                    selected=self.active&(env.mode==mode)
                    self.side_peak[:,side]=torch.maximum(self.side_peak[:,side],torch.where(selected[:,None],unloaded,0.))
                    high=selected[:,None]&~env.reference_contact&(env.reference_height>.022)
                    self.side_trials[:,side]+=high;self.side_unloaded[:,side]+=high&~loaded
                selected_indices=lambda mask:mask.nonzero(as_tuple=False).flatten().cpu().tolist()
                self.masks.append({'time_s':len(self.masks)*env.step_dt,
                    'requested_left_indices':selected_indices(policy.last_requested),
                    'mirrored_indices':selected_indices(policy.last_mask),
                    'infeasible_fallback_indices':selected_indices(policy.last_requested&~policy.last_feasible),
                    'first_episode_valid_after_step_indices':selected_indices(self.active)})
                return result

        trace=FirstEpisodeTrace(wrapped)
        evaluated=evaluate_lift(InferenceRunner(),trace,env,episodes=count,seed=args.seed,stage=1)
        if digest(args.checkpoint)!=checkpoint_sha or digest(contract_path)!=contract_sha:
            raise ValueError('Source checkpoint/contract changed during evaluation; use a retained immutable copy')
        if any(digest(ROOT/relative)!=sha for relative,sha in manifest.items()):
            raise ValueError('Diagnostic sources changed during evaluation')
        to_list=lambda value:value.detach().cpu().tolist()
        for row in evaluated['per_episode']:
            index=row['env_index']
            row.update(variant=VARIANTS[index//block],
                unloaded_peak_clearance_m_per_foot=to_list(trace.unloaded_peak[index]),
                direction_unloaded_peak_clearance_m_per_foot=to_list(trace.side_peak[index]),
                direction_high_swing_samples_per_foot=to_list(trace.side_trials[index]),
                direction_unloaded_swing_samples_per_foot=to_list(trace.side_unloaded[index]),
                canonical_left_requested_samples=int(policy.requested_samples[index]),
                canonical_left_mirrored_samples=int(policy.mirrored_samples[index]),
                canonical_left_fallback_samples=int(policy.requested_samples[index]-policy.mirrored_samples[index]))
        summaries={}
        for name in VARIANTS:
            rows=[row for row in evaluated['per_episode'] if row['variant']==name]
            summary=summarize(rows,cases);checks=check_lift_gates(summary)
            summary.update(diagnostic_gate_preview=checks,diagnostic_metrics_gate_passed=all(checks.values()),
                           diagnostic_score=sum(checks.values())/len(checks),per_episode=rows)
            summaries[name]=summary
        report={'schema':1,'kind':'native_canonical_left_policy_ablation','backend':'isaaclab','proxy':False,
            'ready':False,'production_acceptance':False,'diagnostic_policy_override':True,
            'scope':'Fixed-policy inference diagnostic only; native mass/inertia are approximately symmetric, with no exact dynamics claim. Cannot train, promote or unlock controls.',
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha,
            'checkpoint_contract_sha256':contract_sha,'completed_iterations':int(runner.current_learning_iteration),
            'contract':contract,'source_sha256':manifest,'seed':args.seed,'envs':count,'replicates_per_case_variant':args.replicates,
            'stage':1,'duration_s':20.,'first_episode_only':True,'exclude_done_and_all_post_reset_samples':True,
            'reset_between_transitions':False,'actuator_override':False,
            'selection':'canonical_left group AND requested _manual_mode==2 AND first episode AND production mirror-feasibility mask; all other inputs/actions unchanged',
            'feasibility':'augment_policy actual/reference native joint bounds; shoulder actual abs<=1.1345 and reference abs<=1.1345-.35; infeasible rows use original policy unchanged',
            'observation_timing':'Selection uses the manual request immediately before inference. Reference-conditioned observations still have the same one-frame mode-boundary delay as baseline; no extra refresh/reset was inserted.',
            'policy_mask_trace_semantics':'Action-selection masks include the terminal action, if any; native pose metrics permanently exclude the done frame and all subsequent reset samples.',
            'overall':evaluated,'variants':summaries,'policy_mask_trace':trace.masks}
        write_fresh_json(args.output,report)
        brief={name:{key:value for key,value in summary.items() if key not in ('per_episode','diagnostic_gate_preview')}
               for name,summary in summaries.items()}
        print('LEFT_POLICY_ABLATION '+json.dumps({'output':str(args.output),'checkpoint_sha256':checkpoint_sha,'variants':brief}),flush=True)
    except BaseException as error:
        traceback.print_exc()
        failure={'kind':'native_canonical_left_policy_ablation_error','error':str(error),'type':type(error).__name__,
                 'production_acceptance':False,'ready':False}
        if not error_path.exists():write_fresh_json(error_path,failure)
        print('LEFT_POLICY_ABLATION_ERROR '+json.dumps(failure),flush=True)
        raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()


if __name__=='__main__':main()
