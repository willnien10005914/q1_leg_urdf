#!/usr/bin/env python3
"""One native upper-body-v6 chunk; no model publication or media upload."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--num_envs',type=int,default=10240)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--completed',type=int,default=0)
p.add_argument('--curriculum-stage',type=int,choices=(0,1,2),default=0)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--training-profile',type=Path)
p.add_argument('--checkpoint',type=Path)
initialization_mode=p.add_mutually_exclusive_group()
initialization_mode.add_argument('--initialize-policy',action='store_true',
    help='New V6 run: preserve full 13115 actor MLP; fresh critic/Adam/std, new update counter zero')
initialization_mode.add_argument('--evaluate',action='store_true')
p.add_argument('--episodes',type=int,default=32)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--reward-revision',type=int,choices=(0,1,2),default=0)
p.add_argument('--smoke-steps',type=int,default=0)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.completed<0 or args.iterations<1 or args.smoke_steps<0:
    p.error('Iteration counts must be nonnegative and iterations positive')
if args.initialize_policy and (not args.checkpoint or args.completed or args.curriculum_stage != 0):
    p.error('New V6 initialization needs source checkpoint, completed=0 and stage=0')
if args.completed and not args.checkpoint:
    p.error('V6 resume needs its saved checkpoint')
if args.initialize_policy and (args.output_dir/'resume.pt').resolve()==args.checkpoint.resolve():
    p.error('V6 initialization must preserve its source checkpoint')
if args.evaluate and not args.checkpoint:
    p.error('Evaluation requires a saved V6 checkpoint')
launcher=AppLauncher(args);app=launcher.app

import hashlib
import json
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from leg_humanoid_lab.upper_body_env import UpperBodyLegEnv,UpperBodyEnvCfg,upper_body_contract
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import ROOT
from runtime_compat import runner_cfg,load_checkpoint,reset_noise_std,policy_module,checkpoint_for_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_checkpoint_contract(saved, expected, *, initialize_policy=False):
    """Source V5 may initialize a new experiment; ordinary V6 resume is exact."""
    if (expected.get('robot') != 'q1_leg_upper_body_v6' or expected.get('schema') != 6
            or expected.get('action_size') != 23 or expected.get('observation_size') != 110
            or len(set(expected.get('joint_names', []))) != 23):
        raise ValueError('V6 needs schema 6 and the native 23-action / 110-observation interface')
    keys=('urdf_sha256','gait_sha256','joint_names','action_size','observation_size')
    if initialize_policy:
        if saved.get('robot') != 'q1_leg_lift_v5':
            raise ValueError('V6 initialization must use the frozen reviewed V5 actor')
    else:
        keys+=('robot','schema','control_sha256','control_profile','motor')
    if any(saved.get(key)!=expected.get(key) for key in keys) or saved.get('proxy') is not False:
        raise ValueError('Source or resumable checkpoint contract mismatch')
    if not isinstance(saved.get('control_sha256'),str) or len(saved['control_sha256'])!=64 or not saved.get('control_profile'):
        raise ValueError('Checkpoint must preserve original control profile and SHA256 lineage')


def validate_update_budget(profile, completed, iterations):
    """20,000 NEW V6 updates; the source actor's 13,115 updates are provenance."""
    if (type(completed) is not int or completed < 0 or type(iterations) is not int or iterations < 1
            or profile.get('budget_mode') != 'new_updates' or profile.get('max_iterations') != 20000
            or completed + iterations > 20000):
        raise ValueError('V6 new update count exceeds its explicitly authorized 20000-update budget')
    return completed + iterations


def initialize_complete_actor(runner, path, expected, profile):
    """Copy all actor MLP layers, including output; no critic, Adam or source std."""
    path=Path(path)
    saved=json.loads((path.parent/'contract.json').read_text())
    validate_checkpoint_contract(saved,expected,initialize_policy=True)
    if digest(path) != profile['source_checkpoint_sha256']:
        raise ValueError('V6 source actor SHA256 differs from the reviewed 13115 checkpoint')
    original=torch.load(path,map_location='cpu',weights_only=False)
    source_iteration=profile['source_iteration']
    if (not isinstance(original,dict) or type(original.get('iter')) is not int
            or original['iter'] != source_iteration):
        raise ValueError('V6 source checkpoint iteration is not the reviewed 13115')
    infos=original.get('infos')
    if isinstance(infos,dict) and 'completed_iterations' in infos and infos['completed_iterations']!=source_iteration:
        raise ValueError('Source checkpoint infos contradict its serialized iteration')
    optimizer=getattr(getattr(runner,'alg',None),'optimizer',getattr(runner,'optimizer',None))
    if optimizer is None or optimizer.state:
        raise ValueError('New V6 initialization requires a fresh Adam optimizer')
    compatible=Path(checkpoint_for_runtime(path))
    payload=original if compatible.resolve()==path.resolve() else torch.load(compatible,map_location='cpu',weights_only=False)
    if payload.get('iter') != source_iteration:
        raise ValueError('Runtime conversion changed the recorded source iteration')
    if 'actor_state_dict' in payload:
        source={key.removeprefix('mlp.'):value for key,value in payload['actor_state_dict'].items() if key.startswith('mlp.')}
    elif 'model_state_dict' in payload:
        source={key.removeprefix('actor.'):value for key,value in payload['model_state_dict'].items() if key.startswith('actor.')}
    else:
        raise ValueError('Source checkpoint has no supported actor MLP parameters')
    model=policy_module(runner)
    current=model.state_dict()
    prefixes={name.split('.',1)[0]+'.' for name,module in model.named_modules()
              if isinstance(module,torch.nn.Linear) and name.startswith(('mlp.','actor.'))}
    if len(prefixes)!=1:
        raise ValueError('Actor must expose exactly one supported complete MLP')
    prefix=next(iter(prefixes));copied=[]
    for key,value in current.items():
        if not key.startswith(prefix):continue
        incoming=source.get(key.removeprefix(prefix))
        if incoming is None or incoming.shape!=value.shape or not torch.isfinite(incoming).all():
            raise ValueError(f'Cannot preserve complete actor MLP parameter {key}')
        current[key]=incoming;copied.append(key)
    if not copied:raise ValueError('No compatible complete actor MLP found')
    model.load_state_dict(current)
    reset_noise_std(runner,profile['exploration_std'])
    runner.current_learning_iteration=0
    return {'policy':'new_upper_body_v6_from_complete_actor','source':str(path.resolve()),
            'source_sha256':digest(path),'source_contract_sha256':digest(path.parent/'contract.json'),
            'source_control_sha256':saved['control_sha256'],'source_control_profile':saved['control_profile'],
            'source_payload_iteration':source_iteration,'source_iteration':source_iteration,
            'inherited_completed_iterations':0,'completed_iterations':0,'budget_mode':'new_updates',
            'new_control_sha256':expected['control_sha256'],'new_control_profile':expected['control_profile'],
            'actor_mlp_parameters':copied,'actor_output_zeroed':False,'critic':'fresh','optimizer':'fresh',
            'exploration_std_reset':True,'exploration_std':float(profile['exploration_std']),
            'curriculum_restarts_at':0,'acceptance_inherited':False}


env=None
try:
    args.output_dir.mkdir(parents=True,exist_ok=True)
    profile=json.loads((args.training_profile or ROOT/'config/upper_body_v6.json').read_text())
    if not args.evaluate and not args.smoke_steps:
        validate_update_budget(profile,args.completed,args.iterations)
    cfg=UpperBodyEnvCfg()
    cfg.scene.num_envs=args.episodes if args.evaluate else args.num_envs
    cfg.training_mode=not args.evaluate
    cfg.curriculum_stage=args.curriculum_stage
    cfg.episode_length_s=20.5 if args.evaluate else 20.
    if profile.get('training_episode_lengths_s') != [20,20,20]:
        raise ValueError('Every V6 training stage must retain full 20-second recovery')
    cfg.reward_revision=args.reward_revision
    cfg.seed=args.seed;cfg.sim.device=args.device
    gait_path=(ROOT/profile.get('gait_file','config/v5_gaits_lift6.json')).resolve()
    # LiftEnvCfg owns its native geometry/control defaults; configure its gait
    # path only through the explicit field when the environment exposes one.
    if hasattr(cfg,'gait_file'):cfg.gait_file=str(gait_path)
    env=UpperBodyLegEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    agent=PPOCfg();agent.seed=args.seed
    agent.num_steps_per_env=profile['num_steps_per_env']
    agent.algorithm.num_mini_batches=profile['num_mini_batches']
    agent.algorithm.learning_rate=profile['learning_rate']*(profile.get('reward_revision_learning_rate_factor',.7)**args.reward_revision)
    agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=profile['entropy_coef']
    agent.save_interval=100
    configuration=runner_cfg(agent)
    if not args.evaluate and profile.get('symmetry'):
        configuration['algorithm']['symmetry_cfg']={key:value for key,value in profile['symmetry'].items() if key!='scope'}
    runner=OnPolicyRunner(wrapped,configuration,log_dir=None if args.evaluate or args.smoke_steps else str(args.output_dir),device=args.device)
    expected=upper_body_contract()
    if expected.get('robot')!='q1_leg_upper_body_v6' or expected.get('schema')!=6 or expected.get('gait_sha256')!=digest(gait_path):
        raise ValueError('Upper-body environment does not match its declared V6 gait/robot contract')
    initialization=None
    if args.checkpoint:
        saved=json.loads((args.checkpoint.parent/'contract.json').read_text())
        validate_checkpoint_contract(saved,expected,initialize_policy=args.initialize_policy)
        if args.initialize_policy:
            initialization=initialize_complete_actor(runner,args.checkpoint,expected,profile)
        else:
            compatible=checkpoint_for_runtime(args.checkpoint)
            if not args.evaluate and Path(compatible).resolve()!=args.checkpoint.resolve():
                raise ValueError('V6 training resume requires identical runtime format to preserve Adam')
            payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            initialization=(payload.get('infos') or {}).get('initialization')
            if (not isinstance(initialization,dict) or initialization.get('source_sha256') != profile['source_checkpoint_sha256']
                    or initialization.get('budget_mode') != 'new_updates'):
                raise ValueError('V6 resume checkpoint has missing or incorrect source initialization lineage')
            load_checkpoint(runner,args.checkpoint,load_optimizer=not args.evaluate,load_iteration=True)
            if runner.current_learning_iteration!=args.completed:
                raise ValueError('V6 checkpoint iteration differs from its completed new update counter')
    elif not args.smoke_steps:
        raise ValueError('V6 training needs the explicit frozen actor initialization checkpoint')
    if args.evaluate:
        from leg_humanoid_lab.upper_body_evaluation import evaluate_upper_body
        report=evaluate_upper_body(runner,wrapped,env,episodes=args.episodes,seed=args.seed,stage=args.curriculum_stage)
        source_dir=ROOT/'source/leg_humanoid_lab/leg_humanoid_lab'
        report.update(checkpoint_sha256=digest(args.checkpoint),checkpoint=str(args.checkpoint),completed_iterations=args.completed,
                      stage=args.curriculum_stage,evaluation_seed=args.seed,environment_sha256=digest(source_dir/'upper_body_env.py'),
                      robot=expected['robot'],schema=expected['schema'],control_sha256=expected['control_sha256'],
                      base_environment_sha256=digest(source_dir/'interactive_env.py'),urdf_sha256=expected['urdf_sha256'],gait_sha256=expected['gait_sha256'])
        destination=args.output_dir/f'evaluation_{args.completed}_s{args.curriculum_stage}_{args.seed}.json'
        atomic_json(destination,report)
        print('V6_EVALUATION '+json.dumps({key:value for key,value in report.items() if key not in ('rollout','per_episode')}),flush=True)
    elif args.smoke_steps:
        wrapped.reset();seen_modes=set();peak=0.
        policy=runner.get_inference_policy(device=env.device) if args.checkpoint else None
        action_source=('frozen_initialized_actor' if args.initialize_policy else 'saved_v6_actor') if policy else 'zero_residual'
        upper_peaks={};upper_sums={};upper_counts={}
        active=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
        failed=torch.zeros_like(active)
        actual_peak=torch.zeros((env.num_envs,2),device=env.device)
        unloaded_peak=actual_peak.clone()
        for _ in range(args.smoke_steps):
            with torch.inference_mode():
                actions=policy(wrapped.get_observations()) if policy else torch.zeros((env.num_envs,23),device=env.device)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite native smoke actor actions')
                _,reward,done,_=wrapped.step(actions)

            failed|=active&env.reset_terminated.bool()
            active&=~done.bool()
            height,_,loaded,_=env.foot_step_metrics()
            actual_peak=torch.maximum(actual_peak,torch.where(active[:,None],height,0.).clamp_min(0))
            unloaded_peak=torch.maximum(unloaded_peak,torch.where(active[:,None]&~loaded,height,0.).clamp_min(0))
            obs=env._get_observations()['policy']
            if not torch.isfinite(reward).all() or not torch.isfinite(obs).all():raise RuntimeError('Nonfinite native smoke rollout')
            seen_modes.update(env.mode.unique().cpu().tolist())
            peak=max(peak,float(env.reference_height.max()))
            measurements=env.upper_body_measurements()
            if any(not torch.isfinite(value).all() for value in measurements.values()):
                raise RuntimeError('Nonfinite native upper-body smoke measurements')
            for key,value in measurements.items():
                valid=value[active].float()
                if valid.numel():
                    upper_sums[key]=upper_sums.get(key,0.)+float(valid.sum())
                    upper_counts[key]=upper_counts.get(key,0)+valid.numel()
                    upper_peaks[key]=max(upper_peaks.get(key,0.),float(valid.max()))
        report={'envs':env.num_envs,'steps':args.smoke_steps,'stage':args.curriculum_stage,'finite':True,
                'observation_shape':list(obs.shape),'seen_modes':sorted(seen_modes),'peak_reference_height_m':peak,
                'first_episode_survival_fraction':float((~failed).float().mean()),'action_source':action_source,'active_first_episode_fraction_at_end':float(active.float().mean()),'actual_peak_sole_m_per_foot_mean':actual_peak.mean(0).cpu().tolist(),
                'actual_unloaded_peak_m_per_foot_mean':unloaded_peak.mean(0).cpu().tolist(),'first_episode_only':True,'contract':expected,
                'upper_body_measurements_finite':True,'upper_body_mean':{key:value/upper_counts[key] for key,value in upper_sums.items()},
                'upper_body_peak':upper_peaks,'initialization':initialization,
                'runner_iteration':runner.current_learning_iteration,'new_update_count':0,'source_iteration':profile['source_iteration']}

        atomic_json(args.output_dir/'smoke.json',report)
        print('V6_SMOKE '+json.dumps(report),flush=True)
    else:
        atomic_json(args.output_dir/'contract.json',expected)
        # Resume retains learned std and every Adam moment. A declared revision
        # changes only the learning rate/reward coefficient, never exploration.
        runner.alg.learning_rate=agent.algorithm.learning_rate
        for group in runner.alg.optimizer.param_groups:group['lr']=agent.algorithm.learning_rate
        runner.current_learning_iteration=args.completed
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        completed=args.completed+args.iterations
        runner.current_learning_iteration=completed
        temporary=args.output_dir/'resume.tmp.pt'
        runner.save(str(temporary),infos={'completed_iterations':completed,'curriculum_stage':args.curriculum_stage,
                    'reward_revision':args.reward_revision,'initialization':initialization,
                    'budget_mode':'new_updates','source_iteration':profile['source_iteration']})
        temporary.replace(args.output_dir/'resume.pt')
        result={'completed_iterations':completed,'num_envs':args.num_envs,'curriculum_stage':args.curriculum_stage,
                'reward_revision':args.reward_revision,'learning_rate':runner.alg.learning_rate,'contract':expected,'initialization':initialization,
                'new_update_count':args.iterations,'checkpoint_sha256':digest(args.output_dir/'resume.pt'),
                'budget_mode':'new_updates','source_iteration':profile['source_iteration']}
        atomic_json(args.output_dir/'training_result.json',result)
        print('V6_TRAINING_RESULT '+json.dumps(result),flush=True)
except BaseException as error:
    import traceback
    traceback.print_exc()
    failure={'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
             'completed_before_chunk':args.completed,'new_updates_requested':args.iterations,'stage':args.curriculum_stage}
    first_error=args.output_dir/'first_worker_error.json'
    if not first_error.exists():atomic_json(first_error,failure)
    atomic_json(args.output_dir/'worker_error.json',failure)
    raise
finally:
    if env:env.close()
    app.close()
