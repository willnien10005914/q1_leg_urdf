#!/usr/bin/env python3
"""One native lift-v5 chunk; no model publication or media upload."""
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
initialization_mode.add_argument('--bootstrap',action='store_true')
initialization_mode.add_argument('--evaluate',action='store_true')
initialization_mode.add_argument('--fork-policy',action='store_true',
    help='Initialize a new control experiment from the complete actor only; inherit the total update counter, with fresh critic/Adam/std')
p.add_argument('--episodes',type=int,default=32)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--reward-revision',type=int,choices=(0,1,2),default=0)
p.add_argument('--smoke-steps',type=int,default=0)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.completed<0 or args.iterations<1 or args.smoke_steps<0:
    p.error('Iteration counts must be nonnegative and iterations positive')
if args.bootstrap and (not args.checkpoint or args.completed or args.evaluate):
    p.error('Bootstrap requires an initialization checkpoint, completed=0 and training/smoke mode')
if args.fork_policy and (not args.checkpoint or args.completed<=0 or args.curriculum_stage!=0):
    p.error('Control fork requires a checkpoint, inherited completed>0 and curriculum-stage=0 for fresh gate validation')
if args.completed and not args.checkpoint:
    p.error('A nonzero inherited update counter requires a saved checkpoint')
if args.fork_policy and (args.output_dir/'resume.pt').resolve()==args.checkpoint.resolve():
    p.error('Control fork must retain its source checkpoint and use a separate output directory')
if args.evaluate and not args.checkpoint:
    p.error('Evaluation requires a saved checkpoint')
launcher=AppLauncher(args);app=launcher.app

import hashlib
import json
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from leg_humanoid_lab.lift_env import LiftLegEnv,LiftEnvCfg,lift_contract
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import ROOT
from runtime_compat import runner_cfg,load_checkpoint,reset_noise_std,policy_module,checkpoint_for_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def initialize_actor_hidden(runner,path):
    """Retain hidden representations only; all old residual outputs are discarded."""
    payload=torch.load(checkpoint_for_runtime(path),map_location='cpu',weights_only=False)
    model=policy_module(runner)
    current=model.state_dict()
    if 'actor_state_dict' in payload:
        source=payload['actor_state_dict']
        prefix='mlp.'
    elif 'model_state_dict' in payload:
        source={key.removeprefix('actor.'):value for key,value in payload['model_state_dict'].items() if key.startswith('actor.')}
        prefix='actor.'
        source={prefix+key:value for key,value in source.items()}
    else:
        raise ValueError('Initialization checkpoint has no supported actor parameters')
    layers=[(name,module) for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)]
    # Legacy ActorCritic contains both actor and critic: select actor MLP only.
    layers=[item for item in layers if item[0].startswith(('mlp.','actor.'))]
    if not layers:raise ValueError('Actor does not expose supported MLP output layer')
    last_name,last_layer=layers[-1]
    copied=[]
    for key,value in source.items():
        if key.startswith(prefix) and not key.startswith(last_name+'.') and key in current and current[key].shape==value.shape:
            current[key]=value;copied.append(key)
    if not copied:raise ValueError('No compatible actor hidden layers were found')
    model.load_state_dict(current)
    with torch.no_grad():
        last_layer.weight.zero_()
        if last_layer.bias is not None:last_layer.bias.zero_()
    runner.current_learning_iteration=0
    return {'source':str(path),'source_sha256':digest(path),'actor_hidden_parameters':copied,
            'actor_output_zeroed':True,'critic':'fresh','optimizer':'fresh','completed_iterations':0}


def validate_checkpoint_contract(saved,expected,*,bootstrap=False,fork_policy=False):
    """A control mismatch is an explicit fork, never an ordinary resume."""
    if bootstrap:
        keys=('urdf_sha256','joint_names','observation_size')
    else:
        keys=('robot','schema','urdf_sha256','gait_sha256','joint_names','action_size','observation_size')
        if (expected.get('robot')!='q1_leg_lift_v5' or expected.get('action_size')!=23
                or expected.get('observation_size')!=110 or len(set(expected.get('joint_names',[])))!=23):
            raise ValueError('Native lift policy requires the exact 23-joint / 110-observation interface')
        if not fork_policy:
            keys+=('control_sha256','control_profile','motor')
    if any(saved.get(key)!=expected.get(key) for key in keys) or saved.get('proxy') is not False:
        raise ValueError('Checkpoint/source contract mismatch; control changes require explicit --fork-policy initialization')
    if fork_policy and (not isinstance(saved.get('control_sha256'),str)
                        or len(saved['control_sha256'])!=64 or not saved.get('control_profile')):
        raise ValueError('Control fork needs the original control SHA256 and profile for lineage')


def validate_update_budget(profile,completed,iterations):
    """Fork initialization inherits prior work; it grants no extra updates."""
    total=completed+iterations
    if total>min(15000,int(profile.get('max_iterations',15000))):
        raise ValueError('Inherited plus new PPO updates would exceed the total authorized training budget')
    return total


def initialize_control_fork(runner,path,expected,completed,exploration_std):
    """Keep every actor MLP layer; initialize no critic, optimizer or noise state.

    The caller constructs a new runner. The source's actual serialized iter is
    inherited for the user's total budget, even though Adam and curriculum
    validation restart. A runtime-format conversion may translate parameter
    names but may not invent or change that source update counter.
    """
    path=Path(path)
    contract_path=path.parent/'contract.json'
    saved=json.loads(contract_path.read_text())
    validate_checkpoint_contract(saved,expected,fork_policy=True)
    if type(completed) is not int or completed<=0:
        raise ValueError('Control fork needs a positive inherited completed update count')
    original=torch.load(path,map_location='cpu',weights_only=False)
    if (not isinstance(original,dict) or type(original.get('iter')) is not int
            or original['iter']!=completed):
        raise ValueError('Control fork source checkpoint iter does not match inherited completed update count')
    infos=original.get('infos')
    if isinstance(infos,dict) and 'completed_iterations' in infos and infos['completed_iterations']!=completed:
        raise ValueError('Control fork checkpoint infos contradict its actual serialized iter')
    optimizer=getattr(getattr(runner,'alg',None),'optimizer',getattr(runner,'optimizer',None))
    if optimizer is None or optimizer.state:
        raise ValueError('Control fork requires a newly constructed runner with a fresh optimizer')
    compatible=Path(checkpoint_for_runtime(path))
    payload=original if compatible.resolve()==path.resolve() else torch.load(compatible,map_location='cpu',weights_only=False)
    if type(payload.get('iter')) is not int or payload['iter']!=completed:
        raise ValueError('Runtime conversion changed the inherited checkpoint update count')
    if 'actor_state_dict' in payload:
        source={key.removeprefix('mlp.'):value for key,value in payload['actor_state_dict'].items() if key.startswith('mlp.')}
    elif 'model_state_dict' in payload:
        source={key.removeprefix('actor.'):value for key,value in payload['model_state_dict'].items() if key.startswith('actor.')}
    else:
        raise ValueError('Control fork checkpoint has no supported actor MLP parameters')
    model=policy_module(runner)
    current=model.state_dict()
    prefixes={name.split('.',1)[0]+'.' for name,module in model.named_modules()
              if isinstance(module,torch.nn.Linear) and name.startswith(('mlp.','actor.'))}
    if len(prefixes)!=1:
        raise ValueError('Control fork actor does not expose one supported complete MLP')
    prefix=next(iter(prefixes))
    copied=[]
    for key,value in current.items():
        if not key.startswith(prefix):continue
        incoming=source.get(key.removeprefix(prefix))
        if incoming is None or incoming.shape!=value.shape or not torch.isfinite(incoming).all():
            raise ValueError(f'Control fork cannot preserve complete actor MLP parameter {key}')
        current[key]=incoming;copied.append(key)
    if not copied:
        raise ValueError('Control fork found no compatible actor MLP parameters')
    model.load_state_dict(current)
    # The source distribution, normalizers, critic and Adam are deliberately
    # excluded. This is the only fork-specific explicit exploration reset.
    reset_noise_std(runner,exploration_std)
    runner.current_learning_iteration=completed
    return {'policy':'control_fork','source':str(path.resolve()),'source_sha256':digest(path),
            'source_contract':str(contract_path.resolve()),'source_contract_sha256':digest(contract_path),
            'source_control_sha256':saved['control_sha256'],'source_control_profile':saved['control_profile'],
            'source_payload_iteration':original['iter'],'inherited_completed_iterations':completed,
            'completed_iterations':completed,'new_control_sha256':expected['control_sha256'],
            'new_control_profile':expected['control_profile'],'actor_mlp_parameters':copied,
            'actor_output_zeroed':False,'critic':'fresh','optimizer':'fresh',
            'exploration_std_reset':True,'exploration_std':float(exploration_std),
            'curriculum_restarts_at':0,'acceptance_inherited':False}


env=None
try:
    args.output_dir.mkdir(parents=True,exist_ok=True)
    profile=json.loads((args.training_profile or ROOT/'config/lift_v5.json').read_text())
    if not args.evaluate and not args.smoke_steps:
        validate_update_budget(profile,args.completed,args.iterations)
    cfg=LiftEnvCfg()
    cfg.scene.num_envs=args.episodes if args.evaluate else args.num_envs
    cfg.training_mode=not args.evaluate
    cfg.curriculum_stage=args.curriculum_stage
    cfg.episode_length_s=20.5 if args.evaluate else float(profile.get('training_episode_lengths_s',[8,8,20])[args.curriculum_stage])
    cfg.reward_revision=args.reward_revision
    cfg.seed=args.seed;cfg.sim.device=args.device
    gait_path=(ROOT/profile.get('gait_file','config/v5_gaits_lift6.json')).resolve()
    # LiftEnvCfg owns its native geometry/control defaults; configure its gait
    # path only through the explicit field when the environment exposes one.
    if hasattr(cfg,'gait_file'):cfg.gait_file=str(gait_path)
    env=LiftLegEnv(cfg)
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
    expected=lift_contract()
    if expected.get('robot')!='q1_leg_lift_v5' or expected.get('gait_sha256')!=digest(gait_path):
        raise ValueError('Lift environment does not match the declared v5 gait/robot contract')
    initialization=None
    if args.checkpoint:
        saved=json.loads((args.checkpoint.parent/'contract.json').read_text())
        validate_checkpoint_contract(saved,expected,bootstrap=args.bootstrap,fork_policy=args.fork_policy)
        if args.bootstrap:
            initialization=initialize_actor_hidden(runner,args.checkpoint)
        elif args.fork_policy:
            initialization=initialize_control_fork(runner,args.checkpoint,expected,args.completed,profile['exploration_std'])
            initialization['new_update_count']=args.iterations if not args.smoke_steps else 0
        else:
            # Cross-runtime compatibility conversions intentionally discard Adam.
            # Refuse those conversions for actual training resume.
            compatible=checkpoint_for_runtime(args.checkpoint)
            if not args.evaluate and Path(compatible).resolve()!=args.checkpoint.resolve():
                raise ValueError('True training resume requires identical RSL checkpoint format to retain Adam')
            load_checkpoint(runner,args.checkpoint,load_optimizer=not args.evaluate,load_iteration=True)
            if not args.evaluate and runner.current_learning_iteration!=args.completed:
                raise ValueError('Checkpoint iteration does not match completed update count')
    if args.evaluate:
        if args.curriculum_stage<2:
            from leg_humanoid_lab.lift_evaluation import evaluate_lift
            report=evaluate_lift(runner,wrapped,env,episodes=args.episodes,seed=args.seed,stage=args.curriculum_stage)
        else:
            from leg_humanoid_lab.interactive_evaluation import evaluate_interactive
            report=evaluate_interactive(runner,wrapped,env,episodes=args.episodes,seed=args.seed)
        source_dir=ROOT/'source/leg_humanoid_lab/leg_humanoid_lab'
        report.update(checkpoint_sha256=digest(args.checkpoint),checkpoint=str(args.checkpoint),completed_iterations=args.completed,
                      stage=args.curriculum_stage,evaluation_seed=args.seed,environment_sha256=digest(source_dir/'lift_env.py'),
                      base_environment_sha256=digest(source_dir/'interactive_env.py'),urdf_sha256=expected['urdf_sha256'],gait_sha256=expected['gait_sha256'])
        destination=args.output_dir/f'evaluation_{args.completed}_s{args.curriculum_stage}_{args.seed}.json'
        atomic_json(destination,report)
        print('V5_EVALUATION '+json.dumps({key:value for key,value in report.items() if key not in ('rollout','per_episode')}),flush=True)
    elif args.smoke_steps:
        wrapped.reset();seen_modes=set();peak=0.
        active=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
        failed=torch.zeros_like(active)
        actual_peak=torch.zeros((env.num_envs,2),device=env.device)
        unloaded_peak=actual_peak.clone()
        for _ in range(args.smoke_steps):
            _,reward,done,_=wrapped.step(torch.zeros((env.num_envs,23),device=env.device))
            failed|=active&env.reset_terminated.bool()
            active&=~done.bool()
            height,_,loaded,_=env.foot_step_metrics()
            actual_peak=torch.maximum(actual_peak,torch.where(active[:,None],height,0.).clamp_min(0))
            unloaded_peak=torch.maximum(unloaded_peak,torch.where(active[:,None]&~loaded,height,0.).clamp_min(0))
            obs=env._get_observations()['policy']
            if not torch.isfinite(reward).all() or not torch.isfinite(obs).all():raise RuntimeError('Nonfinite native smoke rollout')
            seen_modes.update(env.mode.unique().cpu().tolist())
            peak=max(peak,float(env.reference_height.max()))
        report={'envs':env.num_envs,'steps':args.smoke_steps,'stage':args.curriculum_stage,'finite':True,
                'observation_shape':list(obs.shape),'seen_modes':sorted(seen_modes),'peak_reference_height_m':peak,
                'zero_residual_survival_fraction':float((~failed).float().mean()),'active_first_episode_fraction_at_end':float(active.float().mean()),'actual_peak_sole_m_per_foot_mean':actual_peak.mean(0).cpu().tolist(),
                'actual_unloaded_peak_m_per_foot_mean':unloaded_peak.mean(0).cpu().tolist(),'first_episode_only':True,'contract':expected}
        atomic_json(args.output_dir/'smoke.json',report)
        print('V5_SMOKE '+json.dumps(report),flush=True)
    else:
        atomic_json(args.output_dir/'contract.json',expected)
        if not args.checkpoint or args.bootstrap:
            reset_noise_std(runner,profile['exploration_std'])
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
                    'reward_revision':args.reward_revision,'initialization':initialization})
        temporary.replace(args.output_dir/'resume.pt')
        result={'completed_iterations':completed,'num_envs':args.num_envs,'curriculum_stage':args.curriculum_stage,
                'reward_revision':args.reward_revision,'learning_rate':runner.alg.learning_rate,'contract':expected,'initialization':initialization,
                'new_update_count':args.iterations,'checkpoint_sha256':digest(args.output_dir/'resume.pt')}
        atomic_json(args.output_dir/'training_result.json',result)
        print('V5_TRAINING_RESULT '+json.dumps(result),flush=True)
except BaseException as error:
    import traceback
    traceback.print_exc()
    atomic_json(args.output_dir/'worker_error.json',{'type':type(error).__name__,'error':str(error)})
    raise
finally:
    if env:env.close()
    app.close()
