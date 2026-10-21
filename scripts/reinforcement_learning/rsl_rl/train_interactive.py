#!/usr/bin/env python3
"""One resumable native v4 train/evaluation chunk. No policy promotion here."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=10240)
p.add_argument('--iterations',type=int,default=500)
p.add_argument('--completed',type=int,default=0)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--training-profile',type=Path)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--bootstrap-v3',action='store_true')
p.add_argument('--evaluate',action='store_true')
p.add_argument('--episodes',type=int,default=32)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--reward-revision',type=int,default=0)
p.add_argument('--smoke-steps',type=int,default=0)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
launcher=AppLauncher(args);app=launcher.app
import json
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from leg_humanoid_lab.interactive_env import InteractiveLegEnv,InteractiveEnvCfg,interactive_contract
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import ROOT
from runtime_compat import runner_cfg,load_checkpoint,reset_noise_std

env=None
try:
    args.output_dir.mkdir(parents=True,exist_ok=True)
    profile=json.loads((args.training_profile or ROOT/'config/interactive_v4.json').read_text())
    cfg=InteractiveEnvCfg()
    cfg.scene.num_envs=args.episodes if args.evaluate else args.num_envs
    cfg.training_mode=not args.evaluate
    cfg.episode_length_s=20.5 if args.evaluate else 20.
    cfg.seed=args.seed;cfg.sim.device=args.device
    cfg.curriculum_progress=min(1.,max(float(profile.get('minimum_curriculum_progress',.25)),.25+args.completed/2000.))
    cfg.reward_revision=args.reward_revision
    env=InteractiveLegEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    agent=PPOCfg();agent.seed=args.seed
    agent.algorithm.num_mini_batches=profile['num_mini_batches']
    agent.algorithm.learning_rate=profile['learning_rate']*(.7**args.reward_revision)
    agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=profile['entropy_coef']
    agent.save_interval=100
    runner=OnPolicyRunner(wrapped,runner_cfg(agent),log_dir=None if args.evaluate or args.smoke_steps else str(args.output_dir),device=args.device)
    expected=interactive_contract()
    if args.checkpoint:
        saved=json.loads((args.checkpoint.parent/'contract.json').read_text())
        keys=('urdf_sha256','joint_names','observation_size') if args.bootstrap_v3 else ('robot','urdf_sha256','gait_sha256','joint_names','observation_size')
        if any(saved.get(k)!=expected[k] for k in keys) or saved.get('proxy') is not False:raise ValueError('Checkpoint/source contract mismatch')
        if args.bootstrap_v3 and (saved.get('robot')!='q1_leg_native_v3' or args.completed):raise ValueError('v3 bootstrap is initialization only')
        load_checkpoint(runner,args.checkpoint,load_optimizer=not args.bootstrap_v3 and not args.evaluate,load_iteration=not args.bootstrap_v3)
    if args.evaluate:
        from leg_humanoid_lab.interactive_evaluation import evaluate_interactive
        report=evaluate_interactive(runner,wrapped,env,episodes=args.episodes,seed=args.seed)
        import hashlib
        report.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),checkpoint=str(args.checkpoint),completed_iterations=args.completed)
        (args.output_dir/f'evaluation_{args.completed}_{args.seed}.json').write_text(json.dumps(report,indent=2)+'\n')
        print('V4_EVALUATION '+json.dumps({k:v for k,v in report.items() if k!='rollout'}),flush=True)
    elif args.smoke_steps:
        obs=wrapped.reset()[0]
        seen_modes=set();peak_reference_height=0.
        for _ in range(args.smoke_steps):
            obs,reward,done,_=wrapped.step(torch.zeros((env.num_envs,23),device=env.device))
            assert torch.isfinite(reward).all() and torch.isfinite(env._get_observations()['policy']).all()
            seen_modes.update(env.mode.unique().cpu().tolist())
            peak_reference_height=max(peak_reference_height,float(env.reference_height.max()))
        report={'envs':env.num_envs,'steps':args.smoke_steps,'observation_shape':list(env._get_observations()['policy'].shape),'finite':True,'seen_modes':sorted(seen_modes),'peak_reference_height_m':peak_reference_height,'gait_sha256':expected['gait_sha256']}
        (args.output_dir/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
        print('V4_SMOKE '+json.dumps(report),flush=True)
    else:
        (args.output_dir/'contract.json').write_text(json.dumps(expected,indent=2)+'\n')
        previous_revision=0
        if args.checkpoint and (args.checkpoint.parent/'training_result.json').exists():
            previous_revision=json.loads((args.checkpoint.parent/'training_result.json').read_text()).get('reward_revision',0)
        if args.completed==0 or args.reward_revision>previous_revision:
            reset_noise_std(runner,profile['exploration_std']+min(args.reward_revision,1)*.03)
        # Loading Adam also restores its old param-group learning rates. Apply
        # the declared revision rate while retaining all optimizer moments.
        runner.alg.learning_rate=agent.algorithm.learning_rate
        for group in runner.alg.optimizer.param_groups:group['lr']=agent.algorithm.learning_rate
        print('V4_OPTIMIZER '+json.dumps({'learning_rate':runner.alg.learning_rate,'reward_revision':args.reward_revision,'previous_reward_revision':previous_revision,'completed_before_chunk':args.completed}),flush=True)
        runner.current_learning_iteration=args.completed
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        # RSL-RL stores the last zero-based index. Store the completed count for
        # the next chunk so no update is repeated or omitted on resume.
        completed=args.completed+args.iterations
        runner.current_learning_iteration=completed
        runner.save(str(args.output_dir/'resume.pt'),infos={'completed_iterations':completed,'reward_revision':args.reward_revision})
        (args.output_dir/'training_result.json').write_text(json.dumps({'completed_iterations':completed,'num_envs':args.num_envs,'reward_revision':args.reward_revision,'learning_rate':runner.alg.learning_rate,'contract':expected},indent=2)+'\n')
finally:
    if env:env.close()
    app.close()
