#!/usr/bin/env python3
"""Isaac Lab / RSL-RL GPU training, with the wheel runtime discovery convention."""
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument("--skill",choices=["stand","walk","sidestep"],default="stand")
p.add_argument("--allow-proxy",action="store_true")
p.add_argument("--num_envs",type=int,default=512)
p.add_argument("--seed",type=int,default=42)
p.add_argument('--learning-rate',type=float)
p.add_argument('--schedule',choices=['fixed','adaptive'])
p.add_argument('--exploration-std',type=float)
p.add_argument('--entropy-coef',type=float)
p.add_argument("--max_iterations",type=int,default=2000)
p.add_argument("--checkpoint",type=Path)
p.add_argument("--smoke-steps",type=int,default=0)
p.add_argument("--evaluate-only",action="store_true")
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.evaluate_only and args.checkpoint is None:
    p.error('--evaluate-only requires --checkpoint')
launcher=AppLauncher(args)
app=launcher.app

import json
import shutil
from datetime import datetime
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,RslRlOnPolicyRunnerCfg,RslRlPpoActorCriticCfg,RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass
from leg_humanoid_lab.isaac_env import LegEnvCfg,IsaacLegEnv
from leg_humanoid_lab.robot import ROOT,JOINTS,contract
from runtime_compat import runner_cfg,load_checkpoint,reset_noise_std
from leg_humanoid_lab.isaac_evaluation import evaluate
from leg_humanoid_lab.agent_cfg import PPOCfg

env=None
try:
    cfg=LegEnvCfg()
    cfg.skill=args.skill
    cfg.allow_proxy=args.allow_proxy
    cfg.scene.num_envs=args.num_envs
    cfg.seed=args.seed
    cfg.sim.device=args.device
    if args.skill=="sidestep":
        cfg.episode_length_s=7.8
    env=IsaacLegEnv(cfg)
    if args.smoke_steps:
        env.reset()
        for _ in range(args.smoke_steps):
            env.step(torch.zeros((env.num_envs,len(JOINTS)),device=env.device))
        report={"backend":"isaaclab","skill":args.skill,"proxy":args.allow_proxy,"steps":args.smoke_steps,"envs":env.num_envs,"observation_shape":list(env._get_observations()["policy"].shape)}
        (ROOT/"outputs/isaac_smoke.json").write_text(json.dumps(report,indent=2))
        print(report,flush=True)
    else:
        wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
        folder=ROOT/"checkpoints/isaac"/args.skill
        folder.mkdir(parents=True,exist_ok=True)
        snapshots=list(folder.glob('model_*.pt'))
        if snapshots and not args.evaluate_only:
            history=folder/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            history.mkdir(parents=True,exist_ok=True)
            for file in snapshots:shutil.copyfile(file,history/file.name)
            for name in ('contract.json','evaluation.json','latest.pt'):
                if (folder/name).exists():shutil.copyfile(folder/name,history/name)
        agent=PPOCfg()
        agent.seed=args.seed
        if args.skill!='stand':
            agent.algorithm.entropy_coef=.0001
        if args.learning_rate is not None:agent.algorithm.learning_rate=args.learning_rate
        if args.schedule is not None:agent.algorithm.schedule=args.schedule
        if args.entropy_coef is not None:agent.algorithm.entropy_coef=args.entropy_coef
        runner=OnPolicyRunner(wrapped,runner_cfg(agent),log_dir=str(folder),device=args.device)
        if args.checkpoint:
            saved=json.loads((args.checkpoint.parent/'contract.json').read_text())
            expected=contract()
            for key in ('robot','urdf_sha256','joint_names','observation_size'):
                if saved.get(key)!=expected[key]:
                    raise ValueError(f'Checkpoint contract mismatch: {key}. Use weights from this native robot revision.')
            if saved.get('proxy')!=args.allow_proxy:
                raise ValueError('Do not mix native CAD and proxy checkpoints')
            load_checkpoint(runner,args.checkpoint,load_optimizer=False,load_iteration=False)
            if not args.evaluate_only:
                reset_noise_std(runner,args.exploration_std if args.exploration_std is not None else (.15 if args.skill=='walk' else .20))
        (folder/"contract.json").write_text(json.dumps({**contract(),"backend":"isaaclab","proxy":args.allow_proxy},indent=2))
        if not args.evaluate_only:
            runner.learn(num_learning_iterations=args.max_iterations,init_at_random_ep_len=False)
            runner.save(str(folder/"latest.pt"))
        report=evaluate(runner,wrapped,env,checkpoint=args.checkpoint if args.evaluate_only else folder/'latest.pt')
        (ROOT/'outputs/training_status.json').write_text(json.dumps({'state':'passed' if report['passed'] else 'gate_failed','skill':args.skill,'backend':'isaaclab','proxy':args.allow_proxy,'evaluation':report},indent=2)+'\n')
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
