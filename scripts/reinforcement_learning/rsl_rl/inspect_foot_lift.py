#!/usr/bin/env python3
"""Evaluate old weights against new foot goals without publishing or training."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--episodes',type=int,default=32)
p.add_argument('--seed',type=int,default=60000)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
launcher=AppLauncher(args)
app=launcher.app
import hashlib
import json
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from leg_humanoid_lab.isaac_env import IsaacLegEnv,LegEnvCfg
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import ROOT,contract
from leg_humanoid_lab.isaac_evaluation import evaluate
from runtime_compat import runner_cfg,load_checkpoint
env=None
try:
    folder=ROOT/'checkpoints/isaac/sidestep'
    saved=json.loads((folder/'contract.json').read_text());expected=contract()
    if saved.get('proxy') is not False or any(saved.get(k)!=expected[k] for k in ('robot','urdf_sha256','joint_names','observation_size')):
        raise ValueError('Native checkpoint contract mismatch')
    cfg=LegEnvCfg();cfg.scene.num_envs=args.episodes;cfg.skill='sidestep';cfg.episode_length_s=7.8;cfg.seed=args.seed;cfg.sim.device=args.device
    env=IsaacLegEnv(cfg)
    if env.foot_targets is None:raise ValueError('Set Q1_LEG_CURRICULUM_CONFIG to a foot-lift profile')
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=args.device)
    checkpoint=folder/'latest.pt'
    load_checkpoint(runner,checkpoint,load_optimizer=False,load_iteration=False)
    report=evaluate(runner,wrapped,env,episodes=args.episodes,seed=args.seed,checkpoint=checkpoint,publish=False)
    out={'purpose':'Old sidestep policy assessed against proposed foot-lift goals; no training and no published policy replacement','evaluation':report}
    (ROOT/'outputs/foot_lift_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
finally:
    if env:env.close()
    app.close()
