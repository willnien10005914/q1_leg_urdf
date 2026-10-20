#!/usr/bin/env python3
"""Play a confirmed native PPO policy in Isaac Sim, optionally recording one episode."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--skill',choices=['stand','sidestep'],default='sidestep')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--video',type=Path,help='Record one finite episode to this MP4')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.video:args.enable_cameras=True
launcher=AppLauncher(args)
app=launcher.app

import hashlib
import json
import time
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from leg_humanoid_lab.isaac_env import IsaacLegEnv,LegEnvCfg
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import ROOT,contract
from runtime_compat import runner_cfg,load_checkpoint

env=None
writer=None
try:
    folder=ROOT/'checkpoints/isaac'/args.skill
    checkpoint=folder/'latest.pt'
    saved=json.loads((folder/'contract.json').read_text())
    evaluation=json.loads((folder/'evaluation.json').read_text())
    expected=contract()
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not (evaluation.get('passed') and evaluation.get('proxy') is False and
            evaluation.get('episodes',0)>=32 and evaluation.get('checkpoint_sha256')==digest and
            evaluation.get('urdf_sha256')==expected['urdf_sha256'] and saved.get('proxy') is False and
            all(saved.get(k)==expected[k] for k in ('robot','urdf_sha256','joint_names','observation_size'))):
        raise ValueError('A matching native policy with a passed 32-episode evaluation is required')
    cfg=LegEnvCfg()
    cfg.skill=args.skill
    cfg.episode_length_s=7.8 if args.skill=='sidestep' else 10.
    cfg.scene.num_envs=1
    cfg.seed=args.seed
    cfg.sim.device=args.device
    cfg.viewer.eye=(3.,-2.6,1.9)
    cfg.viewer.lookat=(0.,.10,.75)
    cfg.viewer.resolution=(1280,720)
    env=IsaacLegEnv(cfg,render_mode='rgb_array' if args.video else None)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=args.device)
    load_checkpoint(runner,checkpoint,load_optimizer=False,load_iteration=False)
    policy=runner.get_inference_policy(device=env.device)
    torch.manual_seed(args.seed)
    obs=wrapped.reset()[0]
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(str(args.video),fps=50,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
        for _ in range(12):
            env.sim.render()
            env.render()
    lateral=[]
    failed=False
    frames=0
    with torch.inference_mode():
        while app.is_running():
            start=time.monotonic()
            if writer:
                # The viewport annotator is not an Isaac Lab RTX sensor. In
                # headless mode env.step() therefore does not render for it.
                env.sim.render()
                frame=env.render(recompute=True)
                writer.append_data(np.asarray(frame))
                lateral.append(float(env.robot.data.root_pos_w[0,1]-env.scene.env_origins[0,1]))
                frames+=1
                if env.episode_length_buf[0]>=env.max_episode_length-2:
                    break
            obs,_,done,_=wrapped.step(policy(obs))
            if bool(done[0]):
                if writer:
                    failed=bool(env.reset_terminated[0])
                    break
            if not args.headless:
                time.sleep(max(0.,env.step_dt-(time.monotonic()-start)))
    if writer:
        writer.close();writer=None
        metadata={'skill':args.skill,'source':'live native Isaac Sim PhysX PPO','checkpoint':str(checkpoint),
                  'checkpoint_sha256':digest,'selected_from':evaluation.get('selected_from'),'seed':args.seed,
                  'fps':50,'frames':frames,'duration':frames/50,'survived':not failed,
                  'lateral_range_m':float(np.ptp(lateral)) if lateral else 0.,'evaluation':evaluation}
        args.video.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('VIDEO_RESULT '+json.dumps(metadata),flush=True)
        if failed:raise RuntimeError('Recorded episode fell; inspect video metadata')
finally:
    if writer:writer.close()
    if env:env.close()
    app.close()
