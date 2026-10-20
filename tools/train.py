#!/usr/bin/env python3
"""PPO curriculum with persisted evaluation gates and recoverable checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/leg_humanoid_lab"))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from leg_humanoid_lab.robot import ROOT, SKILLS, contract, JOINTS
from leg_humanoid_lab.env import LegEnv
from leg_humanoid_lab.evaluation import passes

def evaluate(model,skill,episodes=8,allow_proxy=False):
    env=LegEnv(skill,allow_proxy=allow_proxy,randomize=True)
    rows=[]
    frames=[]
    for e in range(episodes):
        obs,_=env.reset(seed=10000+e)
        history=[]
        episode_frames=[{**env.frame(),"time":0.}]
        lateral=[]
        for _ in range(env.max_steps):
            action,_=model.predict(obs,deterministic=True)
            obs,_,done,timeout,info=env.step(action)
            history.append(info)
            lateral.append(float(env.data.qpos[1]))
            episode_frames.append({**env.frame(),"time":info["time"]})
            if done or timeout:
                break
        rows.append({"survived":not done,"duration":info["time"],"displacement":info["displacement"],
            "lateral_range":float(np.ptp(lateral)),
            **{k:float(np.mean([h[k] for h in history])) for k in ("upright","height_error","pose_rmse","velocity_error","root_error")}})
        if e==0:
            frames=episode_frames
    report={"skill":skill,"backend":"mujoco","proxy":allow_proxy,"episodes":episodes,"survival_rate":float(np.mean([r["survived"] for r in rows])),
        **{k:float(np.mean([r[k] for r in rows])) for k in ("duration","displacement","upright","height_error","pose_rmse","velocity_error","root_error")},"episode_metrics":rows}
    if skill=='sidestep':
        report['lateral_range']=float(np.mean([r['lateral_range'] for r in rows]))
        report['reference_lateral_range']=float(np.ptp(env.motion.root[:,1]))
    report['passed']=passes(report)
    report["scope"]="Proxy simulation only" if allow_proxy else "Native CAD convex-hull simulation; real hardware unvalidated."
    env.close()
    return report,frames

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--skill",choices=[*SKILLS,"curriculum"],default="curriculum")
    p.add_argument("--allow-proxy",action="store_true")
    p.add_argument("--steps",type=int,default=262144,help="PPO timesteps per stage per attempt")
    p.add_argument("--attempts",type=int,default=3)
    p.add_argument("--envs",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--warm-start",type=Path)
    p.add_argument("--evaluate-only",action="store_true")
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    torch.set_num_threads(1)
    stages=SKILLS if args.skill=="curriculum" else (args.skill,)
    checkpoint=args.warm_start
    status_path=ROOT/"outputs/training_status.json"
    for stage in stages:
        folder=ROOT/"checkpoints"/stage
        folder.mkdir(parents=True,exist_ok=True)
        def make_env():
            return LegEnv(stage,allow_proxy=args.allow_proxy)
        if checkpoint:
            saved=json.loads((checkpoint.parent/'contract.json').read_text())
            expected=contract()
            if any(saved.get(k)!=expected[k] for k in ('robot','urdf_sha256','joint_names','observation_size')) or saved.get('proxy')!=args.allow_proxy:
                raise ValueError('Checkpoint belongs to a different robot or geometry mode')
        env=SubprocVecEnv([make_env for _ in range(args.envs)],start_method="spawn") if args.envs>1 else DummyVecEnv([make_env])
        model=PPO.load(str(checkpoint),env=env,device=args.device) if checkpoint else PPO("MlpPolicy",env,learning_rate=3e-4,n_steps=512,batch_size=256,n_epochs=5,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.001,policy_kwargs={"net_arch":dict(pi=[256,256],vf=[256,256]),"log_std_init":-1.5},seed=args.seed,device=args.device,verbose=1)
        env.seed(args.seed)
        for attempt in range(args.attempts):
            status_path.write_text(json.dumps({"state":"evaluating" if args.evaluate_only else "training","skill":stage,"attempt":attempt+1,"proxy":args.allow_proxy,"checkpoint":str(checkpoint) if checkpoint else None},indent=2))
            if not args.evaluate_only:
                callback=CheckpointCallback(save_freq=max(32768//args.envs,1),save_path=str(folder/"snapshots"),name_prefix="ppo")
                model.learn(total_timesteps=args.steps,reset_num_timesteps=False,callback=callback)
                model.save(str(folder/"latest"))
            report,frames=evaluate(model,stage,args.episodes,args.allow_proxy)
            report["timesteps"]=int(model.num_timesteps)
            (folder/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
            (folder/"contract.json").write_text(json.dumps({**contract(),"backend":"mujoco","proxy":args.allow_proxy},indent=2)+"\n")
            (ROOT/"outputs"/f"{stage}_policy_rollout.json").write_text(json.dumps({"source":"ppo_mujoco","proxy":args.allow_proxy,"skill":stage,"joint_names":list(JOINTS),"passed":report["passed"],"frames":frames}))
            print(json.dumps(report,indent=2),flush=True)
            status_path.write_text(json.dumps({"state":"passed" if report["passed"] else "gate_failed","skill":stage,"proxy":args.allow_proxy,"evaluation":report},indent=2))
            checkpoint=folder/"latest.zip" if not args.evaluate_only else args.warm_start
            if report["passed"] or args.evaluate_only:
                break
        env.close()
        if not report["passed"]:
            print(f"{stage} did not pass. Curriculum stopped; checkpoint retained. More training or geometry tuning is required.",flush=True)
            return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
