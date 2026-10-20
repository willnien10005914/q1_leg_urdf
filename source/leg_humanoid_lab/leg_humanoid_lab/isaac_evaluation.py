"""First-episode evaluation; reset frames are excluded from the saved playback."""
import json
import hashlib
import numpy as np
import torch
from .robot import ROOT,JOINTS,contract
from .evaluation import passes

@torch.inference_mode()
def evaluate(runner,wrapped,env,episodes=8,checkpoint=None,seed=10000,publish=True):
    torch.manual_seed(seed)
    obs=wrapped.reset()[0]
    count=min(episodes,env.num_envs)
    policy=runner.get_inference_policy(device=env.device)
    active=np.ones(count,dtype=bool)
    survived=np.zeros(count,dtype=bool)
    history=[[] for _ in range(count)]
    lateral=[[] for _ in range(count)]
    frames=[]
    starts=env.robot.data.root_pos_w[:count,:2].clone()
    duration=np.zeros(count)
    displacement=np.zeros(count)
    foot_peak=torch.zeros((count,2),device=env.device)
    lift_hits=torch.zeros_like(foot_peak);lift_trials=torch.zeros_like(foot_peak)
    slip_sum=slip_count=contact_sum=contact_count=0.
    for step in range(env.max_episode_length):
        before=active.copy()
        up,height,pose,velocity,root=env._errors()
        values=torch.stack((up,height,pose.sqrt(),velocity.sqrt(),root.sqrt()),dim=-1)[:count].cpu().numpy()
        if env.cfg.skill=='sidestep':
            clearance,speed,loaded,target=env.foot_step_metrics()
            alive=torch.as_tensor(active,device=env.device)[:,None]
            clearance,speed,loaded=clearance[:count],speed[:count],loaded[:count]
            foot_peak=torch.maximum(foot_peak,torch.where(alive,clearance,0.))
            if target is not None:
                target=target[:count]
                peak=(target>=env.foot_objective['peak_target_m'])&alive
                lift_trials+=peak
                lift_hits+=peak&(clearance>=env.foot_objective['success_clearance_m'])&~loaded
                stance=(target<=.005)&alive
                slip_sum+=(speed*stance*loaded).sum()
                slip_count+=(stance*loaded).sum()
                contact_sum+=((loaded==stance)&alive).sum()
                contact_count+=alive.sum()*2
        for i in np.flatnonzero(active):
            history[i].append(values[i])
            lateral[i].append(float(env.robot.data.root_pos_w[i,1]-env.scene.env_origins[i,1]))
        if active[0]:
            data=env.robot.data
            frames.append({"time":step*env.step_dt,"q":data.joint_pos[0,env.jids].cpu().tolist(),
                "root":(data.root_pos_w[0]-env.scene.env_origins[0]).cpu().tolist(),"quaternion":data.root_quat_w[0].cpu().tolist()})
        with torch.inference_mode():
            action=policy(obs)
            obs,_,done,_=wrapped.step(action)
        ended=done[:count].cpu().numpy().astype(bool)&before
        failed=env.reset_terminated[:count].cpu().numpy().astype(bool)
        # Track displacement before a simulator auto-reset can erase it.
        current=(env.robot.data.root_pos_w[:count,:2]-starts).norm(dim=-1).cpu().numpy()
        for i in np.flatnonzero(before):
            duration[i]=(step+1)*env.step_dt
            if not ended[i]:
                displacement[i]=current[i]
        survived[ended]=~failed[ended]
        active[ended]=False
        if not active.any():
            break
    survived[active]=True
    metrics=np.array([np.mean(h,axis=0) for h in history])
    report={"skill":env.cfg.skill,"backend":"isaaclab","proxy":env.cfg.allow_proxy,"episodes":count,
        "evaluation_seed":seed,
        "urdf_sha256":contract()['urdf_sha256'],
        "survival_rate":float(survived.mean()),"duration":float(duration.mean()),"displacement":float(displacement.mean()),
        **dict(zip(("upright","height_error","pose_rmse","velocity_error","root_error"),metrics.mean(axis=0).tolist()))}
    if env.cfg.skill=='sidestep':
        report['lateral_range']=float(np.mean([np.ptp(path) for path in lateral]))
        report['reference_lateral_range']=float(np.ptp(env.motion.root[:,1]))
        report['foot_peak_clearance_m_per_foot']=foot_peak.mean(dim=0).cpu().tolist()
        if env.foot_targets is not None:
            report['foot_lift_success_rate_per_foot']=(lift_hits.sum(0)/lift_trials.sum(0).clamp(min=1)).cpu().tolist()
            report['stance_slip_m_s']=float(slip_sum/max(float(slip_count),1))
            report['planned_contact_match']=float(contact_sum/max(float(contact_count),1))
    report["passed"]=passes(report)
    report["scope"]="Proxy simulation" if env.cfg.allow_proxy else "Native CAD simulation; real hardware unvalidated"
    if checkpoint is not None:
        report['checkpoint']=checkpoint.name
        report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if publish:
        file=ROOT/"checkpoints/isaac"/env.cfg.skill/"evaluation.json"
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(report,indent=2)+"\n")
        rollout=f"/outputs/isaac_{env.cfg.skill}_policy_rollout.json"
        (ROOT/rollout.lstrip('/')).write_text(json.dumps({"source":"ppo_isaaclab","proxy":env.cfg.allow_proxy,"skill":env.cfg.skill,"joint_names":list(JOINTS),"passed":report["passed"],"frames":frames}))
    print(json.dumps(report,indent=2),flush=True)
    return report
