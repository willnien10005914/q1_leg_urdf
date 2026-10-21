#!/usr/bin/env python3
"""Causal native-PhysX ablation; never promotes or hides failed episodes."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--alternate-gaits',type=Path)
p.add_argument('--expanded-control',action='store_true')
p.add_argument('--replicates',type=int,default=4)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();launcher=AppLauncher(args);app=launcher.app
import hashlib,json,torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from leg_humanoid_lab.interactive_env import InteractiveLegEnv,InteractiveEnvCfg
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import JOINTS
from runtime_compat import runner_cfg,load_checkpoint

variants=[
 {'name':'candidate_v4','actor':True,'kp':160.,'kd':8.,'cap':.35,'imu':1.},
 {'name':'reference_only','actor':False,'kp':160.,'kd':8.,'cap':.35,'imu':1.},
 {'name':'candidate_swing_cap','actor':True,'kp':160.,'kd':8.,'cap':.05,'imu':.2},
 {'name':'reference_kp400','actor':False,'kp':400.,'kd':20.,'cap':.35,'imu':1.},
 {'name':'candidate_kp400','actor':True,'kp':400.,'kd':20.,'cap':.35,'imu':1.},
 {'name':'candidate_cap_kp400','actor':True,'kp':400.,'kd':20.,'cap':.05,'imu':.2},
 {'name':'reference_kp800','actor':False,'kp':800.,'kd':32.,'cap':.35,'imu':1.},
 {'name':'candidate_cap_kp800','actor':True,'kp':800.,'kd':32.,'cap':.05,'imu':.2},
 {'name':'reference_no_feedback','actor':False,'kp':400.,'kd':20.,'cap':.35,'imu':0.},
]
if args.expanded_control:
 variants += [
  {'name':'reference_kp800_swing_imu_weak','actor':False,'kp':800.,'kd':32.,'cap':.35,'imu':.2},
  {'name':'reference_kp800_velocity_ff','actor':False,'kp':800.,'kd':32.,'cap':.35,'imu':.2,'velocity_ff':True},
  {'name':'reference_torque_gravity400','actor':False,'kp':400.,'kd':20.,'cap':.35,'imu':.2,'explicit':True,'gravity_ff':True,'velocity_ff':True},
  {'name':'reference_torque_gravity800','actor':False,'kp':800.,'kd':32.,'cap':.35,'imu':.2,'explicit':True,'gravity_ff':True,'velocity_ff':True},
  {'name':'candidate_cap_torque_gravity400','actor':True,'kp':400.,'kd':20.,'cap':.05,'imu':.2,'explicit':True,'gravity_ff':True,'velocity_ff':True},
  {'name':'reference_torque_gravity400_d8','actor':False,'kp':400.,'kd':8.,'cap':.35,'imu':.2,'explicit':True,'gravity_ff':True,'velocity_ff':True},
  {'name':'reference_implicit_gravity400','actor':False,'kp':400.,'kd':20.,'cap':.35,'imu':.2,'implicit_gravity_ff':True,'velocity_ff':True},
  {'name':'reference_implicit_gravity800','actor':False,'kp':800.,'kd':32.,'cap':.35,'imu':.2,'implicit_gravity_ff':True,'velocity_ff':True},
  {'name':'candidate_cap_implicit_gravity400','actor':True,'kp':400.,'kd':20.,'cap':.02,'imu':.2,'implicit_gravity_ff':True,'velocity_ff':True},
 ]
if args.alternate_gaits:
 variants += [dict(v,name=v['name']+'_alternate',alternate=True) for v in list(variants) if v['name'] in ('reference_kp400','candidate_cap_kp400','reference_kp800','candidate_cap_kp800') or v.get('velocity_ff')]
count=len(variants)*3*args.replicates

class ProbeEnv(InteractiveLegEnv):
 allow_reset=True
 def _reset_idx(self,ids):
  if self.allow_reset:super()._reset_idx(ids)
 def _sample(self,table,indices,phase):
  value=super()._sample(table,indices,phase)
  if hasattr(self,'alternate_tables') and id(table) in self.alternate_tables:
   other=super()._sample(self.alternate_tables[id(table)],indices,phase)
   shape=(self.num_envs,)+(1,)*(value.ndim-1)
   value=torch.where(self.use_alternate.reshape(shape),other,value)
  return value
 def _pre_physics_step(self,actions):
  old_reference=self.refq.clone()
  if hasattr(self,'use_actor'):actions=actions*self.use_actor[:,None]
  super()._pre_physics_step(actions)
  if not hasattr(self,'swing_caps'):return
  feedback=self.target-self.refq-.35*self.actions
  for foot,side in enumerate(('left','right')):
   swing=(~self.reference_contact[:,foot])&(self.reference_height[:,foot]>.008)
   ids=[JOINTS.index(f'{side}_{part}_joint') for part in ('hip_Pitch','hip_Roll','hip_Yaw','knee_Pitch','ankle_Pitch','ankle_Roll')]
   scale=torch.where(swing,self.swing_caps,.35)
   self.target[:,ids]=self.refq[:,ids]+scale[:,None]*self.actions[:,ids]+feedback[:,ids]*torch.where(swing,self.swing_imu,1.)[:,None]
  self.target=self.target.clamp(self.lower,self.upper)
  self.target_velocity=((self.refq-old_reference)/self.step_dt).clamp(-10,10)*self.velocity_ff[:,None]
 def _apply_action(self):
  super()._apply_action()
  if not hasattr(self,'explicit_control'):return
  self.robot.set_joint_velocity_target(self.target_velocity,joint_ids=self.jids)
  q=self.robot.data.joint_pos;qd=self.robot.data.joint_vel
  desired=self.robot.data.default_joint_pos.clone();desired[:,self.jids]=self.target
  desired_velocity=torch.zeros_like(desired);desired_velocity[:,self.jids]=self.target_velocity
  torque=self.controller_stiffness*(desired-q)+self.controller_damping*(desired_velocity-qd)
  gravity=self.robot.root_physx_view.get_gravity_compensation_forces()[:,-self.robot.num_joints:]
  torque+=gravity*self.gravity_ff[:,None]
  torque=torque.clamp(-100,100)*self.explicit_control[:,None]
  # Implicit drive is limited to 60 Nm, plus at most 40 Nm of joint-only
  # gravity feedforward. The triangle bound keeps the total at <=100 Nm.
  torque+=gravity.clamp(-40,40)*self.implicit_gravity_ff[:,None]
  self.robot.set_joint_effort_target(torque)

env=None
try:
 cfg=InteractiveEnvCfg();cfg.scene.num_envs=count;cfg.training_mode=False;cfg.episode_length_s=10.;cfg.seed=17421;cfg.sim.device=args.device
 env=ProbeEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=1.);env.allow_reset=False
 device=env.device
 variant=torch.arange(count,device=device)//(3*args.replicates)
 task=(torch.arange(count,device=device)//args.replicates)%3
 env.use_actor=torch.tensor([v['actor'] for v in variants],device=device)[variant]
 env.swing_caps=torch.tensor([v['cap'] for v in variants],device=device)[variant]
 env.swing_imu=torch.tensor([v['imu'] for v in variants],device=device)[variant]
 env.velocity_ff=torch.tensor([v.get('velocity_ff',False) for v in variants],device=device)[variant]
 env.explicit_control=torch.tensor([v.get('explicit',False) for v in variants],device=device)[variant]
 env.gravity_ff=torch.tensor([v.get('gravity_ff',False) for v in variants],device=device)[variant]
 env.implicit_gravity_ff=torch.tensor([v.get('implicit_gravity_ff',False) for v in variants],device=device)[variant]
 env.use_alternate=torch.tensor([v.get('alternate',False) for v in variants],device=device)[variant]
 if args.alternate_gaits:
  data=json.loads(args.alternate_gaits.read_text());plans=data['plans']
  env.alternate_tables={id(table):torch.tensor([plans[n][key] for n in env.names],device=device,dtype=torch.float32)
      for table,key in ((env.qtable,'q'),(env.htable,'foot_height'),(env.ctable,'contact'),(env.ptable,'foot_position_body'),(env.rtable,'root_offset_body'))}
 stiffness=env.robot.data.default_joint_stiffness.clone();damping=env.robot.data.default_joint_damping.clone()
 legs=[i for i,n in enumerate(JOINTS) if any(s in n for s in ('hip_','knee_','ankle_'))]
 knee=[JOINTS.index(s+'_knee_Pitch_joint') for s in ('left','right')]
 kp=torch.tensor([v['kp'] for v in variants],device=device)[variant]
 kd=torch.tensor([v['kd'] for v in variants],device=device)[variant]
 for i in legs:stiffness[:,env.jids[i]]=kp;damping[:,env.jids[i]]=kd
 env.controller_stiffness=stiffness.clone();env.controller_damping=damping.clone()
 stiffness[env.explicit_control]=0;damping[env.explicit_control]=0
 env.robot.write_joint_stiffness_to_sim(stiffness);env.robot.write_joint_damping_to_sim(damping)
 actuator=env.robot.actuators['all'];actuator.stiffness.copy_(stiffness);actuator.damping.copy_(damping)
 effort=env.robot.data.joint_effort_limits.clone();effort[env.implicit_gravity_ff]=60.
 env.robot.write_joint_effort_limit_to_sim(effort);actuator.effort_limit.copy_(effort)
 runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=device)
 load_checkpoint(runner,args.checkpoint,load_optimizer=False,load_iteration=False)
 policy=runner.get_inference_policy(device=device)
 active=torch.ones(count,device=device,dtype=torch.bool);samples=torch.zeros(count,device=device)
 peak=torch.zeros((count,2),device=device);unloaded_peak=peak.clone();reference_peak=peak.clone();swing_samples=peak.clone();lift_samples=peak.clone()
 minimum_height=torch.full((count,),10.,device=device);swing_root_sum=torch.zeros(count,device=device);swing_root_count=swing_root_sum.clone()
 tracking_sum=torch.zeros((count,2),device=device);target_residual_sum=tracking_sum.clone();knee_count=tracking_sum.clone();torque_peak=torch.zeros(count,device=device)
 failure_time=torch.full((count,),8.,device=device)
 snapshots=[]
 with torch.inference_mode():
  for step in range(400):
   if step%100==0:print('PROBE_STEP '+str(step),flush=True)
   t=step*env.step_dt
   mode=torch.zeros(count,device=device,dtype=torch.long);command=torch.zeros((count,3),device=device);phase=torch.zeros(count,device=device)
   if 1<=t<6:mode[task==0]=1;command[task==0,0]=.22
   if 1<=t<2.6:
    mode[task==1]=2;mode[task==2]=3;phase[task>0]=(t-1)/1.6
   elif 2.6<=t<3.6:mode[task>0]=4
   env.set_manual_control(mode,command,phase)
   _obs,_reward,done,_info=wrapped.step(policy(wrapped.get_observations()))
   failed=done.bool()&active;failure_time[failed]=t+env.step_dt;active&=~done.bool()
   height,speed,loaded,target=env.foot_step_metrics();d=env.robot.data
   mask=active[:,None];swing=(target>.022)&(~env.reference_contact)&mask
   peak=torch.maximum(peak,torch.where(mask,height,0.).clamp(min=0))
   unloaded_peak=torch.maximum(unloaded_peak,torch.where(mask&~loaded,height,0.).clamp(min=0))
   reference_peak=torch.maximum(reference_peak,torch.where(mask,target,0.))
   swing_samples+=swing;lift_samples+=swing&(height>=.025)&~loaded
   rootheight=d.root_pos_w[:,2]-env.scene.env_origins[:,2]
   minimum_height=torch.minimum(minimum_height,torch.where(active,rootheight,10.))
   in_swing=swing.any(-1);swing_root_sum+=rootheight*in_swing;swing_root_count+=in_swing
   actual=d.joint_pos[:,env.jids][:,knee];targetknees=env.target[:,knee]
   tracking_sum+=(targetknees-actual).abs()*swing;target_residual_sum+=(targetknees-env.refq[:,knee]).abs()*swing;knee_count+=swing
   torque_peak=torch.maximum(torque_peak,d.applied_torque[:,[env.jids[i] for i in legs]].abs().max(-1).values*active)
   samples+=active
   if step in (65,75,85,100):
    i=torch.arange(0,count,args.replicates,device=device)
    snapshots.append({'time':t+env.step_dt,'env_index':i.cpu().tolist(),'alive':active[i].cpu().tolist(),'root_height':rootheight[i].cpu().tolist(),'root_quaternion':d.root_quat_w[i].cpu().tolist(),'root_position':(d.root_pos_w[i]-env.scene.env_origins[i]).cpu().tolist(),'foot_position':(d.body_pos_w[i][:,env.foot_body_ids]-env.scene.env_origins[i,None,:]).cpu().tolist(),'foot_quaternion':d.body_quat_w[i][:,env.foot_body_ids].cpu().tolist(),'clearance':height[i].cpu().tolist(),'force_z':env.contacts.data.net_forces_w[i][:,env.feet,2].cpu().tolist(),'q_reference':env.refq[i].cpu().tolist(),'q_target':env.target[i].cpu().tolist(),'q_actual':d.joint_pos[i][:,env.jids].cpu().tolist()})
 summaries=[]
 for v,spec in enumerate(variants):
  for task_id,name in enumerate(('forward','sidestep_left','sidestep_right')):
   mask=(variant==v)&(task==task_id)
   entry={'variant':spec,'task':name,'episodes':int(mask.sum()),'survival':float(active[mask].float().mean()),'duration_mean_s':float(failure_time[mask].mean()),'peak_sole_m':peak[mask].mean(0).cpu().tolist(),'peak_unloaded_sole_m':unloaded_peak[mask].mean(0).cpu().tolist(),'reference_peak_m':reference_peak[mask].mean(0).cpu().tolist(),'swing_success_per_foot':(lift_samples[mask].sum(0)/swing_samples[mask].sum(0).clamp_min(1)).cpu().tolist(),'minimum_pelvis_height_m':float(minimum_height[mask].mean()),'swing_pelvis_height_m':float((swing_root_sum[mask]/swing_root_count[mask].clamp_min(1)).mean()),'knee_target_tracking_error_rad':(tracking_sum[mask].sum(0)/knee_count[mask].sum(0).clamp_min(1)).cpu().tolist(),'knee_target_reference_residual_rad':(target_residual_sum[mask].sum(0)/knee_count[mask].sum(0).clamp_min(1)).cpu().tolist(),'peak_leg_torque_nm':float(torque_peak[mask].mean())}
   summaries.append(entry);print('PROBE '+json.dumps(entry),flush=True)
 report={'backend':'isaaclab','proxy':False,'accepted':False,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'native_motor_effort_limit_nm':100.,'episodes':count,'duration_s':8.,'first_episode_only':True,'reset_after_start':False,'summaries':summaries,'snapshots':snapshots}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
except BaseException as error:
 import traceback
 traceback.print_exc()
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.with_suffix('.error.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__})+'\n')
 raise
finally:
 if env:env.close()
 app.close()
