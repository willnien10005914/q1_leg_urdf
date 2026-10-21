#!/usr/bin/env python3
"""Read native first-failure states before Isaac's ordinary automatic resets.

This diagnostic does not change collisions, done rules, controls or rewards.
It accepts unpromoted v5 checkpoints and never publishes them as a success.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--contract',type=Path,help='Defaults to checkpoint sibling contract.json')
p.add_argument('--output',type=Path,required=True,help='Diagnostic JSON; no media upload')
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--curriculum-stage','--stage',dest='stage',type=int,choices=(0,1),default=0)
p.add_argument('--seed',type=int,default=271100)
p.add_argument('--trace-hz',type=float,default=10.)
p.add_argument('--full-trace',action='store_true',help='Include joints, native root pose and contact forces in every first-episode trace sample')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.num_envs<4 or args.num_envs%4:
    p.error('Use a positive multiple of four native environments')
if not 0.<args.trace_hz<=50.:
    p.error('trace-hz must be in (0,50]')
launcher=AppLauncher(args);app=launcher.app

import hashlib
import json
import math
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.lift_env import LiftLegEnv,LiftEnvCfg,lift_contract
from leg_humanoid_lab.lift_evaluation import STAGE_CASES,evaluate_lift
from leg_humanoid_lab.robot import ROOT,URDF,LINKS,JOINTS,mesh_vertices,vector
from runtime_compat import runner_cfg,load_checkpoint


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    """Preserve nonfinite-failure evidence in standards-compliant JSON."""
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {key:finite_json(item) for key,item in value.items()}
    if isinstance(value,list):return [finite_json(item) for item in value]
    return value


class DiagnoseLiftEnv(LiftLegEnv):
    """Capture the unchanged done predicate before _reset_idx overwrites it."""
    def __init__(self,cfg,**kwargs):
        self.diagnostic_active=None
        super().__init__(cfg,**kwargs)
        self.diagnostic_step=0
        self.diagnostic_active=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.diagnostic_trace=[];self.first_failures=[];self.first_terminations=[]
        self.diagnostic_trace_stride=max(1,round(1./(args.trace_hz*self.step_dt)))
        self.ankle_names=[f'{side}_{part}_link' for side in ('left','right')
                          for part in ('ankle_pitch','ankle_roll')]
        self.ankle_body_ids,_=self.robot.find_bodies(self.ankle_names,preserve_order=True)
        if len(self.ankle_body_ids)!=len(self.ankle_names):
            raise ValueError('Native ankle bodies are required for source-CAD diagnostics')
        self.ankle_vertices=[]
        for name in self.ankle_names:
            collision=LINKS[name].find('collision')
            vertices=mesh_vertices(collision.find('geometry/mesh'))
            origin=collision.find('origin')
            rotation=Rotation.from_euler('xyz',vector(origin,'rpy')).as_matrix()
            vertices=vertices@rotation.T+vector(origin)
            vertices=vertices[ConvexHull(vertices).vertices]
            self.ankle_vertices.append(torch.tensor(vertices,device=self.device,dtype=torch.float32))

    def _ankle_height(self):
        data=self.robot.data
        up=torch.zeros((self.num_envs,3),device=self.device);up[:,2]=1.
        height=[]
        for body,vertices in zip(self.ankle_body_ids,self.ankle_vertices):
            local_up=quat_apply_inverse(data.body_quat_w[:,body],up)
            height.append(data.body_pos_w[:,body,2]-self.scene.env_origins[:,2]
                          +(local_up@vertices.T).min(-1).values)
        return torch.stack(height,-1)

    def _snapshot(self,indices,full=False):
        data=self.robot.data
        clearance,speed,loaded,target=self.foot_step_metrics()
        ankle=self._ankle_height()
        height=data.root_pos_w[:,2]-self.scene.env_origins[:,2]
        desired_height=self.root_height+self.reference_root_offset[:,2]
        q=data.joint_pos[:,self.jids];qd=data.joint_vel[:,self.jids]
        actual_target_error=q-self.target;actual_reference_error=q-self.refq
        torque=data.applied_torque[:,self.jids]
        forces=self.contacts.data.net_forces_w
        rows=[]
        for index in indices:
            index=int(index)
            row={'env_index':index,'case':STAGE_CASES[args.stage][index%len(STAGE_CASES[args.stage])],
                 'time_s':self.diagnostic_step*self.step_dt,'mode':int(self.mode[index]),
                 'gait_phase':float(self.gait_phase[index].remainder(1)),
                 'gesture_phase':float(self.gesture_phase[index]),
                 'root_height_m':float(height[index]),'reference_root_height_m':float(desired_height[index]),
                 'root_height_error_m':float(height[index]-desired_height[index]),
                 'upright':float(-data.projected_gravity_b[index,2]),
                 'root_velocity_body_m_s':data.root_lin_vel_b[index].cpu().tolist(),
                 'sole_clearance_m_per_foot':clearance[index].cpu().tolist(),
                 'target_clearance_m_per_foot':target[index].cpu().tolist(),
                 'loaded_per_foot':loaded[index].cpu().tolist(),
                 'planned_contact_per_foot':self.reference_contact[index].cpu().tolist(),
                 'foot_link_horizontal_speed_m_s':speed[index].cpu().tolist(),
                 'ankle_source_cad_min_height_m':dict(zip(self.ankle_names,ankle[index].cpu().tolist())),
                 'joint_target_rmse_rad':float(actual_target_error[index].square().mean().sqrt()),
                 'joint_reference_rmse_rad':float(actual_reference_error[index].square().mean().sqrt()),
                 'max_estimated_applied_joint_torque_nm':float(torque[index].abs().max()),
                 'estimated_torque_near_100nm_joint_count':int((torque[index].abs()>=99.).sum())}
            if full:
                contacts=[]
                for body in self.other_bodies:
                    force=forces[index,body]
                    magnitude=float(force.norm())
                    if magnitude>25.:
                        contacts.append({'body':self.contacts.body_names[body],
                                         'force_world_n':force.cpu().tolist(),'norm_n':magnitude})
                row.update(nonfoot_contacts_over_25n=contacts,
                           failure_predicates={
                               'root_below_052m':bool(height[index]<.52),
                               'upright_below_065':bool(-data.projected_gravity_b[index,2]<.65),
                               'nonfoot_contact_over_25n':bool(contacts),
                               'nonfinite_root_state':not bool(torch.isfinite(data.root_state_w[index]).all())},
                           root_pose_world=data.root_state_w[index,:7].cpu().tolist(),
                           joint_names=list(JOINTS),joint_position_rad=q[index].cpu().tolist(),
                           joint_velocity_rad_s=qd[index].cpu().tolist(),
                           reference_joint_position_rad=self.refq[index].cpu().tolist(),
                           target_joint_position_rad=self.target[index].cpu().tolist(),
                           joint_error_actual_minus_target_rad=actual_target_error[index].cpu().tolist(),
                           joint_error_actual_minus_reference_rad=actual_reference_error[index].cpu().tolist(),
                           actuator_estimated_applied_torque_nm=torque[index].cpu().tolist(),
                           foot_force_world_n=forces[index,self.feet].cpu().tolist())
            rows.append(row)
        return rows

    def _get_dones(self):
        terminated,time_out=super()._get_dones()
        if self.diagnostic_active is not None:
            self.diagnostic_step+=1
            active=self.diagnostic_active
            failures=active&terminated.bool()
            endings=active&(terminated.bool()|time_out.bool())
            if self.diagnostic_step%self.diagnostic_trace_stride==0:
                self.diagnostic_trace.extend(self._snapshot(torch.where(active)[0].cpu().tolist(),full=args.full_trace))
            if bool(failures.any()):
                self.first_failures.extend(self._snapshot(torch.where(failures)[0].cpu().tolist(),full=True))
            for index in torch.where(endings)[0].cpu().tolist():
                self.first_terminations.append({'env_index':index,'time_s':self.diagnostic_step*self.step_dt,
                                               'terminated':bool(terminated[index]),'time_out':bool(time_out[index])})
            self.diagnostic_active[endings]=False
        return terminated,time_out


env=None
try:
    contract_path=args.contract or args.checkpoint.parent/'contract.json'
    saved=json.loads(contract_path.read_text());expected=lift_contract()
    keys=('robot','urdf_sha256','gait_sha256','control_sha256','control_profile','joint_names','observation_size')
    if saved.get('proxy') is not False or any(saved.get(key)!=expected.get(key) for key in keys):
        raise ValueError('Checkpoint contract does not match the current frozen native v5 environment/control')
    cfg=LiftEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.training_mode=False
    cfg.curriculum_stage=args.stage;cfg.episode_length_s=20.5;cfg.seed=args.seed;cfg.sim.device=args.device
    cfg.gait_file=str(ROOT/expected['gait_file'])
    env=DiagnoseLiftEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    agent=PPOCfg();agent.seed=args.seed
    runner=OnPolicyRunner(wrapped,runner_cfg(agent),log_dir=None,device=args.device)
    load_checkpoint(runner,args.checkpoint,load_optimizer=False,load_iteration=True)
    # RslRlVecEnvWrapper may reset once during construction. Begin the
    # diagnostic collection immediately before the evaluator's explicit reset.
    env.diagnostic_step=0;env.diagnostic_active.fill_(True)
    env.diagnostic_trace.clear();env.first_failures.clear();env.first_terminations.clear()
    report=evaluate_lift(runner,wrapped,env,episodes=args.num_envs,seed=args.seed,stage=args.stage)
    result={'schema':1,'diagnostic':True,'accepted':False,'source':'native_physx_checkpoint_first_failure_pre_reset',
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint),
            'contract_path':str(contract_path.resolve()),'contract':expected,'stage':args.stage,
            'num_envs':args.num_envs,'seed':args.seed,'trace_hz_requested':args.trace_hz,
            'trace_stride_steps':env.diagnostic_trace_stride,
            'first_episode_only':True,'done_or_collision_rules_modified':False,
            'root_teleports_between_commands':False,'captures_before_native_auto_reset':True,
            'torque_measurement':'Implicit actuator model applied-torque estimate, not an independent solver or force-sensor torque measurement',
            'ankle_geometry_measurement':'Minimum transformed source collision-CAD hull vertex height above each environment origin',
            'environment_sha256':sha(Path(__file__).resolve().parents[3]/'source/leg_humanoid_lab/leg_humanoid_lab/lift_env.py'),
            'diagnostic_script_sha256':sha(__file__),'urdf_sha256':sha(URDF),
            'evaluation':report,'first_failures':env.first_failures,
            'first_terminations':env.first_terminations,'trace':env.diagnostic_trace}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(finite_json(result),indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    counts={}
    for item in env.first_failures:
        for body in item['nonfoot_contacts_over_25n']:
            counts[body['body']]=counts.get(body['body'],0)+1
    print('V5_FAILURE_DIAGNOSTIC '+json.dumps({'output':str(args.output),'first_failures':len(env.first_failures),
          'failure_body_counts':counts,'stage_evaluation_passed':report['passed']}),flush=True)
finally:
    if env is not None:env.close()
    app.close()
