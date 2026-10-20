"""GPU Isaac Lab backend for native CAD or explicitly requested proxy geometry.

Load only after AppLauncher. SB3 and RSL-RL checkpoint formats are different;
stages can warm-start within one backend, never by mixing checkpoint formats.
"""
import json
import os
import hashlib
from pathlib import Path
import torch
import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .robot import ROOT,URDF,JOINTS,JMAP,DEFAULT,LOWER,UPPER,HEIGHT,SKILLS,OBSERVATION_SIZE,FOOT_VERTICES
from .motion import Motion

def robot_cfg(proxy=False):
    if not proxy and not json.loads((ROOT/"config/robot_contract.json").read_text())["native_cad_ready"]:
        raise FileNotFoundError("Native foot CAD missing; install meshes/LOWERBODY_LEGGED/MESH or explicitly use --allow-proxy")
    return ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
        spawn=sim.UrdfFileCfg(asset_path=str(ROOT/"urdf/leg_humanoid_proxy.urdf" if proxy else URDF),
            fix_base=False,merge_fixed_joints=False,activate_contact_sensors=True,
            usd_dir=str(ROOT/"outputs/usd_proxy" if proxy else ROOT/"outputs/usd_native"),force_usd_conversion=False,
            joint_drive=sim.UrdfConverterCfg.JointDriveCfg(gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.)),
            rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,HEIGHT+.003),joint_pos={**{n:0. for n in JMAP},**dict(zip(JOINTS,DEFAULT.tolist()))},joint_vel={".*":0.}),
        soft_joint_pos_limit_factor=1.,
        actuators={"all":ImplicitActuatorCfg(joint_names_expr=[".*"],effort_limit_sim=100.,velocity_limit_sim=10.,
            stiffness={".*hip_.*|.*knee_.*|.*ankle_.*":160.,".*waist_.*|.*shoulder_.*|.*elbow_.*":70.,".*wrist_.*|.*finger_.*":200.},
            damping={".*hip_.*|.*knee_.*|.*ankle_.*":8.,".*waist_.*|.*shoulder_.*|.*elbow_.*":4.,".*wrist_.*|.*finger_.*":10.})})

@configclass
class LegEnvCfg(DirectRLEnvCfg):
    decimation=4
    episode_length_s=10.
    action_space=len(JOINTS)
    observation_space=OBSERVATION_SIZE
    state_space=0
    sim:SimulationCfg=SimulationCfg(dt=.005,render_interval=4)
    scene:InteractiveSceneCfg=InteractiveSceneCfg(num_envs=512,env_spacing=3.,replicate_physics=True)
    contact_cfg:ContactSensorCfg=ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*",update_period=0.,history_length=3,track_air_time=True)
    skill:str="stand"
    allow_proxy:bool=False

class IsaacLegEnv(DirectRLEnv):
    cfg:LegEnvCfg
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self.jids,_=self.robot.find_joints(list(JOINTS),preserve_order=True)
        self.feet,_=self.contacts.find_bodies(["left_foot_link","right_foot_link"],preserve_order=True)
        self.foot_body_ids,_=self.robot.find_bodies(["left_foot_link","right_foot_link"],preserve_order=True)
        from scipy.spatial import ConvexHull
        self.sole_vertices=[torch.tensor(v[ConvexHull(v).vertices],device=self.device,dtype=torch.float32) for v in FOOT_VERTICES.values()]
        self.other_bodies=[i for i in range(len(self.contacts.body_names)) if i not in self.feet]
        self.default=torch.tensor(DEFAULT,device=self.device,dtype=torch.float32)
        self.lower=torch.tensor(LOWER,device=self.device,dtype=torch.float32)
        self.upper=torch.tensor(UPPER,device=self.device,dtype=torch.float32)
        self.actions=torch.zeros((self.num_envs,len(JOINTS)),device=self.device)
        self.previous_actions=torch.zeros_like(self.actions)
        self.set_skill(cfg.skill)

    def set_skill(self,skill):
        if skill not in SKILLS:
            raise ValueError(skill)
        self.cfg.skill=skill
        self.cfg.episode_length_s=7.8 if skill=='sidestep' else 10.
        self.motion=Motion(skill)
        duration=self.motion.duration if skill=="sidestep" else self.cfg.episode_length_s+.1
        samples=[self.motion.sample(i*.02) for i in range(round(duration/.02)+2)]
        self.refq=torch.tensor(__import__('numpy').stack([x[0] for x in samples]),device=self.device,dtype=torch.float32)
        self.refroot=torch.tensor(__import__('numpy').stack([x[1] for x in samples]),device=self.device,dtype=torch.float32)
        self.refcmd=torch.tensor(__import__('numpy').stack([x[2] for x in samples]),device=self.device,dtype=torch.float32)
        self.skill=torch.nn.functional.one_hot(torch.full((self.num_envs,),SKILLS.index(skill),device=self.device),3).float()
        self.foot_targets=None
        self.foot_objective={}
        profile=os.environ.get('Q1_LEG_CURRICULUM_CONFIG')
        if skill=='sidestep' and profile:
            config=json.loads(Path(profile).read_text())
            target_file=config.get('sidestep_foot_targets')
            if target_file:
                data=json.loads((ROOT/target_file).read_text())
                if data['robot_urdf_sha256']!=hashlib.sha256(URDF.read_bytes()).hexdigest() or data['motion_sha256']!=hashlib.sha256((ROOT/'docs/reference/sidestep_motion.json').read_bytes()).hexdigest():
                    raise ValueError('Foot targets belong to a different robot or motion revision')
                import numpy as np
                time=np.asarray(data['time']);height=np.asarray(data['height_m'])
                sample_time=np.arange(round(self.cfg.episode_length_s/self.step_dt)+2)*self.step_dt
                self.foot_targets=torch.tensor(np.stack([np.interp(sample_time,time,height[:,i]) for i in (0,1)],axis=1),device=self.device,dtype=torch.float32)
                self.foot_objective=config['foot_objective']

    def foot_step_metrics(self):
        """Minimum source-CAD sole height, load and world-frame foot velocity."""
        d=self.robot.data
        heights=[]
        z=torch.zeros((self.num_envs,3),device=self.device);z[:,2]=1
        for body,vertices in zip(self.foot_body_ids,self.sole_vertices):
            local_up=quat_apply_inverse(d.body_quat_w[:,body],z)
            heights.append(d.body_pos_w[:,body,2]-self.scene.env_origins[:,2]+(local_up@vertices.T).min(dim=1).values)
        height=torch.stack(heights,dim=1)
        speed=d.body_lin_vel_w[:,self.foot_body_ids,:2].norm(dim=-1)
        loaded=self.contacts.data.net_forces_w[:,self.feet,2]>5.
        target=None if self.foot_targets is None else self.foot_targets[self.episode_length_buf.clamp(max=len(self.foot_targets)-1).long()]
        return height,speed,loaded,target

    def _setup_scene(self):
        self.robot=Articulation(robot_cfg(self.cfg.allow_proxy))
        self.contacts=ContactSensor(self.cfg.contact_cfg)
        sim.spawn_ground_plane("/World/ground",sim.GroundPlaneCfg(physics_material=sim.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.9)))
        self.scene.clone_environments(copy_from_source=False)
        if self.device=="cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"]=self.robot
        self.scene.sensors["contacts"]=self.contacts
        sim.DomeLightCfg(intensity=2000.).func("/World/Light",sim.DomeLightCfg(intensity=2000.))

    def _reference(self):
        idx=self.episode_length_buf.clamp(0,len(self.refq)-1).long()
        return self.refq[idx],self.refroot[idx],self.refcmd[idx]

    def _pre_physics_step(self,actions):
        self.previous_actions=self.actions.clone()
        self.actions=actions.clone().clamp(-1,1)
        ref,_,_=self._reference()
        self.target=ref+.2*self.actions
        gravity=self.robot.data.projected_gravity_b
        angular=self.robot.data.root_ang_vel_b
        roll=torch.atan2(-gravity[:,1],-gravity[:,2])
        pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        for side,sign in (("left",1),("right",-1)):
            self.target[:,JOINTS.index(f"{side}_ankle_Pitch_joint")]+=sign*(pitch+.15*angular[:,1]).clamp(-.15,.15)
            self.target[:,JOINTS.index(f"{side}_ankle_Roll_joint")]+=(roll+.15*angular[:,0]).clamp(-.15,.15)
        self.target=self.target.clamp(self.lower,self.upper)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target,joint_ids=self.jids)

    def _get_observations(self):
        ref,_,cmd=self._reference()
        phase=(self.episode_length_buf*self.step_dt/(self.motion.duration if self.cfg.skill=="sidestep" else .9))
        if self.cfg.skill=="sidestep":
            phase=phase.clamp(max=1)
        data=self.robot.data
        obs=torch.cat((data.root_lin_vel_b,data.root_ang_vel_b,data.projected_gravity_b,
            data.joint_pos[:,self.jids]-self.default,data.joint_vel[:,self.jids]*.1,self.actions,
            cmd,self.skill,torch.sin(2*torch.pi*phase)[:,None],torch.cos(2*torch.pi*phase)[:,None],
            (data.root_pos_w[:,2]-self.scene.env_origins[:,2])[:,None],ref-self.default),dim=-1)
        return {"policy":obs}

    def _errors(self):
        ref,root,cmd=self._reference()
        d=self.robot.data
        upright=-d.projected_gravity_b[:,2]
        height=abs(d.root_pos_w[:,2]-self.scene.env_origins[:,2]-root[:,2])
        pose=((d.joint_pos[:,self.jids]-ref)**2).mean(dim=-1)
        velocity=((d.root_lin_vel_b[:,:2]-cmd[:,:2])**2).sum(dim=-1)
        root_error=((d.root_pos_w[:,:2]-self.scene.env_origins[:,:2]-root[:,:2])**2).sum(dim=-1)
        return upright,height,pose,velocity,root_error

    def _get_rewards(self):
        up,height,pose,velocity,root=self._errors()
        forces=self.contacts.data.net_forces_w
        loaded=forces[:,self.feet,2]>5.
        foot_ids,_=self.robot.find_bodies(["left_foot_link","right_foot_link"],preserve_order=True)
        slip=((self.robot.data.body_lin_vel_w[:,foot_ids,:2]**2).sum(-1)*loaded).sum(-1)
        reward=(1+2*up.clamp(min=0)**4+2*torch.exp(-height**2/.012)+2*torch.exp(-pose/.10)+2*torch.exp(-velocity/.16)
            -.04*((self.actions-self.previous_actions)**2).mean(-1)-.02*(self.actions**2).mean(-1)-.08*slip
            -.00001*(self.robot.data.applied_torque[:,self.jids]**2).mean(-1)-10*self.reset_terminated.float())
        reward+=2*torch.exp(-root/.09)-.3*velocity
        if self.cfg.skill=='walk':
            phase=self.episode_length_buf*self.step_dt/.9*2*torch.pi
            expected=torch.stack((torch.sin(phase)<=0,torch.sin(phase+torch.pi)<=0),dim=-1)
            reward+=1.5*(loaded==expected).float().mean(-1)
            reward+=.5*torch.exp(-self.robot.data.root_ang_vel_b[:,2]**2/.25)
        if self.cfg.skill=='sidestep':
            reward+=4*torch.exp(-root/.04)
            reward+=4*torch.exp(-pose/.03)-2*torch.exp(-pose/.10)
            reward+=4*torch.exp(-velocity/.09)-2*torch.exp(-velocity/.16)
            if self.foot_targets is not None:
                height,speed,loaded,target=self.foot_step_metrics()
                swing=target>.005
                goal=self.foot_objective
                # Penalize skating while loaded and reward actual airborne clearance.
                reward+=goal['height_weight']*torch.exp(-((height-target)/goal['height_sigma_m'])**2).mean(-1)
                reward+=goal['contact_weight']*(loaded==~swing).float().mean(-1)
                reward-=goal['stance_slip_weight']*(speed.square()*loaded*(~swing)).mean(-1)
                reward-=goal['scuff_weight']*(swing*loaded*speed).mean(-1)
        self.extras["log"]={"pose_rmse":pose.sqrt().mean(),"upright":up.mean(),"height_error":height.mean(),"velocity_error":velocity.sqrt().mean()}
        return reward/10.

    def _get_dones(self):
        d=self.robot.data
        bad_contact=(self.contacts.data.net_forces_w[:,self.other_bodies,:].norm(dim=-1)>20).any(dim=-1)
        fail=(d.root_pos_w[:,2]-self.scene.env_origins[:,2]<.56)|(-d.projected_gravity_b[:,2]<.65)|bad_contact|~torch.isfinite(d.root_state_w).all(dim=-1)
        return fail,self.episode_length_buf>=self.max_episode_length-1

    def _reset_idx(self,env_ids):
        if env_ids is None:
            env_ids=self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        q=self.robot.data.default_joint_pos[env_ids].clone()
        v=torch.zeros_like(q)
        ref,root,_=self.motion.sample(0)
        q[:,self.jids]=(torch.tensor(ref,device=self.device,dtype=q.dtype)+torch.randn((len(env_ids),len(JOINTS)),device=self.device)*.008).clamp(self.lower,self.upper)
        state=self.robot.data.default_root_state[env_ids].clone()
        state[:,:3]=self.scene.env_origins[env_ids]+torch.tensor(root,device=self.device,dtype=state.dtype)
        state[:,2]+=.003
        self.robot.write_root_pose_to_sim(state[:,:7],env_ids)
        self.robot.write_root_velocity_to_sim(state[:,7:],env_ids)
        self.robot.write_joint_state_to_sim(q,v,env_ids=env_ids)
        self.actions[env_ids]=0
        self.previous_actions[env_ids]=0
