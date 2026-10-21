"""V5: protect swing geometry and learn real unloaded steps before locomotion."""
import hashlib
import json
import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .interactive_env import InteractiveEnvCfg,InteractiveLegEnv,interactive_contract
from .lift_objective import CONTROL_PROFILE,control_targets,swing_terms
from .robot import ROOT,JOINTS,FOOT_VERTICES,fk

CONTROL_FILE=ROOT/'config/lift_control_v5.json'

def lift_contract():
    control=json.loads(CONTROL_FILE.read_text())
    gait_path=ROOT/control['gait_file'];gait=json.loads(gait_path.read_text())
    base=interactive_contract()
    return {**base,'robot':'q1_leg_lift_v5','schema':5,
            'gait_sha256':hashlib.sha256(gait_path.read_bytes()).hexdigest(),
            'gait_file':control['gait_file'],'control_sha256':hashlib.sha256(CONTROL_FILE.read_bytes()).hexdigest(),
            'control_profile':CONTROL_PROFILE,'motor':control,'default':gait['default'],'root_height':gait['root_height'],
            'action':'Native IK + bounded phase-dependent residual + support/swing IMU feedback; 100 Nm implicit PD, no root wrench',
            'height_target':'Per-phase actual source-CAD FK recomputed from native bounded joint reference',
            'curriculum':['inplace_lift','finite_gesture','interactive'],
            'proxy':False}

@configclass
class LiftEnvCfg(InteractiveEnvCfg):
    curriculum_stage:int=0
    gait_file:str='config/v5_gaits_lift6.json'

class LiftLegEnv(InteractiveLegEnv):
    cfg:LiftEnvCfg
    def __init__(self,cfg,**kwargs):
        self.control_config=json.loads(CONTROL_FILE.read_text())
        super().__init__(cfg,**kwargs)
        self.gait_path=ROOT/cfg.gait_file
        self.library=json.loads(self.gait_path.read_text())
        if self.library.get('contract')!='q1_leg_command_v5':raise ValueError('V5 requires its independent native gait library')
        plans=self.library['plans']
        self.qtable=torch.tensor([plans[n]['q'] for n in self.names],device=self.device,dtype=torch.float32)
        self.ctable=torch.tensor([plans[n]['contact'] for n in self.names],device=self.device,dtype=torch.float32)
        self.rtable=torch.tensor([plans[n]['root_offset_body'] for n in self.names],device=self.device,dtype=torch.float32)
        self.root_height=float(self.library['root_height'])
        heights=[];positions=[]
        for name in self.names:
            plan=plans[name];hs=[];ps=[]
            for q,root in zip(plan['q'],plan['root_offset_body']):
                transforms=fk(q);row=[];pr=[]
                for side in ('left','right'):
                    t=transforms[side+'_foot_link']
                    row.append(float(np.min(FOOT_VERTICES[side]@t[2,:3]+t[2,3])+self.root_height+root[2]))
                    pr.append(t[:3,3])
                hs.append(row);ps.append(pr)
            heights.append(hs);positions.append(ps)
        self.htable=torch.tensor(heights,device=self.device,dtype=torch.float32)
        self.ptable=torch.tensor(np.asarray(positions),device=self.device,dtype=torch.float32)
        self.stand_q=self.qtable[0,0]
        self.refq=self.stand_q.repeat(self.num_envs,1)
        self.transition_q=self.refq.clone()
        self.target_velocity=torch.zeros_like(self.refq)
        # Native implicit drive retains the URDF's 100-Nm cap. No external
        # generalized/base force or root pose write is used during any step.
        stiffness=self.robot.data.default_joint_stiffness.clone()
        damping=self.robot.data.default_joint_damping.clone()
        for i,name in enumerate(JOINTS):
            if any(part in name for part in ('hip_','knee_','ankle_')):
                stiffness[:,self.jids[i]]=self.control_config['leg_stiffness']
                damping[:,self.jids[i]]=self.control_config['leg_damping']
        self.robot.write_joint_stiffness_to_sim(stiffness);self.robot.write_joint_damping_to_sim(damping)
        self.robot.actuators['all'].stiffness.copy_(stiffness);self.robot.actuators['all'].damping.copy_(damping)

    def _training_schedule(self):
        stage=int(self.cfg.curriculum_stage)
        if stage==2:
            super()._training_schedule();return
        t=self.episode_length_buf*self.step_dt-self.schedule_start
        self.mode[:]=0;self.command_targets[:]=0;self.gesture_phase[:]=0
        if stage==0:
            moving=(t>=0)&(t<5.5)&(self.scenario!=3)
            self.mode[moving]=1
        elif stage==1:
            # 0=pair, 1=left, 2=right, 3=stand; every sequence remains finite.
            duration=float(self.library['sidestep_duration_s'])
            left=(t>=0)&(t<duration)&((self.scenario==0)|(self.scenario==1))
            right=((t>=duration)&(t<2*duration)&(self.scenario==0))|((t>=0)&(t<duration)&(self.scenario==2))
            self.mode[left]=2;self.mode[right]=3
            self.gesture_phase[left]=t[left]/duration
            self.gesture_phase[right]=torch.where(self.scenario[right]==0,t[right]-duration,t[right])/duration
            end=torch.where(self.scenario==0,2*duration,duration)
            recover=(t>=end)&(t<end+1)&(self.scenario!=3)
            self.mode[recover]=4
        else:raise ValueError('curriculum_stage must be 0, 1 or 2')

    def _update_reference(self):
        # In-place lifting is explicitly conditioned by WALK mode with a zero
        # velocity command. A zero command in STAND still uses the neutral pose.
        phase=self.gait_phase.remainder(1.)
        scaled=self.commands/self.commands.new_tensor([.22,.18,.6])
        magnitude=scaled.abs();weights=magnitude/magnitude.sum(-1).clamp_min(1.)[:,None]
        for table,name in ((self.qtable,'refq'),(self.htable,'reference_height'),(self.ctable,'_reference_contact_float'),(self.ptable,'reference_foot_position'),(self.rtable,'reference_root_offset')):
            base=table[0,0].expand((self.num_envs,)+table.shape[2:])
            inplace=(self._sample(table,torch.ones_like(self.mode),phase)+self._sample(table,torch.full_like(self.mode,2),phase))*.5
            value=inplace.clone();shape=(self.num_envs,)+(1,)*(value.ndim-1)
            for axis,(positive,negative) in enumerate(((1,2),(3,4),(5,6))):
                indices=torch.where(scaled[:,axis]>=0,positive,negative)
                value+=(self._sample(table,indices,phase)-inplace)*weights[:,axis].reshape(shape)
            value=torch.where((self.mode==1).reshape(shape),value,base)
            gesture=(self.mode==2)|(self.mode==3)
            gesture_value=self._sample(table,torch.where(self.mode==2,7,8),self.gesture_phase)
            setattr(self,name,torch.where(gesture.reshape(shape),gesture_value,value))
        self.reference_contact=self._reference_contact_float>.5
        self.refq=self.refq.clamp(self.lower,self.upper)

    def _pre_physics_step(self,actions):
        self.previous_actions=self.actions.clone();self.actions=actions.clone().clamp(-1,1)
        old_mode=self.mode.clone();old_reference=self.refq.clone()
        if self.cfg.training_mode:self._training_schedule()
        else:
            self.mode[:]=self._manual_mode;self.command_targets[:]=self._manual_commands;self.gesture_phase[:]=self._manual_phase
        limits=self.commands.new_tensor([.22,.18,.6])
        self.command_targets/=((self.command_targets/limits).abs().sum(-1).clamp_min(1.))[:,None]
        self.commands+=(self.command_targets-self.commands)*min(1.,self.step_dt/.18)
        self.gait_phase+=(self.mode==1)*self.step_dt/float(self.library['walk_cycle_s'])
        changed=self.mode!=old_mode
        self.transition_q[changed]=old_reference[changed];self.transition_time[changed]=0
        self.transition_time=(self.transition_time+self.step_dt).clamp(max=.2)
        self._update_reference()
        blend=(1-torch.cos(torch.pi*self.transition_time/.2))*.5
        self.refq=self.transition_q*(1-blend[:,None])+self.refq*blend[:,None]
        d=self.robot.data
        self.target,self.action_scales=control_targets(self.refq,self.stand_q,self.actions,d.projected_gravity_b,d.root_ang_vel_b,self.reference_contact,self.reference_height,self.lower,self.upper)

    def _get_rewards(self):
        d=self.robot.data;up=-d.projected_gravity_b[:,2]
        height_error=d.root_pos_w[:,2]-self.scene.env_origins[:,2]-self.root_height-self.reference_root_offset[:,2]
        clearance,speed,loaded,target=self.foot_step_metrics()
        terms=swing_terms(clearance,target,loaded,self.reference_contact)
        desired=self.commands.clone();gesture=(self.mode==2)|(self.mode==3)
        for mode,index in ((2,7),(3,8)):
            ahead=(self.gesture_phase+.01).clamp(max=1.);behind=(self.gesture_phase-.01).clamp(min=0.)
            velocity=(self._sample(self.rtable,torch.full_like(self.mode,index),ahead)-self._sample(self.rtable,torch.full_like(self.mode,index),behind))/((ahead-behind).clamp_min(.001)*float(self.library['sidestep_duration_s']))[:,None]
            desired[self.mode==mode,:2]=velocity[self.mode==mode,:2]
        velocity_error=(d.root_lin_vel_b[:,:2]-desired[:,:2]).square().sum(-1)
        yaw_error=(d.root_ang_vel_b[:,2]-desired[:,2]).square()
        feet=quat_apply_inverse(d.root_quat_w[:,None,:].expand(-1,2,-1).reshape(-1,4),(d.body_pos_w[:,self.foot_body_ids]-d.root_pos_w[:,None,:]).reshape(-1,3)).reshape(self.num_envs,2,3)
        foot_error=(feet-self.reference_foot_position).square().sum(-1).mean(-1)
        pose_error=(d.joint_pos[:,self.jids]-self.refq).square().mean(-1)
        rev=int(self.cfg.reward_revision)
        reward=(.2+2*up.clamp_min(0)**4
            +3*torch.exp(-height_error.square()/self.control_config['height_sigma_m']**2)
            -2*(height_error.abs()/.02).clamp(max=3.)
            +3*torch.exp(-velocity_error/.035)+torch.exp(-yaw_error/.16)
            +.5*torch.exp(-pose_error/.04)+1.5*torch.exp(-foot_error/.004)
            +(6+2*rev)*terms['progress']+terms['unload']-(8+2*rev)*terms['miss']-(4+rev)*terms['wrong_contact']
            +1.*(loaded==self.reference_contact).float().mean(-1)
            -8*(speed.square()*loaded*self.reference_contact).mean(-1)
            -4*(~loaded).all(-1).float()
            -.5*d.root_ang_vel_b[:,:2].square().sum(-1)
            -.04*(self.actions-self.previous_actions).square().mean(-1)
            -.01*self.actions.square().mean(-1)-.00001*d.applied_torque[:,self.jids].square().mean(-1)
            -400*self.reset_terminated.float())
        arm_error=(d.joint_pos[:,[self.jids[i] for i in self.arm_ids]]-self.refq[:,self.arm_ids]).square().mean(-1)
        reward+=gesture*1.*torch.exp(-arm_error/.04)
        self.extras['log']={'velocity_error':velocity_error.sqrt().mean(),'upright':up.mean(),'height_error':height_error.abs().mean(),'foot_clearance':clearance.mean(),'swing_progress':terms['progress'].mean(),'missed_swing':terms['miss'].mean(),'loaded_swing':terms['wrong_contact'].mean(),'stance_slip':(speed*loaded).mean(),'pose_rmse':pose_error.sqrt().mean()}
        return reward/10.

    def foot_step_metrics(self):
        height,_,loaded,target=super().foot_step_metrics()
        # Isaac Lab's legacy body_lin_vel_w is the COM velocity. Report the
        # foot link-origin velocity explicitly, matching the slip proxy label.
        speed=self.robot.data.body_link_lin_vel_w[:,self.foot_body_ids,:2].norm(dim=-1)
        return height,speed,loaded,target

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:env_ids=self.robot._ALL_INDICES
        if int(self.cfg.curriculum_stage)==0:
            self.scenario[env_ids]=torch.where(torch.rand(len(env_ids),device=self.device)<.12,3,0)
        elif int(self.cfg.curriculum_stage)==1:
            self.scenario[env_ids]=torch.multinomial(torch.tensor([.4,.25,.25,.1],device=self.device),len(env_ids),replacement=True)
        self.schedule_start[env_ids]=.3+.4*torch.rand(len(env_ids),device=self.device)
