"""Command-conditioned native DirectRLEnv with continuous mode transitions.

The base is never forced or translated. Joint targets, gravity and ground
contacts produce all movement. References are independent of world position.
"""
import hashlib
import json
from pathlib import Path
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .isaac_env import IsaacLegEnv,LegEnvCfg
from .robot import ROOT,JOINTS,DEFAULT,LOWER,UPPER,contract
from .interactive_control import MODES

@configclass
class InteractiveEnvCfg(LegEnvCfg):
    episode_length_s=20.
    skill='stand'
    training_mode:bool=True
    curriculum_progress:float=1.
    reward_revision:int=0

def interactive_contract():
    base=contract()
    gait_bytes=(ROOT/'config/v4_gaits.json').read_bytes()
    gait=json.loads(gait_bytes)
    return {**base,'robot':'q1_leg_interactive_v4','schema':4,'gait_sha256':hashlib.sha256(gait_bytes).hexdigest(),
            'default':gait['default'],'root_height':gait['root_height'],
            'skills':['stand','command_walk','sidestep_left','sidestep_right','recover'],
            'commands':{'vx':[-.22,.22],'vy':[-.18,.18],'wz':[-.6,.6]},
            'observation_note':'110-D layout retained; skill channels stand/walk/gesture, phase and reference are v4 commands, not v3 fixed world paths',
            'action':'native IK reference + IMU ankle feedback + 0.35 * residual action; native limits',
            'transition':'continuous shared policy; no reset except episode failure or explicit restart','proxy':False}

class InteractiveLegEnv(IsaacLegEnv):
    cfg:InteractiveEnvCfg
    def set_skill(self,skill):
        # Base initialization requires its legacy caches. They are not used by
        # this environment after initialization.
        super().set_skill('stand')

    def __init__(self,cfg,**kwargs):
        episode_seconds=float(cfg.episode_length_s)
        super().__init__(cfg,**kwargs)
        # Base set_skill sets a legacy duration; restore the caller's v4 limit.
        self.cfg.episode_length_s=episode_seconds
        self.library=json.loads((ROOT/'config/v4_gaits.json').read_text())
        self.names=['stand','forward','backward','left','right','yaw_left','yaw_right','sidestep_left','sidestep_right']
        plans=self.library['plans']
        self.qtable=torch.tensor([plans[n]['q'] for n in self.names],device=self.device,dtype=torch.float32)
        self.htable=torch.tensor([plans[n]['foot_height'] for n in self.names],device=self.device,dtype=torch.float32)
        self.ctable=torch.tensor([plans[n]['contact'] for n in self.names],device=self.device,dtype=torch.float32)
        self.ptable=torch.tensor([plans[n]['foot_position_body'] for n in self.names],device=self.device,dtype=torch.float32)
        self.rtable=torch.tensor([plans[n]['root_offset_body'] for n in self.names],device=self.device,dtype=torch.float32)
        self.stand_q=self.qtable[0,0]
        self.root_height=float(self.library['root_height'])
        self.mode=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.commands=torch.zeros((self.num_envs,3),device=self.device)
        self.command_targets=torch.zeros_like(self.commands)
        self.gait_phase=torch.zeros(self.num_envs,device=self.device)
        self.gesture_phase=torch.zeros_like(self.gait_phase)
        self.refq=self.stand_q.repeat(self.num_envs,1)
        self.transition_q=self.refq.clone()
        self.transition_time=torch.full_like(self.gait_phase,.2)
        self.reference_height=self.htable[0,0].repeat(self.num_envs,1)
        self.reference_contact=torch.ones_like(self.reference_height,dtype=torch.bool)
        self.reference_foot_position=self.ptable[0,0].repeat(self.num_envs,1,1)
        self.reference_root_offset=self.rtable[0,0].repeat(self.num_envs,1)
        self.scenario=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.walk_command=torch.zeros_like(self.commands)
        self.walk_command2=torch.zeros_like(self.commands)
        self.walk_duration=torch.full_like(self.gait_phase,5.)
        self.schedule_start=torch.ones_like(self.gait_phase)
        self._manual_mode=torch.zeros_like(self.mode)
        self._manual_commands=torch.zeros_like(self.commands)
        self._manual_phase=torch.zeros_like(self.gait_phase)
        self.arm_ids=[i for i,n in enumerate(JOINTS) if 'shoulder' in n or 'elbow' in n]
        self.hand_ids,_=self.robot.find_bodies(['left_wrist_yaw_link','right_wrist_yaw_link'],preserve_order=True)

    def set_manual_control(self,mode,command,phase=0.):
        if isinstance(mode,str):mode=MODES[mode]
        self._manual_mode[:]=torch.as_tensor(mode,device=self.device,dtype=torch.long)
        self._manual_commands[:]=torch.as_tensor(command,device=self.device,dtype=torch.float32)
        self._manual_phase[:]=torch.as_tensor(phase,device=self.device,dtype=torch.float32)

    def _training_schedule(self):
        t=self.episode_length_buf*self.step_dt-self.schedule_start
        walk1=(t>=0)&(t<self.walk_duration)&(self.scenario!=2)&(self.scenario!=3)
        walk2=(t>=12)&(t<12+self.walk_duration)&(self.scenario!=2)&(self.scenario!=3)
        duration=float(self.library['sidestep_duration_s'])
        left=(t>=6)&(t<6+duration)&(self.scenario!=1)&(self.scenario!=3)
        right=(t>=6+duration)&(t<6+2*duration)&(self.scenario!=1)&(self.scenario!=3)
        self.mode[:]=0
        self.mode[walk1|walk2]=1;self.mode[left]=2;self.mode[right]=3
        recover=(t>=6+2*duration)&(t<7+2*duration)&(self.scenario!=1)&(self.scenario!=3)
        self.mode[recover]=4
        progress=max(.20,min(1.,float(self.cfg.curriculum_progress)))
        # Mid-stride redirects, reversals and key release are part of training,
        # rather than evaluating only a single constant command per episode.
        first=torch.where((t<self.walk_duration/2)[:,None],self.walk_command,self.walk_command2)
        second=torch.where((t-12<self.walk_duration/2)[:,None],-self.walk_command2,-self.walk_command)
        self.command_targets[:]=torch.where(walk1[:,None],first,torch.where(walk2[:,None],second,0.))*progress
        self.mode[(walk1|walk2)&(self.command_targets.abs().sum(-1)<1e-6)]=0
        self.gesture_phase[:]=0
        self.gesture_phase[left]=(t[left]-6)/duration
        self.gesture_phase[right]=(t[right]-6-duration)/duration

    def _sample(self,table,indices,phase):
        x=phase.clamp(0,1)*(table.shape[1]-1)
        i=x.long().clamp(max=table.shape[1]-2);a=x-i
        shape=(len(a),)+(1,)*(table.ndim-2)
        return table[indices,i]*(1-a.reshape(shape))+table[indices,i+1]*a.reshape(shape)

    def _update_reference(self):
        phase=self.gait_phase.remainder(1.)
        # Signed plans share stance phases. Their forward/backward mean cancels
        # stride translation while preserving knee flexion: slow commands must
        # still lift the feet rather than blend them down toward standing.
        limits=torch.tensor([.22,.18,.6],device=self.device)
        scaled=self.commands/limits
        magnitudes=scaled.abs()
        total=magnitudes.sum(-1).clamp(min=1.)
        weights=magnitudes/total[:,None]
        activation=(magnitudes.sum(-1)/.12).clamp(max=1.)
        for table,name in ((self.qtable,'refq'),(self.htable,'reference_height'),(self.ctable,'_reference_contact_float'),(self.ptable,'reference_foot_position'),(self.rtable,'reference_root_offset')):
            base=table[0,0].expand((self.num_envs,)+table.shape[2:]).clone()
            inplace=(self._sample(table,torch.ones_like(self.mode),phase)+self._sample(table,torch.full_like(self.mode,2),phase))*.5
            value=inplace.clone()
            for axis,(positive,negative) in enumerate(((1,2),(3,4),(5,6))):
                selected=torch.where(scaled[:,axis]>=0,positive,negative)
                sample=self._sample(table,selected,phase)
                shape=(self.num_envs,)+(1,)*(sample.ndim-1)
                value+=(sample-inplace)*weights[:,axis].reshape(shape)
            walking=self.mode==1
            shape=(self.num_envs,)+(1,)*(value.ndim-1)
            value=base+(value-base)*activation.reshape(shape)
            value=torch.where(walking.reshape(shape),value,base)
            gesture=(self.mode==2)|(self.mode==3)
            selected=torch.where(self.mode==2,7,8)
            gesture_value=self._sample(table,selected,self.gesture_phase)
            value=torch.where(gesture.reshape(shape),gesture_value,value)
            setattr(self,name,value)
        self.reference_contact=self._reference_contact_float>.5
        self.refq=self.refq.clamp(self.lower,self.upper)

    def _pre_physics_step(self,actions):
        self.previous_actions=self.actions.clone()
        self.actions=actions.clone().clamp(-1,1)
        old_mode=self.mode.clone();old_ref=self.refq.clone()
        if self.cfg.training_mode:self._training_schedule()
        else:
            self.mode[:]=self._manual_mode
            self.command_targets[:]=self._manual_commands
            self.gesture_phase[:]=self._manual_phase
        limits=torch.tensor([.22,.18,.6],device=self.device)
        self.command_targets/=((self.command_targets/limits).abs().sum(-1).clamp(min=1.))[:,None]
        self.commands+=(self.command_targets-self.commands)*min(1.,self.step_dt/.18)
        self.gait_phase+=(self.mode==1)*self.step_dt/float(self.library['walk_cycle_s'])
        changed=self.mode!=old_mode
        self.transition_q[changed]=old_ref[changed]
        self.transition_time[changed]=0
        self.transition_time=(self.transition_time+self.step_dt).clamp(max=.2)
        self._update_reference()
        blend=(1-torch.cos(torch.pi*self.transition_time/.2))*.5
        self.refq=self.transition_q*(1-blend[:,None])+self.refq*blend[:,None]
        self.target=self.refq+.35*self.actions
        gravity=self.robot.data.projected_gravity_b;angular=self.robot.data.root_ang_vel_b
        roll=torch.atan2(-gravity[:,1],-gravity[:,2]);pitch=torch.atan2(gravity[:,0],-gravity[:,2])
        for side,sign in (('left',1),('right',-1)):
            self.target[:,JOINTS.index(f'{side}_ankle_Pitch_joint')]+=sign*(pitch+.15*angular[:,1]).clamp(-.15,.15)
            self.target[:,JOINTS.index(f'{side}_ankle_Roll_joint')]+=(roll+.15*angular[:,0]).clamp(-.15,.15)
        self.target=self.target.clamp(self.lower,self.upper)

    def _get_observations(self):
        phase=torch.where((self.mode==2)|(self.mode==3),self.gesture_phase,self.gait_phase)
        skill=torch.nn.functional.one_hot(torch.where((self.mode==2)|(self.mode==3),2,torch.where(self.mode==1,1,0)),3).float()
        d=self.robot.data
        # Gesture direction is explicit in commands even while no keyboard
        # velocity is requested; reference q supplies the full finite motion.
        command=self.commands.clone()
        command[:,1]=torch.where(self.mode==2,.18,torch.where(self.mode==3,-.18,command[:,1]))
        return {'policy':torch.cat((d.root_lin_vel_b,d.root_ang_vel_b,d.projected_gravity_b,d.joint_pos[:,self.jids]-self.stand_q,
            d.joint_vel[:,self.jids]*.1,self.actions,command,skill,torch.sin(2*torch.pi*phase)[:,None],torch.cos(2*torch.pi*phase)[:,None],
            (d.root_pos_w[:,2]-self.scene.env_origins[:,2])[:,None],self.refq-self.stand_q),dim=-1)}

    def foot_step_metrics(self):
        height,speed,loaded,_=super().foot_step_metrics()
        return height,speed,loaded,self.reference_height

    def _get_rewards(self):
        d=self.robot.data
        up=-d.projected_gravity_b[:,2]
        height=(d.root_pos_w[:,2]-self.scene.env_origins[:,2]-self.root_height-self.reference_root_offset[:,2]).abs()
        clearance,speed,loaded,target=self.foot_step_metrics()
        bodyfeet=quat_apply_inverse(d.root_quat_w[:,None,:].expand(-1,2,-1).reshape(-1,4),(d.body_pos_w[:,self.foot_body_ids]-d.root_pos_w[:,None,:]).reshape(-1,3)).reshape(self.num_envs,2,3)
        gesture=(self.mode==2)|(self.mode==3)
        desired=self.commands.clone()
        # The finite maneuver follows its IK root motion derivative; support
        # forces, rather than world-root constraints, must generate it.
        for mode,index in ((2,7),(3,8)):
            mask=self.mode==mode
            ahead=(self.gesture_phase+.01).clamp(max=1.);behind=(self.gesture_phase-.01).clamp(min=0.)
            indices=torch.full_like(self.mode,index)
            velocity=(self._sample(self.rtable,indices,ahead)-self._sample(self.rtable,indices,behind))/((ahead-behind).clamp(min=.001)*float(self.library['sidestep_duration_s']))[:,None]
            desired[mask,:2]=velocity[mask,:2]
        err=(d.root_lin_vel_b[:,:2]-desired[:,:2]).square().sum(-1)
        yaw=(d.root_ang_vel_b[:,2]-desired[:,2]).square()
        pose=(d.joint_pos[:,self.jids]-self.refq).square().mean(-1)
        foot_error=(bodyfeet-self.reference_foot_position).square().sum(-1).mean(-1)
        swing=target>.012
        rev=int(self.cfg.reward_revision)
        lift_weight=6.+2*rev;track_weight=5.+rev
        reward=(1.+2*up.clamp(min=0)**4+1.5*torch.exp(-height.square()/.008)
                +track_weight*torch.exp(-err/.035)+1.5*torch.exp(-yaw/.16)
                +(1.2/(1+rev))*torch.exp(-pose/.12)+2*torch.exp(-foot_error/.008)
                +lift_weight*torch.exp(-((clearance-target)/.02).square()).mean(-1)
                +2*(loaded==self.reference_contact).float().mean(-1)
                -(5.+2*rev)*(speed.square()*loaded*self.reference_contact).mean(-1)
                -2*(loaded*swing*speed).mean(-1)-.08*(self.actions-self.previous_actions).square().mean(-1)
                -.015*self.actions.square().mean(-1)-.00001*d.applied_torque[:,self.jids].square().mean(-1)
                -20*self.reset_terminated.float())
        arm_error=(d.joint_pos[:,[self.jids[i] for i in self.arm_ids]]-self.refq[:,self.arm_ids]).square().mean(-1)
        reward+=gesture*2.*torch.exp(-arm_error/.04)
        push=gesture&(self.gesture_phase>=.45)&(self.gesture_phase<=.65)
        rear=torch.where(self.mode==2,1,0)
        rear_loaded=loaded.gather(1,rear[:,None]).squeeze(1)
        diagonal=torch.stack((torch.ones_like(up),torch.where(self.mode==2,1.,-1.)),dim=1)/2**.5
        forward_push=(d.root_lin_vel_b[:,:2]*diagonal).sum(-1)
        # This is a load-and-progress proxy, not a claim that the contact sensor
        # separately measures tangential friction force.
        reward+=push*(2*rear_loaded.float()*(forward_push/.08).clamp(0,1)-2*(~rear_loaded).float())
        # Progress is commanded in body axes and has no fixed world endpoint.
        direction=desired[:,:2]/desired[:,:2].norm(dim=-1,keepdim=True).clamp(min=.01)
        reward+=2*(d.root_lin_vel_b[:,:2]*direction).sum(-1).clamp(-.3,.4)
        self.extras['log']={'velocity_error':err.sqrt().mean(),'upright':up.mean(),'foot_clearance':clearance.mean(),'stance_slip':(speed*loaded).mean(),'pose_rmse':pose.sqrt().mean()}
        return reward/10.

    def _get_dones(self):
        d=self.robot.data
        bad=(self.contacts.data.net_forces_w[:,self.other_bodies].norm(dim=-1)>25).any(-1)
        fail=(d.root_pos_w[:,2]-self.scene.env_origins[:,2]<.52)|(-d.projected_gravity_b[:,2]<.65)|bad|~torch.isfinite(d.root_state_w).all(-1)
        return fail,self.episode_length_buf>=self.max_episode_length-1

    def _reset_idx(self,env_ids):
        if env_ids is None:env_ids=self.robot._ALL_INDICES
        # Avoid the legacy fixed-world motion reset.
        from isaaclab.envs import DirectRLEnv
        DirectRLEnv._reset_idx(self,env_ids)
        q=self.robot.data.default_joint_pos[env_ids].clone()
        q[:,self.jids]=(self.stand_q+torch.randn((len(env_ids),len(JOINTS)),device=self.device)*.006).clamp(self.lower,self.upper)
        state=self.robot.data.default_root_state[env_ids].clone()
        state[:,:3]=self.scene.env_origins[env_ids];state[:,2]+=self.root_height+.003
        self.robot.write_root_pose_to_sim(state[:,:7],env_ids);self.robot.write_root_velocity_to_sim(torch.zeros_like(state[:,7:]),env_ids)
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
        self.actions[env_ids]=0;self.previous_actions[env_ids]=0
        self.commands[env_ids]=0;self.command_targets[env_ids]=0
        self.gait_phase[env_ids]=0;self.gesture_phase[env_ids]=0;self.mode[env_ids]=0
        self.scenario[env_ids]=torch.multinomial(torch.tensor([.35,.35,.20,.10],device=self.device),len(env_ids),replacement=True)
        def draw_commands():
            angle=torch.rand(len(env_ids),device=self.device)*2*torch.pi
            commands=torch.stack((.22*torch.cos(angle),.18*torch.sin(angle),(torch.rand(len(env_ids),device=self.device)-.5)*.8),dim=1)
            cardinal=torch.tensor([[.22,0,0],[-.22,0,0],[0,.18,0],[0,-.18,0],[0,0,.6],[0,0,-.6]],device=self.device)
            selected=torch.randint(6,(len(env_ids),),device=self.device)
            return torch.where((torch.rand(len(env_ids),device=self.device)<.35)[:,None],cardinal[selected],commands)
        first=draw_commands();second=draw_commands()
        reverse=torch.rand(len(env_ids),device=self.device)<.4
        second[reverse]=-first[reverse]
        second[torch.rand(len(env_ids),device=self.device)<.15]=0
        self.walk_command[env_ids]=first;self.walk_command2[env_ids]=second
        # Stop at many gait phases, not only the same end of a five-second walk.
        self.walk_duration[env_ids]=3.5+2*torch.rand(len(env_ids),device=self.device)
        self.schedule_start[env_ids]=torch.rand(len(env_ids),device=self.device)+.5
        self.refq[env_ids]=self.stand_q;self.reference_height[env_ids]=self.htable[0,0]
        self.transition_q[env_ids]=self.stand_q;self.transition_time[env_ids]=.2
        self.reference_contact[env_ids]=True;self.reference_foot_position[env_ids]=self.ptable[0,0]
