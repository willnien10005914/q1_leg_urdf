"""V7 balance fork. Frozen V6 environment/control/termination are unchanged."""
import torch
import numpy as np
from isaaclab.utils.math import quat_apply
from .upper_body_env import UpperBodyLegEnv,UpperBodyEnvCfg,upper_body_contract
from .balance_objective import support_capture_error,graded_cost
from .robot import FOOT_VERTICES

class BalanceLegEnv(UpperBodyLegEnv):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        mass=self.robot.data.default_mass.to(self.device)
        self.balance_mass=mass/mass.sum(-1,keepdim=True)
        self.sole_centers=torch.tensor(np.stack([FOOT_VERTICES[s].mean(0) for s in ('left','right')]),device=self.device,dtype=torch.float32)

    def balance_measurements(self):
        d=self.robot.data
        com=(d.body_com_pos_w*self.balance_mass[:,:,None]).sum(1)
        velocity=(d.body_com_lin_vel_w*self.balance_mass[:,:,None]).sum(1)
        feet=d.body_link_pos_w[:,self.foot_body_ids]+quat_apply(d.body_link_quat_w[:,self.foot_body_ids].reshape(-1,4),self.sole_centers.expand(self.num_envs,-1,-1).reshape(-1,3)).reshape(self.num_envs,2,3)
        _,_,loaded,_=self.foot_step_metrics()
        error=support_capture_error(com,velocity,feet,loaded)
        badforces=self.contacts.data.net_forces_w[:,self.other_bodies].norm(dim=-1)
        return error,loaded,badforces

    def _get_rewards(self):
        reward=super()._get_rewards()
        error,loaded,force=self.balance_measurements()
        d=self.robot.data
        tilt=torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1))
        height=d.root_pos_w[:,2]-self.scene.env_origins[:,2]
        costs=2*graded_cost(error,.04)+graded_cost((tilt-.1).clamp_min(0),.15)
        costs+=graded_cost((self.root_height-.035-height).clamp_min(0),.03)
        costs+=2*(~loaded.any(-1)).float()+2*graded_cost(force.max(-1).values,25)
        reward+=(1-costs-40*self.reset_terminated.float())/10
        hands=self.upper_body_measurements()['hands_return_distance_m']
        quiet=(self.mode==0)|(self.mode==4)
        reward+=quiet*2*torch.exp(-hands.square().mean(-1)/.015**2)/10
        self.extras['log'].update(balance_capture_excess_m=error.mean(),balance_cost=costs.mean(),balance_nonfoot_force_n=force.max(-1).values.mean())
        return reward

    def _training_schedule(self):
        from .interactive_env import InteractiveLegEnv
        InteractiveLegEnv._training_schedule(self)
        t=self.episode_length_buf*self.step_dt-self.schedule_start
        sequence=self.balance_task<5
        single=(self.balance_task==5)|(self.balance_task==6)
        stand=self.balance_task==9
        selected=sequence|single|stand
        self.mode[selected]=0;self.command_targets[selected]=0;self.gesture_phase[selected]=0
        duration=float(self.library['sidestep_duration_s']);pair=.5+2*duration+1.
        for pair_id in (0,1):
            local=t-pair_id*pair-.5
            left=sequence&(local>=0)&(local<duration)
            right=sequence&(local>=duration)&(local<2*duration)
            self.mode[left]=2;self.mode[right]=3
            self.gesture_phase[left]=local[left]/duration
            self.gesture_phase[right]=(local[right]-duration)/duration
            self.mode[sequence&(local>=2*duration)&(local<2*duration+1.)]=4
        for task,mode in ((5,2),(6,3)):
            selected=(self.balance_task==task)&(t>=.5)&(t<.5+duration)
            self.mode[selected]=mode;self.gesture_phase[selected]=(t[selected]-.5)/duration
            self.mode[(self.balance_task==task)&(t>=.5+duration)&(t<1.5+duration)]=4

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:env_ids=self.robot._ALL_INDICES
        if not hasattr(self,'balance_task'):self.balance_task=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.balance_task[env_ids]=torch.randint(10,(len(env_ids),),device=self.device)
