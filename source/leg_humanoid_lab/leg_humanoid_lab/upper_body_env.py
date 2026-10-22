"""V6: one native Direct PPO for stable torso, controlled arms and real steps.

V5 stays frozen. This subclass retains its leg control, observation layout and
contact predicates and learns with a separately hashed upper control contract.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .lift_env import LiftLegEnv, LiftEnvCfg, lift_contract, CONTROL_FILE as BASE_CONTROL_FILE
from .robot import ROOT, JOINTS, fk
from .upper_body_objective import (ARM_IDS, WAIST_IDS, CONTROL_PROFILE,
    guarded_upper_targets, nominal_upper_velocity, upper_reward_terms)

CONTROL_FILE = ROOT / 'config/upper_body_control_v6.json'


def upper_body_contract():
    profile = json.loads(CONTROL_FILE.read_text())
    base = lift_contract()
    if hashlib.sha256(BASE_CONTROL_FILE.read_bytes()).hexdigest() != profile['base_control_sha256']:
        raise ValueError('V6 requires its declared frozen V5 leg control source')
    return {**base, 'robot': 'q1_leg_upper_body_v6', 'schema': 6,
            'control_sha256': hashlib.sha256(CONTROL_FILE.read_bytes()).hexdigest(),
            'control_profile': CONTROL_PROFILE,
            'base_control_sha256': base['control_sha256'],
            'motor': {**base['motor'], 'upper_body': profile},
            'action': 'V5 leg control + guarded upper residuals; original upper PD; native 100 Nm; no root wrench',
            'observation_note': '110-D command observation retained; chest state observable from root IMU and waist q/qdot',
            'curriculum': ['inplace_upper_recovery', 'finite_gesture_upper_return', 'interactive_upper'],
            'upper_body_gates': profile['evaluation_gates']}


@configclass
class UpperBodyEnvCfg(LiftEnvCfg):
    episode_length_s = 20.


class UpperBodyLegEnv(LiftLegEnv):
    cfg: UpperBodyEnvCfg

    def __init__(self, cfg, **kwargs):
        self.upper_body_profile = json.loads(CONTROL_FILE.read_text())
        self.upper_body_control_sha256 = hashlib.sha256(CONTROL_FILE.read_bytes()).hexdigest()
        self.upper_body_contract = upper_body_contract()
        super().__init__(cfg, **kwargs)
        ids, _ = self.robot.find_bodies(['center_torso_link'], preserve_order=True)
        self.torso_id = ids[0]
        self.arm_native_ids = [self.jids[i] for i in ARM_IDS]
        self.waist_native_ids = [self.jids[i] for i in WAIST_IDS]
        transforms = fk(self.stand_q.cpu().numpy())
        self.stand_hand_reference = torch.tensor(np.stack([transforms[n][:3, 3]
            for n in ('left_wrist_yaw_link', 'right_wrist_yaw_link')]),
            dtype=torch.float32, device=self.device)
        self.upper_reference_velocity = torch.zeros_like(self.refq)
        self.world_down = torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, -1)

    def _pre_physics_step(self, actions):
        previous = self.refq.clone()
        super()._pre_physics_step(actions)
        self.target = guarded_upper_targets(self.target, self.refq, self.actions,
            self.lower, self.upper, self.upper_body_profile['residual_rad'])
        self.upper_reference_velocity = nominal_upper_velocity(self.refq, previous,
            self.episode_length_buf, self.step_dt, self.control_config['motor_velocity_limit_rad_s'])

    def upper_body_measurements(self):
        d = self.robot.data
        torso_q = d.body_quat_w[:, self.torso_id]
        down = quat_apply_inverse(torso_q, self.world_down)
        # atan2 avoids acos's loss of resolution near an upright torso.
        tilt = torch.atan2(down[:, :2].norm(dim=-1), -down[:, 2])
        angular = quat_apply_inverse(torso_q, d.body_ang_vel_w[:, self.torso_id])
        q = d.joint_pos[:, self.jids]
        qvel = d.joint_vel[:, self.jids]
        hand = quat_apply_inverse(d.root_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
            (d.body_link_pos_w[:, self.hand_ids] - d.root_pos_w[:, None, :]).reshape(-1, 3)).reshape(-1, 2, 3)
        return {
            'chest_tilt_rad': tilt,
            'chest_angular_xy_rad_s': angular[:, :2].norm(dim=-1),
            'chest_angular_xyz_rad_s': angular.norm(dim=-1),
            'arm_reference_mse_rad2': (q[:, ARM_IDS] - self.refq[:, ARM_IDS]).square().mean(-1),
            'waist_reference_mse_rad2': (q[:, WAIST_IDS] - self.refq[:, WAIST_IDS]).square().mean(-1),
            'arm_velocity_error_mse_rad2_s2': (qvel[:, ARM_IDS] - self.upper_reference_velocity[:, ARM_IDS]).square().mean(-1),
            'arm_joint_speed_mse_rad2_s2': qvel[:, ARM_IDS].square().mean(-1),
            'waist_joint_speed_mse_rad2_s2': qvel[:, WAIST_IDS].square().mean(-1),
            'hands_return_distance_m': (hand - self.stand_hand_reference).norm(dim=-1),
        }

    def _get_rewards(self):
        reward = super()._get_rewards()
        measurements = self.upper_body_measurements()
        reward_cfg = {**self.upper_body_profile['reward'],
                      **self.upper_body_profile['reward_revisions'][str(int(self.cfg.reward_revision))]}
        terms = upper_reward_terms(measurements, self.mode, self.gesture_phase, reward_cfg)
        self.extras['log'].update({f'upper/{key}': value.mean() for key, value in terms.items()})
        self.extras['log'].update(chest_tilt_rad=measurements['chest_tilt_rad'].mean(),
            chest_angular_xy_rad_s=measurements['chest_angular_xy_rad_s'].mean(),
            arm_reference_rmse_rad=measurements['arm_reference_mse_rad2'].mean().sqrt())
        return reward + sum(terms.values()) / 10.

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, 'upper_reference_velocity'):
            self.upper_reference_velocity[env_ids] = 0.
