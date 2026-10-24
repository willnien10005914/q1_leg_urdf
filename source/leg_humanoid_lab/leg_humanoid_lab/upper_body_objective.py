"""V6 physical upper-body control and bounded tracking rewards."""
from __future__ import annotations

import math
import torch

from .robot import JOINTS

CONTROL_PROFILE = 'upper_v6r1_waist_damping_contact_energy_rehearsal'
ARM_IDS = tuple(i for i, n in enumerate(JOINTS) if 'shoulder_' in n or 'elbow_' in n)
WAIST_IDS = tuple(i for i, n in enumerate(JOINTS) if 'waist_' in n)


def guarded_upper_targets(original, reference, actions, lower, upper, caps):
    """Apply declared upper residual caps, preserving every leg target exactly."""
    target = original.clone()
    for i in ARM_IDS + WAIST_IDS:
        name = JOINTS[i]
        cap = next((v for k, v in caps.items() if k != 'arm' and k in name), caps['arm'])
        target[:, i] = (reference[:, i] + cap * actions[:, i].clamp(-1., 1.)).clamp(lower[i], upper[i])
    return target


def nominal_upper_velocity(reference, previous, episode_steps, dt, limit=10.):
    """Track the reference's outgoing AND returning motion, never policy targets."""
    if dt <= 0:
        raise ValueError('A positive native control step is required')
    velocity = ((reference - previous) / dt).clamp(-limit, limit)
    return torch.where((episode_steps > 0)[:, None], velocity, 0.)


def upper_reward_terms(measurements, mode, phase, cfg):
    """Yaw-tolerant rewards; no reward for freezing an arm during its swing."""
    tilt = measurements['chest_tilt_rad']
    excess = (tilt - math.radians(cfg['tilt_deadband_deg'])).clamp_min(0.)
    return_phase = ((mode == 2) | (mode == 3)) & (phase >= .65)
    recovering = return_phase | (mode == 4)
    rate_weight = torch.where(recovering, cfg['return_velocity_weight'], cfg['velocity_weight'])
    return {
        'chest_posture': cfg['chest_weight'] * torch.exp(-(excess / cfg['tilt_sigma_rad']).square()),
        'chest_rate': cfg['chest_rate_weight'] * torch.exp(-measurements['chest_angular_xy_rad_s'].square() / cfg['chest_rate_sigma_rad_s'] ** 2),
        'arm_posture': cfg['arm_weight'] * torch.exp(-measurements['arm_reference_mse_rad2'] / cfg['arm_sigma_rad'] ** 2),
        'arm_rate': rate_weight * torch.exp(-measurements['arm_velocity_error_mse_rad2_s2'] / cfg['arm_rate_sigma_rad_s'] ** 2),
        'waist_posture': cfg['waist_weight'] * torch.exp(-measurements['waist_reference_mse_rad2'] / cfg['waist_sigma_rad'] ** 2),
    }


def stability_costs(chest_rate, clearance, height_target, loaded, reference_contact, foot_speed, cfg):
    """Graded motion/contact costs retain pressure beyond an exponential tail."""
    normalized = (chest_rate / cfg['angular_scale_rad_s']).clamp(max=cfg['angular_clip'])
    huber = torch.where(normalized <= 1., .5 * normalized.square(), normalized - .5)
    swing = ~reference_contact & (height_target > .022)
    excess = ((clearance - height_target - cfg['height_tolerance_m']).clamp_min(0.)
              / cfg['height_scale_m']).clamp(max=cfg['height_clip'])
    overshoot = (excess.square() * swing).sum(-1) / swing.sum(-1).clamp_min(1)
    stance = loaded & reference_contact
    sliding = ((foot_speed / cfg['slip_scale_m_s']).clamp(max=cfg['slip_clip']).square()
               * stance).sum(-1) / stance.sum(-1).clamp_min(1)
    return {'chest_energy': -cfg['angular_weight'] * huber,
            'swing_overshoot': -cfg['height_weight'] * overshoot,
            'stance_sliding': -cfg['slip_weight'] * sliding}
