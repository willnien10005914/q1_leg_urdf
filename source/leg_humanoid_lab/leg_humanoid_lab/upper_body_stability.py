"""Experimental joint-drive variants; never modify the accepted V5 contract.

These helpers do not constrain the base, change native effort limits, or filter
rendered poses. They are for physical ablation before any retraining decision.
"""
from __future__ import annotations

import torch

from .robot import JOINTS


VARIANTS = ('baseline', 'upper_pd', 'upper_residual_guard', 'upper_pd_guard', 'waist_pd_guard')
UPPER_IDS = tuple(i for i, name in enumerate(JOINTS)
                  if any(part in name for part in ('waist_', 'shoulder_', 'elbow_')))
PROFILE = {
    'schema': 1,
    'experimental': True,
    'revision': 'waist_isolation_r3',
    'waist': {'stiffness': 160., 'damping': 14.},
    'shoulder': {'stiffness': 100., 'damping': 8.},
    'elbow': {'stiffness': 90., 'damping': 8.},
    'waist_only': {'stiffness': 120., 'damping': 10.},
    'residual_rad': {'waist_Yaw': .08, 'waist_Roll': .025,
                     'waist_Pitch': .04, 'arm': .08},
    'scope': 'Native joint targets and PD only; same actor, legs, contacts and motor limits; no root force or pose filtering',
}


def guarded_targets(original, reference, actions, variant, lower, upper):
    """Protect upper reference excursions; keep every leg target bit-identical."""
    if original.shape != reference.shape or actions.shape != original.shape:
        raise ValueError('Joint target/reference/action shapes must match')
    if original.shape[-1] != len(JOINTS) or variant.shape != original.shape[:-1]:
        raise ValueError('Expected 23 joints and a variant ID per environment')
    if bool(((variant < 0) | (variant >= len(VARIANTS))).any()):
        raise ValueError('Unknown upper-body variant')
    selected = (variant == 2) | (variant == 3) | (variant == 4)
    result = original.clone()
    for joint in UPPER_IDS:
        name = JOINTS[joint]
        cap = next((value for part, value in PROFILE['residual_rad'].items()
                    if part != 'arm' and part in name), PROFILE['residual_rad']['arm'])
        value = (reference[..., joint] + cap * actions[..., joint].clamp(-1., 1.)).clamp(
            min=lower[joint], max=upper[joint])
        result[..., joint] = torch.where(selected, value, result[..., joint])
    return result


def joint_drive_variants(stiffness, damping, joint_names, variant):
    """Return drive buffers; wrists/fingers/legs and baseline rows are exact."""
    if stiffness.shape != damping.shape or stiffness.shape[-1] != len(joint_names):
        raise ValueError('Full native drive buffers must match joint names')
    if variant.shape != stiffness.shape[:-1]:
        raise ValueError('Expected a variant ID per native drive row')
    kp, kd = stiffness.clone(), damping.clone()
    selected = (variant == 1) | (variant == 3)
    for i, name in enumerate(joint_names):
        for part in ('waist', 'shoulder', 'elbow'):
            if part + '_' in name:
                kp[..., i] = torch.where(selected, PROFILE[part]['stiffness'], kp[..., i])
                kd[..., i] = torch.where(selected, PROFILE[part]['damping'], kd[..., i])
                break
        if 'waist_' in name:
            kp[..., i] = torch.where(variant == 4, PROFILE['waist_only']['stiffness'], kp[..., i])
            kd[..., i] = torch.where(variant == 4, PROFILE['waist_only']['damping'], kd[..., i])
    return kp, kd


def tilt_rad(quaternion):
    """Inclination from vertical of a wxyz body frame; insensitive to yaw."""
    normalized = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (1. - 2. * normalized[..., 1:3].square().sum(-1)).clamp(-1., 1.).acos()
