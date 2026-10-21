"""Batched lift control and rewards, independent of the simulator runtime.

These joint targets still require real actuators and ground contacts to move
the robot. They neither alter a root state nor establish dynamic feasibility.
Training, evaluation and playback must use the same control profile.
"""
from __future__ import annotations

import torch

from .robot import JOINTS


CONTROL_PROFILE = "swing_pitch_002_support_035_imu_020_v1"
SWING_HEIGHT_M = .008
_SWING_SCALES = {
    "hip_Pitch": .02,
    "hip_Roll": .10,
    "hip_Yaw": .08,
    "knee_Pitch": .02,
    "ankle_Pitch": .02,
    "ankle_Roll": .05,
}
_LEG_IDS = tuple(
    tuple(JOINTS.index(f"{side}_{part}_joint") for part in _SWING_SCALES)
    for side in ("left", "right")
)


def _swing_mask(reference_contact, reference_height):
    return (~reference_contact.bool()) & (reference_height > SWING_HEIGHT_M)


def control_targets(
    refq,
    stand_q,
    actions,
    gravity_b,
    angular_b,
    reference_contact,
    reference_height,
    lower,
    upper,
):
    """Return native bounded targets and per-joint residual scales.

    Inputs have a common batch dimension; the final dimensions are 23 for
    joints, 3 for body vectors, and 2 for feet (left, right). ``stand_q`` and
    the limits may be unbatched. Stance legs, the waist and the arms retain a
    0.35-rad residual. Small swing pitch residuals prevent a policy from
    cancelling the complete reference knee lift. IMU feedback is applied with
    the native left/right pitch signs and is attenuated on the swing leg.
    """
    swing = _swing_mask(reference_contact, reference_height)
    scales = torch.full_like(refq, .35)
    swing_scales = refq.new_tensor(tuple(_SWING_SCALES.values()))
    for foot, ids in enumerate(_LEG_IDS):
        scales[..., list(ids)] = torch.where(swing[..., foot, None], swing_scales, .35)

    # Express the commanded excursion about the same neutral pose used in the
    # observations; this remains a joint target, never a root-pose constraint.
    target = stand_q + (refq - stand_q) + scales * actions.clamp(-1., 1.)
    roll = torch.atan2(-gravity_b[..., 1], -gravity_b[..., 2])
    pitch = torch.atan2(gravity_b[..., 0], -gravity_b[..., 2])
    pitch_feedback = (pitch + .15 * angular_b[..., 1]).clamp(-.15, .15)
    roll_feedback = (roll + .15 * angular_b[..., 0]).clamp(-.15, .15)
    for foot, (side, sign) in enumerate((("left", 1.), ("right", -1.))):
        gain = torch.where(swing[..., foot], torch.full_like(pitch_feedback, .2), 1.)
        target[..., JOINTS.index(f"{side}_ankle_Pitch_joint")] += sign * pitch_feedback * gain
        target[..., JOINTS.index(f"{side}_ankle_Roll_joint")] += roll_feedback * gain
    return torch.maximum(torch.minimum(target, upper), lower), scales


def swing_terms(clearance, target, loaded, reference_contact):
    """Return per-environment terms normalized only over active swing feet.

    ``progress`` requires an actually unloaded foot; a loaded or stationary
    foot cannot earn clearance reward. ``miss`` is a linear fractional height
    deficit, so a grounded foot misses a 4-cm target by 1 rather than entering
    the flat tail of a Gaussian. ``wrong_contact`` penalizes loaded swings
    even when horizontal foot speed is zero. All terms are zero in stance.
    Heights are the true minimum native-CAD sole heights above the ground;
    ``loaded`` must come from physical contacts, not the reference schedule.
    """
    swing = _swing_mask(reference_contact, target)
    active = swing.to(dtype=clearance.dtype)
    count = active.sum(dim=-1).clamp_min(1.)
    safe_target = target.clamp_min(SWING_HEIGHT_M)
    airborne = (~loaded.bool()).to(dtype=clearance.dtype)
    height_fraction = (clearance / safe_target).clamp(0., 1.)
    return {
        "progress": (active * height_fraction * airborne).sum(dim=-1) / count,
        "miss": (active * ((target - clearance).clamp_min(0.) / safe_target)).sum(dim=-1) / count,
        "wrong_contact": (active * loaded.to(dtype=clearance.dtype)).sum(dim=-1) / count,
        "unload": (active * airborne).sum(dim=-1) / count,
    }
