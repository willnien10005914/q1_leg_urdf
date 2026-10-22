import math

import pytest
import torch

from leg_humanoid_lab.robot import JOINTS, LOWER, UPPER
from leg_humanoid_lab.upper_body_stability import guarded_targets, joint_drive_variants, tilt_rad


def test_guard_preserves_leg_feedback_and_original_rows_exactly():
    generator = torch.Generator().manual_seed(42)
    original = torch.randn(4, 23, generator=generator)
    before = original.clone()
    reference = torch.zeros_like(original)
    actions = torch.ones_like(original)
    result = guarded_targets(original, reference, actions, torch.arange(4),
                             torch.tensor(LOWER), torch.tensor(UPPER))
    assert torch.equal(result[:2], original[:2])
    leg = [i for i, name in enumerate(JOINTS) if any(p in name for p in ('hip_', 'knee_', 'ankle_'))]
    assert torch.equal(result[:, leg], original[:, leg])
    assert result[3, JOINTS.index('center_waist_Roll_joint')] == .025
    assert result[3, JOINTS.index('center_waist_Pitch_joint')] == .04
    assert result[3, JOINTS.index('right_shoulder_Pitch_joint')] == .08
    assert torch.equal(original, before)


def test_upper_targets_still_obey_native_asymmetric_joint_limits():
    reference = torch.tensor(UPPER).repeat(4, 1)
    result = guarded_targets(reference, reference, torch.ones_like(reference), torch.arange(4),
                             torch.tensor(LOWER), torch.tensor(UPPER))
    assert (result <= torch.tensor(UPPER)).all()
    assert (result >= torch.tensor(LOWER)).all()
    with pytest.raises(ValueError):
        guarded_targets(reference, reference, reference, torch.full((4,), 9),
                        torch.tensor(LOWER), torch.tensor(UPPER))


def test_drive_changes_never_touch_legs_or_locked_wrist_and_finger_joints():
    names = list(JOINTS) + ['left_wrist_Yaw_joint', 'right_finger_1_Roll_joint']
    kp = torch.full((4, len(names)), 70.)
    kd = torch.full_like(kp, 4.)
    changed_kp, changed_kd = joint_drive_variants(kp, kd, names, torch.arange(4))
    assert torch.equal(changed_kp[[0, 2]], kp[[0, 2]])
    assert torch.equal(changed_kd[[0, 2]], kd[[0, 2]])
    for i, name in enumerate(names):
        if not any(part in name for part in ('waist_', 'shoulder_', 'elbow_')):
            assert torch.equal(changed_kp[:, i], kp[:, i])
            assert torch.equal(changed_kd[:, i], kd[:, i])
    assert changed_kp[3, 0] == 160 and changed_kd[3, 0] == 14
    assert torch.equal(kp, torch.full_like(kp, 70.))


def test_torso_tilt_detects_chest_lean_independently_of_pelvis_and_yaw():
    # A vertical pelvis does not imply the chest above its waist is vertical.
    q = torch.tensor([[1., 0., 0., 0.],
                      [math.cos(.15), math.sin(.15), 0., 0.],
                      [math.cos(.8), 0., 0., math.sin(.8)]], dtype=torch.float64)
    assert torch.allclose(tilt_rad(q), torch.tensor([0., .3, 0.], dtype=torch.float64), atol=1e-7)


def test_waist_only_variant_leaves_arms_and_legs_drives_exact():
    kp = torch.full((1, len(JOINTS)), 70.)
    kd = torch.full_like(kp, 4.)
    changed_kp, changed_kd = joint_drive_variants(kp, kd, JOINTS, torch.tensor([4]))
    for i, name in enumerate(JOINTS):
        assert changed_kp[0, i] == (120. if 'waist_' in name else 70.)
        assert changed_kd[0, i] == (10. if 'waist_' in name else 4.)
    reference = torch.zeros_like(kp)
    target = guarded_targets(reference, reference, torch.ones_like(kp), torch.tensor([4]),
                             torch.tensor(LOWER), torch.tensor(UPPER))
    assert target[0, JOINTS.index('center_waist_Pitch_joint')] == .04
    assert target[0, JOINTS.index('right_shoulder_Pitch_joint')] == .08
