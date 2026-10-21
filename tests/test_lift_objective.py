"""Physical invariants for swing-only scoring and bounded native targets."""
from itertools import product

import numpy as np
import torch

from leg_humanoid_lab.lift_objective import control_targets, swing_terms
from leg_humanoid_lab.robot import JOINTS, LOWER, UPPER, FOOT_VERTICES, fk
from leg_humanoid_lab.v4_reference import V4Reference


def _score(terms):
    return 6 * terms["progress"] + terms["unload"] - 8 * terms["miss"] - 3 * terms["wrong_contact"]


def _targets(q, contact, height, actions=None, gravity=None, angular=None):
    q = torch.as_tensor(q, dtype=torch.float64)
    if q.ndim == 1:
        q = q[None, :]
    batch = q.shape[0]
    return control_targets(
        q,
        torch.as_tensor(V4Reference().default, dtype=q.dtype),
        torch.zeros_like(q) if actions is None else actions,
        torch.tensor([[0., 0., -1.]], dtype=q.dtype).expand(batch, -1) if gravity is None else gravity,
        torch.zeros((batch, 3), dtype=q.dtype) if angular is None else angular,
        torch.as_tensor(contact, dtype=torch.bool).expand(batch, -1),
        torch.as_tensor(height, dtype=q.dtype).expand(batch, -1),
        torch.as_tensor(LOWER, dtype=q.dtype),
        torch.as_tensor(UPPER, dtype=q.dtype),
    )


def test_loaded_stationary_foot_loses_to_real_unloaded_clearance():
    # The second foot is supporting throughout. It must not inflate lift score.
    clearance = torch.tensor([[0., 0.], [.025, 0.], [.04, 0.], [.04, 0.]])
    target = torch.tensor([[.04, .003]]).expand(4, -1)
    loaded = torch.tensor([[True, True], [False, True], [False, True], [True, True]])
    contact = torch.tensor([[False, True]]).expand(4, -1)
    terms = swing_terms(clearance, target, loaded, contact)
    score = _score(terms)
    assert score[0] < score[1] < score[2]
    assert score[0] < 0 and score[1] > 0
    assert terms["miss"][0] == 1  # Four centimetres missed, not a near-full reward.
    assert terms["progress"][3] == 0  # Height alone is insufficient without unloading.
    assert terms["wrong_contact"][3] == 1


def test_support_foot_height_and_number_do_not_dilute_active_swing():
    target = torch.tensor([[.04, .003], [.04, .003], [.04, .04]])
    contact = torch.tensor([[False, True], [False, True], [False, False]])
    clearance = torch.tensor([[.025, 0.], [.025, .30], [.025, .025]])
    loaded = torch.tensor([[False, True], [False, False], [False, False]])
    terms = swing_terms(clearance, target, loaded, contact)
    for term in terms.values():
        assert torch.allclose(term, term[:1].expand_as(term))


def test_no_swing_gets_no_free_lift_reward_or_penalty():
    terms = swing_terms(
        torch.tensor([[0., .2], [0., 0.]]),
        torch.tensor([[.04, .003], [.008, .008]]),
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([[True, True], [False, False]]),
    )
    assert all(torch.equal(term, torch.zeros(2)) for term in terms.values())
    assert all(torch.isfinite(term).all() for term in terms.values())


def test_native_residual_scales_and_signed_support_imu_feedback():
    reference = V4Reference()
    gravity = torch.tensor([[.10, -.20, -.97]], dtype=torch.float64)
    angular = torch.tensor([[.3, .4, 0.]], dtype=torch.float64)
    target, scales = _targets(reference.default, [False, True], [.04, .003], gravity=gravity, angular=angular)
    for side, foot in (("left", 0), ("right", 1)):
        expected = {"hip_Pitch": .02, "hip_Roll": .10, "hip_Yaw": .08,
                    "knee_Pitch": .02, "ankle_Pitch": .02, "ankle_Roll": .05}
        for part, swing_scale in expected.items():
            assert scales[0, JOINTS.index(f"{side}_{part}_joint")] == (swing_scale if foot == 0 else .35)
    for joint, scale in zip(JOINTS, scales[0]):
        if "waist" in joint or "shoulder" in joint or "elbow" in joint:
            assert scale == .35
    delta = target[0] - torch.as_tensor(reference.default, dtype=target.dtype)
    assert torch.isclose(delta[JOINTS.index("left_ankle_Pitch_joint")], torch.tensor(.03, dtype=target.dtype))
    assert torch.isclose(delta[JOINTS.index("right_ankle_Pitch_joint")], torch.tensor(-.15, dtype=target.dtype))
    assert torch.isclose(delta[JOINTS.index("left_ankle_Roll_joint")], torch.tensor(.03, dtype=target.dtype))
    assert torch.isclose(delta[JOINTS.index("right_ankle_Roll_joint")], torch.tensor(.15, dtype=target.dtype))


def test_action_clipping_and_native_joint_limits_remain_enforced():
    reference = V4Reference()
    batch = np.stack([LOWER, UPPER])
    actions = torch.tensor([[-10.] * 23, [10.] * 23], dtype=torch.float64)
    target, _ = _targets(batch, [False, False], [.04, .04], actions=actions)
    assert torch.all(target >= torch.as_tensor(LOWER))
    assert torch.all(target <= torch.as_tensor(UPPER))
    # Clipping is at the action boundary as well as the final joint boundary.
    clipped, _ = _targets(batch, [False, False], [.04, .04], actions=actions.clamp(-1, 1))
    assert torch.equal(target, clipped)


def test_adversarial_swing_residuals_cannot_flatten_true_native_fk_peak():
    reference = V4Reference()
    # Enumerate every corner of the swing leg's 6-D residual box. FK uses the
    # original STL vertices and native URDF axes; the pelvis is held only for
    # this offline geometric bound, not as a simulation or training constraint.
    corners = torch.tensor(list(product((-1., 1.), repeat=6)), dtype=torch.float64)
    for name in ("forward", "sidestep_left", "sidestep_right"):
        plan = reference.data["plans"][name]
        for foot, side in enumerate(("left", "right")):
            peak = int(np.argmax(np.asarray(plan["foot_height"])[:, foot]))
            q = np.asarray(plan["q"][peak])
            actions = torch.zeros((len(corners), len(JOINTS)), dtype=torch.float64)
            leg_ids = [JOINTS.index(f"{side}_{part}_joint") for part in (
                "hip_Pitch", "hip_Roll", "hip_Yaw", "knee_Pitch", "ankle_Pitch", "ankle_Roll")]
            actions[:, leg_ids] = corners
            targets, _ = _targets(np.repeat(q[None, :], len(corners), axis=0),
                                  plan["contact"][peak], plan["foot_height"][peak], actions=actions)
            minimum = float("inf")
            for candidate in targets.numpy():
                transform = fk(candidate)[side + "_foot_link"]
                sole_z = np.min(FOOT_VERTICES[side] @ transform[2, :3] + transform[2, 3])
                clearance = sole_z + reference.root_height + plan["root_offset_body"][peak][2]
                minimum = min(minimum, clearance)
            assert minimum > .02, (name, side, minimum)
