import json
from pathlib import Path

import torch

from leg_humanoid_lab.robot import JOINTS, LOWER, UPPER
from leg_humanoid_lab.upper_body_objective import (
    guarded_upper_targets, nominal_upper_velocity, upper_reward_terms)


PROFILE = json.loads((Path(__file__).resolve().parents[1] / 'config/upper_body_control_v6.json').read_text())


def test_new_control_retains_every_leg_target_and_native_limits():
    torch.manual_seed(17)
    original = torch.randn(7, 23)
    before = original.clone()
    reference = torch.tensor(UPPER, dtype=original.dtype).expand(7, -1)
    result = guarded_upper_targets(original, reference, torch.ones_like(original),
        torch.tensor(LOWER, dtype=original.dtype), torch.tensor(UPPER, dtype=original.dtype), PROFILE['residual_rad'])
    leg = [i for i, name in enumerate(JOINTS) if any(x in name for x in ('hip_', 'knee_', 'ankle_'))]
    assert torch.equal(original, before)
    assert torch.equal(result[:, leg], before[:, leg])
    upper = [i for i in range(23) if i not in leg]
    assert (result[:, upper] <= torch.tensor(UPPER, dtype=result.dtype)[upper]).all()
    assert (result[:, upper] >= torch.tensor(LOWER, dtype=result.dtype)[upper]).all()


def test_reference_rate_rewards_controlled_return_instead_of_freezing():
    reference = torch.zeros(2, 23)
    previous = torch.full_like(reference, .01)
    rate = nominal_upper_velocity(reference, previous, torch.tensor([0, 5]), .02)
    assert not rate[0].any()  # Reset cannot create a velocity impulse.
    assert torch.allclose(rate[1], torch.full((23,), -.5))
    base = {'chest_tilt_rad': torch.zeros(2), 'chest_angular_xy_rad_s': torch.zeros(2),
            'arm_reference_mse_rad2': torch.zeros(2), 'waist_reference_mse_rad2': torch.zeros(2),
            'arm_velocity_error_mse_rad2_s2': torch.tensor([0., .25])}
    terms = upper_reward_terms(base, torch.tensor([2, 2]), torch.tensor([.8, .8]), PROFILE['reward'])
    assert terms['arm_rate'][0] > terms['arm_rate'][1]
    assert sum(t.max() for t in terms.values()) <= 5.75


def test_torso_wobble_and_arm_sag_are_independently_penalized():
    m = {'chest_tilt_rad': torch.tensor([.03, .5, .03]),
         'chest_angular_xy_rad_s': torch.tensor([0., 1.5, 0.]),
         'arm_reference_mse_rad2': torch.tensor([0., 0., .04]),
         'waist_reference_mse_rad2': torch.zeros(3),
         'arm_velocity_error_mse_rad2_s2': torch.zeros(3)}
    t = upper_reward_terms(m, torch.tensor([2, 2, 2]), torch.tensor([.3, .3, .3]), PROFILE['reward'])
    assert t['chest_posture'][0] > t['chest_posture'][1]
    assert t['chest_rate'][0] > t['chest_rate'][1]
    assert t['arm_posture'][0] > t['arm_posture'][2]
    assert t['arm_posture'][0] == t['arm_posture'][1]
    assert t['chest_posture'][0] == t['chest_posture'][2]


def test_predeclared_plateau_revisions_are_bounded_and_keep_motion_targets():
    measurements = {'chest_tilt_rad': torch.zeros(1), 'chest_angular_xy_rad_s': torch.zeros(1),
                    'arm_reference_mse_rad2': torch.zeros(1), 'waist_reference_mse_rad2': torch.zeros(1),
                    'arm_velocity_error_mse_rad2_s2': torch.zeros(1)}
    for revision, bound in [('0', 5.75), ('1', 6.75), ('2', 7.5)]:
        cfg = {**PROFILE['reward'], **PROFILE['reward_revisions'][revision]}
        terms = upper_reward_terms(measurements, torch.tensor([4]), torch.tensor([0.]), cfg)
        assert sum(terms.values()).item() <= bound
        assert cfg['arm_sigma_rad'] == .08 and cfg['tilt_deadband_deg'] == 4.
