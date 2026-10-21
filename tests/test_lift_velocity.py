"""Nominal feedforward must never inject reset/action discontinuities."""
import pytest
import torch

from leg_humanoid_lab.lift_objective import reference_velocity_targets
from leg_humanoid_lab.robot import JOINTS


def run(mode, previous_mode=None, episode_length=None, difference=.01):
    n=len(mode)
    ref=torch.full((n,23),difference)
    previous=torch.zeros_like(ref)
    mode=torch.tensor(mode)
    old=mode.clone() if previous_mode is None else torch.tensor(previous_mode)
    age=torch.ones(n) if episode_length is None else torch.tensor(episode_length)
    return reference_velocity_targets(ref,previous,mode,old,age,
                                      torch.tensor([[False,True]]*n),
                                      torch.tensor([[.04,0.]]*n),.02)


def test_feedforward_uses_adjacent_nominal_reference_on_swing_only():
    target=run([2,3])
    swing=[i for i,n in enumerate(JOINTS) if n.startswith('left_') and any(p in n for p in ('hip_','knee_','ankle_'))]
    others=[i for i in range(23) if i not in swing]
    assert torch.allclose(target[:,swing],torch.full((2,6),.5))
    assert torch.equal(target[:,others],torch.zeros((2,17)))


def test_stand_walk_recovery_reset_and_mode_switch_have_zero_speed():
    assert not run([0,1,4]).any()
    assert not run([2,3],previous_mode=[0,2]).any()
    assert not run([2,3],episode_length=[0,0]).any()


def test_native_velocity_limit_and_signed_difference():
    assert run([2],difference=1.).max()==10.
    assert run([2],difference=-1.).min()==-10.
    with pytest.raises(ValueError):
        reference_velocity_targets(torch.zeros(1,23),torch.zeros(1,23),torch.tensor([2]),
            torch.tensor([2]),torch.ones(1),torch.zeros(1,2,dtype=torch.bool),torch.ones(1,2),0.)
