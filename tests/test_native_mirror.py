"""Geometric proof and explicit domain restrictions for native augmentation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from leg_humanoid_lab.lift_symmetry import (
    LIMIT_EXCEPTIONS,MIRROR_INDICES,MIRROR_SIGNS,NativeMirror,
    SYMMETRIC_LOWER,SYMMETRIC_UPPER,augment_policy,
)
from leg_humanoid_lab.robot import DEFAULT,JOINTS,LOWER,ROOT,UPPER,fk


def test_joint_signs_are_derived_from_native_axial_vectors():
    assert MIRROR_SIGNS==(-1,-1,1)+(-1,)*20
    assert tuple(MIRROR_INDICES[i] for i in MIRROR_INDICES)==tuple(range(23))


def test_random_native_fk_foot_wrist_and_gripper_reflections_are_exact():
    rng=np.random.default_rng(20261021)
    mirror=NativeMirror().double();reflection=np.diag([1.,-1.,1.])
    # Full native range proves the kinematic transform even for the cases
    # where shoulder limits make the mirrored pose physically inadmissible.
    for _ in range(32):
        q=rng.uniform(LOWER,UPPER)
        mirrored=mirror.joints(torch.tensor(q)).numpy()
        actual=fk(mirrored);source=fk(q)
        for side,other in (('left','right'),('right','left')):
            for suffix in ('foot_link','wrist_yaw_link','gripper_link'):
                a=actual[side+'_'+suffix];b=source[other+'_'+suffix]
                assert np.allclose(a[:3,3],reflection@b[:3,3],atol=1e-11)
                assert np.allclose(a[:3,:3],reflection@b[:3,:3]@reflection,atol=1e-11)


def test_neutral_poses_and_joint_action_involution():
    mirror=NativeMirror().double()
    default=torch.tensor(DEFAULT)
    stand=torch.tensor(json.loads((ROOT/'config/v5_gaits_lift6.json').read_text())['default'],dtype=torch.float64)
    assert torch.equal(mirror.joints(default),default)
    assert torch.allclose(mirror.joints(stand),stand,atol=1e-7,rtol=0.)
    actions=torch.rand((17,23),dtype=torch.float64)*2-1
    assert torch.equal(mirror.actions(mirror.actions(actions)),actions)
    assert mirror.actions(actions).abs().max()<=1


def test_native_limit_asymmetry_is_exposed_and_never_clamped():
    assert LIMIT_EXCEPTIONS==('left_shoulder_Pitch_joint','right_shoulder_Pitch_joint')
    mirror=NativeMirror();q=torch.tensor(DEFAULT,dtype=torch.float32)
    q[JOINTS.index('left_shoulder_Pitch_joint')]=2.
    assert bool(((q>=mirror.lower)&(q<=mirror.upper)).all())
    assert not bool(mirror.positions_feasible(q))
    mirrored=mirror.joints(q)
    assert mirrored[JOINTS.index('right_shoulder_Pitch_joint')]==-2.
    assert mirrored[JOINTS.index('right_shoulder_Pitch_joint')]<mirror.lower[JOINTS.index('right_shoulder_Pitch_joint')]


def test_symmetric_native_subdomain_is_limit_invariant():
    mirror=NativeMirror();rng=np.random.default_rng(5)
    positions=torch.tensor(rng.uniform(SYMMETRIC_LOWER,SYMMETRIC_UPPER,(128,23)),dtype=torch.float32)
    assert bool(mirror.positions_feasible(positions).all())
    assert bool(mirror.positions_feasible(mirror.joints(positions)).all())
    for name in LIMIT_EXCEPTIONS:
        i=JOINTS.index(name)
        assert SYMMETRIC_LOWER[i]==pytest.approx(-1.1345)
        assert SYMMETRIC_UPPER[i]==pytest.approx(1.1345)


def test_native_residual_imu_control_targets_obey_the_same_mirror():
    from leg_humanoid_lab.lift_objective import control_targets
    mirror=NativeMirror()
    lib=json.loads((ROOT/'config/v5_gaits_lift6.json').read_text())
    stand=torch.tensor(lib['default']);ref=torch.tensor([lib['plans']['sidestep_left']['q'][19]])
    actions=torch.linspace(-.7,.7,23).unsqueeze(0)
    gravity=torch.tensor([[.03,.06,-.9977475]])
    angular=torch.tensor([[.1,-.2,.05]])
    contact=torch.tensor([[False,True]]);height=torch.tensor([[.06,.003]])
    lower=torch.tensor(LOWER,dtype=ref.dtype);upper=torch.tensor(UPPER,dtype=ref.dtype)
    target,_=control_targets(ref,stand,actions,gravity,angular,contact,height,lower,upper)
    mirrored,_=control_targets(mirror.joints(ref),stand,mirror.actions(actions),
        gravity*torch.tensor([1.,-1.,1.]),angular*torch.tensor([-1.,1.,-1.]),
        contact.flip(-1),height.flip(-1),lower,upper)
    assert torch.allclose(mirrored,mirror.joints(target),atol=1e-6,rtol=0.)


def test_observation_layout_vectors_commands_phase_and_involution():
    mirror=NativeMirror();obs=torch.arange(3*110,dtype=torch.float32).reshape(3,110)/100
    obs[:,81:84]=torch.eye(3)
    phase=torch.tensor([.12,.24,.36]);obs[:,84]=torch.sin(2*torch.pi*phase);obs[:,85]=torch.cos(2*torch.pi*phase)
    result=mirror.observations(obs)
    assert torch.equal(mirror.observations(result),obs)
    assert torch.equal(result[:,81:84],obs[:,81:84])
    assert torch.equal(result[:,86],obs[:,86])
    assert torch.equal(result[[0,2],84:86],obs[[0,2],84:86])
    assert torch.equal(result[1,84:86],-obs[1,84:86])
    assert torch.equal(result[:,3:6],obs[:,3:6]*torch.tensor([-1.,1.,-1.]))
    assert torch.equal(result[:,78:81],obs[:,78:81]*torch.tensor([1.,-1.,-1.]))
    for start in (9,32,55,87):
        assert torch.equal(result[:,start:start+23],mirror.joints(obs[:,start:start+23]))


def test_observation_feasibility_checks_actual_and_reference_shoulders():
    mirror=NativeMirror();stand=torch.tensor(DEFAULT,dtype=torch.float32)
    obs=torch.zeros(3,110);obs[:,81]=1
    assert bool(mirror.observations_feasible(obs,stand).all())
    shoulder=JOINTS.index('left_shoulder_Pitch_joint')
    obs[1,9+shoulder]=2.
    obs[2,87+shoulder]=2.
    assert mirror.observations_feasible(obs,stand).tolist()==[True,False,False]
    bad_stand=stand.clone();bad_stand[shoulder]=.1
    with pytest.raises(ValueError,match='mirror-invariant'):
        mirror.observations_feasible(obs,bad_stand)


def callback_fixture():
    stand=torch.tensor(DEFAULT,dtype=torch.float32)
    env=SimpleNamespace(unwrapped=SimpleNamespace(stand_q=stand))
    obs=torch.zeros(3,110);obs[:,81]=1;obs[:,0:9]=torch.tensor([.1,.2,.3,.4,.5,.6,.1,.2,-.97])
    obs[:,78:81]=torch.tensor([.1,.12,.2])
    shoulder=JOINTS.index('left_shoulder_Pitch_joint')
    obs[1,9+shoulder]=2.  # Legal source q, illegal exact mirrored q.
    obs[2,87+shoulder]=.9  # Legal mirrored ref, but not safe for +/- .35 residual.
    return env,{'policy':obs}


def test_rsl_callback_invalid_rows_duplicate_identically_and_have_zero_loss():
    env,obs=callback_fixture();augmented,_=augment_policy(env,obs=obs)
    assert env._q1_mirror_feasible.tolist()==[True,False,False]
    assert augmented['policy'].shape==(6,110)
    assert torch.equal(augmented['policy'][4:],obs['policy'][1:])
    assert torch.equal(augmented['policy'][3],NativeMirror().observations(obs['policy'][0]))
    actor=torch.nn.Sequential(torch.nn.Linear(110,31),torch.nn.ELU(),torch.nn.Linear(31,23))
    predicted=actor(augmented['policy'])
    _,requested=augment_policy(env,obs=None,actions=predicted[:3])
    per_sample=(predicted[3:]-requested[3:].detach()).square().mean(-1)
    assert torch.equal(per_sample[1:],torch.zeros(2))
    assert per_sample[0]>0


def test_rsl_callback_both_arguments_share_mask_and_validate_cache():
    env,obs=callback_fixture();actions=torch.rand(3,23)
    augmented,augmented_actions=augment_policy(env,obs=obs,actions=actions)
    assert torch.equal(augmented_actions[:3],actions)
    assert torch.equal(augmented_actions[4:],actions[1:])
    assert torch.equal(augmented_actions[3],NativeMirror().actions(actions[0]))
    with pytest.raises(ValueError,match='preceding observation'):
        augment_policy(SimpleNamespace(),obs=None,actions=actions)
    with pytest.raises(ValueError,match='batch size/device'):
        augment_policy(env,obs=None,actions=actions[:2])


def test_rsl_callback_tensordict_batch_metadata_when_dependency_available():
    TensorDict=pytest.importorskip('tensordict').TensorDict
    env,obs=callback_fixture();batch=TensorDict(obs,batch_size=[3])
    result,_=augment_policy(env,obs=batch)
    assert isinstance(result,TensorDict)
    assert result.batch_size==torch.Size([6])
    assert torch.equal(result['policy'][4:],batch['policy'][1:])
