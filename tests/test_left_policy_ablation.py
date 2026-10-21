"""Canonical-left inference is conditional and never changes physical controls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from leg_humanoid_lab.lift_symmetry import NativeMirror
from leg_humanoid_lab.robot import DEFAULT,JOINTS


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('left_policy_ablation',ROOT/'scripts/reinforcement_learning/rsl_rl/ablate_left_policy.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class CloneDict(dict):
    def clone(self):return CloneDict({key:value.clone() for key,value in self.items()})


def fixtures():
    env=SimpleNamespace(stand_q=torch.tensor(DEFAULT,dtype=torch.float32),
                        _manual_mode=torch.tensor([2,2,3,0,2,2,3,2]),mode=torch.zeros(8,dtype=torch.long))
    obs=torch.zeros(8,110);obs[:,81]=1.;obs[:,8]=-1.
    obs[:,:6]=torch.tensor([.1,.2,.3,.4,.5,.6])
    shoulder=JOINTS.index('left_shoulder_Pitch_joint')
    obs[5,9+shoulder]=2.  # Its exact reflected physical pose exceeds limits.
    obs[7,87+shoulder]=.9  # Mirror pose is legal but its residual margin is not.
    return env,CloneDict(policy=obs)


def means(obs):
    return torch.tanh(obs['policy'][:,:23]*.7+torch.arange(23)*.03)


def test_only_feasible_canonical_requested_left_rows_are_reflected_and_mapped_back():
    env,obs=fixtures();before=obs['policy'].clone();seen=[]
    def policy(batch):seen.append(batch['policy'].clone());return means(batch)
    wrapper=MODULE.CanonicalLeftPolicy(policy,env,torch.arange(8)>=4)
    actual=wrapper(obs)
    mirror=NativeMirror();expected_obs=before.clone();expected_obs[4]=mirror.observations(before[4])
    expected=means(CloneDict(policy=expected_obs));expected[4]=mirror.actions(expected[4])
    assert torch.equal(actual,expected)
    assert torch.equal(seen[0],expected_obs) and torch.equal(obs['policy'],before)
    assert wrapper.last_mask.tolist()==[False,False,False,False,True,False,False,False]
    assert wrapper.last_requested.tolist()==[False,False,False,False,True,True,False,True]
    assert wrapper.mirrored_samples.tolist()==[0,0,0,0,1,0,0,0]
    # env.mode deliberately remains stand: selection follows the manual request
    # before the native _pre_physics_step applies the next left maneuver.
    assert torch.all(env.mode==0)


def test_other_modes_and_post_first_episode_rows_use_original_policy_unchanged():
    env,obs=fixtures();wrapper=MODULE.CanonicalLeftPolicy(means,env,torch.arange(8)>=4)
    wrapper.active[4:]=False
    assert torch.equal(wrapper(obs),means(obs))
    assert not wrapper.last_mask.any() and not wrapper.last_requested.any()
    wrapper.active[:]=True;env._manual_mode[4:]=3
    assert torch.equal(wrapper(obs),means(obs))
    assert not wrapper.last_mask.any() and not wrapper.last_requested.any()


def test_source_control_and_native_interface_must_match_exactly():
    current={'robot':'q1_leg_lift_v5','schema':5,'urdf_sha256':'u','gait_sha256':'g',
             'control_sha256':'c','control_profile':'current','motor':{'reference_velocity_feedforward':True},
             'joint_names':[str(index) for index in range(23)],'action_size':23,'observation_size':110,'proxy':False}
    MODULE.validate_contract(dict(current),current)
    for key,value in [('urdf_sha256','changed'),('gait_sha256','changed'),('control_sha256','changed'),
                      ('control_profile','old'),('action_size',22),('observation_size',109),('proxy',True)]:
        saved={**current,key:value}
        with pytest.raises(ValueError):MODULE.validate_contract(saved,current)
    malformed={**current,'joint_names':['duplicate']*23}
    with pytest.raises(ValueError):MODULE.validate_contract(malformed,malformed)


def test_diagnostic_cannot_overwrite_previous_results(tmp_path):
    path=tmp_path/'diagnostic.json';MODULE.write_fresh_json(path,{'ready':False})
    with pytest.raises(FileExistsError):MODULE.write_fresh_json(path,{'ready':True})
