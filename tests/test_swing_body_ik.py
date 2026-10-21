"""Geometry/limit checks for the independent body-aware swing ablation."""
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from leg_humanoid_lab.robot import JOINTS, LOWER, ROOT, UPPER, fk
from leg_humanoid_lab.swing_body_ik import BodyAwareSwingIK


def peak_fixture():
    # Original 1805 native left-leading frame, t=1.52 s. This contains poses,
    # not a fabricated policy rollout or an assertion of dynamic feasibility.
    q=torch.tensor([[.1227937043,0.,.0422103405,.5597350597,.1125556007,
        .0703261122,1.0653752089,-.4999989867,-.1326413006,0.,0.,0.,-.15,
        -.3375603855,.0435256138,.0152732786,-.7193333507,.3821058869,
        -.0461258180,.5908583999,.3214885294,-1.1924864054,.4660361111]])
    quat=torch.tensor([[.9964994192,-.0688302293,-.0080420785,.0467603728]])
    height=torch.tensor([.8280601501])
    target=torch.tensor([[.0607712418,.0029999157]])
    contact=torch.tensor([[False,True]])
    return q,quat,height,target,contact


def test_native_torch_leg_fk_matches_authoritative_signed_axes():
    lib=json.loads((ROOT/'config/v5_gaits_lift6.json').read_text())
    q=torch.tensor([lib['plans'][name]['q'][19] for name in ('sidestep_left','sidestep_right')])
    ik=BodyAwareSwingIK(iterations=3)
    pos,rot=ik.leg_fk(q[...,ik.joint_ids])
    for sample in range(len(q)):
        native=fk(q[sample].numpy())
        for side,name in enumerate(('left','right')):
            assert np.allclose(pos[sample,side],native[name+'_foot_link'][:3,3],atol=1e-6)
            assert np.allclose(rot[sample,side],native[name+'_foot_link'][:3,:3],atol=1e-6)


def test_measured_peak_pose_has_feasible_25mm_bounded_swing_solution():
    q,quat,height,target,contact=peak_fixture();target[0,0]=.025
    ik=BodyAwareSwingIK(iterations=3)
    result=ik(q,quat,height,target,contact)
    clearance=ik.sole_height(result[...,ik.joint_ids],quat,height)
    assert abs(float(clearance[0,0])-.025)<.0001
    assert torch.all(result>=torch.tensor(LOWER)-1e-6)
    assert torch.all(result<=torch.tensor(UPPER)+1e-6)
    changed={int(i) for i in torch.where((result-q).abs()[0]>0)[0]}
    assert changed.issubset(set(ik.joint_ids[0].tolist()))
    assert torch.equal(result[:,ik.joint_ids[1]],q[:,ik.joint_ids[1]])


def test_infeasible_full_pose_retains_limits_and_reports_achieved_cad_height():
    q,quat,height,target,contact=peak_fixture()
    ik=BodyAwareSwingIK(iterations=3)
    result=ik(q,quat,height,target,contact)
    clearance=ik.sole_height(result[...,ik.joint_ids],quat,height)
    assert .044<float(clearance[0,0])<.048
    assert float(clearance[0,0])<float(target[0,0])-.01
    assert abs(float(result[0,JOINTS.index('left_ankle_Pitch_joint')])+.5)<1e-5
    # A requested target is never substituted for the achieved CAD geometry.
    assert not torch.allclose(clearance,target)


def test_roll_pitch_height_grid_is_finite_bounded_and_stance_unchanged():
    q,_,_,target,contact=peak_fixture()
    rotations=[];heights=[]
    for roll in (-.15,0.,.15):
        for pitch in (-.08,0.,.08):
            for sag in (0.,.015,.03):
                x,y,z,w=Rotation.from_euler('xyz',[roll,pitch,.35]).as_quat()
                rotations.append([w,x,y,z]);heights.append(.848-sag)
    count=len(heights);q=q.repeat(count,1)
    quat=torch.tensor(rotations,dtype=q.dtype);height=torch.tensor(heights,dtype=q.dtype)
    target=target.repeat(count,1);target[:,0]=.025
    contact=contact.repeat(count,1)
    ik=BodyAwareSwingIK(iterations=3);result=ik(q,quat,height,target,contact)
    clearance=ik.sole_height(result[...,ik.joint_ids],quat,height)
    assert torch.isfinite(result).all() and torch.isfinite(clearance).all()
    assert torch.all(result>=torch.tensor(LOWER)-1e-6)
    assert torch.all(result<=torch.tensor(UPPER)+1e-6)
    assert torch.equal(result[:,ik.joint_ids[1]],q[:,ik.joint_ids[1]])
    # Joint limits make some 3-cm-sag/negative-roll targets infeasible. The
    # ablation must expose this rather than claim every requested pose works.
    assert float(clearance[:,0].min())<.024
    assert int((clearance[:,0]>=.024).sum())>=24


def test_no_planned_swing_is_exact_identity():
    q,quat,height,target,contact=peak_fixture();contact[:]=True
    assert torch.equal(BodyAwareSwingIK(iterations=3)(q,quat,height,target,contact),q)
