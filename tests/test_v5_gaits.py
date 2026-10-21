"""Native CAD invariants for the reduced-sway ablation reference.

These tests deliberately make no claim that a bounded IK plan is dynamically
stable. The support approximation records the adverse reduced-sway result.
"""
import hashlib
import json

import numpy as np

from leg_humanoid_lab.robot import ROOT, FOOT_VERTICES, JOINTS, LOWER, UPPER, URDF, fk


def _library():
    return json.loads((ROOT/'config/v5_gaits.json').read_text())


def test_v5_provenance_and_original_references_are_explicit():
    data = _library()
    assert data['contract']=='q1_leg_command_v5' and data['schema']==5
    for key, path in (
        ('robot_urdf_sha256', URDF),
        ('source_v4_builder_sha256', ROOT/'tools/build_v4_gaits.py'),
        ('source_v4_gaits_sha256', ROOT/'config/v4_gaits.json'),
    ):
        assert data[key]==hashlib.sha256(path.read_bytes()).hexdigest()
    # This .02 m / .04 m lift baseline is immutable. Later CLI extensions
    # preserve its historical generation hash rather than rewriting its data.
    assert data['builder_sha256']=='0659a341e2fe5a4b6e4beaf370d687a6843efcd8e04303768bd58a9b49a93be0'
    assert hashlib.sha256((ROOT/'config/v5_gaits.json').read_bytes()).hexdigest()=='37376238740a3955ddc815f83e2832e555cb7ab7babac50afdb7fb4ea3701e83'
    old=json.loads((ROOT/'config/v4_gaits.json').read_text())
    assert data['joint_names']==list(JOINTS)
    assert data['default']==old['default']
    for name in ('stand', 'sidestep_left', 'sidestep_right'):
        assert data['plans'][name]==old['plans'][name]


def test_v5_true_cad_both_feet_lift_with_native_limits_and_speed():
    data=_library()
    for name, plan in data['plans'].items():
        q=np.asarray(plan['q'])
        assert q.shape==(61,23)
        assert np.all(q>=LOWER-1e-7) and np.all(q<=UPPER+1e-7)
        assert np.max(np.abs(np.diff(q,axis=0)))/(plan['duration_s']/60)<10.
        assert np.allclose(q[0],q[-1],atol=1e-7)
        peaks=np.zeros(2)
        for pose, root, targets in zip(q,plan['root_offset_body'],plan['foot_height']):
            transforms=fk(pose)
            for index,side in enumerate(('left','right')):
                transform=transforms[side+'_foot_link']
                actual=np.min(FOOT_VERTICES[side]@transform[2,:3]+transform[2,3])+data['root_height']+root[2]
                # Bounded ankle orientation IK has a documented <=4.2 mm
                # source-CAD clearance residual in the unchanged gestures.
                assert abs(actual-targets[index])<.005, (name,side,actual,targets[index])
                peaks[index]=max(peaks[index],actual)
        if name!='stand':
            assert np.min(peaks)>.035, (name,peaks)


def test_v5_sway_is_ablation_and_approximation_records_support_tradeoff():
    data=_library()
    assert data['walk_root_sway_amplitude_m']==.02
    for name in ('forward','backward','left','right','yaw_left','yaw_right'):
        offset=np.asarray(data['plans'][name]['root_offset_body'])
        assert np.isclose(offset[:,1].min(),-.02)
        assert np.isclose(offset[:,1].max(),.02)
    comparison=json.loads((ROOT/'config/v5_gait_comparison.json').read_text())
    old,new=comparison['v4']['forward'],comparison['v5']['forward']
    assert new['root_sway_velocity_peak_m_s']<old['root_sway_velocity_peak_m_s']
    assert new['single_support_lateral_inside_fraction']<old['single_support_lateral_inside_fraction']
    assert 'approximate' in comparison['scope']
    assert 'neither reference has passed dynamic validation' in comparison['scope']
