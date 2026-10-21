"""Source CAD checks for a 6 cm command experiment, not dynamic acceptance."""
import hashlib
import json

import numpy as np

from leg_humanoid_lab.robot import ROOT, FOOT_VERTICES, JOINTS, LOWER, UPPER, fk


def test_lift6_has_explicit_experiment_parameters_and_current_builder_hash():
    data=json.loads((ROOT/'config/v5_gaits_lift6.json').read_text())
    assert data['contract']=='q1_leg_command_v5'
    assert data['walk_root_sway_amplitude_m']==.045
    assert data['swing_lift_amplitude_m']==.06
    assert data['builder_sha256']==hashlib.sha256((ROOT/'tools/build_v5_gaits.py').read_bytes()).hexdigest()
    original=json.loads((ROOT/'config/v4_gaits.json').read_text())
    assert data['plans']['stand']==original['plans']['stand']
    assert data['default']==original['default']
    for name,plan in data['plans'].items():
        assert plan['contact']==original['plans'][name]['contact']
        assert plan['root_offset_body']==original['plans'][name]['root_offset_body']


def test_lift6_true_cad_clearance_limits_and_native_velocity_cap():
    data=json.loads((ROOT/'config/v5_gaits_lift6.json').read_text())
    original=json.loads((ROOT/'config/v4_gaits.json').read_text())
    assert data['joint_names']==list(JOINTS)
    for name,plan in data['plans'].items():
        q=np.asarray(plan['q'])
        assert q.shape==(61,23)
        assert np.all(q>=LOWER-1e-7) and np.all(q<=UPPER+1e-7)
        assert np.max(np.abs(np.diff(q,axis=0)))/(plan['duration_s']/60)<10.
        assert np.allclose(q[0],q[-1],atol=1e-7)
        peaks=np.zeros(2)
        for pose,root in zip(q,plan['root_offset_body']):
            transforms=fk(pose)
            for index,side in enumerate(('left','right')):
                transform=transforms[side+'_foot_link']
                clearance=np.min(FOOT_VERTICES[side]@transform[2,:3]+transform[2,3])+data['root_height']+root[2]
                peaks[index]=max(peaks[index],clearance)
                assert clearance>-.001, (name,side,clearance)
        if name!='stand':
            # Native ankle orientation limits lose up to 13.7 mm of nominal
            # clearance; prove measured CAD improvement, not a 63 mm claim.
            assert min(peaks)>.049, (name,peaks)
            previous=np.asarray(original['diagnostics'][name]['maximum_fk_clearance_m'])
            assert np.min(peaks-previous)>.009, (name,peaks,previous)
