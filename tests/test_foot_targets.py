import json
import numpy as np
from pathlib import Path
from leg_humanoid_lab.robot import ROOT
from leg_humanoid_lab.evaluation import passes

def test_foot_targets_lift_both_sides_with_single_support_and_finite_tail():
    data=json.loads((ROOT/'docs/reference/sidestep_foot_targets.json').read_text())
    heights=np.array(data['height_m']);time=np.array(data['time'])
    assert len(data['steps'])==16
    assert (heights>=0).all() and heights.max()<=.040001
    assert (heights.max(axis=0)>.035).all()
    assert not ((heights[:,0]>.005)&(heights[:,1]>.005)).any()
    assert all(sum(s['side']==side for s in data['steps'])==8 for side in ('left','right'))
    assert not heights[time>6.5].any()

def test_new_gate_rejects_dragging_or_one_sided_lift(tmp_path,monkeypatch):
    path=ROOT/'config/sidestep_foot_lift.json'
    monkeypatch.setenv('Q1_LEG_CURRICULUM_CONFIG',str(path))
    report={'skill':'sidestep','survival_rate':1.,'upright':1.,'height_error':.01,'pose_rmse':.1,'root_error':.1,'reference_lateral_range':.4,'lateral_range':.3}
    assert not passes(report)
    report.update(foot_lift_success_rate_per_foot=[.8,.8],stance_slip_m_s=.04,planned_contact_match=.9)
    assert passes(report)
    report['foot_lift_success_rate_per_foot']=[.8,.1]
    assert not passes(report)
    report['foot_lift_success_rate_per_foot']=[.8,.8];report['stance_slip_m_s']=.4
    assert not passes(report)
