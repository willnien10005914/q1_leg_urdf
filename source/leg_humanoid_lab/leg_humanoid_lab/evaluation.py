"""One set of configurable gates for both physics backends."""
import json
import os
from pathlib import Path
from .robot import ROOT

def passes(report):
    config=json.loads(Path(os.environ.get('Q1_LEG_CURRICULUM_CONFIG',ROOT/'config/curriculum.json')).read_text())
    common=config['common_gates']
    ok=report['survival_rate']>=common['survival_rate_min'] and report['upright']>=common['upright_min'] and report['height_error']<common['height_error_m_max']
    if report['skill']=='walk':
        gate=config['walk_gates']
        ok=ok and report['velocity_error']<gate['mean_velocity_error_m_s_max'] and report['displacement']>gate['displacement_m_min']
    if report['skill']=='sidestep':
        gate=config['sidestep_gates']
        ok=ok and report['pose_rmse']<gate['joint_rmse_rad_max'] and report['root_error']<gate['root_path_error_m_max']
        expected=report.get('reference_lateral_range',1)
        ratio=report.get('lateral_range',0)/max(expected,1e-6)
        ok=ok and ratio>=gate.get('lateral_excursion_ratio_min',.5) and ratio<=gate.get('lateral_excursion_ratio_max',1.5)
        if 'foot_lift_success_rate_min' in gate:
            lifts=report.get('foot_lift_success_rate_per_foot',[])
            slip=report.get('stance_slip_m_s')
            contact=report.get('planned_contact_match')
            ok=ok and len(lifts)==2 and min(lifts)>=gate['foot_lift_success_rate_min'] and slip is not None and slip<=gate['stance_slip_m_s_max'] and contact is not None and contact>=gate['planned_contact_match_min']
    return bool(ok)
