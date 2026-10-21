"""Real native 501-update / seed-71501 evaluation captured 2026-10-21 08:42 Taipei."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deadline_summary',ROOT/'tools/summarize_interactive_deadline.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

REAL_NATIVE_EVALUATION=json.loads(r'''{
  "episodes": 32,
  "evaluation_seed": 71501,
  "backend": "isaaclab",
  "proxy": false,
  "passed": false,
  "score": 0.3486238532110092,
  "checkpoint_sha256": "246c66218942d4ed300f17d61d8f318a8d5855089f92625dc468aeda2b8a2992",
  "completed_iterations": 501,
  "cases": {
    "stand": {
      "episodes": 4,
      "completed_episodes": 4,
      "survival_rate": 1.0,
      "duration_s": 20.0,
      "upright": 0.9996700286865234,
      "velocity_rmse_m_s": 0.0,
      "yaw_rmse_rad_s": 0.0,
      "directional_displacement_m": 0.0,
      "signed_yaw_displacement_rad": 0.0,
      "final_stand_speed_m_s": 0.003805286716669798,
      "final_stand_drift_m": 7.183926209108904e-05,
      "foot_peak_clearance_m_per_foot": [
        0.0,
        0.0
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.9992499947547913,
      "loaded_foot_speed_m_s": 0.012851236388087273,
      "stance_slip_m_s": 0.014314906671643257
    },
    "w": {
      "episodes": 4,
      "completed_episodes": 0,
      "survival_rate": 0.0,
      "duration_s": 3.3550000190734863,
      "upright": 0.9771527647972107,
      "velocity_rmse_m_s": 0.37367483973503113,
      "yaw_rmse_rad_s": 0.12195746600627899,
      "directional_displacement_m": 0.7270419001579285,
      "signed_yaw_displacement_rad": 0.0,
      "final_stand_speed_m_s": 1000000.0,
      "final_stand_drift_m": 0.0,
      "foot_peak_clearance_m_per_foot": [
        0.00020342133939266205,
        0.00015402212738990784
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.7451519966125488,
      "loaded_foot_speed_m_s": 0.05345512926578522,
      "stance_slip_m_s": 0.11206300556659698,
      "redirection": {
        "command": [
          0.0,
          0.18,
          0.0
        ],
        "window_s": [
          4.0,
          5.0
        ],
        "directional_displacement_m": 0.0,
        "velocity_rmse_m_s": 1000000.0
      }
    },
    "s": {
      "episodes": 4,
      "completed_episodes": 0,
      "survival_rate": 0.0,
      "duration_s": 2.880000114440918,
      "upright": 0.9749758839607239,
      "velocity_rmse_m_s": 0.4263937473297119,
      "yaw_rmse_rad_s": 0.25356972217559814,
      "directional_displacement_m": 0.6934601068496704,
      "signed_yaw_displacement_rad": 0.0,
      "final_stand_speed_m_s": 1000000.0,
      "final_stand_drift_m": 0.0,
      "foot_peak_clearance_m_per_foot": [
        8.03731381893158e-05,
        3.124121576547623e-05
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.7647345066070557,
      "loaded_foot_speed_m_s": 0.049674998968839645,
      "stance_slip_m_s": 0.1091991439461708,
      "redirection": {
        "command": [
          0.0,
          -0.18,
          0.0
        ],
        "window_s": [
          4.0,
          5.0
        ],
        "directional_displacement_m": 0.0,
        "velocity_rmse_m_s": 1000000.0
      }
    },
    "a": {
      "episodes": 4,
      "completed_episodes": 0,
      "survival_rate": 0.0,
      "duration_s": 6.885000228881836,
      "upright": 0.9836260080337524,
      "velocity_rmse_m_s": 0.32768839597702026,
      "yaw_rmse_rad_s": 0.2086116373538971,
      "directional_displacement_m": -0.09148597717285156,
      "signed_yaw_displacement_rad": 0.0,
      "final_stand_speed_m_s": 1000000.0,
      "final_stand_drift_m": 0.0,
      "foot_peak_clearance_m_per_foot": [
        0.00017117895185947418,
        8.215010166168213e-05
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.6844589114189148,
      "loaded_foot_speed_m_s": 0.029907599091529846,
      "stance_slip_m_s": 0.05956341326236725,
      "redirection": {
        "command": [
          -0.22,
          0.0,
          0.0
        ],
        "window_s": [
          4.0,
          5.0
        ],
        "directional_displacement_m": 0.09318329393863678,
        "velocity_rmse_m_s": 0.15501748025417328
      }
    },
    "d": {
      "episodes": 4,
      "completed_episodes": 2,
      "survival_rate": 0.5,
      "duration_s": 14.655000686645508,
      "upright": 0.990038275718689,
      "velocity_rmse_m_s": 0.16827577352523804,
      "yaw_rmse_rad_s": 0.07160431891679764,
      "directional_displacement_m": 0.09608456492424011,
      "signed_yaw_displacement_rad": 0.0,
      "final_stand_speed_m_s": 500000.0,
      "final_stand_drift_m": 5.6597680668346584e-05,
      "foot_peak_clearance_m_per_foot": [
        5.168747156858444e-05,
        8.048489689826965e-06
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.7972748279571533,
      "loaded_foot_speed_m_s": 0.024269267916679382,
      "stance_slip_m_s": 0.046924903988838196,
      "redirection": {
        "command": [
          0.22,
          0.0,
          0.0
        ],
        "window_s": [
          4.0,
          5.0
        ],
        "directional_displacement_m": 0.053980764001607895,
        "velocity_rmse_m_s": 0.16904157400131226
      }
    },
    "yaw_left": {
      "episodes": 4,
      "completed_episodes": 0,
      "survival_rate": 0.0,
      "duration_s": 6.600000381469727,
      "upright": 0.9878923892974854,
      "velocity_rmse_m_s": 0.22661054134368896,
      "yaw_rmse_rad_s": 0.49676650762557983,
      "directional_displacement_m": 0.0,
      "signed_yaw_displacement_rad": 1.4107786417007446,
      "final_stand_speed_m_s": 1000000.0,
      "final_stand_drift_m": 0.0,
      "foot_peak_clearance_m_per_foot": [
        7.385388016700745e-05,
        4.6574510633945465e-05
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.6917240619659424,
      "loaded_foot_speed_m_s": 0.03365302458405495,
      "stance_slip_m_s": 0.06519101560115814
    },
    "yaw_right": {
      "episodes": 4,
      "completed_episodes": 0,
      "survival_rate": 0.0,
      "duration_s": 6.949999809265137,
      "upright": 0.9970594644546509,
      "velocity_rmse_m_s": 0.06101620942354202,
      "yaw_rmse_rad_s": 0.6768137216567993,
      "directional_displacement_m": 0.0,
      "signed_yaw_displacement_rad": -0.2592949867248535,
      "final_stand_speed_m_s": 1000000.0,
      "final_stand_drift_m": 0.0,
      "foot_peak_clearance_m_per_foot": [
        5.6206248700618744e-05,
        5.9536658227443695e-05
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.6857953667640686,
      "loaded_foot_speed_m_s": 0.02714030258357525,
      "stance_slip_m_s": 0.03848046436905861
    },
    "sequence": {
      "episodes": 4,
      "completed_episodes": 0,
      "survival_rate": 0.0,
      "duration_s": 6.75,
      "upright": 0.9753676056861877,
      "velocity_rmse_m_s": 0.0,
      "yaw_rmse_rad_s": 0.0,
      "directional_displacement_m": 0.0,
      "signed_yaw_displacement_rad": 0.0,
      "final_stand_speed_m_s": 1000000.0,
      "final_stand_drift_m": 0.0,
      "foot_peak_clearance_m_per_foot": [
        4.6516768634319305e-05,
        9.121466428041458e-05
      ],
      "foot_lift_success_rate_per_foot": [
        0.0,
        0.0
      ],
      "planned_contact_match": 0.8026528358459473,
      "loaded_foot_speed_m_s": 0.04108383506536484,
      "stance_slip_m_s": 0.0801241546869278,
      "resumed_displacement_x_m": 0.05897463113069534,
      "resume_velocity_rmse_m_s": 500000.0625,
      "finite_sequence_completed": false,
      "directions": {
        "left": {
          "lead_forward_m": 0.0,
          "lead_signed_lateral_m": 0.0,
          "lead_landing_rate": 0.0,
          "lead_peak_clearance_m": 1.1442229151725769e-05,
          "lead_lift_success_rate": 0.0,
          "trailing_peak_clearance_m": 1.701153814792633e-05,
          "trailing_lift_success_rate": 0.0,
          "hand_cross_success_rate": 0.6666666865348816,
          "push_support_forward_proxy_rate": 0.5625,
          "root_forward_m": 0.05689217522740364,
          "root_signed_lateral_m": 0.006532859057188034
        },
        "right": {
          "lead_forward_m": 0.0,
          "lead_signed_lateral_m": 0.0,
          "lead_landing_rate": 0.0,
          "lead_peak_clearance_m": 9.121466428041458e-05,
          "lead_lift_success_rate": 0.0,
          "trailing_peak_clearance_m": 3.052968531847e-05,
          "trailing_lift_success_rate": 0.0,
          "hand_cross_success_rate": 0.7285714149475098,
          "push_support_forward_proxy_rate": 0.6470588445663452,
          "root_forward_m": 0.23398159444332123,
          "root_signed_lateral_m": 0.02262742444872856
        }
      }
    }
  },
  "checks": {
    "stand.sample_count": true,
    "stand.survival": true,
    "stand.upright": true,
    "stand.stop": true,
    "stand.stance_slip": true,
    "w.sample_count": true,
    "w.survival": false,
    "w.upright": true,
    "w.stop": false,
    "w.stance_slip": false,
    "w.direction": true,
    "w.velocity": false,
    "w.redirection_progress": false,
    "w.redirection_velocity": false,
    "w.foot_0.clearance": false,
    "w.foot_0.lift": false,
    "w.foot_1.clearance": false,
    "w.foot_1.lift": false,
    "s.sample_count": true,
    "s.survival": false,
    "s.upright": true,
    "s.stop": false,
    "s.stance_slip": false,
    "s.direction": true,
    "s.velocity": false,
    "s.redirection_progress": false,
    "s.redirection_velocity": false,
    "s.foot_0.clearance": false,
    "s.foot_0.lift": false,
    "s.foot_1.clearance": false,
    "s.foot_1.lift": false,
    "a.sample_count": true,
    "a.survival": false,
    "a.upright": true,
    "a.stop": false,
    "a.stance_slip": true,
    "a.direction": false,
    "a.velocity": false,
    "a.redirection_progress": true,
    "a.redirection_velocity": true,
    "a.foot_0.clearance": false,
    "a.foot_0.lift": false,
    "a.foot_1.clearance": false,
    "a.foot_1.lift": false,
    "d.sample_count": true,
    "d.survival": false,
    "d.upright": true,
    "d.stop": false,
    "d.stance_slip": true,
    "d.direction": false,
    "d.velocity": false,
    "d.redirection_progress": true,
    "d.redirection_velocity": false,
    "d.foot_0.clearance": false,
    "d.foot_0.lift": false,
    "d.foot_1.clearance": false,
    "d.foot_1.lift": false,
    "yaw_left.sample_count": true,
    "yaw_left.survival": false,
    "yaw_left.upright": true,
    "yaw_left.stop": false,
    "yaw_left.stance_slip": true,
    "yaw_left.yaw_direction": true,
    "yaw_left.yaw_tracking": false,
    "yaw_left.foot_0.clearance": false,
    "yaw_left.foot_0.lift": false,
    "yaw_left.foot_1.clearance": false,
    "yaw_left.foot_1.lift": false,
    "yaw_right.sample_count": true,
    "yaw_right.survival": false,
    "yaw_right.upright": true,
    "yaw_right.stop": false,
    "yaw_right.stance_slip": true,
    "yaw_right.yaw_direction": false,
    "yaw_right.yaw_tracking": false,
    "yaw_right.foot_0.clearance": false,
    "yaw_right.foot_0.lift": false,
    "yaw_right.foot_1.clearance": false,
    "yaw_right.foot_1.lift": false,
    "sequence.sample_count": true,
    "sequence.survival": false,
    "sequence.upright": true,
    "sequence.stop": false,
    "sequence.stance_slip": true,
    "sequence.resume_progress": true,
    "sequence.resume_velocity": false,
    "sequence.left.lead_forward_m": false,
    "sequence.left.lead_signed_lateral_m": false,
    "sequence.left.lead_peak_clearance_m": false,
    "sequence.left.lead_lift_success_rate": false,
    "sequence.left.trailing_peak_clearance_m": false,
    "sequence.left.trailing_lift_success_rate": false,
    "sequence.left.hand_cross_success_rate": true,
    "sequence.left.root_forward_m": true,
    "sequence.left.root_signed_lateral_m": false,
    "sequence.left.push_support_forward_proxy_rate": true,
    "sequence.left.landing": false,
    "sequence.right.lead_forward_m": false,
    "sequence.right.lead_signed_lateral_m": false,
    "sequence.right.lead_peak_clearance_m": false,
    "sequence.right.lead_lift_success_rate": false,
    "sequence.right.trailing_peak_clearance_m": false,
    "sequence.right.trailing_lift_success_rate": false,
    "sequence.right.hand_cross_success_rate": true,
    "sequence.right.root_forward_m": true,
    "sequence.right.root_signed_lateral_m": true,
    "sequence.right.push_support_forward_proxy_rate": true,
    "sequence.right.landing": false,
    "sequence.finite_completion": false
  },
  "failed_gates": [
    "w.survival",
    "w.stop",
    "w.stance_slip",
    "w.velocity",
    "w.redirection_progress",
    "w.redirection_velocity",
    "w.foot_0.clearance",
    "w.foot_0.lift",
    "w.foot_1.clearance",
    "w.foot_1.lift",
    "s.survival",
    "s.stop",
    "s.stance_slip",
    "s.velocity",
    "s.redirection_progress",
    "s.redirection_velocity",
    "s.foot_0.clearance",
    "s.foot_0.lift",
    "s.foot_1.clearance",
    "s.foot_1.lift",
    "a.survival",
    "a.stop",
    "a.direction",
    "a.velocity",
    "a.foot_0.clearance",
    "a.foot_0.lift",
    "a.foot_1.clearance",
    "a.foot_1.lift",
    "d.survival",
    "d.stop",
    "d.direction",
    "d.velocity",
    "d.redirection_velocity",
    "d.foot_0.clearance",
    "d.foot_0.lift",
    "d.foot_1.clearance",
    "d.foot_1.lift",
    "yaw_left.survival",
    "yaw_left.stop",
    "yaw_left.yaw_tracking",
    "yaw_left.foot_0.clearance",
    "yaw_left.foot_0.lift",
    "yaw_left.foot_1.clearance",
    "yaw_left.foot_1.lift",
    "yaw_right.survival",
    "yaw_right.stop",
    "yaw_right.yaw_direction",
    "yaw_right.yaw_tracking",
    "yaw_right.foot_0.clearance",
    "yaw_right.foot_0.lift",
    "yaw_right.foot_1.clearance",
    "yaw_right.foot_1.lift",
    "sequence.survival",
    "sequence.stop",
    "sequence.resume_velocity",
    "sequence.left.lead_forward_m",
    "sequence.left.lead_signed_lateral_m",
    "sequence.left.lead_peak_clearance_m",
    "sequence.left.lead_lift_success_rate",
    "sequence.left.trailing_peak_clearance_m",
    "sequence.left.trailing_lift_success_rate",
    "sequence.left.root_signed_lateral_m",
    "sequence.left.landing",
    "sequence.right.lead_forward_m",
    "sequence.right.lead_signed_lateral_m",
    "sequence.right.lead_peak_clearance_m",
    "sequence.right.lead_lift_success_rate",
    "sequence.right.trailing_peak_clearance_m",
    "sequence.right.trailing_lift_success_rate",
    "sequence.right.landing",
    "sequence.finite_completion"
  ],
  "gates": {
    "survival_rate_min": 0.875,
    "upright_min": 0.97,
    "stop_speed_m_s_max": 0.07,
    "walk_directional_displacement_m_min": 0.35,
    "walk_velocity_rmse_m_s_max": 0.12,
    "foot_clearance_m_min": 0.025,
    "foot_lift_success_min": 0.4,
    "yaw_displacement_rad_min": 0.5,
    "yaw_rmse_rad_s_max": 0.2,
    "lead_forward_m_min": 0.02,
    "lead_lateral_m_min": 0.025,
    "hand_cross_success_min": 0.5,
    "root_forward_m_min": 0.015,
    "root_lateral_m_min": 0.015,
    "push_proxy_success_min": 0.25,
    "stance_slip_m_s_max": 0.1,
    "redirection_displacement_m_min": 0.025,
    "redirection_velocity_rmse_m_s_max": 0.16,
    "resume_displacement_x_m_min": 0.05,
    "resume_velocity_rmse_m_s_max": 0.12
  }
}''')


def test_real_failed_checkpoint_report_exports_truthfully_and_refuses_active_finalization(tmp_path):
    evaluation=tmp_path/'evaluation_501_71501.json'
    evaluation.write_text(json.dumps(REAL_NATIVE_EVALUATION))
    state={'state':'training','ready':False,'num_envs':10240,'completed_iterations':501,
        'max_iterations':801,'adopted_worker_iterations':501,
        'profile':{'num_envs':10240,'max_iterations':801,'guestToken':'NEVER_PUBLISH'},
        'current_command':['private command'], 'guestToken':'NEVER_PUBLISH',
        'last_evaluation':{'report':str(evaluation),'completed_iterations':501}}
    statefile=tmp_path/'state.json';statefile.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='still active'):
        module.summarize(tmp_path,statefile)
    assert not (tmp_path/'docs').exists()
    paths=module.summarize(tmp_path,statefile,allow_active=True)
    assert 'ONGOING' in paths[1].read_text()
    state['state']='deadline_reached';statefile.write_text(json.dumps(state))
    paths=module.summarize(tmp_path,statefile)
    result=json.loads(paths[0].read_text())
    assert result['accepted'] is False and result['ongoing'] is False
    assert result['evaluation']['cases']==REAL_NATIVE_EVALUATION['cases']
    assert result['evaluation']['checks']==REAL_NATIVE_EVALUATION['checks']
    assert result['evaluation']['failed_gates']==REAL_NATIVE_EVALUATION['failed_gates']
    assert result['evaluation']['cases']['stand']['survival_rate']==1.0
    assert result['evaluation']['cases']['w']['survival_rate']==0.0
    assert result['evaluation']['cases']['sequence']['directions']['left']['lead_peak_clearance_m']==1.1442229151725769e-05
    assert result['evaluation']['checkpoint_sha256']==REAL_NATIVE_EVALUATION['checkpoint_sha256']
    combined=''.join(path.read_text() for path in paths)
    assert 'NEVER_PUBLISH' not in combined and 'current_command' not in combined
    assert 'NOT ACCEPTED' in paths[1].read_text() and '2.5 cm' in paths[2].read_text()
    assert '501／801' in paths[1].read_text()
    assert ET.fromstring(paths[2].read_text()).tag.endswith('svg')
