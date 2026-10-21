"""A public diagnostic must bind the real native footage to its evaluated model."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('lift_diagnostic_summary',ROOT/'tools/summarize_lift_diagnostic.py')
summary=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(summary)


def inputs():
    cases={name:{'episodes':16,'survival_rate':.875,'upright':.99,'stance_slip_m_s':.03,
                 'foot_peak_clearance_m_per_foot':[.035,.03],'foot_lift_success_rate_per_foot':[.6,.5],
                 'high_swing_samples_per_foot':[0.,0.] if name=='stand' else [100.,100.]}
           for name in ('stand','inplace')}
    evaluation={'robot':'q1_leg_lift_v5','contract':'q1_leg_command_v5','backend':'isaaclab','proxy':False,
                'stage':0,'completed_iterations':105,'checkpoint_sha256':'a'*64,'environment_sha256':'b'*64,
                'urdf_sha256':'c'*64,'gait_sha256':'d'*64,'episodes':32,'passed':False,
                'score':.5,'checks':{'stand.survival':True,'inplace.right.lift':False},'cases':cases,
                'checkpoint':'/private/run/checkpoint.pt','gofile_admin_token':'secret-must-not-leak'}
    metadata={'robot':'q1_leg_lift_v5','diagnostic':True,'gate_passed':False,'accepted':False,'proxy':False,
              'source':'native Isaac Sim PhysX PPO','overlay':'EXPERIMENTAL / NOT ACCEPTED','task':'inplace_curriculum',
              'technical_error':None,'reset_count_after_start':0,'checkpoint_sha256':'a'*64,
              'video_sha256':'e'*64,'frames':400,'fps':50,'actual_duration_s':8.,'failure':False,
              'control_profile':'native_bounded_v5','source_sha256':{summary.ENV_SOURCE:'b'*64,summary.CONTROL_SOURCE:'f'*64},
              'contract':{'robot':'q1_leg_lift_v5','observation_size':110,'joint_names':[str(n) for n in range(23)],
                          'urdf_sha256':'c'*64,'gait_sha256':'d'*64,'control_sha256':'f'*64,'control_profile':'native_bounded_v5'},
              'foot_metrics':{'valid_first_episode_frames':400,'unloaded_peak_clearance_m_per_foot':[.04,.03],
                              'unloaded_samples_per_foot':[100,80],'planned_high_swing_samples_per_foot':[50,40],
                              'planned_high_swing_success_rate_per_foot':[.6,.4],
                              'foot_world_peak_displacement_m_per_foot':[.06,.05],'foot_world_path_length_m_per_foot':[.4,.3],
                              'foot_world_displacement_xyz_m_per_foot':[[.01,.0,.0],[-.01,.0,.0]]},
              'failure_reason':'/private/run/irrelevant-secret','gofile_token':'secret-must-not-leak'}
    return evaluation,metadata


def test_cpu_cli_reports_actual_measurements_and_never_exports_secrets_paths_or_acceptance(tmp_path):
    evaluation,metadata=inputs()
    source=tmp_path/'evaluation.json';movie=tmp_path/'movie.json';output=tmp_path/'public.md'
    source.write_text(json.dumps(evaluation));movie.write_text(json.dumps(metadata))
    assert summary.main(['--evaluation',str(source),'--metadata',str(movie),'--output',str(output),
                         '--gofile-url','https://gofile.io/d/publicClip'])==0
    text=output.read_text()
    assert '105 次新 V5 PPO 更新' in text
    assert '6 cm 是抬腳命令' in text and '不是任務成功率' in text
    assert '4.00' in text and '3.00' in text and '60.0%' in text
    assert 'Stage 2' in text and 'NOT ACCEPTED' in text
    assert 'https://gofile.io/d/publicClip' in text
    assert '/private/' not in text and 'secret-must-not-leak' not in text and 'gofile_token' not in text
    with pytest.raises(ValueError,match='public'):
        summary.render(evaluation,metadata,'https://gofile.io/d/publicClip?token=private')


@pytest.mark.parametrize('side,key,value',[
    ('metadata','checkpoint_sha256','0'*64),('evaluation','robot','q1_leg_interactive_v4'),
    ('evaluation','environment_sha256','0'*64),('evaluation','gait_sha256','0'*64),
    ('evaluation','stage',2),('metadata','video_sha256',None)])
def test_mismatched_policy_or_source_cannot_become_a_public_v5_report(side,key,value):
    evaluation,metadata=inputs()
    (evaluation if side=='evaluation' else metadata)[key]=value
    with pytest.raises(ValueError):summary.render(evaluation,metadata)


@pytest.mark.parametrize('side,key,value',[
    ('metadata','accepted',True),('metadata','gate_passed',True),('metadata','proxy',True),
    ('evaluation','proxy',True),('metadata','reset_count_after_start',1),('evaluation','passed',True)])
def test_accepted_proxy_reset_or_contradictory_claims_are_refused(side,key,value):
    evaluation,metadata=inputs()
    (evaluation if side=='evaluation' else metadata)[key]=value
    with pytest.raises(ValueError):summary.render(evaluation,metadata)
