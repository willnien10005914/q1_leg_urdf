"""Accepted media cannot bypass native stage-2 confirmation or hide bad feet."""
import builtins
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
sys.path.insert(0,str(ROOT/'source/leg_humanoid_lab'))
from record_lift_v5 import direction_summary,output_path,preflight_recording,recording_plan,validate_recording,verify_artifact
import record_lift_v5 as RECORDER
from leg_humanoid_lab.interactive_control import KeyboardController

SPEC=importlib.util.spec_from_file_location('lift_gate_test_fixtures',ROOT/'tests/test_lift_player.py')
FIXTURES=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(FIXTURES)


def test_recording_plan_executes_one_finite_pair_then_stands_without_restart():
    plan=recording_plan();assert abs(plan[-1]['end']-10.2)<1e-8
    assert all(a['end']==b['start'] for a,b in zip(plan,plan[1:]))
    controller=KeyboardController();last=None;modes=[]
    for index in range(round(plan[-1]['end']/.02)):
        t=index*.02;segment=next(s for s in plan if s['start']<=t<s['end'])
        if segment['name']!=last:
            if segment['name']=='stop':controller.stop(t)
            elif segment['name']=='sidestep_sequence':assert controller.sequence(t) is True
            last=segment['name']
        if segment['name']=='W_forward':assert controller.velocity(*segment['command'],t) is True
        control=controller.tick(.02,t)
        if not modes or modes[-1]!=control['mode']:modes.append(control['mode'])
    assert modes.count('sidestep_left')==modes.count('sidestep_right')==1
    assert 'walk' in modes and control['mode']=='stand' and not control['sequence_active']


@pytest.mark.parametrize('change',[{'stage':0},{'ready':False},{'confirmation_episodes':32}])
def test_preflight_rejects_foundation_and_unconfirmed_candidates(tmp_path,change):
    folder,_,report=FIXTURES.accepted_artifacts(tmp_path)
    FIXTURES.write_json(folder/'evaluation.json',{**report,**change})
    with pytest.raises(ValueError):preflight_recording(tmp_path)


def test_preflight_rejects_changed_context_and_non_native_timing(tmp_path):
    FIXTURES.accepted_artifacts(tmp_path)
    assert preflight_recording(tmp_path)['confirmation_episodes']==128
    with pytest.raises(ValueError,match='duration'):preflight_recording(tmp_path,2.)
    path=tmp_path/'config/lift_control_v5.json';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):preflight_recording(tmp_path)


def test_recording_output_cannot_overwrite_weights_or_checkpoint_metadata(tmp_path):
    with pytest.raises(ValueError):output_path(tmp_path/'checkpoints/isaac/lift_v5/latest.pt',tmp_path)
    with pytest.raises(ValueError):output_path(tmp_path/'checkpoints/isaac/lift_v5/movie.mp4',tmp_path)
    assert output_path(tmp_path/'docs/results/movie.mp4',tmp_path).suffix=='.mp4'


def test_cli_unready_preflight_never_imports_isaac_or_torch(tmp_path,monkeypatch):
    monkeypatch.setattr(RECORDER,'ROOT',tmp_path)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert not name.startswith(('isaaclab','isaacsim','torch'))
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert RECORDER.main(['--validate-only'])==1


def accepted_capture():
    # Synthetic JSON exercises publication validation only; never inference.
    frames=510;plan=recording_plan();controller=KeyboardController();last=None;observations=[];modes=[]
    for index in range(frames):
        t=index*.02;segment=next(s for s in plan if s['start']<=t<s['end'])
        if segment['name']!=last:
            if segment['name']=='stop':controller.stop(t)
            elif segment['name']=='sidestep_sequence':controller.sequence(t)
            last=segment['name']
        if segment['name']=='W_forward':controller.velocity(*segment['command'],t)
        control=controller.tick(.02,t);mode=control['mode'];phase=control['phase']
        contact=[True,True]
        if mode in ('sidestep_left','sidestep_right'):
            lead=0 if mode=='sidestep_left' else 1
            if .18<phase<.44:contact[lead]=False
            if .66<phase<.9:contact[1-lead]=False
        observations.append({'time':(index+1)/50.,'mode':mode,'gesture_phase':phase,
            'valid_first_episode':True,'reset_count_after_start':0,'auto_resets_suppressed':0,
            'clearance_m':[.003 if c else .03 for c in contact],'loaded':contact[:],
            'reference_height_m':[.003 if c else .04 for c in contact],'reference_contact':contact[:],
            'foot_link_positions_world_m':[[.1,.15,.08],[0.,-.15,.05]]})
        if not modes or modes[-1]['mode']!=mode:modes.append({'mode':mode,'time':t})
    return {'accepted':True,'gate_passed':True,'robot':'q1_leg_lift_v5','controller_version':'q1_leg_v5',
        'backend':'isaaclab','proxy':False,'confirmation_episodes':128,'confirmation_stage':2,
        'gate_unchanged_after_recording':True,'completed_plan':True,'failure':False,
        'technical_error':None,'error':None,'reset_count_after_start':0,'auto_resets_suppressed':0,
        'unexpected_episode_reset':False,'final_mode':'stand','final_sequence_active':False,
        'fps':50,'resolution':[1280,720],'frames':frames,'expected_frames':frames,
        'foot_observations':observations,
        'plan':recording_plan(),'sidestep_duration_s':1.6,
        'actual_duration_s':10.2,'expected_duration_s':10.2,'final_stand_speed_m_s':.02,
        'walk_forward_displacement_m':.2,'mode_changes':modes,
        'foot_metrics':{'valid_first_episode_frames':frames},
        'directions':direction_summary(observations)}


def test_capture_rejects_grounded_lead_even_if_full_video_survives():
    report=accepted_capture();assert validate_recording(report)
    for row in report['foot_observations']:
        if row['mode']=='sidestep_left':row['clearance_m'][0]=.004;row['loaded'][0]=True
    report['directions']=direction_summary(report['foot_observations'])
    assert not validate_recording(report)


def test_capture_recomputes_directional_unloaded_metrics_and_rejects_forged_summary():
    report=accepted_capture()
    report['directions']['left']['lead_peak_clearance_m']=.08
    assert not validate_recording(report)
    report=accepted_capture()
    for row in report['foot_observations']:
        if row['mode']=='sidestep_left':row['loaded'][0]=True
    report['directions']=direction_summary(report['foot_observations'])
    assert report['directions']['left']['lead_peak_clearance_m']==0.
    assert report['directions']['left']['lead_lift_success_rate']==0.
    assert not validate_recording(report)


def test_capture_rejects_missing_or_reset_frame_measurements():
    report=accepted_capture();report['foot_observations'][42]['clearance_m']=[float('nan'),.003]
    assert not validate_recording(report)
    report=accepted_capture();report['foot_observations'][42]['reset_count_after_start']=1
    assert not validate_recording(report)


def test_artifact_rejects_stale_output_even_if_old_movie_is_valid(tmp_path):
    video,_=synthetic_artifact(tmp_path)
    assert verify_artifact(video,tmp_path,run_id='completed-previous-attempt',check_frames=False)
    with pytest.raises(ValueError,match='stale'):
        verify_artifact(video,tmp_path,run_id='failed-new-attempt',check_frames=False)


def synthetic_artifact(tmp_path):
    FIXTURES.accepted_artifacts(tmp_path);gate=preflight_recording(tmp_path)
    report=accepted_capture();report.update({key:gate[key] for key in ('checkpoint_sha256','contract','source_sha256')})
    report['recording_run_id']='completed-previous-attempt';report['recorder_sources']={}
    for relative in RECORDER.RECORDER_SOURCES:
        destination=tmp_path/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,destination);report['recorder_sources'][relative]=RECORDER.sha256(destination)
    video=tmp_path/'movie.mp4';video.write_bytes(b'CPU-only synthetic artifact fixture '*100)
    report['video_sha256']=RECORDER.sha256(video);video.with_suffix('.json').write_text(json.dumps(report))
    return video,report


def test_frame_count_must_match_full_plan_even_if_truncated_metadata_agrees():
    report=accepted_capture();report['frames']=report['expected_frames']=500
    report['actual_duration_s']=report['expected_duration_s']=10.
    report['foot_observations']=report['foot_observations'][:500]
    report['foot_metrics']['valid_first_episode_frames']=500
    assert not validate_recording(report)


def test_artifact_rejects_actual_decoded_frame_count_mismatch(tmp_path,monkeypatch):
    video,_=synthetic_artifact(tmp_path)
    probe={'streams':[{'width':1280,'height':720,'avg_frame_rate':'50/1','nb_read_frames':'509'}]}
    monkeypatch.setattr(RECORDER.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=json.dumps(probe)))
    with pytest.raises(ValueError,match='decoded frame count'):
        verify_artifact(video,tmp_path)


def test_artifact_rejects_frozen_viewport_even_when_timestamp_overlay_changes(tmp_path,monkeypatch):
    import numpy as np
    video,_=synthetic_artifact(tmp_path)
    def media_result(command,**kwargs):
        if command[0]=='ffprobe':
            return SimpleNamespace(stdout=json.dumps({'streams':[{'width':1280,'height':720,
                'avg_frame_rate':'50/1','nb_read_frames':'510'}]}))
        image=np.zeros((720,1280,3),dtype=np.uint8)
        image[:96]=round(float(command[command.index('-ss')+1])*20)
        return SimpleNamespace(stdout=image.tobytes())
    monkeypatch.setattr(RECORDER.subprocess,'run',media_result)
    with pytest.raises(ValueError,match='viewport appears frozen'):
        verify_artifact(video,tmp_path)


@pytest.mark.parametrize('change',[
    {'failure':True},{'reset_count_after_start':1},{'gate_unchanged_after_recording':False},
    {'frames':100},{'walk_forward_displacement_m':.01},{'final_stand_speed_m_s':.2},
])
def test_capture_rejects_incomplete_or_unsuccessful_physics(change):
    report=accepted_capture();report.update(change);assert not validate_recording(report)
