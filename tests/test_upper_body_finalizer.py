"""CPU-only pipeline tests: subprocess boundaries replace Sim/network/media.

Temporary Git repositories are local fixtures. No real upload, push, accepted
checkpoint inference or native movie generation occurs in these tests.
"""
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
sys.path.insert(0,str(ROOT/'source/leg_humanoid_lab'))
SPEC=importlib.util.spec_from_file_location('upper_body_finalizer',ROOT/'tools/finalize_upper_body_v6.py')
F=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(F)
PLAYER_SPEC=importlib.util.spec_from_file_location('v6_player_fixtures',ROOT/'tests/test_upper_body_player.py')
PLAYER=importlib.util.module_from_spec(PLAYER_SPEC);PLAYER_SPEC.loader.exec_module(PLAYER)
BASE_SPEC=importlib.util.spec_from_file_location('base_record_fixtures',ROOT/'tests/test_lift_recording.py')
BASE=importlib.util.module_from_spec(BASE_SPEC);BASE_SPEC.loader.exec_module(BASE)
import record_upper_body_v6 as RECORDER


def accepted_capture():
    report=BASE.accepted_capture()
    report.update(robot='q1_leg_upper_body_v6',controller_version='q1_leg_v6',
        contract={'upper_body_gates':json.loads((ROOT/'config/upper_body_control_v6.json').read_text())['evaluation_gates']},
        root_forces_or_constraints=False,rendered_pose_filtering=False)
    report['upper_observations']=[{'time':row['time'],'mode':row['mode'],'gesture_phase':row['gesture_phase'],
        'valid_first_episode':True,'chest_tilt_deg':5.,'chest_angular_xy_rad_s':.1,'chest_angular_xyz_rad_s':1.5,
        'arm_reference_mse_rad2':.0004,'arm_joint_speed_mse_rad2_s2':.01,
        'hands_return_distance_m':[.005,.006]} for row in report['foot_observations']]
    report['poses']=[{'time':row['time'],'q':[0.]*23,'root':[0.,0.,.86],'quaternion':[1.,0.,0.,0.]} for row in report['foot_observations']]
    report['upper_metrics']=RECORDER.upper_summary(report['upper_observations'],report['plan'])
    assert RECORDER.validate_recording(report)
    return report


def complete_fixture(root):
    folder,_,evaluation=PLAYER.accepted_artifacts(root)
    gate=RECORDER.preflight_recording(root)
    run=root/'checkpoints/isaac/upper_body_v6/runs/test-run'
    run.mkdir(parents=True)
    state={'robot':'q1_leg_upper_body_v6','backend':'isaaclab','proxy':False,'state':'complete','ready':True,
           'curriculum_stage':2,'run':str(run),'evaluation':evaluation,
           'published_checkpoint':str(folder/'latest.pt'),'completed_iterations':evaluation['completed_iterations'],
           'profile':evaluation['training_profile']}
    F.private_json(run/'supervisor_state.json',state)
    F.private_json(root/'outputs/upper_body_training_status.json',state)
    (root/'README.md').write_text('# Project\n'+F.START+'\nOld V6: NOT ACCEPTED\n'+F.END+
        '\n<!-- V4_STATUS_START -->\nKeep old V4 failed history\n<!-- V4_STATUS_END -->\n')
    return state,gate


class FakeRecorder:
    """The real preflight/metadata checker with a mocked decode boundary."""
    def __init__(self):self.verifications=0
    def preflight_recording(self,root):return RECORDER.preflight_recording(root)
    def verify_artifact(self,video,root):
        self.verifications+=1
        report=json.loads(Path(video).with_suffix('.json').read_text())
        if not RECORDER.validate_recording(report) or report['video_sha256']!=F.digest(video):raise ValueError('Invalid simulated boundary capture')
        gate=self.preflight_recording(root)
        for key in ('checkpoint_sha256','contract','source_sha256','confirmation_episodes'):
            if report[key]!=gate[key]:raise ValueError('Mismatched simulated boundary capture')
        return report


class Runner:
    def __init__(self,root):
        self.root=root;self.commands=[];self.remote=None;self.uploads=0;self.pushes=0;self.records=0
        self.gpu='';self.service='inactive';self.bad_upload=False
        self.isaac=''
    def __call__(self,command,**kwargs):
        self.commands.append(command);stdout='';stderr='';code=0
        if command[0]=='systemctl':stdout=self.service;code=3 if self.service=='inactive' else 0
        elif command[0]=='nvidia-smi':stdout=self.gpu
        elif command[0]=='ps':stdout=self.isaac
        elif command[0]=='bash':
            self.records+=1
            video=Path(kwargs['env']['OUT']);video.write_bytes(b'\0CPU mocked native movie boundary'*120)
            report=accepted_capture();gate=RECORDER.preflight_recording(self.root)
            report.update({key:gate[key] for key in ('checkpoint_sha256','contract','source_sha256')})
            report.update(seed=int(command[-1]),recording_run_id='mock-run',video_sha256=F.digest(video))
            video.with_suffix('.json').write_text(json.dumps(report))
        elif len(command)>1 and command[1]=='tools/video_to_gif.py':
            assert command[command.index('--fps')+1]=='10'
            Path(command[3]).write_bytes(b'\0CPU mocked GIF boundary'*100)
        elif len(command)>1 and command[1]=='tools/upload_gofile.py':
            self.uploads+=1
            F.private_json(command[3],{'status':'ok','data':{'md5':'wrong' if self.bad_upload else F.digest(command[2],'md5'),
                'downloadPage':'https://gofile.io/d/PublicV6','guestToken':'PRIVATE_TEST_TOKEN'}})
            stdout='PRIVATE_TEST_TOKEN';stderr='PRIVATE_TEST_TOKEN'
        elif command[:2]==['git','ls-remote']:
            stdout=self.remote+'\trefs/heads/main\n'
        elif command[:2]==['git','push']:
            self.pushes+=1;self.remote=command[-1].split(':')[0]
        elif command[0]=='git':return subprocess.run(command,**kwargs)
        else:raise AssertionError('Unexpected external boundary: '+repr(command))
        return SimpleNamespace(returncode=code,stdout=stdout,stderr=stderr)


def local_git(root,*arguments):
    return subprocess.run(['git',*arguments],cwd=root,check=True,text=True,capture_output=True).stdout.strip()


def git_fixture(root):
    local_git(root,'init','-b','main')
    local_git(root,'config','user.name','CPU test');local_git(root,'config','user.email','cpu@example.invalid')
    local_git(root,'remote','add','origin','git@github.com:willnien10005914/q1_leg_urdf.git')
    local_git(root,'add','README.md');local_git(root,'commit','-m','Initial fixture')
    return local_git(root,'rev-parse','HEAD')


@pytest.mark.parametrize('change',[{'state':'training'},{'ready':False},{'curriculum_stage':1}])
def test_supervisor_must_be_complete_and_both_markers_agree(tmp_path,change):
    state,gate=complete_fixture(tmp_path)
    assert len(F.supervisor_ready(state,state,gate,tmp_path))==64
    with pytest.raises(F.Pending):F.supervisor_ready({**state,**change},state,gate,tmp_path)


def test_supervisor_completion_is_bound_to_checkpoint_and_sources(tmp_path):
    state,gate=complete_fixture(tmp_path)
    bad={**state,'evaluation':{**state['evaluation'],'checkpoint_sha256':'wrong'}}
    with pytest.raises(F.Pending):F.supervisor_ready(bad,state,gate,tmp_path)
    bad={**state,'evaluation':{**state['evaluation'],'passed':False}}
    with pytest.raises(F.Pending):F.supervisor_ready(bad,state,gate,tmp_path)
    bad={**state,'run':str(tmp_path/'elsewhere')}
    with pytest.raises(ValueError):F.job_identity(bad,gate,tmp_path)


def test_default_prepare_never_uploads_mutates_index_or_changes_readme(tmp_path):
    complete_fixture(tmp_path);before=(tmp_path/'README.md').read_text()
    runner=Runner(tmp_path);finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    result=finalizer.execute()
    assert result['phase']=='prepared' and result['published'] is False
    assert runner.records==1 and runner.uploads==runner.pushes==0
    assert not any(command[0]=='git' for command in runner.commands)
    assert (tmp_path/'README.md').read_text()==before and not (tmp_path/F.VIDEO).exists()
    assert finalizer.job.stat().st_mode&0o777==0o700
    assert finalizer.output.stat().st_mode&0o777==0o600
    finalizer.execute();assert runner.records==1


def test_gpu_compute_or_active_training_blocks_recording_but_display_memory_is_ignored(tmp_path):
    complete_fixture(tmp_path);runner=Runner(tmp_path);finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    runner.gpu='1234, python (Isaac)\n'
    with pytest.raises(F.Pending):finalizer.execute()
    assert runner.records==0
    runner.gpu='';runner.service='active'
    with pytest.raises(F.Pending):finalizer.execute()
    runner.service='inactive';finalizer.execute()
    assert all('--query-compute-apps=pid,process_name' in command for command in runner.commands if command[0]=='nvidia-smi')


def test_busy_training_lock_does_not_consume_a_recording_seed(tmp_path):
    complete_fixture(tmp_path);runner=Runner(tmp_path);finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    with (tmp_path/'outputs/upper_body_training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(F.Pending):finalizer.execute()
    assert not finalizer.state.get('recording_attempts') and runner.records==0


def test_publish_uses_only_exact_media_allowlist_and_reuses_upload_commit_and_push(tmp_path):
    complete_fixture(tmp_path);runner=Runner(tmp_path);runner.remote=git_fixture(tmp_path)
    finalizer=F.Finalizer(tmp_path,runner,FakeRecorder());result=finalizer.execute(publish=True)
    assert result['phase']=='complete' and result['published'] is True
    assert runner.uploads==runner.pushes==runner.records==1
    assert set(local_git(tmp_path,'diff-tree','--no-commit-id','--name-only','-r','HEAD').splitlines())==F.PUBLISH_PATHS
    assert all('checkpoint' not in name and '.gofile.' not in name and '.candidate' not in name for name in F.PUBLISH_PATHS)
    text=(tmp_path/'README.md').read_text()
    assert 'Old V6: NOT ACCEPTED' in text and 'Keep old V4 failed history' in text
    assert 'https://gofile.io/d/PublicV6' in text and 'PRIVATE_TEST_TOKEN' not in text
    assert 'PRIVATE_TEST_TOKEN' not in finalizer.logfile.read_text()
    assert (finalizer.job/'upload.gofile.json').stat().st_mode&0o777==0o600
    commit=result['commit'];finalizer.execute(publish=True)
    assert runner.uploads==runner.pushes==runner.records==1 and finalizer.state['commit']==commit


def test_resume_after_commit_crash_adopts_exact_job_commit_without_duplicate_upload(tmp_path):
    class CrashRunner(Runner):
        crashed=False
        def __call__(self,command,**kwargs):
            result=super().__call__(command,**kwargs)
            if command[:2]==['git','commit'] and not self.crashed:
                self.crashed=True
                raise RuntimeError('CPU injected interruption after successful local commit')
            return result
    complete_fixture(tmp_path);runner=CrashRunner(tmp_path);runner.remote=git_fixture(tmp_path)
    first=F.Finalizer(tmp_path,runner,FakeRecorder())
    with pytest.raises(RuntimeError,match='injected interruption'):first.execute(publish=True)
    assert runner.uploads==1 and runner.pushes==0
    assert first.state.get('commit') is None and first.state['git_commit_intent'] is True
    resumed=F.Finalizer(tmp_path,runner,FakeRecorder());result=resumed.execute(publish=True)
    assert result['published'] is True and runner.uploads==runner.records==runner.pushes==1
    assert local_git(tmp_path,'rev-list','--count','HEAD')=='2'


def test_remote_change_after_commit_never_force_pushes_or_claims_publication(tmp_path):
    class ChangedRemote(Runner):
        def __call__(self,command,**kwargs):
            result=super().__call__(command,**kwargs)
            if command[:2]==['git','commit']:self.remote='f'*40
            return result
    complete_fixture(tmp_path);runner=ChangedRemote(tmp_path);runner.remote=git_fixture(tmp_path)
    finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    with pytest.raises(RuntimeError,match='Remote main changed'):finalizer.execute(publish=True)
    assert not finalizer.state.get('published') and runner.pushes==0
    assert not any('--force' in arg for command in runner.commands if command[:2]==['git','push'] for arg in command)


def test_failed_seed_captures_remain_private_and_never_reach_upload(tmp_path):
    class FailedCapture(Runner):
        def __call__(self,command,**kwargs):
            if command[0]=='bash':
                self.records+=1;video=Path(kwargs['env']['OUT'])
                F.private_json(video.with_name('failed-seed-'+command[-1]+'.candidate.json'),{'accepted':False})
                return SimpleNamespace(returncode=1,stdout='',stderr='native physical failure')
            return super().__call__(command,**kwargs)
    complete_fixture(tmp_path);before=(tmp_path/'README.md').read_text()
    runner=FailedCapture(tmp_path);finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    with pytest.raises(RuntimeError,match='No accepted capture'):finalizer.execute()
    assert runner.records==3 and runner.uploads==runner.pushes==0
    assert len(list(finalizer.job.glob('*.candidate.json')))==3
    assert not (tmp_path/F.VIDEO).exists() and (tmp_path/'README.md').read_text()==before


def test_upload_mismatch_never_changes_readme_and_uncertain_attempt_never_reuploads(tmp_path):
    complete_fixture(tmp_path);before=(tmp_path/'README.md').read_text()
    runner=Runner(tmp_path);runner.remote=git_fixture(tmp_path);runner.bad_upload=True
    finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    with pytest.raises(RuntimeError,match='MD5'):finalizer.execute(publish=True)
    assert (tmp_path/'README.md').read_text()==before and not (tmp_path/F.VIDEO).exists()
    with pytest.raises(RuntimeError,match='reconcile'):finalizer.execute(publish=True)
    assert runner.uploads==1 and runner.pushes==0


def test_unrelated_staged_change_is_not_committed_or_uploaded(tmp_path):
    complete_fixture(tmp_path);runner=Runner(tmp_path);runner.remote=git_fixture(tmp_path)
    (tmp_path/'unrelated.txt').write_text('user work');local_git(tmp_path,'add','unrelated.txt')
    finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    with pytest.raises(ValueError,match='Unrelated staged'):finalizer.execute(publish=True)
    assert local_git(tmp_path,'diff','--cached','--name-only')=='unrelated.txt'
    assert runner.records==runner.uploads==runner.pushes==0


def test_private_receipt_creation_and_public_url_validation(tmp_path):
    old=os.umask(0)
    try:F.private_json(tmp_path/'receipt.json',{'token':'PRIVATE_TEST_TOKEN'})
    finally:os.umask(old)
    assert (tmp_path/'receipt.json').stat().st_mode&0o777==0o600
    with pytest.raises(ValueError):F.checked_url('https://gofile.io/d/Public?token=PRIVATE_TEST_TOKEN')


def test_isaac_startup_process_blocks_before_compute_context_exists(tmp_path):
    complete_fixture(tmp_path);runner=Runner(tmp_path);runner.isaac='python3 /project/scripts/reinforcement_learning/rsl_rl/train_lift.py --headless'
    finalizer=F.Finalizer(tmp_path,runner,FakeRecorder())
    with pytest.raises(F.Pending,match='Isaac worker'):finalizer.execute()
    assert runner.records==0


def test_wait_mode_polls_without_recording_bypasses_or_network(tmp_path,monkeypatch):
    class WaitingFinalizer:
        calls=0
        def __init__(self,*args):pass
        def execute(self,publish=False):
            assert publish is False
            self.calls+=1
            if self.calls<3:raise F.Pending('Training still active')
        def progress(self,*args,**kwargs):pass
    sleeps=[];monkeypatch.setattr(F,'Finalizer',WaitingFinalizer);monkeypatch.setattr(F.time,'sleep',sleeps.append)
    assert F.main(['--project-root',str(tmp_path),'--wait','--poll-seconds','1'])==0
    assert sleeps==[1.,1.]


def test_job_identity_changes_with_run_or_checkpoint_and_readme_replacement_preserves_history(tmp_path):
    state,gate=complete_fixture(tmp_path);job=F.job_identity(state,gate,tmp_path)
    assert job!=F.job_identity(state,{**gate,'checkpoint_sha256':'different'},tmp_path)
    other={**state,'run':str(tmp_path/'checkpoints/isaac/upper_body_v6/runs/other')}
    assert job!=F.job_identity(other,gate,tmp_path)
    text=(tmp_path/'README.md').read_text();replaced=F.replace_readme(text,'new verified block')
    assert replaced.endswith('Keep old V4 failed history\n<!-- V4_STATUS_END -->\n')
    with pytest.raises(ValueError):F.replace_readme('No markers','new')
