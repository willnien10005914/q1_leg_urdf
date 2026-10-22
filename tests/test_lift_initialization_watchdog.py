"""Only a stopped, reverified pre-PPO initialization may be aborted."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import watch_lift_initialization as guard


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Reader:
    def __init__(self,processes):self.processes=processes;self.sent=[];self.on_stop=None
    def read(self,pid):return copy.deepcopy(self.processes.get(pid))
    def members(self,pgid):return [copy.deepcopy(row) for row in self.processes.values() if row['pgid']==pgid]
    def send(self,original,sig):
        current=self.read(original['pid'])
        if current is None:return False
        if guard.fingerprint(current)!=guard.fingerprint(original):raise ValueError('PID reused')
        self.sent.append((original['pid'],sig))
        if sig==signal.SIGSTOP:
            self.processes[original['pid']]['state']='T'
            if self.on_stop:self.on_stop()
        elif sig==signal.SIGCONT:self.processes[original['pid']]['state']='S'
        elif sig==signal.SIGKILL:self.processes.pop(original['pid'])
        return True


def setup(tmp_path):
    root=tmp_path.resolve();run=root/'checkpoints/isaac/lift_v5/runs/test'
    run.mkdir(parents=True);checkpoint=run/'resume.pt';checkpoint.write_bytes(b'policy before pending chunk')
    digest=guard.sha256(checkpoint)
    write(run/'training_result.json',{'completed_iterations':13115,'checkpoint_sha256':digest})
    chunk=run/'chunks/0013115_s2_r2';chunk.mkdir(parents=True);(chunk/'resume.pt').write_bytes(checkpoint.read_bytes())
    manifest={}
    for name in ('tools/run_lift_training.py','scripts/reinforcement_learning/rsl_rl/train_lift.py','config/lift_v5.json'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen source')
        manifest[name]=guard.sha256(path)
    command=['bash',str(root/'train_lift_native.sh'),'--output-dir',str(run),
             '--training-profile',str(root/'config/lift_v5.json'),'--num_envs','10240',
             '--iterations','500','--completed','13115','--reward-revision','2',
             '--curriculum-stage','2','--headless','--checkpoint',str(checkpoint)]
    status={'state':'training','robot':'q1_leg_lift_v5','run':str(run),'profile_path':str(root/'config/lift_v5.json'),
            'completed_iterations':13115,'curriculum_stage':2,'num_envs':10240,'chunk_target_iterations':13615,
            'pid':10010,'child_pid':10020,'current_command':command,'source_sha256':manifest,'last_segment':str(chunk),
            'log':'outputs/lift_training.log','updated_at':datetime.now(timezone.utc).isoformat()}
    status_path=root/'outputs/lift_training_status.json';write(status_path,status)
    header='V5_SUPERVISOR '+json.dumps({'state':'training','completed_iterations':13115,'command':command})+'\n'
    stamp=datetime.fromtimestamp(time.time()-700,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    log=root/'outputs/lift_training.log';log.write_text('Learning iteration 13099/13115\nActor Model: old worker\n'+header+stamp+' [INFO] Kit boot\n'+guard.START+'\n')
    def process(pid,ppid,pgid,session,argv,age):
        return {'pid':pid,'ppid':ppid,'pgid':pgid,'session':session,'start_ticks':pid*100,
                'uid':os.getuid(),'state':'S','argv':tuple(argv),'age_s':age,'cwd':str(root),'boot_id':'A'}
    reader=Reader({10010:process(10010,1,10010,10010,['python',str(root/'tools/run_lift_training.py')],1000.),
                   10020:process(10020,10010,10020,10020,['bash',str(root/'train_lift_native.sh'),*command[2:]],700.),
                   10021:process(10021,10020,10020,10020,['python','scripts/reinforcement_learning/rsl_rl/train_lift.py',*command[2:]],700.)})
    return root,run,status_path,status,log,reader


def test_old_learning_logs_do_not_contaminate_exact_new_worker_and_dry_run_is_read_only(tmp_path):
    root,_,_,_,_,reader=setup(tmp_path);saved=[]
    result=guard.intervene(root,reader,{},saved.append)
    assert result['action']=='abort_proven_initialization' and not result['enforced']
    assert not reader.sent and not saved


@pytest.mark.parametrize('marker',guard.PROGRESS)
def test_any_current_progress_marker_prevents_all_signals(tmp_path,marker):
    root,_,_,_,log,reader=setup(tmp_path)
    with log.open('a') as output:output.write(marker+'\n')
    assert guard.intervene(root,reader,{},lambda unused:None,enforce=True)['action']=='idle_or_progressed_beyond_initialization'
    assert not reader.sent


def test_proven_hang_is_paused_rechecked_and_killed_without_requeue_or_checkpoint_write(tmp_path):
    root,run,_,_,_,reader=setup(tmp_path);checkpoint=(run/'resume.pt').read_bytes();journal={};saved=[]
    result=guard.intervene(root,reader,journal,lambda value:saved.append(copy.deepcopy(value)),enforce=True)
    assert result['action']=='abort_proven_initialization' and not result['auto_requeue']
    assert reader.sent==[(10021,signal.SIGSTOP),(10020,signal.SIGSTOP),(10020,signal.SIGKILL),(10021,signal.SIGKILL)]
    assert all(pid!=10010 for pid,_ in reader.sent)
    assert (run/'resume.pt').read_bytes()==checkpoint
    assert len(journal['aborted_chunks'])==1 and 'pending_pause' not in journal
    assert any(row.get('pending_pause',{}).get('kill_authorized') for row in saved)


def test_actor_appearing_in_the_pause_race_is_resumed_and_never_killed(tmp_path):
    root,_,_,_,log,reader=setup(tmp_path);journal={}
    def progress():
        if 'Actor Model: new' not in log.read_text():
            with log.open('a') as output:output.write('Actor Model: new\n')
    reader.on_stop=progress
    result=guard.intervene(root,reader,journal,lambda unused:None,enforce=True)
    assert result['action']=='progress_detected_resumed'
    assert all(sig!=signal.SIGKILL for _,sig in reader.sent)
    assert all(row['state']=='S' for row in reader.processes.values())
    assert not journal['aborted_chunks'] and 'pending_pause' not in journal


def test_source_change_during_pause_is_resumed_without_abort(tmp_path):
    root,_,_,_,_,reader=setup(tmp_path)
    reader.on_stop=lambda:(root/'config/lift_v5.json').write_text('changed after pause')
    with pytest.raises(ValueError,match='Frozen source changed'):
        guard.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert all(sig!=signal.SIGKILL for _,sig in reader.sent)
    assert all(row['state']=='S' for row in reader.processes.values())


def test_periodic_update_artifact_and_wrong_latest_worker_header_fail_closed(tmp_path):
    root,run,_,_,log,reader=setup(tmp_path)
    (run/'model_13115.pt').write_bytes(b'PPO may already have run')
    assert guard.intervene(root,reader,{},lambda unused:None,enforce=True)['action']=='idle_or_progressed_beyond_initialization'
    (run/'model_13115.pt').unlink()
    with log.open('a') as output:output.write('V5_SUPERVISOR '+json.dumps({'state':'validating','completed_iterations':13115,'command':[]})+'\n')
    with pytest.raises(ValueError,match='Newest supervisor log header'):
        guard.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert not reader.sent


def test_checkpoint_change_and_pid_reuse_cannot_be_killed(tmp_path,monkeypatch):
    root,run,_,_,_,reader=setup(tmp_path)
    (run/'resume.pt').write_bytes(b'changed policy')
    with pytest.raises(ValueError,match='checkpoint/result changed'):
        guard.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert not reader.sent
    root,_,_,_,_,reader=setup(tmp_path/'second');original=guard.inspect;calls=[]
    def racing(*args,**kwargs):
        if calls:reader.processes[10020]['start_ticks']+=1
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(guard,'inspect',racing)
    with pytest.raises(ValueError,match='changed before pause'):
        guard.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert not reader.sent


@pytest.mark.parametrize('authorized',[False,True])
def test_guard_restart_recovers_its_persisted_pause_without_starting_anything(tmp_path,authorized):
    _,_,_,_,_,reader=setup(tmp_path)
    members=reader.members(10020)
    for row in members:reader.processes[row['pid']]['state']='T'
    # Simulate JSON round-trip: argv tuples become lists.
    journal=json.loads(json.dumps({'pending_pause':{'members':members,'kill_authorized':authorized}}))
    saved=[];guard.resume_pending(reader,journal,saved.append)
    assert 'pending_pause' not in journal and len(saved)==1
    assert reader.sent==[(10020,signal.SIGKILL if authorized else signal.SIGCONT),
                         (10021,signal.SIGKILL if authorized else signal.SIGCONT)]


def test_authorized_abort_receipt_never_kills_an_externally_resumed_possible_ppo(tmp_path):
    _,_,_,_,_,reader=setup(tmp_path)
    journal={'pending_pause':{'members':reader.members(10020),'kill_authorized':True}}
    with pytest.raises(ValueError,match='externally resumed'):
        guard.resume_pending(reader,journal,lambda unused:None)
    assert not reader.sent
