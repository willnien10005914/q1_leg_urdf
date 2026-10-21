"""Validation timeout guards exercised entirely with fake process identities."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import time

import pytest


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('lift_watchdog',ROOT/'tools/watch_lift_validation.py')
watch=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(watch)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Reader:
    def __init__(self,processes):self.processes=processes;self.sent=[]
    def read(self,pid):return copy.deepcopy(self.processes.get(pid))
    def members(self,pgid):return [copy.deepcopy(row) for row in self.processes.values() if row['pgid']==pgid]
    def send(self,original,sig):
        current=self.read(original['pid'])
        if current is None:return False
        if watch.fingerprint(current)!=watch.fingerprint(original):raise ValueError('PID reused before signal')
        self.sent.append((original['pid'],sig));return True


def setup(tmp_path):
    root=tmp_path.resolve();run=root/'checkpoints/isaac/lift_v5/runs/test'
    run.mkdir(parents=True);checkpoint=run/'resume.pt';checkpoint.write_bytes(b'retained learned policy')
    sha=watch.sha256(checkpoint)
    write(run/'training_result.json',{'completed_iterations':1910,'checkpoint_sha256':sha,'contract':{'robot':'q1_leg_lift_v5'}})
    command=['bash',str(root/'train_lift_native.sh'),'--evaluate','--episodes','32',
             '--training-profile',str(root/'config/lift_v5.json'),'--num_envs','32','--seed','172910',
             '--completed','1910','--checkpoint',str(checkpoint),'--output-dir',str(run),
             '--reward-revision','1','--curriculum-stage','0','--headless']
    status={'robot':'q1_leg_lift_v5','state':'validating','run':str(run),'profile_path':str(root/'config/lift_v5.json'),
            'pid':10010,'child_pid':10020,'completed_iterations':1910,'curriculum_stage':0,
            'updated_at':datetime.now(timezone.utc).isoformat(),
            # Old experiment timestamp is intentional: reboot/resume must work.
            'started_at':'2026-01-01T00:00:00+00:00','current_command':command}
    status_path=root/'outputs/lift_training_status.json';write(status_path,status)
    def process(pid,ppid,pgid,session,argv,age):
        return {'pid':pid,'ppid':ppid,'pgid':pgid,'session':session,'start_ticks':pid*100,
                'uid':os.getuid(),'state':'S','argv':tuple(argv),'age_s':age,'cwd':str(root),'boot_id':'boot-A'}
    processes={10010:process(10010,1,10010,10010,['python',str(root/'tools/run_lift_training.py')],1000.),
               10020:process(10020,10010,10020,10020,['bash',str(root/'train_lift_native.sh'),*command[2:]],400.),
               10021:process(10021,10020,10020,10020,['python','scripts/reinforcement_learning/rsl_rl/train_lift.py',*command[2:]],400.)}
    return root,status_path,status,Reader(processes)


def test_dry_run_and_live_300_second_bound_never_send_or_write(tmp_path):
    root,_,_,reader=setup(tmp_path);journal={};saved=[]
    result=watch.intervene(root,reader,journal,saved.append,sleep=lambda unused:None)
    assert result['action']=='recover_validation' and not result['enforced']
    assert not reader.sent and not saved and not result['evaluation_success_declared']
    reader.processes[10020]['age_s']=300.
    assert watch.intervene(root,reader,{},saved.append,enforce=True)['action']=='waiting'
    assert not reader.sent


def test_only_original_validation_members_terminated_and_saved_policy_unchanged(tmp_path):
    root,_,_,reader=setup(tmp_path);journal={};saved=[]
    checkpoint=root/'checkpoints/isaac/lift_v5/runs/test/resume.pt';before=checkpoint.read_bytes()
    result=watch.intervene(root,reader,journal,lambda value:saved.append(copy.deepcopy(value)),enforce=True,sleep=lambda unused:None)
    assert result['action']=='recover_validation'
    assert set(reader.sent)=={(10020,signal.SIGTERM),(10021,signal.SIGTERM),(10020,signal.SIGKILL),(10021,signal.SIGKILL)}
    assert all(pid!=10010 for pid,_ in reader.sent) and len(saved)>=3
    assert checkpoint.read_bytes()==before and not list(checkpoint.parent.glob('evaluation_*.json'))
    assert not result['training_chunk_repeated'] and not result['evaluation_success_declared']


def test_second_hanging_attempt_aborts_without_another_recovery(tmp_path):
    root,path,status,reader=setup(tmp_path);journal={}
    watch.intervene(root,reader,journal,lambda unused:None,enforce=True,sleep=lambda unused:None)
    status['validation_worker_restart']={'attempt':1,'completed_iterations':1910,'phase':'validating'}
    # Preserve the same checkpoint/stage/seed while giving retry a new session.
    status['child_pid']=10030;write(path,status)
    original=reader.processes.pop(10020);reader.processes.pop(10021)
    original.update(pid=10030,pgid=10030,session=10030,start_ticks=999999)
    reader.processes[10030]=original;reader.sent.clear()
    result=watch.intervene(root,reader,journal,lambda unused:None,enforce=True,sleep=lambda unused:None)
    assert result['action']=='abort_repeated_validation'
    assert next(iter(journal['runs'].values()))['recoveries']==1
    assert reader.sent==[(10030,signal.SIGTERM),(10030,signal.SIGKILL)]


def test_training_and_malformed_or_escaping_arguments_are_never_signalled(tmp_path):
    root,path,status,reader=setup(tmp_path)
    status['state']='training';write(path,status)
    assert watch.intervene(root,reader,{},lambda unused:None,enforce=True)['action']=='idle'
    status['state']='validating'
    for flag,value in [('--checkpoint','/tmp/unrelated/resume.pt'),('--output-dir',str(root/'../escape')),
                       ('--completed','1900'),('--seed','-1'),('--episodes','64')]:
        changed=copy.deepcopy(status);index=changed['current_command'].index(flag)
        changed['current_command'][index+1]=value;write(path,changed)
        with pytest.raises(ValueError):watch.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert not reader.sent


@pytest.mark.parametrize('change',['pid_reuse','phase_change','live_argument_change'])
def test_state_and_process_are_rechecked_immediately_before_term(tmp_path,monkeypatch,change):
    root,path,status,reader=setup(tmp_path);original=watch.inspect;calls=[]
    def racing(*args,**kwargs):
        if calls:
            if change=='pid_reuse':reader.processes[10020]['start_ticks']+=1
            elif change=='phase_change':status['state']='training';write(path,status)
            else:
                argv=list(reader.processes[10020]['argv']);argv.remove('--evaluate');reader.processes[10020]['argv']=tuple(argv)
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(watch,'inspect',racing)
    with pytest.raises(ValueError):watch.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert not reader.sent


def test_after_term_new_training_and_pid_reuse_are_never_killed(tmp_path):
    root,path,status,reader=setup(tmp_path)
    def during_grace(unused):
        # Wrapper PID gets reused for an unrelated process. The old original
        # Kit child is reparented, while a new training worker starts elsewhere.
        reader.processes[10020]['start_ticks']+=1
        reader.processes[10021]['ppid']=1
        status['state']='training';status['child_pid']=10040;write(path,status)
        new=copy.deepcopy(reader.processes[10020]);new.update(pid=10040,pgid=10040,session=10040,start_ticks=888888)
        reader.processes[10040]=new
    result=watch.intervene(root,reader,{},lambda unused:None,enforce=True,sleep=during_grace)
    assert (10020,signal.SIGKILL) not in reader.sent
    assert (10021,signal.SIGKILL) in reader.sent  # Original orphan, not new PPO.
    assert all(pid not in (10010,10040) for pid,_ in reader.sent)
    assert result['refused_reused_pids']==[10020]


def test_original_checkpoint_must_match_retained_training_result(tmp_path):
    root,_,_,reader=setup(tmp_path)
    (root/'checkpoints/isaac/lift_v5/runs/test/resume.pt').write_bytes(b'changed source')
    with pytest.raises(ValueError,match='retained completed'):
        watch.intervene(root,reader,{},lambda unused:None,enforce=True)
    assert not reader.sent


def test_proc_reader_skips_unreadable_unrelated_processes(monkeypatch):
    original=Path.stat
    def denied(path,*args,**kwargs):
        if str(path)=='/proc/999999':raise PermissionError('unrelated root process')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',denied)
    assert watch.ProcReader().read(999999) is None


def test_receipts_persist_without_touching_training_files(tmp_path):
    path=tmp_path/'receipt.json';journal={'schema':1,'runs':{'run':{'recoveries':1}}}
    watch.atomic_receipt(path,journal)
    assert json.loads(path.read_text())==journal
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    assert list(tmp_path.iterdir())==[path]


def test_pidfd_esrch_after_identity_read_does_not_abort_remaining_orphan_cleanup(tmp_path,monkeypatch):
    root,_,_,fake=setup(tmp_path)
    reader=watch.ProcReader();reader.read=fake.read;reader.members=fake.members
    sent=[];closed=[]
    monkeypatch.setattr(watch.os,'pidfd_open',lambda pid,flags:pid)
    monkeypatch.setattr(watch.os,'close',closed.append)
    def pidfd_send(fd,sig,info,flags):
        if fd==10020:
            # Root wrapper exits in the exact interval after read()/identity
            # validation and before pidfd_send_signal(). Its Kit child remains.
            fake.processes.pop(10020,None)
            raise ProcessLookupError('exited after fingerprint read')
        sent.append((fd,sig))
    monkeypatch.setattr(watch.signal,'pidfd_send_signal',pidfd_send)
    result=watch.intervene(root,reader,{},lambda unused:None,enforce=True,sleep=lambda unused:None)
    assert result['action']=='recover_validation'
    assert sent==[(10021,signal.SIGTERM),(10021,signal.SIGKILL)]
    assert closed==[10021,10020,10021]
    assert all(row['pid']==10021 for row in result['signals'])
