#!/usr/bin/env python3
"""External timeout guard for frozen lift validation workers; dry-run by default.

This process never starts PPO, writes checkpoints/evaluations, changes gates or
declares success. The existing supervisor owns its one validation retry and
fresh-JSON checks. A second hanging attempt is terminated so that supervisor
reports an error. Only receipts under outputs/private are written here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time


PHASES=('validating','confirming')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argument(argv,name):
    if argv.count(name)!=1:
        raise ValueError(f'Expected exactly one {name}')
    index=argv.index(name)
    if index+1>=len(argv) or argv[index+1].startswith('--'):
        raise ValueError(f'Missing value for {name}')
    return argv[index+1]


def inside(root,value):
    path=Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute project-local path without parent traversal')
    path=path.resolve()
    if not path.is_relative_to(root):raise ValueError('Path escapes the project')
    return path


def integer(value):
    if type(value) is int:return value
    if not isinstance(value,str) or str(int(value))!=value:raise ValueError('Expected a canonical integer')
    return int(value)


def fingerprint(process):
    # PPID can become init after TERM kills the wrapper; starttime + pidfd
    # still identify an original surviving Kit descendant unambiguously.
    return tuple(process[key] for key in ('pid','start_ticks','uid','pgid','session','argv','boot_id'))


class ProcReader:
    def read(self,pid):
        path=Path('/proc')/str(pid)
        try:
            uid=path.stat().st_uid
            if uid!=os.getuid():return None  # never inspect foreign cwd/argv
            stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
            ticks=int(fields[19]);uptime=float(Path('/proc/uptime').read_text().split()[0])
            return {'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),'session':int(fields[3]),
                    'start_ticks':ticks,'uid':uid,'state':fields[0],
                    'age_s':uptime-ticks/os.sysconf('SC_CLK_TCK'),
                    'argv':tuple(item.decode(errors='replace') for item in (path/'cmdline').read_bytes().split(b'\0') if item),
                    'cwd':str((path/'cwd').resolve()),
                    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        except (FileNotFoundError,ProcessLookupError,PermissionError):return None

    def members(self,pgid):
        result=[]
        for path in Path('/proc').iterdir():
            if path.name.isdigit():
                process=self.read(int(path.name))
                if process and process['pgid']==pgid and process['state']!='Z':result.append(process)
        return result

    def send(self,original,sig):
        """Signal an identity-bound pidfd; never signal a reused numeric PID."""
        try:descriptor=os.pidfd_open(original['pid'],0)
        except ProcessLookupError:return False
        try:
            current=self.read(original['pid'])
            if current is None or current['state']=='Z':return False
            if fingerprint(current)!=fingerprint(original):
                raise ValueError('Process identity changed before signal')
            try:signal.pidfd_send_signal(descriptor,sig,None,0)
            except ProcessLookupError:return False  # exited after identity read
            return True
        finally:os.close(descriptor)


def inspect(root,reader,now=None,uid=None):
    """Return a fully checked validation snapshot, or None for other phases."""
    root=Path(root).resolve();now=time.time() if now is None else now
    uid=os.getuid() if uid is None else uid
    status=json.loads((root/'outputs/lift_training_status.json').read_text())
    if status.get('state') not in PHASES:return None
    if status.get('robot')!='q1_leg_lift_v5':raise ValueError('Unexpected robot status')
    updated=datetime.fromisoformat(status['updated_at']).timestamp()
    if not -5<=now-updated<=60:raise ValueError('Supervisor heartbeat is stale or in the future')
    run=inside(root,status['run'])
    if not run.is_relative_to(root/'checkpoints/isaac/lift_v5/runs'):
        raise ValueError('Unexpected training-run location')
    command=status['current_command']
    if not isinstance(command,list) or not all(isinstance(part,str) for part in command):raise ValueError('Invalid worker command')
    if command.count('--evaluate')!=1 or any(flag in command for flag in ('--bootstrap','--fork-policy','--iterations')):
        raise ValueError('Only an explicit evaluation command may be interrupted')
    if len(command)<2 or Path(command[1]).resolve()!=root/'train_lift_native.sh':
        raise ValueError('Unexpected validation launcher')
    completed=integer(status['completed_iterations']);stage=integer(status['curriculum_stage'])
    if completed<0 or stage not in (0,1,2):raise ValueError('Invalid validation counter/stage')
    if integer(argument(command,'--completed'))!=completed or integer(argument(command,'--curriculum-stage'))!=stage:
        raise ValueError('Command counter/stage differs from status')
    seed=integer(argument(command,'--seed'));episodes=integer(argument(command,'--episodes'))
    expected=128 if status['state']=='confirming' else 32
    if seed<0 or episodes!=expected or integer(argument(command,'--num_envs'))!=episodes:
        raise ValueError('Invalid independent validation seed/environment count')
    if inside(root,argument(command,'--output-dir'))!=run:raise ValueError('Worker output directory differs from run')
    checkpoint=inside(root,argument(command,'--checkpoint'))
    if checkpoint.parent!=run or checkpoint.name!='resume.pt':raise ValueError('Unexpected checkpoint location')
    if inside(root,argument(command,'--training-profile'))!=inside(root,status['profile_path']):
        raise ValueError('Worker training profile differs from status')
    supervisor=reader.read(integer(status['pid']));child=reader.read(integer(status['child_pid']))
    if supervisor is None or child is None:return None
    if supervisor['uid']!=uid or child['uid']!=uid:raise ValueError('Supervisor/worker UID mismatch')
    if supervisor['state']=='Z' or child['state']=='Z':return None
    if supervisor['cwd']!=str(root) or child['cwd']!=str(root):raise ValueError('Process working directory differs from project')
    script=str(root/'tools/run_lift_training.py')
    if script not in supervisor['argv']:raise ValueError('Supervisor executable identity mismatch')
    if (child['ppid']!=supervisor['pid'] or child['pgid']!=child['pid'] or child['session']!=child['pid']
            or child['pgid']==supervisor['pgid'] or child['pid']==os.getpid()):
        raise ValueError('Validation worker is not its own supervisor-owned process session')
    # started_at is the experiment start, preserved across reboot/resume; it
    # must not be mistaken for this supervisor process's own start time.
    if child['age_s']>supervisor['age_s']+1:raise ValueError('Worker predates its claimed supervisor')
    if '--evaluate' not in child['argv'] or any(flag in child['argv'] for flag in ('--bootstrap','--fork-policy','--iterations')):
        raise ValueError('Live child is not a validation process')
    if not any(part.endswith('/train_lift.py') or part.endswith('/train_lift_native.sh') for part in child['argv']):
        raise ValueError('Live child worker script identity mismatch')
    for name in ('--checkpoint','--output-dir','--training-profile','--completed','--curriculum-stage','--seed','--episodes','--num_envs'):
        if argument(child['argv'],name)!=argument(command,name):raise ValueError(f'Live child argument mismatch: {name}')
    result=json.loads((run/'training_result.json').read_text())
    digest=sha256(checkpoint)
    if (result.get('completed_iterations')!=completed or result.get('checkpoint_sha256')!=digest
            or result.get('contract',{}).get('robot')!='q1_leg_lift_v5'):
        raise ValueError('Validation checkpoint does not match the retained completed training chunk')
    segment=status.get('last_segment')
    if segment and sha256(inside(root,segment)/'resume.pt')!=digest:
        raise ValueError('Retained immutable chunk differs from validation checkpoint')
    retry=status.get('validation_worker_restart',{})
    retried=(retry.get('completed_iterations')==completed and retry.get('phase')==status['state'] and retry.get('attempt')==1)
    key=hashlib.sha256(f'{digest}:{stage}:{seed}'.encode()).hexdigest()
    return {'phase':status['state'],'run':str(run),'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
            'completed_iterations':completed,'stage':stage,'seed':seed,'key':key,'retried':retried,
            'supervisor':supervisor,'child':child,'command':tuple(command),
            'expected_report':str(run/f'evaluation_{completed}_s{stage}_{seed}.json')}


def identity(candidate):
    return (candidate['phase'],candidate['run'],candidate['key'],candidate['command'],
            fingerprint(candidate['supervisor']),fingerprint(candidate['child']))


def atomic_receipt(path,journal):
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.watchdog_',delete=False) as output:
        temporary=Path(output.name)
        os.fchmod(output.fileno(),0o600)  # private before the first receipt byte
        json.dump(journal,output,indent=2,allow_nan=False);output.write('\n')
        output.flush();os.fsync(output.fileno())
    temporary.replace(path)


def intervene(root,reader,journal,save,*,enforce=False,timeout=300.,grace=15.,warn_run_recoveries=3,sleep=time.sleep):
    candidate=inspect(root,reader)
    if candidate is None:return {'action':'idle'}
    if candidate['child']['age_s']<=timeout:return {'action':'waiting','worker_pid':candidate['child']['pid'],'age_s':candidate['child']['age_s']}
    run=journal.setdefault('runs',{}).setdefault(candidate['run'],{'recoveries':0,'keys':{},'events':[]})
    key=run['keys'].setdefault(candidate['key'],{'recoveries':0,'terminated_workers':[]})
    child_id=f"{candidate['child']['boot_id']}:{candidate['child']['pid']}:{candidate['child']['start_ticks']}"
    if child_id in key['terminated_workers']:return {'action':'already_signalled','worker_pid':candidate['child']['pid']}
    # A prior retry is never converted into another recovery by restarting this
    # watcher. The frozen supervisor will fail after its second worker exits.
    recovery=not candidate['retried'] and key['recoveries']==0
    event={key:candidate[key] for key in ('phase','run','checkpoint','checkpoint_sha256','completed_iterations','stage','seed','expected_report')}
    event.update(action='recover_validation' if recovery else 'abort_repeated_validation',
                 worker_pid=candidate['child']['pid'],worker_start_ticks=candidate['child']['start_ticks'],
                 age_s=candidate['child']['age_s'],enforced=enforce,signals=[],
                 at=datetime.now(timezone.utc).isoformat(),training_chunk_repeated=False,
                 evaluation_success_declared=False)
    if run['recoveries']>=warn_run_recoveries:
        event['warning']='Repeated validation initialization failures in this run; investigate the simulator/driver. This watcher never signals the supervisor.'
    if not enforce:return event
    fresh=inspect(root,reader)
    if fresh is None or identity(fresh)!=identity(candidate):raise ValueError('Validation state/process changed before TERM')
    members=reader.members(candidate['child']['pgid'])
    if (not any(fingerprint(member)==fingerprint(candidate['child']) for member in members)
            or any(member['uid']!=candidate['child']['uid'] or member['session']!=candidate['child']['pid']
                   or member['pid'] in (candidate['supervisor']['pid'],os.getpid()) for member in members)):
        raise ValueError('Validation process-group membership is ambiguous')
    if recovery:key['recoveries']+=1;run['recoveries']+=1
    run['events'].append(event);save(journal)  # intent durable before any signal
    for member in sorted(members,key=lambda row:row['pid']==candidate['child']['pid']):
        if reader.send(member,signal.SIGTERM):event['signals'].append({'pid':member['pid'],'signal':'TERM'})
    save(journal)
    sleep(grace)
    for member in members:
        current=reader.read(member['pid'])
        if current is None or current['state']=='Z':continue
        if fingerprint(current)!=fingerprint(member):
            event.setdefault('refused_reused_pids',[]).append(member['pid']);continue
        # pidfd binds the original validation member even if a retry has since
        # started in a different session. No new group is ever targeted.
        if reader.send(member,signal.SIGKILL):event['signals'].append({'pid':member['pid'],'signal':'KILL'})
    key['terminated_workers'].append(child_id)
    event['finished_at']=datetime.now(timezone.utc).isoformat();save(journal)
    return event


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--enforce',action='store_true',help='Enable validation-only TERM/KILL; default only reports proposed actions')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--timeout',type=float,default=300.)
    parser.add_argument('--grace',type=float,default=15.)
    parser.add_argument('--poll',type=float,default=10.)
    parser.add_argument('--warn-run-recoveries',type=int,default=3,help='Soft warning after this many distinct validation recoveries; never signals the supervisor')
    args=parser.parse_args()
    if args.timeout<300 or args.grace<=0 or args.poll<=0 or args.warn_run_recoveries<1:
        parser.error('Timeout must be >=300s, grace/poll positive and warning threshold positive')
    root=args.root.resolve();private=(root/'outputs/private').resolve()
    if not private.is_relative_to(root):parser.error('Private receipt path escapes project')
    journal={'schema':1,'root':str(root),'runs':{}}
    path=private/'lift_validation_watchdog.json'
    lock=None
    if args.enforce:
        private.mkdir(parents=True,exist_ok=True)
        lock_path=private/'lift_validation_watchdog.lock'
        if path.is_symlink() or lock_path.is_symlink():parser.error('Receipt and lock must not be symlinks')
        if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
            parser.error('Enforcement requires Linux pidfd support')
        lock=lock_path.open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            journal=json.loads(path.read_text())
            if journal.get('schema')!=1 or journal.get('root')!=str(root):raise ValueError('Receipt belongs to another project/schema')
    save=lambda value:atomic_receipt(path,value)
    while True:
        try:
            result=intervene(root,ProcReader(),journal,save,enforce=args.enforce,timeout=args.timeout,
                             grace=args.grace,warn_run_recoveries=args.warn_run_recoveries)
        except (OSError,ValueError,KeyError,TypeError) as error:
            result={'action':'guard_refused','error':f'{type(error).__name__}: {error}','enforced':args.enforce}
        print('LIFT_VALIDATION_WATCHDOG '+json.dumps(result,allow_nan=False),flush=True)
        if args.once:return
        time.sleep(args.poll)


if __name__=='__main__':main()
