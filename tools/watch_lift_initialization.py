#!/usr/bin/env python3
"""External, fail-closed guard for proven pre-PPO simulation initialization.

Dry-run is the default. Enforcement first pauses only captured worker members,
then repeats every identity/source/checkpoint/log check while they are stopped.
Progress beyond simulation initialization causes CONT, never TERM/KILL. Only a
still-proven initialization hang is killed; the frozen supervisor then fails
the chunk normally. This guard never restarts or requeues training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

from watch_lift_validation import ProcReader, argument, atomic_receipt, fingerprint, inside, integer, sha256


START='[INFO]: Starting the simulation. This may take a few seconds. Please wait...'
PROGRESS=('Actor Model:', 'Critic Model:', 'ActorCritic(', 'Learning iteration',
          'V5_TRAINING_RESULT ', 'V5_EVALUATION ', 'V5_SMOKE ', 'Time taken for simulation start')


def relative_file(root,value):
    if not isinstance(value,str) or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Invalid frozen source path')
    return inside(root,str(root/value))


def worker_log(root,status,child,now):
    path=relative_file(root,status['log']);before=path.stat()
    # A missing/truncated/unreasonably large current segment is ambiguity,
    # never evidence that a worker has not started learning.
    with path.open('rb') as stream:
        start=max(0,before.st_size-8*1024*1024);stream.seek(start);data=stream.read()
    lines=data.splitlines(keepends=True);offset=start;last=None
    for line in lines:
        if line.startswith(b'V5_SUPERVISOR '):last=(offset,line)
        offset+=len(line)
    if last is None:raise ValueError('Latest worker header is not in the log tail')
    offset,line=last;header=json.loads(line[len(b'V5_SUPERVISOR '):])
    if (header.get('state')!='training' or header.get('completed_iterations')!=status['completed_iterations']
            or header.get('command')!=status['current_command']):
        raise ValueError('Newest supervisor log header does not match this exact worker')
    segment=data[offset-start+len(line):].decode(errors='replace')
    if START not in segment:raise ValueError('Current worker has no explicit simulation-start marker')
    if any(token in segment for token in PROGRESS):return None
    stamp=re.search(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ',segment)
    if stamp is None:raise ValueError('Current worker segment lacks a dated Kit startup marker')
    birth=datetime.fromisoformat(stamp.group().replace('Z','+00:00')).timestamp()
    if abs(birth-(now-child['age_s']))>90:raise ValueError('Log segment predates this process identity')
    after=path.stat()
    if (before.st_ino,before.st_dev)!=(after.st_ino,after.st_dev) or after.st_size<before.st_size:
        raise ValueError('Log rotated or truncated during inspection')
    return {'path':str(path),'device':before.st_dev,'inode':before.st_ino,'header_offset':offset,
            'header_sha256':hashlib.sha256(line).hexdigest(),'segment_sha256':hashlib.sha256(segment.encode()).hexdigest()}


def inspect(root,reader,now=None):
    root=Path(root).resolve();now=time.time() if now is None else now
    status=json.loads((root/'outputs/lift_training_status.json').read_text())
    if status.get('state')!='training':return None
    if status.get('robot')!='q1_leg_lift_v5':raise ValueError('Unexpected robot')
    heartbeat=datetime.fromisoformat(status['updated_at']).timestamp()
    if not -5<=now-heartbeat<=60:raise ValueError('Stale supervisor heartbeat')
    run=inside(root,status['run'])
    if not run.is_relative_to(root/'checkpoints/isaac/lift_v5/runs'):raise ValueError('Unexpected run location')
    command=status['current_command']
    if (not isinstance(command,list) or not all(isinstance(part,str) for part in command)
            or len(command)<2 or Path(command[1]).resolve()!=root/'train_lift_native.sh'
            or any(flag in command for flag in ('--evaluate','--bootstrap','--fork-policy','--smoke-steps'))):
        raise ValueError('Guard only supports an existing-checkpoint PPO resume initialization')
    completed=integer(status['completed_iterations']);stage=integer(status['curriculum_stage'])
    iterations=integer(argument(command,'--iterations'))
    if completed<1 or iterations<1 or completed+iterations>15000 or stage not in (0,1,2):raise ValueError('Invalid chunk budget/stage')
    if (integer(argument(command,'--completed'))!=completed or integer(argument(command,'--curriculum-stage'))!=stage
            or status.get('chunk_target_iterations')!=completed+iterations
            or integer(argument(command,'--num_envs'))!=status.get('num_envs')):
        raise ValueError('Status and resume command differ')
    if inside(root,argument(command,'--output-dir'))!=run:raise ValueError('Unexpected output directory')
    checkpoint=inside(root,argument(command,'--checkpoint'))
    if checkpoint.parent!=run or checkpoint.name!='resume.pt':raise ValueError('Unexpected checkpoint')
    if inside(root,argument(command,'--training-profile'))!=inside(root,status['profile_path']):raise ValueError('Profile mismatch')
    supervisor=reader.read(integer(status['pid']));child=reader.read(integer(status['child_pid']))
    if supervisor is None or child is None:return None
    if (supervisor['uid']!=os.getuid() or child['uid']!=os.getuid() or supervisor['state']=='Z' or child['state']=='Z'
            or supervisor['cwd']!=str(root) or child['cwd']!=str(root)):
        raise ValueError('Process ownership/cwd mismatch')
    if str(root/'tools/run_lift_training.py') not in supervisor['argv']:raise ValueError('Wrong supervisor executable')
    if (child['ppid']!=supervisor['pid'] or child['pgid']!=child['pid'] or child['session']!=child['pid']
            or child['pgid']==supervisor['pgid'] or child['pid']==os.getpid() or child['age_s']>supervisor['age_s']+1):
        raise ValueError('Not the supervisor-owned worker session')
    if any(flag in child['argv'] for flag in ('--evaluate','--bootstrap','--fork-policy','--smoke-steps')):
        raise ValueError('Live worker mode differs')
    if not any(part.endswith('/train_lift.py') or part.endswith('/train_lift_native.sh') for part in child['argv']):
        raise ValueError('Wrong live worker executable')
    for name in ('--checkpoint','--output-dir','--training-profile','--completed','--curriculum-stage','--iterations','--num_envs','--reward-revision'):
        if argument(child['argv'],name)!=argument(command,name):raise ValueError('Live worker arguments differ')
    manifest=status.get('source_sha256')
    required={'tools/run_lift_training.py','scripts/reinforcement_learning/rsl_rl/train_lift.py'}
    if not isinstance(manifest,dict) or not required.issubset(manifest):raise ValueError('Missing frozen-source manifest')
    for relative,digest in manifest.items():
        if not isinstance(digest,str) or sha256(relative_file(root,relative))!=digest:raise ValueError('Frozen source changed')
    digest=sha256(checkpoint);result=json.loads((run/'training_result.json').read_text())
    if result.get('completed_iterations')!=completed or result.get('checkpoint_sha256')!=digest:
        raise ValueError('Saved checkpoint/result changed')
    segment=inside(root,status['last_segment'])
    if segment.parent!=run/'chunks' or sha256(segment/'resume.pt')!=digest:raise ValueError('Missing unchanged immutable chunk')
    # A periodic policy at/after the inherited counter proves that PPO may
    # already have run, even if the current console output is incomplete.
    if any(path.stem[6:].isdigit() and int(path.stem[6:])>=completed for path in run.glob('model_*.pt')):return None
    proof=worker_log(root,status,child,now)
    if proof is None:return None
    source_digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    key=hashlib.sha256(f'{digest}:{stage}:{completed}:{iterations}'.encode()).hexdigest()
    return {'run':str(run),'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'completed_iterations':completed,
            'stage':stage,'iterations':iterations,'key':key,'child':child,'supervisor':supervisor,
            'command':tuple(command),'source_manifest_sha256':source_digest,'log':proof}


def identity(candidate):
    return (candidate['run'],candidate['key'],candidate['command'],candidate['source_manifest_sha256'],
            fingerprint(candidate['child']),fingerprint(candidate['supervisor']),
            tuple(candidate['log'][key] for key in ('path','device','inode','header_offset','header_sha256')))


def resume_pending(reader,journal,save):
    """Restore an interrupted pause, or finish an authorized still-stopped abort."""
    pending=journal.get('pending_pause')
    if not pending:return
    for saved in pending['members']:
        original={**saved,'argv':tuple(saved['argv'])};current=reader.read(original['pid'])
        if current and fingerprint(current)==fingerprint(original):
            if pending.get('kill_authorized'):
                if current['state'] not in ('T','t','Z'):
                    raise ValueError('An authorized abort member was externally resumed; refusing to kill possible PPO')
                if current['state']!='Z':reader.send(original,signal.SIGKILL)
            elif current['state'] in ('T','t'):reader.send(original,signal.SIGCONT)
    journal.pop('pending_pause',None);save(journal)


def intervene(root,reader,journal,save,*,enforce=False,timeout=600.,sleep=time.sleep):
    candidate=inspect(root,reader)
    if candidate is None:return {'action':'idle_or_progressed_beyond_initialization'}
    if candidate['child']['age_s']<=timeout:return {'action':'waiting','worker_pid':candidate['child']['pid'],'age_s':candidate['child']['age_s']}
    key=candidate['run']+':'+candidate['key']
    if key in journal.setdefault('aborted_chunks',{}):return {'action':'already_aborted_requires_operator','run':candidate['run'],'completed_iterations':candidate['completed_iterations']}
    event={name:candidate[name] for name in ('run','checkpoint','checkpoint_sha256','completed_iterations','stage','iterations','log','source_manifest_sha256')}
    event.update(action='abort_proven_initialization',enforced=enforce,worker_pid=candidate['child']['pid'],
                 age_s=candidate['child']['age_s'],signals=[],training_updates_started=False,
                 auto_requeue=False,at=datetime.now(timezone.utc).isoformat())
    if not enforce:return event
    fresh=inspect(root,reader)
    if fresh is None or identity(fresh)!=identity(candidate):raise ValueError('Initialization evidence changed before pause')
    members=reader.members(candidate['child']['pgid'])
    if (not any(fingerprint(row)==fingerprint(candidate['child']) for row in members)
            or any(row['uid']!=os.getuid() or row['session']!=candidate['child']['pid']
                   or row['pid'] in (candidate['supervisor']['pid'],os.getpid()) for row in members)):
        raise ValueError('Ambiguous worker group')
    journal['pending_pause']={'members':members,'event':event};save(journal)
    killed=False;kill_authorized=False
    try:
        for row in sorted(members,key=lambda value:value['pid']==candidate['child']['pid']):
            if reader.send(row,signal.SIGSTOP):event['signals'].append({'pid':row['pid'],'signal':'STOP_for_recheck'})
        deadline=time.monotonic()+1.
        while True:
            live=[reader.read(row['pid']) for row in members]
            if all(row and row['state'] in ('T','t') for row in live):break
            if time.monotonic()>=deadline:raise ValueError('Could not confirm a complete worker pause')
            sleep(.05)
        fresh=inspect(root,reader)
        if fresh is None or identity(fresh)!=identity(candidate):
            event['action']='progress_detected_resumed';return event
        if {fingerprint(row) for row in reader.members(candidate['child']['pgid'])}!={fingerprint(row) for row in members}:
            raise ValueError('Worker spawned/changed members during pause')
        # All original workers are STOPPED and no post-initialization marker
        # exists. Do not CONT into a possible first PPO update before KILL.
        kill_authorized=True;journal['pending_pause']['kill_authorized']=True
        journal['aborted_chunks'][key]=event;save(journal)
        for row in members:
            if reader.send(row,signal.SIGKILL):event['signals'].append({'pid':row['pid'],'signal':'KILL_proven_pre_PPO'})
        killed=True;event['finished_at']=datetime.now(timezone.utc).isoformat()
        return event
    finally:
        if not kill_authorized:
            for row in members:
                current=reader.read(row['pid'])
                if current and current['state'] in ('T','t') and fingerprint(current)==fingerprint(row):reader.send(row,signal.SIGCONT)
        if not kill_authorized or killed:journal.pop('pending_pause',None)
        save(journal)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--enforce',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--timeout',type=float,default=600.)
    parser.add_argument('--poll',type=float,default=10.)
    args=parser.parse_args()
    if args.timeout<300 or args.poll<=0:parser.error('Timeout must be >=300 seconds and poll positive')
    root=args.root.resolve();private=(root/'outputs/private').resolve()
    if not private.is_relative_to(root):parser.error('Receipt directory escapes project')
    journal={'schema':1,'root':str(root),'aborted_chunks':{}};path=private/'lift_initialization_watchdog.json'
    reader=ProcReader();save=lambda value:atomic_receipt(path,value)
    if args.enforce:
        private.mkdir(parents=True,exist_ok=True);lock_path=private/'lift_initialization_watchdog.lock'
        if path.is_symlink() or lock_path.is_symlink():parser.error('Receipt/lock symlinks are forbidden')
        if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):parser.error('Linux pidfd support required')
        lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            journal=json.loads(path.read_text())
            if journal.get('schema')!=1 or journal.get('root')!=str(root):raise ValueError('Wrong receipt identity')
        resume_pending(reader,journal,save)
    def interrupt(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while True:
            try:
                if args.enforce:resume_pending(reader,journal,save)
                result=intervene(root,reader,journal,save,enforce=args.enforce,timeout=args.timeout)
            except (OSError,ValueError,KeyError,TypeError) as error:result={'action':'guard_refused','error':f'{type(error).__name__}: {error}'}
            print('LIFT_INITIALIZATION_WATCHDOG '+json.dumps(result),flush=True)
            if args.once:return
            time.sleep(args.poll)
    except KeyboardInterrupt:
        if args.enforce:resume_pending(reader,journal,save)
        print('LIFT_INITIALIZATION_WATCHDOG stopped',flush=True)


if __name__=='__main__':main()
