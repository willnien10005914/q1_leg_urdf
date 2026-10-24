#!/usr/bin/env python3
"""Recover only pre-PPO Isaac startup failures; never restart learning workers.

The frozen V6 trainer is unchanged. User interruptions, source changes, new
snapshots and any evidence that an actor was constructed all prevent retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
UNIT = 'q1-leg-upper-training'


def startup_only(state, tail, result, newer_snapshots=False):
    """Fail closed: the last worker must not have reached actor construction."""
    if state.get('state') not in ('training', 'runtime_error'):
        return False
    if state.get('state') == 'runtime_error' and 'training worker' not in state.get('error', ''):
        return False
    if (type(state.get('completed_iterations')) is not int
            or not 0 < state['completed_iterations'] < 20000
            or state.get('robot') != 'q1_leg_upper_body_v6'
            or result.get('completed_iterations') != state['completed_iterations']
            or newer_snapshots):
        return False
    marker = tail.rfind('V6_SUPERVISOR ')
    if marker < 0:
        return False
    segment = tail[marker:]
    try:
        header = json.loads(segment.splitlines()[0].removeprefix('V6_SUPERVISOR '))
    except (ValueError, IndexError):
        return False
    if (header.get('state') != 'training'
            or header.get('completed_iterations') != state['completed_iterations']
            or header.get('command') != state.get('current_command')):
        return False
    return not any(token in segment for token in (
        'Actor Model:', 'Critic Model:', 'Learning iteration', 'V6_TRAINING_RESULT'))


def read_json(path):
    return json.loads(path.read_text())


def elapsed_process(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    started = int(stat[19]) / os.sysconf('SC_CLK_TCK')
    return float(Path('/proc/uptime').read_text().split()[0]) - started


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit],
                          check=False).returncode == 0


def tick(enforce):
    state = read_json(ROOT / 'outputs/upper_body_training_status.json')
    run = Path(state['run']).resolve()
    if not run.is_relative_to((ROOT / 'checkpoints/isaac/upper_body_v6/runs').resolve()):
        return {'action': 'none', 'reason': 'Unexpected run path'}
    result_path = run / 'training_result.json'
    result = read_json(result_path)
    log = ROOT / 'outputs/upper_body_training.log'
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 131072))
        tail = stream.read().decode(errors='replace')
    newer = any(path.stat().st_mtime_ns > result_path.stat().st_mtime_ns
                for path in run.glob('model_*.pt'))
    if not startup_only(state, tail, result, newer):
        return {'action': 'none', 'reason': 'No proven pre-PPO startup failure'}
    running = active(UNIT)
    if running:
        if state['state'] != 'training' or not state.get('child_pid'):
            return {'action': 'none', 'reason': 'No matching initializing child'}
        elapsed = elapsed_process(state['child_pid'])
        if elapsed < 600:
            return {'action': 'none', 'reason': 'Startup is within 600-second limit'}
    elif state['state'] != 'runtime_error':
        return {'action': 'none', 'reason': 'User interruption or service state mismatch'}
    if any(active(unit) for unit in ('q1-leg-live', 'q1-leg-training', 'q1-leg-lift-training')):
        return {'action': 'none', 'reason': 'Another native Q1 service is active'}
    spec = importlib.util.spec_from_file_location('frozen_upper_monitor', ROOT / 'tools/run_upper_body_training.py')
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    if (monitor.source_manifest(Path(state['profile_path'])) != state['source_sha256']
            or not monitor.checkpoint_valid(run / 'resume.pt')
            or monitor.sha256(run / 'resume.pt') != result['checkpoint_sha256']):
        return {'action': 'none', 'reason': 'Source or checkpoint identity changed'}
    ledger_path = ROOT / 'outputs/upper_body_startup_recovery.json'
    ledger = read_json(ledger_path) if ledger_path.exists() else {'attempts': []}
    key = f'{run.name}:{state["completed_iterations"]}'
    today = datetime.now(timezone.utc).date().isoformat()
    if (sum(item['key'] == key for item in ledger['attempts']) >= 2
            or sum(item['date_utc'] == today for item in ledger['attempts']) >= 4):
        return {'action': 'none', 'reason': 'Bounded startup retry allowance exhausted'}
    decision = {'action': 'retry_startup', 'run': run.name,
                'completed_iterations': state['completed_iterations'],
                'ppo_updates_repeated': False, 'enforced': enforce}
    if not enforce:
        return decision
    # Recheck the exact worker identity immediately before any service action.
    current = read_json(ROOT / 'outputs/upper_body_training_status.json')
    if any(current.get(k) != state.get(k) for k in ('run', 'state', 'child_pid', 'completed_iterations', 'current_command')):
        return {'action': 'none', 'reason': 'Worker changed during audit'}
    ledger['attempts'].append({**decision, 'key': key, 'date_utc': today,
                               'at': datetime.now(timezone.utc).isoformat()})
    monitor.atomic_json(ledger_path, ledger)
    if running:
        subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True, timeout=45)
    subprocess.run(['bash', str(ROOT / 'start_upper_body_training.sh'),
                    '--resume-run', str(run)], check=True, timeout=30)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enforce', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            result = tick(args.enforce)
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as error:
            result = {'action': 'none', 'reason': f'Audit unavailable: {error}'}
        print(json.dumps(result), flush=True)
        if args.once or not args.enforce:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
