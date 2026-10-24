#!/usr/bin/env python3
"""Observe frozen V6 training and publish bounded, sanitized progress snapshots.

This process never changes rewards, runs, checkpoints or training services.
Default/--once only prints; --write saves a local snapshot, --watch persists,
and only explicit --publish permits the two public progress files to reach Git.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ('docs/results/upper_body_v6_progress.json', 'docs/UPPER_BODY_V6_PROGRESS.md')
STATUS_FILE = 'outputs/upper_body_monitor_status.json'
LEDGER_FILE = 'outputs/upper_body_progress_publication.json'
SSH_ORIGIN = 'git@github.com:willnien10005914/q1_leg_urdf.git'
ACTIVE_STATES = {'created', 'training', 'validating', 'confirming'}
TERMINAL_STATES = {'runtime_error', 'budget_exhausted', 'artifact_error', 'interrupted'}
KNOWN_STATES = ACTIVE_STATES | TERMINAL_STATES | {'passed', 'complete'}
CASE_NAMES = {'stand', 'inplace', 'left', 'right', 'pair', 'w', 's', 'a', 'd',
              'yaw_left', 'yaw_right', 'sequence'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timestamp(value):
    try:
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    except (AttributeError, TypeError, ValueError):
        return None


def integer(value, minimum=0, maximum=20000):
    return value if type(value) is int and minimum <= value <= maximum else None


def number(value, minimum=0., maximum=1e9):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum else None


def error_code(state):
    error = str(state.get('error') or '').lower()
    if 'timeout' in error or 'timed out' in error:
        return 'worker_timeout'
    if 'exit code' in error or 'segmentation' in error:
        return 'worker_crash'
    if 'source' in error and ('changed' in error or 'mismatch' in error):
        return 'source_changed'
    if 'checkpoint' in error or 'contract' in error:
        return 'checkpoint_or_contract_error'
    return 'runtime_error' if error else None


def service_active(unit):
    result = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit],
                            check=False, timeout=5, capture_output=True)
    return result.returncode == 0


def observations(root, state):
    """Bind reports to their saved artifacts without importing Isaac or torch."""
    result = {'training_service_active': service_active('q1-leg-upper-training'),
              'startup_watchdog_active': service_active('q1-leg-upper-startup-watchdog'),
              'source_verified': False, 'checkpoint_verified': False,
              'publication_verified': False, 'validation_verified': False}
    report = None
    run = Path(state.get('run', '')).resolve()
    allowed = (root / 'checkpoints/isaac/upper_body_v6/runs').resolve()
    if not run.is_relative_to(allowed):
        return result, report
    helper = load_module(root / 'tools/run_upper_body_training.py', 'upper_progress_frozen_sources')
    helper.ROOT = root
    profile = Path(state.get('profile_path', '')).resolve()
    if profile.is_relative_to(root.resolve()):
        result['source_verified'] = helper.source_manifest(profile) == state.get('source_sha256')
    saved_path = run / 'training_result.json'
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text())
        checkpoint = run / 'resume.pt'
        result['checkpoint_verified'] = bool(
            saved.get('completed_iterations') == state.get('completed_iterations')
            and saved.get('contract', {}).get('robot') == 'q1_leg_upper_body_v6'
            and saved.get('budget_mode') == 'new_updates'
            and saved.get('source_iteration') == 13115
            and checkpoint.is_file() and digest(checkpoint) == saved.get('checkpoint_sha256'))
        if result['checkpoint_verified']:
            result['checkpoint_sha256'] = saved['checkpoint_sha256']
    elif state.get('completed_iterations') == 0:
        result['checkpoint_verified'] = None  # No checkpoint is promised before the first chunk.
    latest = state.get('last_evaluation', {})
    path = Path(latest.get('report', '')).resolve()
    if path.is_relative_to(run) and path.is_file() and digest(path) == latest.get('report_sha256'):
        candidate = json.loads(path.read_text())
        if (candidate.get('completed_iterations') == latest.get('completed_iterations')
                and candidate.get('checkpoint_sha256') == latest.get('checkpoint_sha256')
                and candidate.get('backend') == 'isaaclab' and candidate.get('proxy') is False):
            result['validation_verified'] = True
            report = candidate
    if state.get('ready') is True:
        gate = load_module(root / 'tools/v6_policy_gate.py', 'upper_progress_acceptance_gate')
        result['publication_verified'] = gate.v6_policy_gate(root).get('ready') is True
    return result, report


def public_summary(state, audit, report=None, now=None):
    """Only named fields and bounded metric identifiers leave the private state."""
    now = now or datetime.now(timezone.utc)
    phase = state.get('state') if state.get('state') in KNOWN_STATES else 'unknown'
    completed = integer(state.get('completed_iterations'))
    stage = integer(state.get('curriculum_stage'), maximum=2)
    accepted = bool(state.get('ready') is True and audit.get('publication_verified') is True)
    updated = timestamp(state.get('updated_at'))
    age = max(0., (now - updated).total_seconds()) if updated else None
    run_id = Path(str(state.get('run', ''))).name
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        run_id = 'unavailable'
    alerts = []
    if phase in ACTIVE_STATES and (age is None or age > 120):
        alerts.append('training_heartbeat_stale')
    if phase in ACTIVE_STATES and audit.get('training_service_active') is False:
        alerts.append('training_service_inactive')
    if audit.get('source_verified') is not True:
        alerts.append('training_source_not_verified')
    if audit.get('checkpoint_verified') is False:
        alerts.append('saved_checkpoint_not_verified')
    if state.get('plateau_unresolved') is True and not accepted:
        alerts.append('plateau_requires_review')
    if phase in TERMINAL_STATES:
        alerts.append(phase)
    if state.get('ready') is True and not accepted:
        alerts.append('accepted_publication_not_verified')
    latest = state.get('last_evaluation', {})
    failed = latest.get('failed_gates', [])
    failed = [value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', value)
              else 'invalid_metric_identifier' for value in failed[:30]] if isinstance(failed, list) else []
    validation = None
    if latest:
        validation = {'saved_updates': integer(latest.get('completed_iterations')),
                      'stage': integer(latest.get('stage'), maximum=2),
                      'episodes': integer(latest.get('episodes'), maximum=128),
                      'passed': latest.get('passed') is True,
                      'artifact_verified': audit.get('validation_verified') is True,
                      'failed_gate_count': len(latest.get('failed_gates', [])) if isinstance(latest.get('failed_gates'), list) else None,
                      'failed_gates': failed}
        if report is not None and audit.get('validation_verified') is True:
            validation.update(survival_rate=number(report.get('survival_rate'), maximum=1.),
                              lower_body_passed=report.get('lower_body_passed') is True)
            cases = {}
            for name, case in report.get('upper_body', {}).get('cases', {}).items():
                if name not in CASE_NAMES:
                    continue
                tail = case.get('tail', {})
                hands = tail.get('hands_return_distance_rms_m_per_hand')
                cases[name] = {
                    'chest_peak_deg': number(case.get('all', {}).get('chest_peak_max_deg'), maximum=180.),
                    'chest_rms_deg': number(case.get('all', {}).get('chest_rms_deg'), maximum=180.),
                    'dynamic_chest_angular_xy_rms_rad_s': number(case.get('dynamic', {}).get('chest_angular_xy_rms_rad_s')),
                    'tail_arm_speed_rms_rad_s': number(tail.get('arm_speed_rms_rad_s')),
                    'tail_hand_distance_rms_m_per_hand': [number(value) for value in hands] if isinstance(hands, list) and len(hands) == 2 else None,
                }
            validation['upper_body_cases'] = cases
    return {'schema': 1, 'robot': 'q1_leg_upper_body_v6', 'run_id': run_id,
            'as_of': updated.astimezone(timezone.utc).isoformat() if updated else None,
            'state': phase, 'saved_updates': completed, 'max_new_updates': 20000,
            'saved_progress_percent': round(completed / 200., 2) if completed is not None else None,
            'num_envs': 10240, 'curriculum_stage': stage,
            'stage_name': {0: 'inplace_upper_recovery', 1: 'finite_gesture_upper_return', 2: 'interactive_upper'}.get(stage, 'unknown'),
            'reward_revision': integer(state.get('reward_revision'), maximum=2),
            'trained_and_validated': accepted,
            'confirmation_episodes': 128 if accepted else None,
            'training_service_active': audit.get('training_service_active'),
            'startup_watchdog_active': audit.get('startup_watchdog_active'),
            'source_verified': audit.get('source_verified') is True,
            'checkpoint_verified': audit.get('checkpoint_verified'),
            'checkpoint_sha256': audit.get('checkpoint_sha256') if re.fullmatch(r'[0-9a-f]{64}', str(audit.get('checkpoint_sha256', ''))) else None,
            'latest_validation': validation, 'alerts': alerts,
            'error_code': error_code(state) if phase in TERMINAL_STATES else None,
            'scope': 'Saved updates are not acceptance; only independent full-gate V6 confirmation enables trained_and_validated.'}


def event_key(summary, previous):
    """Publish at 1,000 saved-update milestones, stages, recovery and terminal events."""
    completed = summary.get('saved_updates')
    if completed is None:
        return None
    run, stage, phase = summary['run_id'], summary['curriculum_stage'], summary['state']
    if summary['trained_and_validated']:
        return f'{run}:accepted:{summary.get("checkpoint_sha256")}'
    if phase in TERMINAL_STATES:
        return f'{run}:terminal:{phase}:{completed}:{summary.get("error_code")}'
    if not previous or previous.get('run_id') != run:
        return f'{run}:baseline:{completed // 1000}:{stage}'
    health={'training_heartbeat_stale','training_service_inactive','training_source_not_verified',
            'saved_checkpoint_not_verified','accepted_publication_not_verified'}
    current_health=set(summary.get('alerts',[])) & health
    previous_health=set(previous.get('alerts',[])) & health
    if current_health != previous_health:
        identity=hashlib.sha256(','.join(sorted(current_health)).encode()).hexdigest()[:12]
        return f'{run}:health:{identity}:{completed}'
    if previous.get('state') in TERMINAL_STATES and phase in ACTIVE_STATES:
        return f'{run}:resumed:{completed}:{stage}'
    if previous.get('curriculum_stage') != stage:
        return f'{run}:stage:{stage}:{completed}'
    if completed // 1000 > (previous.get('saved_updates') or 0) // 1000:
        return f'{run}:milestone:{completed // 1000}:{stage}'
    return None


def render_markdown(summary):
    stage = {0: '原地抬腳與回穩', 1: '有限左右閃身與回手', 2: '方向控制整合'}.get(summary['curriculum_stage'], '待確認')
    accepted = '已完成獨立完整驗收' if summary['trained_and_validated'] else '尚未完成完整驗收'
    lines = ['# V6 訓練監看', '', f"更新時間（UTC）：{summary['as_of'] or '待確認'}。", '',
             f"已保存 **{summary['saved_updates']}／20,000 次新 PPO 更新**，10,240 envs。",
             f"目前階段：**{stage}**；狀態：`{summary['state']}`；**{accepted}**。", '',
             '保存更新數與檢查項目通過比例均不是任務成功率。只有三次完整整合驗證與獨立 128 回合確認通過，才會開放 V6 正式控制及成果錄影。', '']
    validation = summary.get('latest_validation')
    if validation:
        lines += [f"最近驗證：{validation['saved_updates']} 次保存點，{validation['episodes']} 回合，"
                  + ('通過本階段。' if validation['passed'] else '尚未通過本階段。'), '']
        if validation.get('survival_rate') is not None:
            lines += [f"第一回合存活率：{validation['survival_rate'] * 100:.1f}%；"
                      f"下半身門檻：{'通過' if validation.get('lower_body_passed') else '尚未通過'}。", '']
        if validation['failed_gates']:
            lines += ['未達標項目：', '', *[f'- `{gate}`' for gate in validation['failed_gates']], '']
    if summary['alerts']:
        lines += ['監看旗標：', '', *[f'- `{alert}`' for alert in summary['alerts']], '']
    lines += ['[機器可讀進度](results/upper_body_v6_progress.json) · [訓練架構與指令](UPPER_BODY_V6_TRAINING.md)', '',
              '此監看程式只讀取進度與來源，不修改 reward、門檻、模型或訓練程序。初始化失敗由另一個有界的啟動防護處理；手動暫停不會被本監看恢復。', '']
    return '\n'.join(lines)


def write_public(root, summary):
    atomic_json(root / PUBLIC_FILES[0], summary)
    path = root / PUBLIC_FILES[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(render_markdown(summary))
    temporary.replace(path)


@contextmanager
def publication_lock(root):
    path = root / 'outputs/upper_body_publication.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, text=True,
                          capture_output=True, timeout=120).stdout.strip()


def remote_head(root):
    output = git(root, 'ls-remote', 'origin', 'refs/heads/main').split()
    if len(output) != 2 or not re.fullmatch(r'[0-9a-f]{40,64}', output[0]):
        raise RuntimeError('SSH main head could not be verified')
    return output[0]


def publish_progress(root, summary, ledger):
    key = event_key(summary, ledger.get('published_summary'))
    if key == ledger.get('published_event') and not ledger.get('pending'):
        return ledger, 'publication_event_already_delivered'
    if not key and not ledger.get('pending'):
        return ledger, 'no_publication_milestone'
    with publication_lock(root):
        if git(root, 'remote', 'get-url', 'origin') != SSH_ORIGIN or git(root, 'branch', '--show-current') != 'main':
            raise RuntimeError('Progress publication requires the authorized SSH origin and main branch')
        remote = remote_head(root)
        pending = ledger.get('pending')
        if pending:
            head = git(root, 'rev-parse', 'HEAD')
            if remote not in (pending['commit'], head):
                if head != pending['commit']:
                    raise RuntimeError('Pending progress commit needs review before another publication')
                git(root, 'push', 'origin', f"{pending['commit']}:refs/heads/main")
                remote = remote_head(root)
            if remote == pending['commit'] or (remote == head and git(root, 'merge-base', pending['commit'], head) == pending['commit']):
                ledger.update(published_summary=pending['summary'], published_event=pending['event'],
                              published_commit=pending['commit'])
                ledger.pop('pending')
                atomic_json(root / LEDGER_FILE, ledger)
            else:
                raise RuntimeError('Pending SSH progress delivery could not be verified')
            key = event_key(summary, ledger.get('published_summary'))
            if not key:
                return ledger, 'pending_publication_delivered'
        head = git(root, 'rev-parse', 'HEAD')
        if head != remote:
            raise RuntimeError('Unpublished local commits exist; progress monitor will not push unrelated work')
        changed = git(root, 'status', '--porcelain', '--untracked-files=all', '--', *PUBLIC_FILES)
        if not changed:
            ledger.update(published_summary=summary, published_event=key, published_commit=head)
            atomic_json(root / LEDGER_FILE, ledger)
            return ledger, 'snapshot_already_published'
        git(root, 'add', '--', *PUBLIC_FILES)
        git(root, 'commit', '--only', '-m', f"Update V6 monitored progress at {summary['saved_updates']} updates", '--', *PUBLIC_FILES)
        commit = git(root, 'rev-parse', 'HEAD')
        ledger['pending'] = {'commit': commit, 'event': key, 'summary': summary}
        atomic_json(root / LEDGER_FILE, ledger)
        git(root, 'push', 'origin', f'{commit}:refs/heads/main')
        if remote_head(root) != commit:
            raise RuntimeError('SSH progress push did not verify at remote main')
        ledger.update(published_summary=summary, published_event=key, published_commit=commit)
        ledger.pop('pending')
        atomic_json(root / LEDGER_FILE, ledger)
        return ledger, 'published'


def tick(root, write=False, publish=False, now=None):
    now = now or datetime.now(timezone.utc)
    state = json.loads((root / 'outputs/upper_body_training_status.json').read_text())
    audit, report = observations(root, state)
    summary = public_summary(state, audit, report, now)
    publication = 'read_only'
    if write or publish:
        write_public(root, summary)
        ledger_path = root / LEDGER_FILE
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        try:
            if publish:
                ledger, publication = publish_progress(root, summary, ledger)
            else:
                publication = 'local_snapshot_written'
            status = {'observed_at': now.isoformat(), 'publication': publication,
                      'progress': summary, 'last_published_commit': ledger.get('published_commit')}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            # Detailed execution errors remain local. Public output only says
            # publication needs review, never prints command/path/token text.
            status = {'observed_at': now.isoformat(), 'publication': 'publication_error',
                      'progress': summary, 'private_error': str(error)}
        atomic_json(root / STATUS_FILE, status)
        publication = status['publication']
    return {'saved_updates': summary['saved_updates'], 'max_new_updates': 20000,
            'stage': summary['curriculum_stage'], 'state': summary['state'],
            'trained_and_validated': summary['trained_and_validated'],
            'alerts': summary['alerts'], 'publication': publication}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--write', action='store_true', help='Write local status and the two public snapshot files')
    parser.add_argument('--publish', action='store_true', help='Authorize bounded progress commits/pushes to the configured SSH origin')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if not 15 <= args.interval <= 60:
        parser.error('Monitor interval must be between 15 and 60 seconds')
    while True:
        try:
            result = tick(ROOT, write=args.write or args.watch, publish=args.publish)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            result = {'state': 'audit_unavailable', 'error_type': type(error).__name__,
                      'publication': 'none', 'training_changed': False}
            if args.write or args.watch or args.publish:
                atomic_json(ROOT / STATUS_FILE, {**result, 'observed_at': datetime.now(timezone.utc).isoformat(),
                                               'private_error': str(error)})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
