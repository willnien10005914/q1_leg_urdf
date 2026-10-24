import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('v6_progress', Path(__file__).resolve().parents[1] / 'tools/monitor_upper_body_v6.py')
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def summary():
    state = {'state': 'training', 'completed_iterations': 12800, 'curriculum_stage': 0,
             'run': '/private/home/SECRET/run_123', 'updated_at': '2026-10-24T01:00:00+00:00',
             'current_command': ['SECRET'], 'error': 'SECRET', 'ready': False}
    audit = {'source_verified': True, 'checkpoint_verified': True, 'training_service_active': True}
    return MONITOR.public_summary(state, audit)


def test_public_progress_does_not_expose_paths_commands_or_accept_pending_weights():
    value = summary();text = json.dumps(value)
    assert 'SECRET' not in text and '/private/' not in text and 'current_command' not in text
    assert value['saved_updates'] == 12800 and value['max_new_updates'] == 20000
    assert not value['trained_and_validated'] and value['run_id'] == 'run_123'


def test_only_milestones_stages_and_terminal_events_trigger_publication():
    previous = summary()
    assert MONITOR.event_key({**previous, 'saved_updates': 12900}, previous) is None
    assert 'milestone' in MONITOR.event_key({**previous, 'saved_updates': 13000}, previous)
    assert 'stage' in MONITOR.event_key({**previous, 'curriculum_stage': 1}, previous)
    assert 'terminal' in MONITOR.event_key({**previous, 'state': 'runtime_error'}, previous)


def test_same_delivered_terminal_event_does_not_touch_git_or_network(tmp_path, monkeypatch):
    value = {**summary(), 'state': 'runtime_error'}
    event = MONITOR.event_key(value, None)
    monkeypatch.setattr(MONITOR, 'git', lambda *args: (_ for _ in ()).throw(AssertionError('Unexpected Git boundary')))
    ledger = {'published_event': event}
    result, label = MONITOR.publish_progress(tmp_path, value, ledger)
    assert result is ledger and label == 'publication_event_already_delivered'
