import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('watch_upper_startup', Path(__file__).resolve().parents[1] / 'tools/watch_upper_body_startup.py')
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)


def scenario():
    state = {'state': 'runtime_error', 'error': 'RuntimeError: training worker failed with exit code 1',
             'robot': 'q1_leg_upper_body_v6', 'completed_iterations': 7800,
             'current_command': ['bash', 'train_upper_body_native.sh', '--completed', '7800']}
    header = {'state': 'training', 'completed_iterations': 7800, 'command': state['current_command']}
    return state, 'V6_SUPERVISOR ' + json.dumps(header) + '\nFatal: segmentation fault during app startup', {'completed_iterations': 7800}


def test_pre_actor_native_crash_is_retryable_without_repeating_updates():
    state, log, result = scenario()
    assert WATCH.startup_only(state, log, result)


def test_user_pause_and_post_actor_crash_cannot_restart():
    state, log, result = scenario()
    for evidence in ('Actor Model:', 'Critic Model:', 'Learning iteration 7800/7900', 'V6_TRAINING_RESULT'):
        assert not WATCH.startup_only(state, log + '\n' + evidence, result)
    state['state'] = 'interrupted'
    assert not WATCH.startup_only(state, log, result)


def test_new_snapshot_or_wrong_segment_prevents_repeating_ppo():
    state, log, result = scenario()
    assert not WATCH.startup_only(state, log, result, newer_snapshots=True)
    assert not WATCH.startup_only(state, log, {'completed_iterations': 7700})
    assert not WATCH.startup_only(state, 'old worker only', result)
    state['current_command'] = ['another-worker']
    assert not WATCH.startup_only(state, log, result)
