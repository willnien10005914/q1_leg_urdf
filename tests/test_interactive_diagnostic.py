"""Diagnostic media retains failures and can never masquerade as accepted media."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/reinforcement_learning/rsl_rl'))
from record_interactive_diagnostic import diagnostic_plan, validate_metadata


def report():
    return {'diagnostic': True, 'gate_passed': False, 'accepted': False, 'proxy': False,
            'source': 'native Isaac Sim PhysX PPO', 'overlay': 'EXPERIMENTAL / NOT ACCEPTED',
            'reset_count_after_start': 0, 'seed': 42, 'checkpoint_sha256': 'a' * 64,
            'frames': 10, 'actual_duration_s': .2, 'failure': True,
            'failure_reason': 'Native episode terminated', 'technical_error': None}


def test_actual_failure_is_a_valid_diagnostic_but_acceptance_or_reset_is_not():
    value = report()
    assert validate_metadata(value)
    for key, bad in [('gate_passed', True), ('accepted', True), ('proxy', True),
                     ('reset_count_after_start', 1), ('overlay', ''),
                     ('failure_reason', None), ('frames', 0), ('actual_duration_s', .3),
                     ('technical_error', 'camera unavailable')]:
        assert not validate_metadata({**value, key: bad}), key
    assert validate_metadata({**value, 'failure': False, 'failure_reason': None})


def test_plan_is_finite_and_contains_only_native_control_requests():
    plan = diagnostic_plan()
    assert abs(plan[-1]['end'] - 9.7) < 1e-7
    assert all(a['end'] == b['start'] for a, b in zip(plan, plan[1:]))
    assert [item['name'] for item in plan] == ['stand', 'W_forward', 'stop', 'sidestep_sequence', 'final_stand']


def test_sequence_diagnostic_does_not_hide_a_failed_walk_in_the_same_episode():
    plan=diagnostic_plan(sequence_only=True)
    assert [item['name'] for item in plan]==['stand','sidestep_sequence','final_stand']
    assert abs(plan[-1]['end']-7.2)<1e-7
    assert all(a['end']==b['start'] for a,b in zip(plan,plan[1:]))
