import hashlib
import json

import pytest
import torch

from test_upper_body_training_monitor import worker_functions


def fixture(tmp_path):
    root_sha = 'b' * 64
    expected = {'robot': 'q1_leg_upper_body_v6', 'urdf_sha256': 'urdf', 'gait_sha256': 'gait',
                'joint_names': ['native'], 'action_size': 23, 'observation_size': 110}
    parent = tmp_path / 'budget.pt'
    torch.save({'iter': 12800, 'infos': {'completed_iterations': 12800, 'budget_mode': 'new_updates',
               'initialization': {'source_sha256': root_sha}}}, parent)
    (tmp_path / 'contract.json').write_text(json.dumps({**expected, 'proxy': False, 'control_sha256': 'old-control'}))
    profile = {'source_checkpoint_sha256': root_sha, 'control_fork': {
        'inherited_v6_updates': 12800, 'budget_checkpoint': 'budget.pt',
        'budget_checkpoint_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
        'budget_control_sha256': 'old-control'}}
    ns = worker_functions();ns['ROOT'] = tmp_path
    ns['initialize_complete_actor'] = lambda *args: {'source_sha256': root_sha, 'budget_mode': 'new_updates',
                                                   'actor_output_zeroed': False, 'critic': 'fresh', 'optimizer': 'fresh'}
    return ns, expected, profile


def test_actor_recovery_carries_all_spent_updates_without_inheriting_acceptance(tmp_path):
    ns, expected, profile = fixture(tmp_path)
    class Runner: current_learning_iteration = 0
    runner = Runner()
    result = ns['initialize_budgeted_fork'](runner, 'actor.pt', expected, profile, 12800)
    assert runner.current_learning_iteration == 12800
    assert result['inherited_v6_updates'] == 12800
    assert result['source_sha256'] == profile['source_checkpoint_sha256']
    assert not result['acceptance_inherited'] and not result['actor_output_zeroed']


def test_budget_reset_or_tampered_parent_is_rejected(tmp_path):
    ns, expected, profile = fixture(tmp_path)
    for counter in (0, 12799, 20000):
        with pytest.raises(ValueError): ns['initialize_budgeted_fork'](object(), 'actor.pt', expected, profile, counter)
    (tmp_path / 'budget.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        ns['initialize_budgeted_fork'](object(), 'actor.pt', expected, profile, 12800)
