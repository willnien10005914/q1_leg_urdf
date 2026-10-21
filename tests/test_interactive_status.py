import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('interactive_status',ROOT/'tools/publish_interactive_status.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_budget_exhaustion_is_not_success_or_success_media():
    block=module.status_block({'state':'budget_exhausted','completed_iterations':15000,'last_evaluation':{'score':.5}})
    assert '尚未通過' in block and '15,000／15,000' in block
    assert '未發布新版成功 GIF' in block and '![' not in block
    with pytest.raises(ValueError):module.status_block({'ready':True})


def test_deadline_status_uses_actual_budget_and_saved_progress():
    block=module.status_block({'state':'deadline_reached','num_envs':10240,
        'max_iterations':801,'completed_iterations':701,'adopted_worker_iterations':501,
        'started_at':'2026-10-21T00:38:00+00:00','updated_at':'2026-10-21T00:52:00+00:00',
        'last_evaluation':{'score':.75,'episodes':32}})
    assert '701／801 iterations' in block and '501 iterations' in block
    assert '09:00' in block and '不代表動作已學會' in block
    assert '2026-10-21T00:38:00+00:00' in block
    assert '2026-10-21T00:52:00+00:00' in block
    assert '15,000' not in block and '未發布新版成功 GIF' in block and '![' not in block


def test_status_can_read_budget_from_nested_profile():
    block=module.status_block({'state':'budget_exhausted','completed_iterations':400,
        'profile':{'num_envs':2048,'max_iterations':400},'last_evaluation':{'episodes':128,'score':.9}})
    assert '2,048 envs' in block and '400／400 iterations' in block
    assert '最新 128 回合' in block
