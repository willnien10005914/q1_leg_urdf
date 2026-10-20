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
