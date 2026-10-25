"""Store projection preserves truthful saved progress and excludes local paths."""
import importlib.util,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'web'))
spec=importlib.util.spec_from_file_location('motion_store',Path(__file__).parents[1]/'web/store_server.py')
store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)

@pytest.fixture
def root(tmp_path):
 p=tmp_path/'web/store';p.mkdir(parents=True)
 (p/'catalog.json').write_text('[]')
 (tmp_path/'outputs').mkdir()
 return tmp_path

def status(root,record):
 (root/'outputs/spark_balance_status.json').write_text(json.dumps(record))

def test_not_started_is_not_fabricated_training(root):
 result=store.store_payload(root,now=200)
 assert result['training']['state']=='not_started'
 assert result['training']['accepted'] is False
 assert not any(result['features'].values())

def test_saved_and_live_updates_are_distinct_and_paths_are_not_exposed(root):
 status(root,{'state':'training','saved_iterations':1250,'completed_iterations':1250,
     'worker_log_iteration':1420,'started_at':100,'updated_at':198,'max_new_iterations':15000,
     'worker_output':'/private/path','child_pid':123,'checkpoint':'/private/model.pt',
     'evaluations':[{'completed_iterations':1000,'four_step_survival':.78,'report':'/private/report'}]})
 result=store.store_payload(root,now=200)['training']
 assert result['saved_iterations']==1250 and result['live_iteration']==1420
 assert not result['stale'] and result['estimated_remaining_s']>0
 assert not {'worker_output','child_pid','checkpoint'}&result.keys()
 assert 'report' not in result['evaluations'][0]

def test_evaluating_never_reuses_stale_live_iteration(root):
 status(root,{'state':'evaluating','saved_iterations':1500,'worker_log_iteration':1499,'updated_at':50})
 result=store.store_payload(root,now=200)['training']
 assert result['live_iteration']==1500 and result['stale']

def test_partial_status_json_does_not_crash_store(root):
 (root/'outputs/spark_balance_status.json').write_text('{')
 result=store.store_payload(root,now=200)['training']
 assert result['state']=='updating' and result['stale']
