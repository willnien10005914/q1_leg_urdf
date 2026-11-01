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

def test_product_specs_do_not_invent_electrical_or_safe_operating_limits():
 catalog=json.loads((store.ROOT/'web/store/catalog.json').read_text())
 assert {item['family'] for item in catalog}=={'sidestep','stability','inplace'}
 for item in catalog:
  specs={spec['id']:spec for spec in item['specs']}
  assert specs['battery']['value']=='待量測' and specs['battery']['status']=='unmeasured'
  assert specs['energy']['value']=='待量測' and specs['energy']['status']=='unmeasured'
  assert specs['workspace']['value']=='待驗證' and specs['workspace']['status']=='unverified'
  assert specs['speed']['status']=='command'
  assert item['accepted'] is False
  assert item['product_name'] and item['tagline'] and len(item['scenes'])==3
  for spec in specs.values():
   if spec['status'] in ('measured','source','command'):
    assert spec['source'] and (store.ROOT/spec['source'].lstrip('/')).is_file()

def test_robot_specs_are_derived_from_local_model_not_hardcoded(root):
 model=root/'urdf/q1_inertia/quanta_robot_legged.urdf';model.parent.mkdir(parents=True)
 model.write_text('<robot name="fixture"><link name="body"><inertial><mass value="3.25"/></inertial></link>'
  '<link name="leg"><inertial><mass value="1.5"/></inertial></link>'
  '<joint name="hip" type="revolute"/><joint name="finger" type="revolute"/></robot>')
 config=root/'config/robot_contract.json';config.parent.mkdir()
 config.write_text(json.dumps({'joint_names':['hip'],'locked_joints':['finger']}))
 result=store.robot_specs(root)
 assert result['available'] and result['model_only']
 assert result['mass_kg']==4.75 and result['active_joints']==1 and result['source_movable_joints']==2
 assert result['battery_capacity_wh'] is None and result['maximum_measured_speed_m_s'] is None
 assert len(result['urdf_sha256'])==64

def test_missing_robot_model_is_unavailable_not_fabricated(root):
 result=store.store_payload(root,now=200)['robot']
 assert result['available'] is False and result['model_only'] is True
