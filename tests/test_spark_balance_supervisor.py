"""Budget and snapshot safety for long-running Spark jobs."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('spark_supervisor',Path(__file__).parents[1]/'tools/run_spark_balance.py')
supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)

def test_chunks_exhaust_exact_authorized_budget_once():
    completed=0;chunks=[]
    while completed<15000:
        count=supervisor.next_chunk(completed,15000)
        assert 0<count<=250
        completed+=count;chunks.append(completed)
    assert chunks[:3]==[10,250,500]
    assert completed==15000 and len(chunks)==61
    with pytest.raises(ValueError):supervisor.next_chunk(15000,15000)

def test_resume_near_budget_never_overruns():
    assert supervisor.next_chunk(14999,15000)==1
    assert supervisor.next_chunk(14760,15000)==240

def test_frozen_source_tampering_is_detected(tmp_path):
    source=tmp_path/'source_snapshot/config/task.json';source.parent.mkdir(parents=True)
    source.write_text('{"reward":1}')
    supervisor.write(tmp_path/'source_manifest.json',{'config/task.json':supervisor.sha(source)})
    supervisor.verify_snapshot(tmp_path)
    source.write_text('{"reward":2}')
    with pytest.raises(ValueError,match='Frozen source changed'):
        supervisor.verify_snapshot(tmp_path)

def test_config_rejects_unapproved_engine_or_budget():
    config=supervisor.read(supervisor.ROOT/'config/spark_balance_6144_15000.json')
    supervisor.validate_config(config)
    for key,value in [('max_new_iterations',20000),('num_envs',10240),('backend','newton')]:
        with pytest.raises(ValueError):supervisor.validate_config({**config,key:value})

def test_read_only_evaluation_retries_preserve_failed_attempts(tmp_path):
    base=tmp_path/'audit';base.mkdir()
    attempt=supervisor.evaluation_attempt(base)
    assert attempt.name=='audit_attempt_002'
    attempt.mkdir()
    assert supervisor.evaluation_attempt(base).name=='audit_attempt_003'

def test_publish_persists_state_transition_without_argument_collision(tmp_path,monkeypatch):
    monkeypatch.setattr(supervisor,'STATE',tmp_path/'public_status.json')
    record={'completed_iterations':0}
    supervisor.publish(tmp_path/'run',record,state='prepared',accepted=False)
    assert supervisor.read(tmp_path/'run/status.json')['state']=='prepared'
    assert supervisor.read(supervisor.STATE)==record
