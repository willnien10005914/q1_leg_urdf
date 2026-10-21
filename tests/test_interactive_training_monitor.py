"""Budget, plateau and publication rules independent of Isaac/torch imports."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("interactive_training_monitor", ROOT/"tools/run_interactive_training.py")
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def profile():
    return monitor.validate_profile(json.loads((ROOT/"config/interactive_v4.json").read_text()))


def evaluations(scores, revision=0, passed=False):
    return [{"kind":"validation", "reward_revision":revision,
             "completed_iterations":(index+1)*500, "score":score,
             "passed":passed, "failed_gates":[] if passed else ["sequence.right.foot_lift"]}
            for index,score in enumerate(scores)]


def complete_report(episodes=128):
    return {"passed":True, "episodes":episodes, "backend":"isaaclab", "proxy":False,
            "score":1., "checks":{"walk.direction":True,"sequence.left.lift":True},
            "failed_gates":[], "cases":{name:{"episodes":episodes//8} for name in monitor.CASE_NAMES}}


def test_requested_budget_is_exact_and_last_chunk_cannot_overshoot():
    plan = profile()
    plan["initial_chunk_iterations"] = 100
    assert monitor.next_chunk_size(0,plan) == 100
    assert monitor.next_chunk_size(10,plan) == 500
    assert monitor.next_chunk_size(14990,plan) == 10
    assert monitor.next_chunk_size(15000,plan) == 0
    assert plan["num_envs"]*plan["num_steps_per_env"]*plan["max_iterations"] == 3_686_400_000
    for key,wrong in (("num_envs",512),("max_iterations",15010),("confirmation_episodes",32)):
        changed = dict(plan);changed[key]=wrong
        with pytest.raises(ValueError):
            monitor.validate_profile(changed)


def test_plateau_requires_four_finite_current_revision_validations_and_failure():
    plan = profile()
    plateau = monitor.plateau_diagnostic(evaluations([.80,.802,.801,.803]),plan,0)
    assert plateau and not plateau["successful"]
    assert not plateau["acceptance_gates_changed"]
    assert plateau["failed_gate_counts"] == {"sequence.right.foot_lift":4}
    assert monitor.plateau_diagnostic(evaluations([.80,.802,.801]),plan,0) is None
    assert monitor.plateau_diagnostic(evaluations([.75,.80,.82,.85]),plan,0) is None
    assert monitor.plateau_diagnostic(evaluations([1.,1.,1.,1.],passed=True),plan,0) is None
    assert monitor.plateau_diagnostic(evaluations([.80,.802,.801,.803]),plan,1) is None
    with pytest.raises(ValueError):
        monitor.plateau_diagnostic(evaluations([.8,.8,.8,float("nan")]),plan,0)


def test_three_passes_do_not_replace_independent_full_confirmation():
    count=0
    for _ in range(3):
        count=monitor.validation_streak(count,{"passed":True})
    assert count==3
    assert monitor.validation_streak(count,{"passed":False})==0
    assert not monitor.confirmation_passed(complete_report(32))
    assert monitor.confirmation_passed(complete_report(128))
    for field,value in (("proxy",True),("passed",False),("score",float("nan"))):
        report=complete_report();report[field]=value
        assert not monitor.confirmation_passed(report)
    report=complete_report();report["checks"]["sequence.left.lift"]=False
    assert not monitor.confirmation_passed(report)
    report=complete_report();del report["cases"]["sequence"]
    assert not monitor.confirmation_passed(report)


def test_adopt_standalone_worker_keeps_its_ten_completed_iterations(tmp_path,monkeypatch):
    plan=profile()
    run=tmp_path/"run";run.mkdir()
    with zipfile.ZipFile(run/"resume.pt","w") as archive:
        archive.writestr("resume/data.pkl",b"checkpoint-fixture"*200)
    (run/"training_result.json").write_text(json.dumps({
        "completed_iterations":10,"num_envs":10240,"reward_revision":0,
        "contract":{"robot":"q1_leg_interactive_v4"}}))
    monkeypatch.setattr(monitor,"ROOT",tmp_path)
    monkeypatch.setattr(monitor,"source_manifest",lambda path:{"fixture":"stable"})
    supervisor=monitor.Supervisor(plan,tmp_path/"profile.json",run,resume=True)
    assert supervisor.state["completed_iterations"]==10
    assert supervisor.state["adopted_worker_iterations"]==10
    assert supervisor.state["completed_transitions"]==10*10240*24
    assert monitor.next_chunk_size(supervisor.state["completed_iterations"],plan)==500
    assert supervisor.state["evaluations"]==[]


def test_old_result_cannot_mask_a_worker_failure(tmp_path):
    output=tmp_path/"result.json"
    output.write_text('{"completed_iterations":10}')
    before=monitor.timestamp(output)
    with pytest.raises(RuntimeError,match="not freshly written"):
        monitor.fresh_json(output,before,"worker result")
    output.write_text("invalid JSON")
    with pytest.raises(RuntimeError,match="Invalid"):
        monitor.fresh_json(output,None,"worker result")


def test_authorized_deadline_profile_retains_saved_iterations_and_shortens_cap():
    plan=monitor.validate_profile(json.loads((ROOT/'config/interactive_v4_deadline.json').read_text()))
    assert plan['num_envs']==10240 and plan['max_iterations']==801
    assert monitor.next_chunk_size(501,plan)==100
    assert monitor.next_chunk_size(799,plan)==2
    assert monitor.next_chunk_size(801,plan)==0
    changed=dict(plan);changed['training_deadline_utc']='2026-10-21T00:52:00'
    with pytest.raises(ValueError,match='timezone'):
        monitor.validate_profile(changed)
