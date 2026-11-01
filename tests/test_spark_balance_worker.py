"""CPU-only guards for Spark budgets, provenance, and honest gesture survival."""
import importlib.util
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "scripts/reinforcement_learning/rsl_rl/train_spark_balance.py"
spec = importlib.util.spec_from_file_location("spark_balance_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def args(tmp_path, *extra):
    return worker.build_parser().parse_args([
        "--checkpoint", str(tmp_path / "checkpoint.pt"), "--output", str(tmp_path / "result"), *extra
    ])


def contracts():
    expected = {name: "same" for name in worker.NATIVE_KEYS}
    saved = {**expected, "robot": "q1_leg_upper_body_v6"}
    return saved, expected


def test_budget_includes_completed_updates(tmp_path):
    worker.validate_args(args(tmp_path, "--completed", "14999", "--iterations", "1"))
    with pytest.raises(ValueError, match="fit max-new-iterations"):
        worker.validate_args(args(tmp_path, "--completed", "14999", "--iterations", "2"))


@pytest.mark.parametrize("flag,value", [("--envs", "0"), ("--completed", "-1"),
    ("--iterations", "0"), ("--max-new-iterations", "0")])
def test_nonpositive_training_arguments_rejected(tmp_path, flag, value):
    with pytest.raises(ValueError):
        worker.validate_args(args(tmp_path, flag, value))


def test_initialize_never_inherits_spent_v6_counter(tmp_path):
    with pytest.raises(ValueError, match="starts at zero"):
        worker.validate_args(args(tmp_path, "--initialize", "--completed", "20000"))


@pytest.mark.parametrize("count", [1, 3, 6])
def test_evaluation_requires_four_complete_case_groups(tmp_path, count):
    with pytest.raises(ValueError, match="divisible by four"):
        worker.validate_args(args(tmp_path, "--evaluate", "--envs", str(count)))


def test_previous_result_is_not_overwritten(tmp_path):
    options = args(tmp_path)
    options.output.mkdir()
    (options.output / "training_result.json").write_text("{}")
    with pytest.raises(ValueError, match="already contains"):
        worker.validate_args(options)


def test_bootstrap_checks_every_native_hash_and_interface_field():
    saved, expected = contracts()
    worker.validate_contract(saved, expected, initialize=True)
    for key in worker.NATIVE_KEYS:
        with pytest.raises(ValueError, match=key):
            worker.validate_contract({**saved, key: "changed"}, expected, initialize=True)


def test_resume_requires_spark_sources_runtime_and_budget():
    saved, expected = contracts()
    sources = {"worker": "hash"}
    runtime = {"torch": "2.9"}
    saved.update(robot=worker.ROBOT, balance_source_sha256=sources,
        runtime_versions=runtime, max_new_iterations=15000, backend="physx")
    worker.validate_contract(saved, expected, sources=sources, runtime=runtime)
    for changed in [{"balance_source_sha256": {}}, {"runtime_versions": {}},
                    {"max_new_iterations": 20000}, {"backend": "newton"}]:
        with pytest.raises(ValueError):
            worker.validate_contract({**saved, **changed}, expected, sources=sources, runtime=runtime)


def test_timeouts_remain_failed_and_one_lift_cannot_fake_four_gestures():
    peaks = [[[.02, .02] for _ in range(4)] for _ in range(4)]
    seen = [[True] * 4 for _ in range(4)]
    report = worker.audit_summary([False, True, True, True], [0, 1, 2, 3], peaks, seen)
    assert report["four_step_survival"] == .5
    assert report["completed_four_step_rate"] == .5
    peaks[1][3][1] = 0.  # Last gesture's trailing foot never unloaded.
    report = worker.audit_summary([False, True, True, True], [0, 1, 2, 3], peaks, seen)
    assert report["completed_four_step_rate"] == 0.


def test_all_four_gestures_must_be_commanded():
    peaks = [[[.02, .02] for _ in range(4)] for _ in range(4)]
    seen = [[True, True, True, False] for _ in range(4)]
    report = worker.audit_summary([True] * 4, [0, 1, 2, 3], peaks, seen)
    assert report["four_step_survival"] == 1.
    assert report["completed_four_step_rate"] == 0.
