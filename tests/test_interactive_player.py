"""Pure file gates: policy identity stays bound to the evaluated source files."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("interactive_player", ROOT / "scripts/reinforcement_learning/rsl_rl/play_interactive_web.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def artifacts(root):
    files = {
        "config/robot_contract.json": json.dumps({"joint_names": [str(i) for i in range(23)]}),
        "config/v4_gaits.json": "gaits", "urdf/q1_inertia/quanta_robot_legged.urdf": "robot",
        "source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py": "environment",
        "source/leg_humanoid_lab/leg_humanoid_lab/interactive_evaluation.py": "evaluator",
        "source/leg_humanoid_lab/leg_humanoid_lab/interactive_control.py": "controller",
        "checkpoints/isaac/interactive_v4/latest.pt": "weights",
    }
    for relative, text in files.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
    folder = root / "checkpoints/isaac/interactive_v4"
    contract = {"robot": "q1_leg_interactive_v4", "joint_names": [str(i) for i in range(23)],
                "observation_size": 110, "urdf_sha256": MODULE.sha256(root / "urdf/q1_inertia/quanta_robot_legged.urdf"),
                "gait_sha256": MODULE.sha256(root / "config/v4_gaits.json")}
    report = {"passed": True, "episodes": 128, "checkpoint_sha256": MODULE.sha256(folder / "latest.pt"),
              "environment_sha256": MODULE.sha256(root / "source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py"),
              "evaluator_sha256": MODULE.sha256(root / "source/leg_humanoid_lab/leg_humanoid_lab/interactive_evaluation.py")}
    (folder / "contract.json").write_text(json.dumps(contract))
    (folder / "evaluation.json").write_text(json.dumps(report))
    return folder, report


def test_changed_environment_and_evaluator_invalidate_passed_weights(tmp_path):
    folder, report = artifacts(tmp_path)
    assert MODULE.policy_gate(tmp_path)["ready"]
    for key in ("environment_sha256", "evaluator_sha256"):
        (folder / "evaluation.json").write_text(json.dumps({**report, key: "older-source"}))
        assert not MODULE.policy_gate(tmp_path)["ready"]


def test_manifest_is_optional_but_changed_controller_or_external_path_fails(tmp_path):
    folder, report = artifacts(tmp_path)
    relative = "source/leg_humanoid_lab/leg_humanoid_lab/interactive_control.py"
    report["source_sha256"] = {relative: MODULE.sha256(tmp_path / relative)}
    (folder / "evaluation.json").write_text(json.dumps(report))
    assert MODULE.policy_gate(tmp_path)["ready"]
    (tmp_path / relative).write_text("changed-controller")
    assert not MODULE.policy_gate(tmp_path)["ready"]
    report["source_sha256"] = {"../outside": "hash"}
    (folder / "evaluation.json").write_text(json.dumps(report))
    assert not MODULE.policy_gate(tmp_path)["ready"]


def test_validate_only_has_nonzero_exit_for_unready_policy(tmp_path):
    import os
    environment = os.environ.copy()
    environment["Q1_LEG_ROOT"] = str(tmp_path)
    process = subprocess.run([sys.executable, str(SPEC.origin), "--validate-only"],
                             env=environment, text=True, capture_output=True)
    assert process.returncode == 1 and json.loads(process.stdout)["ready"] is False
    artifacts(tmp_path)
    process = subprocess.run([sys.executable, str(SPEC.origin), "--validate-only"],
                             env=environment, text=True, capture_output=True)
    assert process.returncode == 0 and json.loads(process.stdout)["ready"] is True
