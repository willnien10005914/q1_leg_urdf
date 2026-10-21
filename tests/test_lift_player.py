"""An early lift checkpoint cannot unlock the full keyboard controller."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lift_player", ROOT / "scripts/reinforcement_learning/rsl_rl/play_interactive_web.py")
PLAYER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAYER)
GATE = PLAYER._V5_GATE


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def accepted_artifacts(root):
    """A synthetic passing report exercises the actual production checker."""
    control = json.loads((ROOT / "config/lift_control_v5.json").read_text())
    for relative in (*GATE.RUNTIME_FILES, control["gait_file"],
                     "config/robot_contract.json", "config/lift_v5.json"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    base = json.loads((root / "config/robot_contract.json").read_text())
    gait = json.loads((root / control["gait_file"]).read_text())
    profile = json.loads((root / "config/lift_v5.json").read_text())
    folder = root / "checkpoints/isaac/lift_v5"
    folder.mkdir(parents=True)
    (folder / "latest.pt").write_bytes(b"test weights; never used for inference")
    contract = {
        "robot": GATE.ROBOT_VERSION, "schema": 5,
        "urdf_sha256": GATE.sha256(root / "urdf/q1_inertia/quanta_robot_legged.urdf"),
        "gait_sha256": GATE.sha256(root / control["gait_file"]),
        "gait_file": control["gait_file"],
        "control_sha256": GATE.sha256(root / "config/lift_control_v5.json"),
        "control_profile": GATE._constant(root / GATE.SOURCE_DIR / "lift_objective.py", "CONTROL_PROFILE"),
        "motor": control, "joint_names": base["joint_names"],
        "action_size": 23, "observation_size": 110,
        "default": gait["default"], "root_height": gait["root_height"], "proxy": False,
    }
    checker, gates = GATE.acceptance_checker(root)
    cases = {}
    for name in GATE.CASE_NAMES:
        cases[name] = {
            "episodes": 16, "survival_rate": 1., "upright": 1.,
            "final_stand_speed_m_s": .01, "stance_slip_m_s": .01,
            "directional_displacement_m": .5, "velocity_rmse_m_s": .02,
            "redirection": {"directional_displacement_m": .05, "velocity_rmse_m_s": .02},
            "signed_yaw_displacement_rad": .8, "yaw_rmse_rad_s": .02,
            "foot_peak_clearance_m_per_foot": [.03, .03],
            "foot_lift_success_rate_per_foot": [.8, .8],
        }
    cases["sequence"].update(
        resumed_displacement_x_m=.1, resume_velocity_rmse_m_s=.02,
        finite_sequence_completed=True,
        directions={side: {
            "lead_forward_m": .04, "lead_signed_lateral_m": .05,
            "lead_peak_clearance_m": .03, "lead_lift_success_rate": .8,
            "trailing_peak_clearance_m": .03, "trailing_lift_success_rate": .8,
            "hand_cross_success_rate": .8, "root_forward_m": .03,
            "root_signed_lateral_m": .03, "push_support_forward_proxy_rate": .5,
            "lead_landing_rate": 1.,
        } for side in ("left", "right")})
    manifest = GATE.runtime_sources(root)
    manifest["config/lift_v5.json"] = GATE.sha256(root / "config/lift_v5.json")
    report = {
        "ready": True, "passed": True, "stage": 2, "backend": "isaaclab", "proxy": False,
        "reset_between_transitions": False, "duration_s": 20., "episodes": 128,
        "confirmation_episodes": 128, "completed_iterations": 2000,
        "training_profile": profile, "evaluation_seed": profile["confirmation_seed"] + 1000,
        "published_from": "runs/example/resume.pt", "published_at": "2026-10-21T00:00:00Z",
        "source_sha256": manifest, "cases": cases, "gates": gates, "score": 1., "failed_gates": [],
        "environment_sha256": manifest[GATE.SOURCE_DIR + "lift_env.py"],
        "base_environment_sha256": manifest[GATE.SOURCE_DIR + "interactive_env.py"],
        "evaluator_sha256": manifest[GATE.SOURCE_DIR + "interactive_evaluation.py"],
        "urdf_sha256": contract["urdf_sha256"], "gait_sha256": contract["gait_sha256"],
        "checkpoint_sha256": GATE.sha256(folder / "latest.pt"),
    }
    report["checks"] = checker(report)
    assert all(report["checks"].values())
    write_json(folder / "contract.json", contract)
    write_json(folder / "evaluation.json", report)
    return folder, contract, report


def test_confirmed_v5_uses_separate_environment_and_profile(tmp_path, monkeypatch):
    folder, contract, report = accepted_artifacts(tmp_path)
    gate = GATE.v5_policy_gate(tmp_path)
    assert gate["ready"], gate
    version, selected, _ = PLAYER.selected_policy(tmp_path)
    assert (version, selected) == ("v5", folder)

    class LiftCfg:
        pass

    class LiftEnv:
        pass

    class LegacyCfg:
        pass

    class LegacyEnv:
        pass

    monkeypatch.setitem(sys.modules, "leg_humanoid_lab.lift_env",
                        SimpleNamespace(LiftLegEnv=LiftEnv, LiftEnvCfg=LiftCfg))
    monkeypatch.setitem(sys.modules, "leg_humanoid_lab.interactive_env",
                        SimpleNamespace(InteractiveLegEnv=LegacyEnv, InteractiveEnvCfg=LegacyCfg))
    environment, cfg = PLAYER.policy_environment(tmp_path, version)
    assert environment is LiftEnv and isinstance(cfg, LiftCfg)
    assert cfg.curriculum_stage == 2 and cfg.gait_file == contract["gait_file"]
    environment, cfg = PLAYER.policy_environment(tmp_path, "v4")
    assert environment is LegacyEnv and isinstance(cfg, LegacyCfg)
    assert not hasattr(cfg, "gait_file")
    with pytest.raises(ValueError):
        PLAYER.policy_environment(tmp_path, "v3")


@pytest.mark.parametrize("change", [
    {"stage": 1}, {"episodes": 32}, {"confirmation_episodes": 32},
    {"ready": False}, {"reset_between_transitions": True},
    {"evaluation_seed": 70000}, {"duration_s": 8.}, {"proxy": True},
    {"source_sha256": {}}, {"training_profile": {}}, {"published_at": None},
    {"checkpoint_sha256": "different weights"},
])
def test_preparatory_or_mismatched_report_never_unlocks_controls(tmp_path, change):
    folder, _, report = accepted_artifacts(tmp_path)
    write_json(folder / "evaluation.json", {**report, **change})
    assert GATE.v5_policy_gate(tmp_path)["ready"] is False
    # An explicitly requested v5 candidate must never substitute another version.
    version, selected, gate = PLAYER.selected_policy(tmp_path, folder, "v5")
    assert (version, selected, gate["ready"]) == ("v5", folder, False)


def test_metrics_are_recomputed_even_if_saved_checks_claim_success(tmp_path):
    folder, _, report = accepted_artifacts(tmp_path)
    report["cases"]["sequence"]["directions"]["left"]["trailing_peak_clearance_m"] = 0.
    write_json(folder / "evaluation.json", report)
    assert not GATE.v5_policy_gate(tmp_path)["ready"]


@pytest.mark.parametrize("relative", [
    "checkpoints/isaac/lift_v5/latest.pt", "config/lift_control_v5.json",
    GATE.SOURCE_DIR + "lift_env.py", GATE.SOURCE_DIR + "lift_objective.py",
    GATE.SOURCE_DIR + "interactive_evaluation.py",
])
def test_changed_weights_or_runtime_source_invalidate_confirmation(tmp_path, relative):
    accepted_artifacts(tmp_path)
    with (tmp_path / relative).open("a") as file:
        file.write("\n# modified after confirmation\n")
    assert not GATE.v5_policy_gate(tmp_path)["ready"]


def test_unaccepted_v5_keeps_existing_v4_fallback(tmp_path, monkeypatch):
    folder, _, report = accepted_artifacts(tmp_path)
    report["stage"] = 1
    write_json(folder / "evaluation.json", report)
    accepted_v4 = {"ready": True, "error": None, "checkpoint": "original v4 weights"}
    monkeypatch.setattr(PLAYER, "policy_gate", lambda *_args: accepted_v4)
    version, selected, gate = PLAYER.selected_policy(tmp_path)
    assert version == "v4" and selected == tmp_path / "checkpoints/isaac/interactive_v4"
    assert gate is accepted_v4
