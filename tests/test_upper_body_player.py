"""Unaccepted upper-body PPO weights cannot unlock the live keyboard player."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "upper_body_player", ROOT / "scripts/reinforcement_learning/rsl_rl/play_interactive_web.py")
PLAYER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAYER)
GATE = PLAYER._V6_GATE


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def accepted_artifacts(root):
    # Copy the supervisor's real snapshot dependency list, not a parallel list.
    for relative in (*GATE.runtime_sources(ROOT), "config/robot_contract.json"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    expected, _ = GATE.expected_contract(root)
    profile = json.loads((root / GATE.PROFILE_FILE).read_text())
    manifest = GATE.runtime_sources(root)
    folder = root / "checkpoints/isaac/upper_body_v6"
    folder.mkdir(parents=True)
    (folder / "latest.pt").write_bytes(b"Synthetic gate test weights; never inference")
    digest = GATE.sha256(folder / "latest.pt")
    lower_checker, gates, upper_checker, defaults = GATE.acceptance_checkers(root)
    cases = {name: {
        "episodes": 16, "survival_rate": 1., "upright": 1.,
        "final_stand_speed_m_s": .01, "stance_slip_m_s": .01,
        "directional_displacement_m": .5, "velocity_rmse_m_s": .02,
        "redirection": {"directional_displacement_m": .05, "velocity_rmse_m_s": .02},
        "signed_yaw_displacement_rad": .8, "yaw_rmse_rad_s": .02,
        "foot_peak_clearance_m_per_foot": [.04, .04],
        "foot_lift_success_rate_per_foot": [.8, .8],
    } for name in GATE.CASE_NAMES}
    cases["sequence"].update(resumed_displacement_x_m=.1, resume_velocity_rmse_m_s=.02,
        finite_sequence_completed=True, directions={side: {
            "lead_forward_m": .04, "lead_signed_lateral_m": .05,
            "lead_peak_clearance_m": .04, "lead_lift_success_rate": .8,
            "trailing_peak_clearance_m": .04, "trailing_lift_success_rate": .8,
            "hand_cross_success_rate": .8, "root_forward_m": .03,
            "root_signed_lateral_m": .03, "push_support_forward_proxy_rate": .5,
            "lead_landing_rate": 1.,
        } for side in ("left", "right")})
    window = {"observed_episode_rate": 1., "chest_peak_max_deg": 10.,
              "chest_peak_p95_deg": 9., "chest_rms_deg": 4.,
              "chest_angular_xy_rms_rad_s": .2, "arm_reference_rmse_rad": .03,
              "arm_speed_rms_rad_s": .1, "hands_return_distance_rms_m_per_hand": [.005, .006],
              "return_observed_episode_rate": 1., "return_arm_reference_rmse_rad": .03}
    upper_cases = {}
    for name in GATE.CASE_NAMES:
        upper_cases[name] = {"episodes": 16, "finite_measurements": True,
            **{key: copy.deepcopy(window) for key in ("all", "tail", "dynamic", "gesture", "early_return")},
            "directions": {}}
    upper_cases["sequence"]["early_return"]["chest_peak_max_deg"] = 5.
    upper_cases["sequence"]["directions"] = {side: copy.deepcopy(window) for side in ("left", "right")}
    upper = {"schema": 1, "stage": 2, "gates": {**defaults, **expected["upper_body_gates"]},
             "cases": upper_cases, "measurement_errors": [],
             "evaluator_sha256": manifest[GATE.SOURCE_DIR + "upper_body_evaluation.py"],
             "control_sha256": expected["control_sha256"], "control_contract": expected}
    report = {"ready": True, "passed": True, "robot": GATE.ROBOT_VERSION, "schema": 6,
        "stage": 2, "backend": "isaaclab", "proxy": False,
        "reset_between_transitions": False, "duration_s": 20., "episodes": 128,
        "confirmation_episodes": 128, "completed_iterations": 300,
        "training_profile": profile, "evaluation_seed": profile["confirmation_seed"],
        "published_from": "runs/example/resume.pt", "published_at": "2026-10-23T00:00:00Z",
        "source_sha256": manifest, "cases": cases, "gates": gates, "score": 1., "failed_gates": [],
        "environment_sha256": manifest[GATE.SOURCE_DIR + "upper_body_env.py"],
        "base_environment_sha256": manifest[GATE.SOURCE_DIR + "interactive_env.py"],
        "evaluator_sha256": manifest[GATE.SOURCE_DIR + "interactive_evaluation.py"],
        "upper_body_evaluator_sha256": upper["evaluator_sha256"],
        "urdf_sha256": expected["urdf_sha256"], "gait_sha256": expected["gait_sha256"],
        "control_sha256": expected["control_sha256"], "checkpoint_sha256": digest,
        "upper_body_control_sha256": expected["control_sha256"],
        "upper_body_control_contract": expected, "upper_body": upper,
        "lower_body_passed": True, "reward_revision": 0,
        "initial_actor_provenance": {"source_iteration": profile["source_iteration"],
            "source_checkpoint_sha256": profile["source_checkpoint_sha256"],
            "inherited_completed_iterations": 0, "budget_mode": "new_updates"},
        "confirmation_provenance": {"report": "runs/example/confirmation.json",
            "report_sha256": "a" * 64, "checkpoint_sha256": digest,
            "source_sha256": manifest, "evaluation_seed": profile["confirmation_seed"]}}
    report["lower_body_checks"] = lower_checker(report)
    report["upper_body_checks"] = upper_checker(upper)
    report["checks"] = {**report["lower_body_checks"],
                        **{f"upper_body.{key}": value for key, value in report["upper_body_checks"].items()}}
    assert all(report["checks"].values())
    validations = []
    for iteration in (100, 200, 300):
        item = copy.deepcopy(report)
        item.update(episodes=32, completed_iterations=iteration,
                    evaluation_seed=profile["validation_seed_base"] + 200000 + iteration,
                    checkpoint_sha256=digest if iteration == 300 else str(iteration // 100) * 64)
        item["validation_provenance"] = {"report": f"runs/example/evaluation_{iteration}.json",
            "report_sha256": "a" * 64, "immutable_checkpoint": f"runs/example/chunks/{iteration}/resume.pt",
            "checkpoint_sha256": item["checkpoint_sha256"], "source_sha256": manifest,
            "contract_sha256": "b" * 64, "training_result_sha256": "c" * 64,
            "evaluated_at": "2026-10-23T00:00:00Z"}
        for collection in (item["cases"], item["upper_body"]["cases"]):
            for case in collection.values():
                case["episodes"] = 4
        validations.append(item)
    report["preceding_stage2_validations"] = validations
    write_json(folder / "contract.json", expected)
    write_json(folder / "evaluation.json", report)
    return folder, expected, report


def test_confirmed_v6_has_priority_and_its_own_environment(tmp_path, monkeypatch):
    folder, contract, _ = accepted_artifacts(tmp_path)
    gate = GATE.v6_policy_gate(tmp_path)
    assert gate["ready"], gate
    version, selected, _ = PLAYER.selected_policy(tmp_path)
    assert (version, selected) == ("v6", folder)
    assert PLAYER.selected_policy(tmp_path, folder)[0] == "v6"

    class UpperCfg:
        pass

    class UpperEnv:
        pass

    monkeypatch.setitem(sys.modules, "leg_humanoid_lab.upper_body_env",
                        SimpleNamespace(UpperBodyLegEnv=UpperEnv, UpperBodyEnvCfg=UpperCfg))
    environment, cfg = PLAYER.policy_environment(tmp_path, version)
    assert environment is UpperEnv and isinstance(cfg, UpperCfg)
    assert cfg.curriculum_stage == 2 and cfg.gait_file == contract["gait_file"]


@pytest.mark.parametrize("change", [
    {"stage": 1}, {"episodes": 32}, {"confirmation_episodes": 32},
    {"ready": False}, {"proxy": True}, {"reset_between_transitions": True},
    {"duration_s": 8.}, {"training_profile": {}}, {"source_sha256": {}},
    {"published_at": None}, {"checkpoint_sha256": "0" * 64},
    {"preceding_stage2_validations": []}, {"evaluation_seed": 123},
    {"initial_actor_provenance": {}}, {"confirmation_provenance": {}},
])
def test_candidate_or_incomplete_confirmation_never_unlocks_v6(tmp_path, change):
    folder, _, report = accepted_artifacts(tmp_path)
    write_json(folder / "evaluation.json", {**report, **change})
    assert not GATE.v6_policy_gate(tmp_path)["ready"]
    version, selected, gate = PLAYER.selected_policy(tmp_path, folder, "v6")
    assert (version, selected, gate["ready"]) == ("v6", folder, False)


@pytest.mark.parametrize("which", ("feet", "chest", "return", "one_hand", "previous_validation"))
def test_actual_metrics_recomputed_despite_claimed_passing_flags(tmp_path, which):
    folder, _, report = accepted_artifacts(tmp_path)
    if which == "feet":
        report["cases"]["sequence"]["directions"]["right"]["trailing_peak_clearance_m"] = 0.
    elif which == "chest":
        report["upper_body"]["cases"]["sequence"]["all"]["chest_peak_max_deg"] = 40.
    elif which == "return":
        report["upper_body"]["cases"]["sequence"]["early_return"]["arm_speed_rms_rad_s"] = 1.5
    elif which == "one_hand":
        report["upper_body"]["cases"]["sequence"]["tail"]["hands_return_distance_rms_m_per_hand"] = [.001, .05]
    else:
        report["preceding_stage2_validations"][0]["cases"]["w"]["velocity_rmse_m_s"] = .5
    write_json(folder / "evaluation.json", report)
    assert not GATE.v6_policy_gate(tmp_path)["ready"]


@pytest.mark.parametrize("relative", [
    "checkpoints/isaac/upper_body_v6/latest.pt", "config/upper_body_control_v6.json",
    "config/upper_body_v6.json", GATE.SOURCE_DIR + "upper_body_env.py",
    GATE.SOURCE_DIR + "upper_body_evaluation.py", GATE.SOURCE_DIR + "isaac_env.py",
    "tools/run_upper_body_training.py",
])
def test_changed_weights_or_snapshot_dependency_invalidates_acceptance(tmp_path, relative):
    accepted_artifacts(tmp_path)
    with (tmp_path / relative).open("a") as stream:
        stream.write("\n# changed after confirmation\n")
    assert not GATE.v6_policy_gate(tmp_path)["ready"]


def test_confirmation_seed_is_independent_and_validations_are_distinct(tmp_path):
    folder, _, report = accepted_artifacts(tmp_path)
    report["preceding_stage2_validations"][1] = copy.deepcopy(report["preceding_stage2_validations"][0])
    write_json(folder / "evaluation.json", report)
    assert not GATE.v6_policy_gate(tmp_path)["ready"]


def test_preceding_reports_cannot_use_another_source_or_reward_revision(tmp_path):
    folder, _, report = accepted_artifacts(tmp_path)
    report["preceding_stage2_validations"][0]["validation_provenance"]["source_sha256"] = {}
    write_json(folder / "evaluation.json", report)
    assert not GATE.v6_policy_gate(tmp_path)["ready"]
    report["preceding_stage2_validations"][0]["validation_provenance"]["source_sha256"] = report["source_sha256"]
    report["preceding_stage2_validations"][0]["reward_revision"] = 2
    write_json(folder / "evaluation.json", report)
    assert not GATE.v6_policy_gate(tmp_path)["ready"]


def test_unaccepted_v6_preserves_v5_fallback_without_explicit_substitution(tmp_path, monkeypatch):
    folder, _, report = accepted_artifacts(tmp_path)
    report["stage"] = 1
    write_json(folder / "evaluation.json", report)
    accepted_v5 = {"ready": True, "error": None, "checkpoint": "accepted older policy"}
    monkeypatch.setattr(PLAYER._V5_GATE, "v5_policy_gate", lambda *_args: accepted_v5)
    version, selected, gate = PLAYER.selected_policy(tmp_path)
    assert version == "v5" and selected == tmp_path / "checkpoints/isaac/lift_v5"
    assert gate is accepted_v5
    assert PLAYER.selected_policy(tmp_path, folder, "v6")[2]["ready"] is False


def test_validate_only_requires_no_torch_or_isaac(tmp_path):
    folder, _, _ = accepted_artifacts(tmp_path)
    program = f"""import importlib.util,json,sys
s=importlib.util.spec_from_file_location('player', {str(SPEC.origin)!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
v,f,g=m.selected_policy({str(tmp_path)!r}, {str(folder)!r}, 'v6')
assert g['ready'],g
assert not any(n=='torch' or n.startswith('isaac') for n in sys.modules)
print(json.dumps({{'version':v,'ready':g['ready']}}))
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"version": "v6", "ready": True}
    environment = {**os.environ, "Q1_LEG_ROOT": str(tmp_path)}
    result = subprocess.run([sys.executable, str(SPEC.origin), "--validate-only", "--policy-version", "v6"],
                            capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["version"] == "v6"
