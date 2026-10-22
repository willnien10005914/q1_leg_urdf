"""Simulator-free acceptance gate for the independently trained upper-body V6.

A candidate or preparatory pass never enables live controls. Publication needs
three complete stage-2 validations and a fresh 128-episode confirmation, all
bound to the current control, source snapshot and final checkpoint identity.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


ROBOT_VERSION = "q1_leg_upper_body_v6"
CONTROLLER_VERSION = "q1_leg_v6"
SOURCE_DIR = "source/leg_humanoid_lab/leg_humanoid_lab/"
PROFILE_FILE = "config/upper_body_v6.json"
CONTROL_FILE = "config/upper_body_control_v6.json"
CASE_NAMES = ("stand", "w", "s", "a", "d", "yaw_left", "yaw_right", "sequence")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V5 = _load(Path(__file__).with_name("v5_policy_gate.py"), "upper_body_base_gate")


def runtime_sources(root):
    """Use the supervisor's exact manifest definition, without Isaac or torch."""
    root = Path(root).resolve()
    helper = _load(root / "tools/run_upper_body_training.py", "upper_body_source_manifest")
    helper.ROOT = root
    return helper.source_manifest(root / PROFILE_FILE)


def acceptance_checkers(root):
    lower, lower_gates = _V5.acceptance_checker(root)
    path = root / SOURCE_DIR / "upper_body_evaluation.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    defaults = _V5._constant(path, "DEFAULT_GATES")
    namespace = {"math": math, "DEFAULT_GATES": defaults, "CASE_NAMES": CASE_NAMES,
                 "STAGE_CASES": _V5._constant(root / SOURCE_DIR / "lift_evaluation.py", "STAGE_CASES")}
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "check_upper_body_gates"), None)
    if function is None:
        raise ValueError("來源缺少上半身完整驗收函式")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return lower, lower_gates, namespace["check_upper_body_gates"], defaults


def expected_contract(root):
    root = Path(root)
    base = json.loads((root / "config/robot_contract.json").read_text())
    control = json.loads((root / CONTROL_FILE).read_text())
    leg = json.loads((root / "config/lift_control_v5.json").read_text())
    gait_file = _V5.project_file(root, leg["gait_file"])
    gait = json.loads(gait_file.read_text())
    if base.get("native_cad_ready") is not True or base.get("proxy") is not False:
        raise ValueError("v6 即時控制只接受完整原生 CAD")
    if control.get("robot") != ROBOT_VERSION or control.get("schema") != 6:
        raise ValueError("v6 必須使用獨立上半身控制契約")
    base_sha = sha256(root / "config/lift_control_v5.json")
    if control.get("base_control_sha256") != base_sha:
        raise ValueError("v6 原生腿部控制來源已變更")
    expected = {
        "robot": ROBOT_VERSION, "schema": 6,
        "urdf_sha256": sha256(root / "urdf/q1_inertia/quanta_robot_legged.urdf"),
        "gait_sha256": sha256(gait_file), "gait_file": leg["gait_file"],
        "control_sha256": sha256(root / CONTROL_FILE), "base_control_sha256": base_sha,
        "control_profile": _V5._constant(root / SOURCE_DIR / "upper_body_objective.py", "CONTROL_PROFILE"),
        "motor": {**leg, "upper_body": control}, "joint_names": base["joint_names"],
        "action_size": 23, "observation_size": 110, "default": gait["default"],
        "root_height": gait["root_height"], "proxy": False,
        "upper_body_gates": control["evaluation_gates"],
    }
    if (len(set(expected["joint_names"])) != 23 or len(gait["default"]) != 23
            or gait.get("contract") != "q1_leg_command_v5"
            or gait.get("joint_names") != expected["joint_names"]
            or gait.get("robot_urdf_sha256") != expected["urdf_sha256"]):
        raise ValueError("v6 gait、原生 URDF 或 23 關節契約不符")
    return expected, gait


def _check_report(report, episodes, expected, saved, manifest, checkers):
    lower_checker, lower_gates, upper_checker, upper_defaults = checkers
    if (report.get("passed") is not True or report.get("stage") != 2
            or report.get("robot") != ROBOT_VERSION or report.get("schema") != 6
            or report.get("backend") != "isaaclab" or report.get("proxy") is not False
            or type(report.get("episodes")) is not int or report["episodes"] != episodes
            or report.get("reset_between_transitions") is not False or report.get("duration_s") != 20.):
        raise ValueError("v6 必須通過原生 20 秒無動作間重置的完整 stage-2 評估")
    if (type(report.get("completed_iterations")) is not int or not 0 < report["completed_iterations"] <= 20000
            or not isinstance(report.get("checkpoint_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", report["checkpoint_sha256"])):
        raise ValueError("v6 評估缺少實際更新數或 checkpoint 身分")
    identity = {
        "environment_sha256": manifest[SOURCE_DIR + "upper_body_env.py"],
        "base_environment_sha256": manifest[SOURCE_DIR + "interactive_env.py"],
        "evaluator_sha256": manifest[SOURCE_DIR + "interactive_evaluation.py"],
        "upper_body_evaluator_sha256": manifest[SOURCE_DIR + "upper_body_evaluation.py"],
        "urdf_sha256": expected["urdf_sha256"], "gait_sha256": expected["gait_sha256"],
        "control_sha256": expected["control_sha256"],
        "upper_body_control_sha256": expected["control_sha256"],
    }
    if any(report.get(key) != value for key, value in identity.items()):
        raise ValueError("v6 評估與環境、控制或模型來源不符")
    upper = report.get("upper_body", {})
    upper_gates = {**upper_defaults, **expected["upper_body_gates"]}
    if (upper.get("stage") != 2 or upper.get("schema") != 1
            or upper.get("gates") != upper_gates or report.get("gates") != lower_gates
            or upper.get("evaluator_sha256") != identity["upper_body_evaluator_sha256"]
            or upper.get("control_sha256") != expected["control_sha256"]
            or upper.get("control_contract") != saved
            or report.get("upper_body_control_contract") != saved):
        raise ValueError("v6 上下半身驗收門檻或控制契約不符")
    for cases in (report.get("cases", {}), upper.get("cases", {})):
        if any(type(cases.get(name, {}).get("episodes")) is not int
               or cases[name]["episodes"] != episodes // len(CASE_NAMES) for name in CASE_NAMES):
            raise ValueError("v6 確認需要均衡涵蓋八項完整操作")
    lower = lower_checker(report)
    upper_checks = upper_checker(upper)
    combined = {**lower, **{f"upper_body.{key}": value for key, value in upper_checks.items()}}
    if (not lower or not upper_checks or not all(value is True for value in combined.values())
            or report.get("lower_body_passed") is not True
            or report.get("lower_body_checks") != lower or report.get("upper_body_checks") != upper_checks
            or report.get("checks") != combined or report.get("failed_gates") != []
            or type(report.get("score")) not in (int, float) or report["score"] != 1.):
        raise ValueError("v6 實際抬腳、移動、胸口或回手指標未全部合格")


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def v6_policy_gate(root, folder=None):
    root = Path(root)
    folder = Path(folder) if folder is not None else root / "checkpoints/isaac/upper_body_v6"
    try:
        expected, gait = expected_contract(root)
        saved = json.loads((folder / "contract.json").read_text())
        report = json.loads((folder / "evaluation.json").read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("v6 已儲存控制契約不符")
        profile = json.loads((root / PROFILE_FILE).read_text())
        if (profile.get("robot") != ROBOT_VERSION or profile.get("max_iterations") != 20000
                or profile.get("budget_mode") != "new_updates" or profile.get("confirmation_episodes") != 128
                or profile.get("validation_episodes") != 32 or profile.get("consecutive_passes_to_confirm") != 3
                or profile.get("gait_file") != expected["gait_file"]
                or report.get("training_profile") != profile):
            raise ValueError("v6 訓練預算或驗收設定不符")
        manifest = runtime_sources(root)
        if report.get("source_sha256") != manifest:
            raise ValueError("v6 評估後訓練、環境、控制或模型來源已變更")
        if report.get("ready") is not True or not report.get("published_from") or not report.get("published_at"):
            raise ValueError("v6 尚未經監督程式正式發布")
        checkers = acceptance_checkers(root)
        _check_report(report, 128, expected, saved, manifest, checkers)
        if report.get("confirmation_episodes") != 128:
            raise ValueError("v6 缺少獨立 128 回合確認標記")
        revision = report.get("reward_revision")
        if type(revision) is not int or revision not in (0, 1, 2):
            raise ValueError("v6 缺少已宣告的 reward revision")
        actor = report.get("initial_actor_provenance", {})
        if actor != {"source_iteration": profile["source_iteration"],
                     "source_checkpoint_sha256": profile["source_checkpoint_sha256"],
                     "inherited_completed_iterations": 0, "budget_mode": "new_updates"}:
            raise ValueError("v6 未保留原 13115 actor 與新訓練計數的來源證明")
        seed = report.get("evaluation_seed")
        if (type(seed) is not int or seed < profile["confirmation_seed"]
                or (seed - profile["confirmation_seed"]) % 1000):
            raise ValueError("v6 獨立確認 seed 不符")
        confirmation = report.get("confirmation_provenance", {})
        if (confirmation.get("source_sha256") != manifest
                or confirmation.get("checkpoint_sha256") != report["checkpoint_sha256"]
                or confirmation.get("evaluation_seed") != seed or not confirmation.get("report")
                or not _digest(confirmation.get("report_sha256"))):
            raise ValueError("v6 獨立確認缺少封存報告與來源證明")
        validations = report.get("preceding_stage2_validations")
        if not isinstance(validations, list) or len(validations) != 3:
            raise ValueError("v6 需要三次連續完整 stage-2 驗收")
        iterations, seeds = [], []
        for item in validations:
            _check_report(item, 32, expected, saved, manifest, checkers)
            provenance = item.get("validation_provenance", {})
            if (item.get("reward_revision") != revision
                    or provenance.get("source_sha256") != manifest
                    or provenance.get("checkpoint_sha256") != item["checkpoint_sha256"]
                    or not provenance.get("immutable_checkpoint") or not provenance.get("report")
                    or not provenance.get("evaluated_at")
                    or any(not _digest(provenance.get(key)) for key in (
                        "report_sha256", "contract_sha256", "training_result_sha256"))):
                raise ValueError("v6 三次驗收缺少相同 revision 的封存來源證明")
            iteration = item["completed_iterations"]
            validation_seed = item.get("evaluation_seed")
            if (type(validation_seed) is not int
                    or validation_seed != profile["validation_seed_base"] + 200000 + iteration):
                raise ValueError("v6 連續驗收 seed 不符")
            iterations.append(iteration)
            seeds.append(validation_seed)
        if (iterations != sorted(set(iterations)) or len(set(seeds)) != 3 or seed in seeds
                or iterations[-1] != report["completed_iterations"]
                or validations[-1]["checkpoint_sha256"] != report["checkpoint_sha256"]):
            raise ValueError("v6 三次驗收或獨立確認並非同一發布候選的連續流程")
        checkpoint = folder / "latest.pt"
        digest = sha256(checkpoint)
        if report["checkpoint_sha256"] != digest:
            raise ValueError("v6 確認評估與 checkpoint SHA256 不符")
        return {"ready": True, "error": None, "version": "v6", "controller_version": CONTROLLER_VERSION,
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                "confirmation_episodes": 128, "contract": expected,
                "source_sha256": manifest, "sidestep_duration_s": gait["sidestep_duration_s"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, SyntaxError, NameError, ImportError) as error:
        return {"ready": False, "version": "v6", "controller_version": CONTROLLER_VERSION,
                "error": f"v6 控制器尚未就緒：{error}"}
