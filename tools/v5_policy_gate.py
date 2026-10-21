"""Simulator-free identity and acceptance gate for native lift-v5 playback.

Preparatory lift/gesture evaluations never unlock interactive control. A
published stage-2 confirmation must match the weights, control profile, native
gait, current source files and every metric in the full keyboard acceptance
suite. This module intentionally imports neither torch nor Isaac Sim.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path


ROBOT_VERSION = "q1_leg_lift_v5"
CONTROLLER_VERSION = "q1_leg_v5"
SOURCE_DIR = "source/leg_humanoid_lab/leg_humanoid_lab/"
CASE_NAMES = ("stand", "w", "s", "a", "d", "yaw_left", "yaw_right", "sequence")
RUNTIME_FILES = (
    "config/lift_control_v5.json",
    "urdf/q1_inertia/quanta_robot_legged.urdf",
    "scripts/reinforcement_learning/rsl_rl/runtime_compat.py",
) + tuple(SOURCE_DIR + name for name in (
    "lift_env.py", "lift_objective.py", "interactive_env.py",
    "interactive_control.py", "interactive_evaluation.py", "isaac_env.py",
    "robot.py", "agent_cfg.py",
))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_file(root, relative):
    """Resolve only files inside this project's actual filesystem boundary."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("來源路徑必須是專案內相對路徑")
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root.resolve()):
        raise ValueError("來源路徑不可離開專案目錄")
    return resolved


def _constant(path, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id == name for item in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"來源缺少 {name}")


def acceptance_checker(root):
    """Load only the existing pure checker, excluding heavy runtime imports.

    The acceptance thresholds and function come from the same hash-verified
    local source used during evaluation; no duplicate thresholds are maintained
    here. Its function body has no simulator dependency. Project source is
    trusted executable code, just as when constructing the native environment.
    """
    path = root / SOURCE_DIR / "interactive_evaluation.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    namespace = {"math": math, "CASE_NAMES": _constant(path, "CASE_NAMES"),
                 "GATES": _constant(path, "GATES")}
    if tuple(namespace["CASE_NAMES"]) != CASE_NAMES:
        raise ValueError("v5 需要完整八項 keyboard 與閃身步評估")
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "check_interactive_gates"), None)
    if function is None:
        raise ValueError("來源缺少完整互動驗收函式")
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["check_interactive_gates"], namespace["GATES"]


def runtime_sources(root):
    root = Path(root)
    control = json.loads((root / "config/lift_control_v5.json").read_text())
    gait = control["gait_file"]
    files = (*RUNTIME_FILES, gait)
    return {relative: sha256(project_file(root, relative)) for relative in files}


def v5_policy_gate(root, folder=None):
    root = Path(root)
    folder = Path(folder) if folder is not None else root / "checkpoints/isaac/lift_v5"
    try:
        base = json.loads((root / "config/robot_contract.json").read_text())
        control_file = root / "config/lift_control_v5.json"
        control = json.loads(control_file.read_text())
        gait_file = project_file(root, control["gait_file"])
        gait = json.loads(gait_file.read_text())
        saved = json.loads((folder / "contract.json").read_text())
        report = json.loads((folder / "evaluation.json").read_text())
        if base.get("native_cad_ready") is not True or base.get("proxy") is not False:
            raise ValueError("v5 即時控制只接受完整原生 CAD")
        if gait.get("contract") != "q1_leg_command_v5":
            raise ValueError("v5 必須使用其獨立 gait library")
        expected = {
            "robot": ROBOT_VERSION, "schema": 5,
            "urdf_sha256": sha256(root / "urdf/q1_inertia/quanta_robot_legged.urdf"),
            "gait_sha256": sha256(gait_file), "gait_file": control["gait_file"],
            "control_sha256": sha256(control_file),
            "control_profile": _constant(root / SOURCE_DIR / "lift_objective.py", "CONTROL_PROFILE"),
            "motor": control, "joint_names": base["joint_names"],
            "action_size": 23, "observation_size": 110,
            "default": gait["default"], "root_height": gait["root_height"], "proxy": False,
        }
        if len(set(expected["joint_names"])) != 23 or len(gait["default"]) != 23:
            raise ValueError("v5 原生模型需要 23 個動作關節")
        if gait.get("joint_names") != expected["joint_names"] or gait.get("robot_urdf_sha256") != expected["urdf_sha256"]:
            raise ValueError("v5 gait 與原生 URDF 關節契約不符")
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"v5 policy contract 不符：{key}")

        if report.get("ready") is not True or report.get("passed") is not True or report.get("stage") != 2:
            raise ValueError("v5 尚未完成並發布完整互動階段驗收")
        if report.get("backend") != "isaaclab" or report.get("proxy") is not False:
            raise ValueError("v5 確認評估必須使用原生 Isaac/PhysX")
        if report.get("reset_between_transitions") is not False or report.get("duration_s") != 20.:
            raise ValueError("v5 必須通過 20 秒無動作間重置的連續操作評估")
        episodes = report.get("episodes")
        if type(episodes) is not int or episodes < 128 or episodes % len(CASE_NAMES):
            raise ValueError("v5 需要至少 128 回合獨立完整確認")
        if report.get("confirmation_episodes") != episodes:
            raise ValueError("v5 缺少獨立確認回合標記")
        if type(report.get("completed_iterations")) is not int or report["completed_iterations"] <= 0:
            raise ValueError("v5 缺少實際 PPO 更新數")
        profile = report.get("training_profile")
        current_profile = json.loads((root / "config/lift_v5.json").read_text())
        if not isinstance(profile, dict) or profile != current_profile or profile.get("robot") != ROBOT_VERSION:
            raise ValueError("v5 確認評估的訓練設定不符")
        if profile.get("confirmation_episodes") != episodes or profile.get("gait_file") != control["gait_file"]:
            raise ValueError("v5 確認回合數或 gait 設定不符")
        seed, base_seed = report.get("evaluation_seed"), profile.get("confirmation_seed")
        if (type(seed) is not int or type(base_seed) is not int or seed < base_seed
                or (seed-base_seed) % 1000 or seed == profile.get("validation_seed_base", 0) + 200000 + report["completed_iterations"]):
            raise ValueError("v5 缺少獨立確認評估的 seed")
        if not report.get("published_from") or not report.get("published_at"):
            raise ValueError("v5 尚未經訓練監督程式正式發布")

        manifest = report.get("source_sha256")
        required = {*RUNTIME_FILES, control["gait_file"], "config/lift_v5.json"}
        if not isinstance(manifest, dict) or not required.issubset(manifest):
            raise ValueError("v5 缺少控制、環境、評估或模型的完整來源雜湊")
        for relative, digest in manifest.items():
            if not isinstance(digest, str) or sha256(project_file(root, relative)) != digest:
                raise ValueError(f"v5 評估後來源已變更：{relative}")
        for key, relative in (
            ("environment_sha256", SOURCE_DIR+"lift_env.py"),
            ("base_environment_sha256", SOURCE_DIR+"interactive_env.py"),
            ("evaluator_sha256", SOURCE_DIR+"interactive_evaluation.py"),
        ):
            if report.get(key) != manifest[relative]:
                raise ValueError(f"v5 評估來源不符：{key}")
        for key in ("urdf_sha256", "gait_sha256"):
            if report.get(key) != expected[key]:
                raise ValueError(f"v5 評估模型不符：{key}")
        cases = report.get("cases", {})
        if any(type(cases.get(name, {}).get("episodes")) is not int
               or cases[name]["episodes"] != episodes // len(CASE_NAMES) for name in CASE_NAMES):
            raise ValueError("v5 確認需要均衡涵蓋八項完整操作")
        checker, thresholds = acceptance_checker(root)
        checks = checker(report)
        if (not checks or not all(value is True for value in checks.values())
                or report.get("checks") != checks or report.get("gates") != thresholds
                or report.get("failed_gates") != [] or type(report.get("score")) not in (int, float)
                or report["score"] != 1.):
            raise ValueError("v5 評估的實際步態、位移或回穩指標未全部合格")
        checkpoint = folder / "latest.pt"
        digest = sha256(checkpoint)
        if report.get("checkpoint_sha256") != digest:
            raise ValueError("v5 確認評估與 checkpoint SHA256 不符")
        return {"ready": True, "error": None, "version": "v5", "controller_version": CONTROLLER_VERSION,
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                "confirmation_episodes": episodes, "contract": expected,
                "source_sha256": manifest, "sidestep_duration_s": gait["sidestep_duration_s"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, SyntaxError, NameError) as error:
        return {"ready": False, "version": "v5", "controller_version": CONTROLLER_VERSION,
                "error": f"v5 控制器尚未就緒：{error}"}
