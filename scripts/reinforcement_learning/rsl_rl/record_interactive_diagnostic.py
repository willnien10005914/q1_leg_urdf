#!/usr/bin/env python3
"""Record an UNACCEPTED native PPO checkpoint, including its actual failure.

This diagnostic never promotes a policy, writes successful-media names, changes
the acceptance gate, teleports the robot, or restarts a failed episode.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from play_interactive_web import ROOT, sha256


def diagnostic_plan(sidestep_duration=1.6):
    sequence_end = 3.5 + .5 + 2 * sidestep_duration + 1.
    return [
        {"name": "stand", "start": 0., "end": 1., "command": None},
        {"name": "W_forward", "start": 1., "end": 2.5, "command": [.22, 0., 0.]},
        {"name": "stop", "start": 2.5, "end": 3.5, "command": [0., 0., 0.]},
        {"name": "sidestep_sequence", "start": 3.5, "end": sequence_end, "command": None},
        {"name": "final_stand", "start": sequence_end, "end": sequence_end + 1.5, "command": None},
    ]


def validate_metadata(report):
    """A valid diagnostic may show a fall; it must never claim acceptance."""
    return bool(
        report.get("diagnostic") is True
        and report.get("gate_passed") is False
        and report.get("accepted") is False
        and report.get("proxy") is False
        and report.get("source") == "native Isaac Sim PhysX PPO"
        and report.get("overlay") == "EXPERIMENTAL / NOT ACCEPTED"
        and report.get("reset_count_after_start") == 0
        and isinstance(report.get("seed"), int)
        and isinstance(report.get("checkpoint_sha256"), str)
        and len(report["checkpoint_sha256"]) == 64
        and isinstance(report.get("frames"), int) and report["frames"] > 0
        and math.isfinite(report.get("actual_duration_s", float("nan")))
        and abs(report["actual_duration_s"] - report["frames"] / 50.) < 1e-7
        and isinstance(report.get("failure"), bool)
        and (not report["failure"] or bool(report.get("failure_reason")))
        and report.get("technical_error") is None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/interactive_diagnostic.mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sidestep-duration", type=float, default=1.6)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.sidestep_duration) or args.sidestep_duration <= 0:
        parser.error("--sidestep-duration must be finite and positive")
    checkpoint, target = args.checkpoint.resolve(), args.video.resolve()
    if not checkpoint.is_file():
        parser.error("--checkpoint must identify an existing native PPO checkpoint")
    if (target.name == "q1_leg_interactive_v4.mp4"
            or target.is_relative_to((ROOT / "checkpoints").resolve())):
        parser.error("Diagnostic media must use a separate output name outside checkpoints")
    if target.exists() or target.with_suffix(".json").exists():
        parser.error("Diagnostic output already exists; choose another filename to preserve it")
    base = json.loads((ROOT / "config/robot_contract.json").read_text())
    contract_path = args.contract or checkpoint.parent / "contract.json"
    contract = json.loads(contract_path.read_text())
    expected = {"robot": "q1_leg_interactive_v4", "observation_size": 110,
                "joint_names": base["joint_names"],
                "urdf_sha256": sha256(ROOT / "urdf/q1_inertia/quanta_robot_legged.urdf"),
                "gait_sha256": sha256(ROOT / "config/v4_gaits.json")}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Diagnostic requires a matching native v4 robot contract; legacy/proxy weights refused")
    checkpoint_sha = sha256(checkpoint)
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from leg_humanoid_lab.agent_cfg import PPOCfg
    from leg_humanoid_lab.interactive_control import KeyboardController
    from leg_humanoid_lab.interactive_env import InteractiveEnvCfg, InteractiveLegEnv
    from runtime_compat import load_checkpoint, runner_cfg

    class DiagnosticEnv(InteractiveLegEnv):
        _manual_reset_allowed = True
        suppressed_resets = 0

        def _reset_idx(self, env_ids):
            if self._manual_reset_allowed:
                super()._reset_idx(env_ids)
            else:
                self.suppressed_resets += len(env_ids) if env_ids is not None else self.num_envs

    plan = diagnostic_plan(args.sidestep_duration)
    report = {"diagnostic": True, "gate_passed": False, "accepted": False,
        "source": "native Isaac Sim PhysX PPO", "proxy": False,
        "overlay": "EXPERIMENTAL / NOT ACCEPTED", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha, "contract": expected,
        "seed": args.seed, "fps": 50, "resolution": [1280, 720], "frames": 0,
        "actual_duration_s": 0., "expected_duration_s": plan[-1]["end"],
        "plan": plan, "mode_changes": [], "root_positions": [],
        "failure": False, "failure_reason": None, "technical_error": None,
        "completed_plan": False, "reset_count_after_start": 0,
        "auto_resets_suppressed": 0,
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py",
            "source/leg_humanoid_lab/leg_humanoid_lab/interactive_control.py",
            "scripts/reinforcement_learning/rsl_rl/record_interactive_diagnostic.py")},
        "scope": "Unaccepted training candidate; actual failure remains visible; not evidence of learned skill."}
    env, writer = None, None
    try:
        cfg = InteractiveEnvCfg()
        cfg.scene.num_envs = 1
        cfg.training_mode = False
        cfg.episode_length_s = 30.
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.viewer.eye, cfg.viewer.lookat = (3., -3., 1.8), (.1, 0., .8)
        cfg.viewer.resolution = (1280, 720)
        env = DiagnosticEnv(cfg, render_mode="rgb_array")
        env.set_manual_control("stand", [0., 0., 0.], 0.)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        env._manual_reset_allowed = False
        runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
        load_checkpoint(runner, checkpoint, load_optimizer=False, load_iteration=False)
        if sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("Checkpoint changed while loading; record a stable snapshot")
        policy = runner.get_inference_policy(device=env.device)
        if not math.isclose(env.step_dt, .02, rel_tol=0., abs_tol=1e-7):
            raise RuntimeError("Diagnostic recording requires 50 Hz native control")
        controller = KeyboardController(sidestep_duration=args.sidestep_duration)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(target), fps=50, codec="libx264", quality=8,
                                   macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 27)
        except OSError:
            font = ImageFont.load_default()
        for _ in range(12):
            env.sim.render()
            env.render(recompute=True)
        last_segment, last_mode = None, None
        command_clock = time.monotonic()
        count = round(plan[-1]["end"] / env.step_dt)
        with torch.inference_mode():
            for i in range(count):
                if not app.is_running():
                    report.update(failure=True, failure_reason="Isaac Sim closed before the diagnostic finished")
                    break
                t, now = i * env.step_dt, command_clock + i * env.step_dt
                segment = next(item for item in plan if item["start"] <= t < item["end"])
                if segment["name"] != last_segment:
                    if segment["name"] == "stop":
                        controller.stop(now)
                    elif segment["name"] == "sidestep_sequence":
                        controller.sequence(now)
                    last_segment = segment["name"]
                if segment["name"] == "W_forward":
                    controller.velocity(*segment["command"], now)
                control = controller.tick(env.step_dt, now)
                env.set_manual_control(control["mode"], control["command"], control["phase"])
                obs = wrapped.get_observations()
                _obs, _reward, done, _info = wrapped.step(policy(obs))
                data = env.robot.data
                root = (data.root_pos_w[0] - env.scene.env_origins[0]).cpu().tolist()
                quat = data.root_quat_w[0].cpu().tolist()
                finite = all(math.isfinite(value) for value in root + quat)
                if finite:
                    report["root_positions"].append({"time": (i + 1) * env.step_dt,
                                                       "root": root, "quaternion": quat})
                if control["mode"] != last_mode:
                    report["mode_changes"].append({"time": t, "mode": control["mode"]})
                    last_mode = control["mode"]
                failed = bool(done[0]) or not finite
                if failed:
                    report.update(failure=True, failure_reason="Native PhysX episode terminated; recording stops without resetting")
                env.sim.render()
                pixels = np.asarray(env.render(recompute=True))
                if pixels.shape != (720, 1280, 3):
                    raise RuntimeError(f"Unexpected native viewport frame shape: {pixels.shape}")
                picture = Image.fromarray(pixels)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 1280, 86), fill=(18, 18, 18))
                draw.text((16, 7), report["overlay"], fill=(255, 195, 40), font=font)
                label = f"Native PhysX | seed {args.seed} | {t + env.step_dt:.2f}s | {control['mode']}"
                if failed:
                    label += " | FAILED - NO RESET"
                draw.text((16, 46), label, fill=(255, 95, 95) if failed else (245, 245, 245), font=font)
                writer.append_data(np.asarray(picture))
                report["frames"] += 1
                if failed:
                    break
        report["completed_plan"] = report["frames"] == count and not report["failure"]
        report["final_mode"] = last_mode
    except BaseException as exception:
        report["technical_error"] = str(exception)
        raise
    finally:
        if writer is not None:
            writer.close()
        report["actual_duration_s"] = report["frames"] / 50.
        report["auto_resets_suppressed"] = int(env.suppressed_resets) if env else 0
        if target.exists():
            report["video_sha256"] = sha256(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("DIAGNOSTIC_RESULT " + json.dumps({key: report[key] for key in (
            "gate_passed", "seed", "frames", "actual_duration_s", "failure", "failure_reason", "technical_error")}), flush=True)
        if env is not None:
            env.close()
        app.close()
    if not validate_metadata(report):
        raise RuntimeError("Diagnostic capture metadata is incomplete; inspect its JSON before sharing")


if __name__ == "__main__":
    main()
