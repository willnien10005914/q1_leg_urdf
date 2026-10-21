#!/usr/bin/env python3
"""Record native PPO direction changes and one continuous left/right sequence.

Only a policy passing the same 128-episode web gate may be recorded. Failed or
incomplete attempts remain *.candidate files and never replace the final movie.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from play_interactive_web import ROOT, policy_gate, sha256


def recording_plan(sidestep_duration=1.6):
    """Finite deterministic equivalent of held WASD keys and one button press."""
    sequence_end = 9. + .5 + 2. * sidestep_duration + 1.
    return [
        {"name": "stand", "start": 0., "end": 1., "command": [0., 0., 0.]},
        {"name": "W_forward", "start": 1., "end": 3., "command": [.22, 0., 0.]},
        {"name": "S_backward", "start": 3., "end": 5., "command": [-.22, 0., 0.]},
        {"name": "A_left", "start": 5., "end": 6.5, "command": [0., .18, 0.]},
        {"name": "D_right", "start": 6.5, "end": 8., "command": [0., -.18, 0.]},
        {"name": "stop", "start": 8., "end": 9., "command": [0., 0., 0.]},
        {"name": "sidestep_sequence", "start": 9., "end": sequence_end, "command": None},
        {"name": "post_sequence_stand", "start": sequence_end, "end": sequence_end + 1.5, "command": None},
        {"name": "W_resume", "start": sequence_end + 1.5, "end": sequence_end + 3.5, "command": [.22, 0., 0.]},
        {"name": "final_stop", "start": sequence_end + 3.5, "end": sequence_end + 5., "command": [0., 0., 0.]},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "docs/results/q1_leg_interactive_v4.mp4")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/isaac/interactive_v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sidestep-duration", type=float, default=1.6)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.sidestep_duration <= 0:
        parser.error("--sidestep-duration must be positive")
    gate = policy_gate(ROOT, args.checkpoint_dir)
    if not gate["ready"]:
        raise RuntimeError(gate["error"])
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from leg_humanoid_lab.agent_cfg import PPOCfg
    from leg_humanoid_lab.interactive_control import KeyboardController
    from leg_humanoid_lab.interactive_env import InteractiveEnvCfg, InteractiveLegEnv
    from runtime_compat import load_checkpoint, runner_cfg

    class RecordingEnv(InteractiveLegEnv):
        _manual_reset_allowed = True
        suppressed_resets = 0

        def _reset_idx(self, env_ids):
            if self._manual_reset_allowed:
                super()._reset_idx(env_ids)
            else:
                self.suppressed_resets += len(env_ids) if env_ids is not None else self.num_envs

    env, writer = None, None
    target = args.video.resolve()
    candidate = target.with_name(target.stem + ".candidate.mp4")
    candidate_metadata = candidate.with_suffix(".json")
    plan = recording_plan(args.sidestep_duration)
    expected_duration = plan[-1]["end"]
    metadata = {
        "source": "live native Isaac Sim PhysX PPO", "controller_version": "q1_leg_v4",
        "robot": "q1_leg_interactive_v4", "checkpoint": gate["checkpoint"],
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "urdf_sha256": gate["contract"]["urdf_sha256"],
        "gait_sha256": gate["contract"]["gait_sha256"],
        "confirmation_episodes": gate["confirmation_episodes"], "seed": args.seed,
        "fps": 50, "resolution": [1280, 720], "frames": 0, "duration": 0.,
        "expected_duration": expected_duration, "survived": False,
        "actual_failed": False, "succeeded": False, "completed": False,
        "mode_changes": [], "root_positions": [], "plan": plan,
        "no_reset_count": 0, "reset_count_after_start": 0,
        "no_resets_after_start": True, "auto_resets_suppressed": 0,
        "final_mode": None, "final_sequence_active": False,
        "post_sequence_resume": {"accepted": False, "full_period": False,
            "expected_duration_s": 2., "duration_s": 0., "frames": 0,
            "accepted_frames": 0, "walk_frames": 0, "displacement_m": 0.,
            "forward_displacement_m": 0., "sequence_completed_before_resume": False},
        "final_stand_speed_m_s": None,
        "source_sha256": gate.get("source_sha256", {}),
        "scope": "Native CAD simulation; real hardware unvalidated",
    }
    error = None
    try:
        cfg = InteractiveEnvCfg()
        cfg.scene.num_envs = 1
        cfg.training_mode = False
        cfg.episode_length_s = 30.
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.viewer.eye = (3., -3., 1.8)
        cfg.viewer.lookat = (.1, 0., .8)
        cfg.viewer.resolution = (1280, 720)
        env = RecordingEnv(cfg, render_mode="rgb_array")
        env.set_manual_control("stand", [0., 0., 0.], 0.)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        env._manual_reset_allowed = False
        runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
        load_checkpoint(runner, gate["checkpoint"], load_optimizer=False, load_iteration=False)
        policy = runner.get_inference_policy(device=env.device)
        if sha256(gate["checkpoint"]) != gate["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed while loading; rerun recording against a stable evaluation")
        current_gate = policy_gate(ROOT, args.checkpoint_dir)
        if (not current_gate["ready"] or current_gate["contract"] != gate["contract"]
                or current_gate.get("source_sha256") != gate.get("source_sha256")):
            raise RuntimeError("Confirmed simulation sources changed while loading; restart recording")
        if not math.isclose(env.step_dt, .02, rel_tol=0., abs_tol=1e-7):
            raise RuntimeError("Recording requires 50 Hz control / .02 s step_dt")
        controller = KeyboardController(sidestep_duration=args.sidestep_duration)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(candidate), fps=50, codec="libx264", quality=8,
                                   macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
        for _ in range(12):
            env.sim.render()
            env.render(recompute=True)

        count = round(expected_duration / env.step_dt)
        metadata["expected_frames"] = count
        last_segment, last_mode, control = None, None, None
        resume_start, resume_end, resume_forward = None, None, None
        final_speeds = []
        command_clock = time.monotonic()
        with torch.inference_mode():
            for i in range(count):
                if not app.is_running():
                    error = "Isaac Sim closed before completing the finite recording"
                    break
                t = i * env.step_dt
                now = command_clock + t
                segment = next(item for item in plan if item["start"] <= t < item["end"])
                if segment["name"] != last_segment:
                    if segment["name"] in ("stop", "final_stop"):
                        controller.stop(now)
                    elif segment["name"] == "sidestep_sequence":
                        controller.sequence(now)
                    elif segment["name"] == "W_resume":
                        data = env.robot.data
                        resume_start = (data.root_pos_w[0] - env.scene.env_origins[0]).cpu().tolist()
                        w, x, y, z = data.root_quat_w[0].cpu().tolist()
                        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                        resume_forward = (math.cos(yaw), math.sin(yaw))
                        metadata["post_sequence_resume"]["sequence_completed_before_resume"] = bool(
                            control and control["mode"] == "stand" and not control["sequence_active"])
                    last_segment = segment["name"]
                if segment["name"] in ("W_forward", "S_backward", "A_left", "D_right", "W_resume"):
                    accepted = controller.velocity(*segment["command"], now)
                    if segment["name"] == "W_resume":
                        metadata["post_sequence_resume"]["accepted_frames"] += int(accepted is True)
                control = controller.tick(env.step_dt, now)
                env.set_manual_control(control["mode"], control["command"], control["phase"])
                obs = wrapped.get_observations()
                _obs, _reward, done, _info = wrapped.step(policy(obs))
                root = (env.robot.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().tolist()
                quat = env.robot.data.root_quat_w[0].cpu().tolist()
                velocity = env.robot.data.root_lin_vel_b[0, :2].cpu().tolist()
                finite = all(math.isfinite(value) for value in root + quat + velocity)
                if segment["name"] == "W_resume":
                    resume = metadata["post_sequence_resume"]
                    resume["frames"] += 1
                    resume["walk_frames"] += int(control["mode"] == "walk")
                    resume_end = root if finite else None
                if segment["name"] == "final_stop" and t >= segment["end"] - .5 and finite:
                    final_speeds.append(math.hypot(*velocity))
                if finite:
                    metadata["root_positions"].append({"time": (i + 1) * env.step_dt, "root": root, "quaternion": quat})
                if control["mode"] != last_mode:
                    metadata["mode_changes"].append({"time": t, "mode": control["mode"],
                                                     "root": root if finite else None})
                    last_mode = control["mode"]
                # Headless viewport annotators need this explicit render before
                # every captured frame; env.step alone may produce a frozen MP4.
                env.sim.render()
                pixels = np.asarray(env.render(recompute=True))
                if pixels.shape != (720, 1280, 3):
                    raise RuntimeError(f"Unexpected viewport frame shape: {pixels.shape}")
                writer.append_data(pixels)
                metadata["frames"] += 1
                if bool(done[0]) or not finite:
                    metadata["actual_failed"] = True
                    error = "Recorded native PhysX episode failed before completing the command sequence"
                    break
        metadata["duration"] = metadata["frames"] / 50.
        metadata["final_mode"] = control["mode"] if control else None
        metadata["final_sequence_active"] = bool(control and control["sequence_active"])
        metadata["auto_resets_suppressed"] = int(env.suppressed_resets)
        metadata["completed"] = metadata["frames"] == count
        resume = metadata["post_sequence_resume"]
        resume["duration_s"] = resume["frames"] * env.step_dt
        resume["accepted"] = resume["accepted_frames"] == round(2. / env.step_dt)
        resume["full_period"] = resume["frames"] == round(2. / env.step_dt)
        if resume_start is not None and resume_end is not None:
            delta = [resume_end[i] - resume_start[i] for i in (0, 1)]
            resume["displacement_m"] = math.hypot(*delta)
            resume["forward_displacement_m"] = sum(a*b for a, b in zip(delta, resume_forward))
        metadata["final_stand_speed_m_s"] = float(np.mean(final_speeds)) if final_speeds else None
        modes = {item["mode"] for item in metadata["mode_changes"]}
        metadata["survived"] = not metadata["actual_failed"] and metadata["completed"]
        metadata["succeeded"] = bool(
            metadata["survived"] and error is None and env.suppressed_resets == 0
            and {"sidestep_left", "sidestep_right"}.issubset(modes)
            and resume["sequence_completed_before_resume"] and resume["accepted"] and resume["full_period"]
            and resume["walk_frames"] >= .9 * resume["frames"]
            and resume["displacement_m"] > .05 and resume["forward_displacement_m"] > .05
            and metadata["final_stand_speed_m_s"] is not None and metadata["final_stand_speed_m_s"] < .07
            and metadata["final_mode"] == "stand" and not metadata["final_sequence_active"])
        if not metadata["succeeded"] and error is None:
            error = "Recording must finish both sidesteps, resume keyboard walking >.05m for 2s and stop without reset"
    except BaseException as exception:
        error = str(exception)
        metadata["succeeded"] = False
        raise
    finally:
        if writer is not None:
            writer.close()
        metadata["duration"] = metadata["frames"] / 50.
        if error is not None:
            metadata["error"] = error
        if candidate.exists():
            metadata["video_sha256"] = sha256(candidate)
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        if metadata["succeeded"]:
            os.replace(candidate, target)
            os.replace(candidate_metadata, target.with_suffix(".json"))
        print("VIDEO_RESULT " + json.dumps(metadata), flush=True)
        if env is not None:
            env.close()
        app.close()
    if not metadata["succeeded"]:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
