#!/usr/bin/env python3
"""Continuous v4 PPO inference with a local keyboard / sidestep command API.

The HTTP endpoint remains available when the v4 policy has not passed its gate.
Legacy v3 weights never unlock these controls. Ordinary mode transitions do not
reset simulation; a failed episode can be reset only by an explicit restart.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

ROOT = Path(os.environ.get("Q1_LEG_ROOT", Path(__file__).resolve().parents[3]))
CONTROLLER_VERSION = "q1_leg_v4"
ROBOT_VERSION = "q1_leg_interactive_v4"
SPEED_LIMITS = (.22, .18, .6)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_gate(root=ROOT, folder=None):
    """Check source identity, v4 contract, confirmation count and exact weights."""
    root = Path(root)
    folder = Path(folder) if folder else root / "checkpoints/isaac/interactive_v4"
    try:
        base = json.loads((root / "config/robot_contract.json").read_text())
        saved = json.loads((folder / "contract.json").read_text())
        report = json.loads((folder / "evaluation.json").read_text())
        expected = {
            "robot": ROBOT_VERSION,
            "urdf_sha256": sha256(root / "urdf/q1_inertia/quanta_robot_legged.urdf"),
            "gait_sha256": sha256(root / "config/v4_gaits.json"),
            "joint_names": base["joint_names"], "observation_size": 110,
        }
        if len(expected["joint_names"]) != 23:
            raise ValueError("原生模型必須使用 23 個動作關節")
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"v4 policy contract 不符：{key}")
        if report.get("passed") is not True:
            raise ValueError("v4 策略尚未通過控制與閃身步評估")
        count = report.get("confirmation_episodes", report.get("episodes", 0))
        if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 128:
            raise ValueError("v4 策略需要至少 128 回合獨立確認評估")
        if report.get("proxy", False) is not False:
            raise ValueError("v4 即時控制只接受原生 CAD 策略")
        for key in ("urdf_sha256", "gait_sha256"):
            if key in report and report[key] != expected[key]:
                raise ValueError(f"v4 評估來源不符：{key}")
        for key, name in (("environment_sha256", "interactive_env.py"),
                          ("evaluator_sha256", "interactive_evaluation.py")):
            if key in report and report[key] != sha256(root / "source/leg_humanoid_lab/leg_humanoid_lab" / name):
                raise ValueError(f"v4 評估程式來源不符：{key}")
        manifest = report.get("source_sha256", {})
        if not isinstance(manifest, dict):
            raise ValueError("v4 source_sha256 必須是來源檔案雜湊對照表")
        for relative, expected_hash in manifest.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise ValueError("v4 source manifest 格式不符")
            path = root / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("v4 source manifest 必須使用專案內相對路徑")
            if sha256(path) != expected_hash:
                raise ValueError(f"v4 訓練來源已變更：{relative}")
        checkpoint = folder / "latest.pt"
        digest = sha256(checkpoint)
        if report.get("checkpoint_sha256") != digest:
            raise ValueError("v4 評估與 checkpoint SHA256 不符")
        return {"ready": True, "error": None, "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest, "confirmation_episodes": int(count),
                "contract": expected, "source_sha256": manifest}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"ready": False, "error": f"v4 控制器尚未就緒：{error}"}


def normalized_velocity(payload):
    values = []
    for key in ("vx", "vy", "wz"):
        value = payload.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("velocity must contain finite numeric vx, vy, wz")
        values.append(float(value))
    values = [max(-limit, min(limit, value)) for value, limit in zip(values, SPEED_LIMITS)]
    norm = sum(abs(value) / limit for value, limit in zip(values, SPEED_LIMITS))
    return [value / max(1., norm) for value in values]


def runtime_sources(root):
    """Pin sources used to construct this process; hot file edits need a restart."""
    paths = ["config/v4_gaits.json", "urdf/q1_inertia/quanta_robot_legged.urdf",
             "scripts/reinforcement_learning/rsl_rl/runtime_compat.py"]
    paths += ["source/leg_humanoid_lab/leg_humanoid_lab/" + name for name in (
        "interactive_env.py", "interactive_control.py", "interactive_evaluation.py",
        "isaac_env.py", "robot.py", "agent_cfg.py")]
    return {name: sha256(root / name) for name in paths}


def handler_type(commands, state, lock):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            content = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path != "/state":
                self.send_json({"error": "unknown endpoint"}, 404)
                return
            with lock:
                value = dict(state)
            self.send_json(value)

        def do_POST(self):
            if self.path != "/command":
                self.send_json({"error": "unknown endpoint"}, 404)
                return
            # Browser writes must arrive through the same-origin web proxy.
            if self.headers.get("Origin"):
                self.send_json({"error": "use the same-origin web command proxy"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid body length")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("expected a command object")
                action = body.get("action")
                if action not in ("velocity", "sidestep_sequence", "stop", "play", "pause", "restart"):
                    raise ValueError("unknown command")
                velocity = normalized_velocity(body) if action == "velocity" else None
                with lock:
                    ready, status = state["ready"], state["status"]
                if action not in ("stop", "pause") and not ready:
                    self.send_json({"error": "v4 策略尚未通過 128 回合確認，控制已鎖定"}, 409)
                    return
                if status == "failed" and action in ("velocity", "sidestep_sequence", "play"):
                    self.send_json({"error": "機器人已跌倒，請明確按回到起點後再控制"}, 409)
                    return
                if action == "restart" and status != "failed":
                    self.send_json({"error": "只有回合失敗後可重置；移動與閃身步全程保持物理狀態"}, 409)
                    return
                commands.put_nowait((action, velocity, time.monotonic()))
            except (ValueError, UnicodeDecodeError, queue.Full):
                self.send_json({"error": "invalid or busy command"}, 400)
                return
            self.send_json({"accepted": True, "action": action})

        def log_message(self, *_args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/isaac/interactive_v4")
    parser.add_argument("--sidestep-duration", type=float, default=1.6)
    parser.add_argument("--validate-only", action="store_true", help="Print the gate without starting Isaac Sim")
    if "--validate-only" in sys.argv:
        args = parser.parse_args()
        gate = policy_gate(ROOT, args.checkpoint_dir)
        print(json.dumps(gate, indent=2, ensure_ascii=False))
        raise SystemExit(0 if gate["ready"] else 1)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.sidestep_duration <= 0:
        parser.error("--sidestep-duration must be positive")

    commands = queue.Queue(maxsize=32)
    lock, stopping = threading.Lock(), threading.Event()
    state = {"controller_version": CONTROLLER_VERSION, "ready": False,
             "status": "starting", "skill": "interactive_v4", "mode": "stand",
             "backend": "isaaclab", "proxy": False, "sequence_active": False,
             "command": {"vx": 0., "vy": 0., "wz": 0.}, "frame": None,
             "duration": 3600., "error": None}
    signal.signal(signal.SIGTERM, lambda *_args: stopping.set())
    signal.signal(signal.SIGINT, lambda *_args: stopping.set())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_type(commands, state, lock))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Interactive v4 live API: http://127.0.0.1:{args.port}/state", flush=True)

    app, env = None, None
    try:
        startup_sources = runtime_sources(ROOT)
        launcher = AppLauncher(args)
        app = launcher.app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from leg_humanoid_lab.interactive_control import KeyboardController
        from leg_humanoid_lab.interactive_env import InteractiveLegEnv, InteractiveEnvCfg
        from leg_humanoid_lab.robot import JOINTS
        from runtime_compat import load_checkpoint, runner_cfg

        class LiveInteractiveEnv(InteractiveLegEnv):
            # DirectRLEnv normally resets terminated environments inside step().
            # Suppress that reset so the browser sees the actual terminal frame.
            _manual_reset_allowed = True

            def _reset_idx(self, env_ids):
                if self._manual_reset_allowed:
                    super()._reset_idx(env_ids)

        cfg = InteractiveEnvCfg()
        cfg.scene.num_envs = 1
        cfg.training_mode = False
        cfg.episode_length_s = 3600.
        cfg.sim.device = args.device
        cfg.seed = 42
        env = LiveInteractiveEnv(cfg)
        env.set_manual_control("stand", [0., 0., 0.], 0.)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        env._manual_reset_allowed = False
        obs = wrapped.get_observations()
        controller = KeyboardController(sidestep_duration=args.sidestep_duration)
        runner, policy, loaded_hash = None, None, None
        paused, failed, last_gate_time, retry_at = True, False, -math.inf, 0.
        startup = True

        def frame():
            data = env.robot.data
            result = {"time": float(env.episode_length_buf[0] * env.step_dt),
                      "q": data.joint_pos[0, env.jids].cpu().tolist(),
                      "root": (data.root_pos_w[0] - env.scene.env_origins[0]).cpu().tolist(),
                      "quaternion": data.root_quat_w[0].cpu().tolist()}
            if not all(math.isfinite(value) for key in ("q", "root", "quaternion") for value in result[key]):
                return None
            return result

        with lock:
            state.update(frame=frame(), joint_names=list(JOINTS), status="paused")

        with torch.inference_mode():
            while not stopping.is_set() and app.is_running():
                now = time.monotonic()
                if now - last_gate_time >= 1.:
                    last_gate_time = now
                    gate = policy_gate(ROOT, args.checkpoint_dir)
                    if gate["ready"] and runtime_sources(ROOT) != startup_sources:
                        gate = {"ready": False, "error": "v4 模擬程式或模型素材已變更，請重新啟動即時服務。"}
                    if gate["ready"] and gate["checkpoint_sha256"] != loaded_hash and now >= retry_at:
                        try:
                            if runner is None:
                                runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
                            load_checkpoint(runner, gate["checkpoint"], load_optimizer=False, load_iteration=False)
                            policy = runner.get_inference_policy(device=env.device)
                            loaded_hash = gate["checkpoint_sha256"]
                            obs = wrapped.get_observations()
                            # A boot with an already validated model starts in stand.
                            paused = not startup or failed
                        except Exception as error:
                            gate = {"ready": False, "error": f"v4 checkpoint 載入失敗：{error}"}
                            retry_at = now + 10.
                    ready = gate["ready"] and policy is not None and gate.get("checkpoint_sha256") == loaded_hash
                    if not ready:
                        paused = True
                        controller.stop(now)
                    with lock:
                        state.update(ready=ready, checkpoint_sha256=loaded_hash,
                                     confirmation_episodes=gate.get("confirmation_episodes", 0),
                                     status="failed" if failed else "paused" if paused else "running",
                                     error=state.get("error") if failed else gate.get("error") if not ready else None)
                    startup = False

                while True:
                    try:
                        action, velocity, received_at = commands.get_nowait()
                    except queue.Empty:
                        break
                    if action == "stop":
                        controller.stop(now)
                    elif action == "pause":
                        controller.stop(now)
                        paused = True
                    elif state["ready"]:
                        if action == "restart" and failed:
                            controller = KeyboardController(sidestep_duration=args.sidestep_duration)
                            env.set_manual_control("stand", [0., 0., 0.], 0.)
                            env._manual_reset_allowed = True
                            try:
                                obs = wrapped.reset()[0]
                            finally:
                                env._manual_reset_allowed = False
                            failed, paused = False, False
                            with lock:
                                state.update(error=None, frame=frame())
                        elif not failed:
                            if action == "velocity":
                                controller.velocity(*velocity, received_at)
                            elif action == "sidestep_sequence":
                                controller.sequence(now)
                            elif action == "play":
                                paused = False

                control = controller.tick(env.step_dt if not paused and not failed else 0., now)
                with lock:
                    state.update(mode=control["mode"], sequence_active=control["sequence_active"],
                                 command=dict(zip(("vx", "vy", "wz"), control["command"])),
                                 status="failed" if failed else "paused" if paused else "running")
                if state["ready"] and not paused and not failed:
                    env.set_manual_control(control["mode"], control["command"], control["phase"])
                    # Refresh reference-conditioned observations before inference.
                    obs = wrapped.get_observations()
                    obs, _reward, done, _info = wrapped.step(policy(obs))
                    terminal_frame = frame()
                    with lock:
                        if terminal_frame is not None:
                            state["frame"] = terminal_frame
                    if bool(done[0]) or terminal_frame is None:
                        failed, paused = True, True
                        controller.stop(now)
                        with lock:
                            state.update(status="failed", sequence_active=False,
                                         command={"vx": 0., "vy": 0., "wz": 0.},
                                         error="物理回合終止；保留最後姿態，請按「回到起點」重置。")
                elif not args.headless:
                    env.sim.render()
                stopping.wait(max(0., env.step_dt - (time.monotonic() - now)))
    except BaseException as error:
        import traceback
        traceback.print_exc()
        with lock:
            state.update(ready=False, status="failed", error=f"v4 即時服務初始化失敗：{error}")
        sys.stderr.flush()
        raise
    finally:
        server.shutdown()
        server.server_close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
