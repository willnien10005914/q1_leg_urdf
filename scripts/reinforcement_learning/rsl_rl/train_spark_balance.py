#!/usr/bin/env python3
"""Isolated Spark PhysX balance worker with strict contracts and update budget.

Argument, budget, and contract helpers are importable without Isaac Sim. Training
outputs never replace the selected V6 checkpoint or publish an accepted policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

ROBOT = "q1_leg_spark_balance_v1"
MAX_NEW_ITERATIONS = 15000
NATIVE_KEYS = (
    "control_sha256", "base_control_sha256", "control_profile", "gait_sha256",
    "urdf_sha256", "joint_names", "action_size", "observation_size",
    "observation_order", "limits_lower", "limits_upper", "default",
    "root_height", "motor", "proxy",
)
SOURCE_FILES = (
    "scripts/reinforcement_learning/rsl_rl/train_spark_balance.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/spark_balance_env.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/balance_env.py",
    "source/leg_humanoid_lab/leg_humanoid_lab/balance_objective.py",
)
RUNTIME_PACKAGES = ("torch", "rsl-rl-lib", "isaaclab", "isaacsim")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_parser(app_launcher=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--completed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--full-gates", action="store_true")
    parser.add_argument("--max-new-iterations", type=int, default=MAX_NEW_ITERATIONS)
    if app_launcher is not None:
        app_launcher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--headless", action="store_true")
    return parser


def validate_args(args):
    if args.envs <= 0 or args.completed < 0 or args.max_new_iterations <= 0:
        raise ValueError("envs and max-new-iterations must be positive; completed cannot be negative")
    if args.initialize and args.evaluate:
        raise ValueError("initialize and evaluate are mutually exclusive")
    if args.initialize and args.completed:
        raise ValueError("A new Spark experiment starts at zero; inherited V6 updates are separate")
    if args.full_gates and not args.evaluate:
        raise ValueError("full-gates requires evaluate")
    if args.evaluate:
        if args.envs < 4 or args.envs % 4:
            raise ValueError("Evaluation requires at least four envs and envs divisible by four")
    elif args.iterations <= 0 or args.completed + args.iterations > args.max_new_iterations:
        raise ValueError("Training iterations must be positive and fit max-new-iterations")
    if args.completed > args.max_new_iterations:
        raise ValueError("completed exceeds max-new-iterations")
    if any((args.output / name).exists() for name in ("training_result.json", "evaluation.json", "resume.pt")):
        raise ValueError("Output already contains a saved result; use a new chunk output directory")


def source_hashes(root):
    return {name: sha(Path(root) / name) for name in SOURCE_FILES}


def runtime_versions():
    return {name: version(name) for name in RUNTIME_PACKAGES}


def validate_contract(saved, expected, *, initialize=False, evaluate=False,
                      sources=None, runtime=None, max_new_iterations=MAX_NEW_ITERATIONS):
    for key in NATIVE_KEYS:
        if key not in saved or key not in expected or saved[key] != expected[key]:
            raise ValueError("Native control/interface mismatch: " + key)
    if initialize:
        if saved.get("robot") != "q1_leg_upper_body_v6":
            raise ValueError("Initialization requires the published native V6 actor")
        return
    if evaluate and saved.get("robot") == "q1_leg_upper_body_v6":
        return
    if saved.get("robot") != ROBOT:
        raise ValueError("Resume/evaluation requires a Spark balance checkpoint or explicit V6 baseline")
    if saved.get("balance_source_sha256") != sources:
        raise ValueError("Spark source fingerprint changed; create a new experiment")
    if saved.get("runtime_versions") != runtime:
        raise ValueError("Spark runtime versions changed; resume is forbidden")
    if saved.get("max_new_iterations") != max_new_iterations:
        raise ValueError("Spark update budget changed")
    if saved.get("backend") != "physx":
        raise ValueError("Spark checkpoint backend must remain physx")


def audit_summary(active, cases, gesture_peak, gesture_seen):
    """Summarize first-episode survival and four actual both-foot gestures."""
    sequence = [i for i, case in enumerate(cases) if case < 2]
    valid = [all(gesture_seen[i]) and all(
        all(math.isfinite(height) and height >= .01 for height in feet)
        for feet in gesture_peak[i]
    ) for i in sequence]
    completed = [ok and active[i] for i, ok in zip(sequence, valid)]
    return {
        "survival_rate": sum(active) / len(active),
        "case_survival": {str(case): sum(active[i] for i, value in enumerate(cases) if value == case)
            / sum(value == case for value in cases) for case in range(4)},
        "four_step_survival": sum(active[i] for i in sequence) / len(sequence),
        "completed_four_step_rate": sum(completed) / len(sequence),
        "gesture_both_feet_unloaded_threshold_m": .01,
    }


def audit(runner, wrapped, env, args):
    import torch

    wrapped.reset()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cases = torch.arange(env.num_envs, device=env.device) % 4
    modes = torch.zeros_like(cases)
    phases = torch.zeros(env.num_envs, device=env.device)
    first_failure = []
    modes_seen = set()
    capture_max = torch.zeros_like(phases)
    peak = torch.zeros((env.num_envs, 2), device=env.device)
    chest_peak = phases.clone()
    gesture_peak = torch.zeros((env.num_envs, 4, 2), device=env.device)
    gesture_seen = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.bool)
    policy = runner.get_inference_policy(device=env.device)
    duration = float(env.library["sidestep_duration_s"])
    pair_duration = .5 + 2 * duration + 1.
    steps = round(20. / env.step_dt)
    with torch.inference_mode():
        for step in range(steps):
            t = step * env.step_dt
            modes.zero_()
            phases.zero_()
            gesture_index = None
            for pair_id, start in enumerate((1., 1. + pair_duration)):
                local = t - start - .5
                if 0 <= local < 2 * duration:
                    which = int(local >= duration)
                    gesture_index = 2 * pair_id + which
                    selected = cases < 2
                    right = (cases == 1) if which == 0 else (cases == 0)
                    modes[selected] = torch.where(right[selected], 3, 2)
                    phases[selected] = (local % duration) / duration
                elif 2 * duration <= local < 2 * duration + 1.:
                    modes[cases < 2] = 4
            walk = (cases == 2) & (t >= 1.) & (t < 7.)
            modes[walk] = 1
            commands = torch.zeros((env.num_envs, 3), device=env.device)
            commands[walk, 0] = .15
            env.set_manual_control(modes, commands, phases)
            prior = active.clone()
            _, _, done, _ = wrapped.step(policy(wrapped.get_observations()))
            for idx in torch.where(prior & done.bool())[0].cpu().tolist():
                first_failure.append({"env": idx, "case": int(cases[idx]),
                    "time_s": (step + 1) * env.step_dt, **env.last_termination.get(idx, {})})
            # Snapshot taken by _get_dones before DirectRLEnv performs resets.
            sample = env.audit_sample
            capture_max = torch.maximum(capture_max, torch.where(prior, sample["capture_error"], 0.))
            unloaded = prior[:, None] & ~sample["loaded"]
            observed_height = torch.where(unloaded, sample["height"], 0.)
            peak = torch.maximum(peak, observed_height)
            chest_peak = torch.maximum(chest_peak, torch.where(prior, sample["chest_tilt"], 0.))
            if gesture_index is not None:
                selected = prior & (cases < 2) & ((sample["mode"] == 2) | (sample["mode"] == 3))
                gesture_seen[:, gesture_index] |= selected
                gesture_peak[:, gesture_index] = torch.maximum(gesture_peak[:, gesture_index],
                    torch.where(selected[:, None], observed_height, 0.))
            modes_seen.update(sample["mode"][prior].unique().cpu().tolist())
            active &= ~done.bool()
    report = audit_summary(active.cpu().tolist(), cases.cpu().tolist(),
        gesture_peak.cpu().tolist(), gesture_seen.cpu().tolist())
    return {**report, "diagnostic": True, "accepted": False,
        "episodes": env.num_envs, "duration_s": steps * env.step_dt, "seed": args.seed,
        "case_names": ["LRLR", "RLRL", "forward_walk_stop", "stand"],
        "first_failure_events": first_failure, "modes_seen": sorted(modes_seen),
        "peak_unloaded_sole_m_mean_per_foot": peak.mean(0).cpu().tolist(),
        "gesture_peak_unloaded_sole_m_per_episode": gesture_peak.cpu().tolist(),
        "gesture_seen_per_episode": gesture_seen.cpu().tolist(),
        "survived_per_episode": active.cpu().tolist(),
        "chest_peak_deg_max": float(chest_peak.max() * 180 / torch.pi),
        "capture_excess_m_max": float(capture_max.max()),
        "no_post_reset_measurements": True, "checkpoint_sha256": sha(args.checkpoint)}


def initialize_actor(runner, checkpoint):
    import torch
    from runtime_compat import checkpoint_for_runtime, policy_module, reset_noise_std

    payload = torch.load(checkpoint_for_runtime(checkpoint), map_location=runner.device, weights_only=False)
    if "actor_state_dict" in payload:
        source = {key.removeprefix("mlp."): value for key, value in payload["actor_state_dict"].items()
                  if key.startswith("mlp.")}
    else:
        source = {key.removeprefix("actor."): value for key, value in payload["model_state_dict"].items()
                  if key.startswith("actor.")}
    model = policy_module(runner)
    current = model.state_dict()
    actor_keys = [key for key in current if key.startswith(("mlp.", "actor."))]
    if len(actor_keys) != 6 or set(source) != {key.split(".", 1)[1] for key in actor_keys}:
        raise ValueError("Incomplete actor MLP; all six native network tensors are required")
    for key in actor_keys:
        incoming = source[key.split(".", 1)[1]]
        if incoming.shape != current[key].shape or not torch.isfinite(incoming).all():
            raise ValueError("Actor shape/finite mismatch: " + key)
        current[key] = incoming
    model.load_state_dict(current)
    reset_noise_std(runner, .04)


def main():
    from isaaclab.app import AppLauncher

    parser = build_parser(AppLauncher)
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    app = None
    env = None
    try:
        app = AppLauncher(args).app
        # Validate contracts before building the robot scene or loading weights.
        from leg_humanoid_lab.upper_body_env import upper_body_contract
        expected = upper_body_contract()
        saved = json.loads((args.checkpoint.parent / "contract.json").read_text())
        root = Path(__file__).resolve().parents[3]
        sources = source_hashes(root)
        runtime = runtime_versions()
        validate_contract(saved, expected, initialize=args.initialize, evaluate=args.evaluate,
            sources=sources, runtime=runtime, max_new_iterations=args.max_new_iterations)

        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.spark_balance_env import SparkBalanceEnv
        from leg_humanoid_lab.upper_body_env import UpperBodyEnvCfg
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from runtime_compat import runner_cfg, load_checkpoint

        class MeasuredSparkBalanceEnv(SparkBalanceEnv):
            def _get_dones(self):
                failed, timed_out = super()._get_dones()
                self.last_termination = {}
                if self.cfg.training_mode:
                    return failed, timed_out
                height, _, loaded, _ = self.foot_step_metrics()
                error, _, _ = self.balance_measurements()
                self.audit_sample = {"height": height.clone(), "loaded": loaded.clone(),
                    "capture_error": error.clone(), "mode": self.mode.clone(),
                    "chest_tilt": self.upper_body_measurements()["chest_tilt_rad"].clone()}
                data = self.robot.data
                for idx in torch.where(failed | timed_out)[0].cpu().tolist():
                    forces = self.contacts.data.net_forces_w[idx]
                    self.last_termination[idx] = {"terminated": bool(failed[idx]),
                        "timeout": bool(timed_out[idx]),
                        "root_height_m": float(data.root_pos_w[idx, 2] - self.scene.env_origins[idx, 2]),
                        "upright_cosine": float(-data.projected_gravity_b[idx, 2]),
                        "nonfoot_contacts": [{"body": self.contacts.body_names[body],
                            "force_n": float(forces[body].norm())} for body in self.other_bodies
                            if float(forces[body].norm()) > 25.]}
                return failed, timed_out

        cfg = UpperBodyEnvCfg()
        cfg.scene.num_envs = args.envs
        cfg.training_mode = not args.evaluate
        cfg.curriculum_stage = 2
        cfg.reward_revision = 2
        cfg.seed = args.seed
        cfg.episode_length_s = 20.5 if args.evaluate else 20.
        cfg.sim.device = args.device
        env = MeasuredSparkBalanceEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        agent = PPOCfg()
        agent.seed = args.seed
        agent.algorithm.learning_rate = 5e-5
        agent.algorithm.schedule = "fixed"
        agent.algorithm.entropy_coef = .0005
        agent.algorithm.num_mini_batches = 8
        agent.save_interval = 100
        runner = OnPolicyRunner(wrapped, runner_cfg(agent),
            log_dir=None if args.evaluate else str(args.output), device=args.device)
        if args.initialize:
            initialize_actor(runner, args.checkpoint)
        else:
            load_checkpoint(runner, args.checkpoint, load_optimizer=not args.evaluate,
                load_iteration=not args.evaluate)
            if not args.evaluate and runner.current_learning_iteration != args.completed:
                raise ValueError("Checkpoint iteration differs from completed counter")
        if args.evaluate:
            if args.full_gates:
                from leg_humanoid_lab.upper_body_evaluation import evaluate_upper_body
                report = evaluate_upper_body(runner, wrapped, env, episodes=args.envs,
                    seed=args.seed, stage=2)
                report.update(checkpoint_sha256=sha(args.checkpoint), balance_experiment=True,
                    accepted=False, backend="physx")
            else:
                report = audit(runner, wrapped, env, args)
                report.update(backend="physx")
            write_json(args.output / "evaluation.json", report)
            print("SPARK_BALANCE_EVALUATION " + json.dumps(report), flush=True)
        else:
            spark_contract = {**expected, "robot": ROBOT, "schema": 1,
                "backend": "physx", "max_new_iterations": args.max_new_iterations,
                "balance_source_sha256": sources, "runtime_versions": runtime,
                "balance_scope": "Native COM/contact costs; randomized LRLR/RLRL; not accepted"}
            write_json(args.output / "contract.json", spark_contract)
            runner.current_learning_iteration = args.completed
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            completed = args.completed + args.iterations
            runner.current_learning_iteration = completed
            runner.save(str(args.output / "resume.tmp.pt"), infos={
                "completed_iterations": completed, "prior_v6_updates": 20000,
                "experiment": ROBOT, "accepted": False})
            (args.output / "resume.tmp.pt").replace(args.output / "resume.pt")
            write_json(args.output / "training_result.json", {
                "completed_iterations": completed, "prior_v6_updates": 20000,
                "checkpoint_sha256": sha(args.output / "resume.pt"),
                "num_envs": args.envs, "accepted": False, "backend": "physx",
                "max_new_iterations": args.max_new_iterations})
    except BaseException as error:
        write_json(args.output / "worker_error.json", {"error": str(error)})
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
