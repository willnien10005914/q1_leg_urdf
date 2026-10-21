#!/usr/bin/env python3
"""Compare native checkpoint controls without training or policy publication.

Default batch: four variants x four stage-1 cases x two replicates = 32 envs.
Only joint position/velocity targets are changed. Root state, contacts, motor
limits, reset/done logic and evaluator are the unchanged LiftLegEnv versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback


VARIANTS = ("baseline", "protected_support", "swing_velocity_ff", "protected_support_velocity_ff")
BODY_IK_VARIANTS = ("baseline", "swing_velocity_ff", "body_ik", "body_ik_velocity_ff")
PITCH_CAPS = {"hip_Pitch": .15, "knee_Pitch": .10, "ankle_Pitch": .15}
OTHER_CAPS = {"hip_Roll": .15, "hip_Yaw": .15, "ankle_Roll": .10}
LEG_PARTS = ("hip_Pitch", "hip_Roll", "hip_Yaw", "knee_Pitch", "ankle_Pitch", "ankle_Roll")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fresh_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def summarize(rows, names):
    """Aggregate only real first-episode measurements, retaining missing stops."""
    def mean(values):
        return sum(values) / len(values) if values else 1e6

    def ratio(hits, trials):
        return sum(hits) / max(1., sum(trials))

    cases = {}
    for name in names:
        selected = [row for row in rows if row["case"] == name]
        observed = [row for row in selected if row["stop_observed"]]
        case = {
            "episodes": len(selected), "completed_episodes": sum(row["survived"] for row in selected),
            "survival_rate": mean([float(row["survived"]) for row in selected]),
            "duration_s": mean([row["duration_s"] for row in selected]),
            "upright": mean([row["upright"] for row in selected]),
            "stop_observed_rate": len(observed) / max(1, len(selected)),
            "final_stand_speed_m_s": mean([row["final_stand_speed_m_s"] for row in observed]),
            "stance_slip_m_s": mean([row["stance_slip_m_s"] for row in selected]),
        }
        for key in ("foot_peak_clearance_m_per_foot", "unloaded_peak_clearance_m_per_foot"):
            case[key] = [mean([row[key][foot] for row in selected]) for foot in (0, 1)]
        case["high_swing_samples_per_foot"] = [sum(row["high_swing_samples_per_foot"][foot] for row in selected) for foot in (0, 1)]
        case["foot_lift_success_rate_per_foot"] = [ratio(
            [row["foot_lift_success_rate_per_foot"][foot] * row["high_swing_samples_per_foot"][foot] for row in selected],
            [row["high_swing_samples_per_foot"][foot] for row in selected]) for foot in (0, 1)]
        if name != "stand":
            case["directions"] = {}
            for side_name in (("left", "right") if name == "pair" else (name,)):
                side = 0 if side_name == "left" else 1
                event = {}
                for role, foot in (("lead", side), ("trailing", 1-side)):
                    trials = [row["direction_high_swing_samples_per_foot"][side][foot] for row in selected]
                    event[f"{role}_peak_clearance_m"] = mean([row["direction_peak_clearance_m_per_foot"][side][foot] for row in selected])
                    event[f"{role}_unloaded_peak_clearance_m"] = mean([row["direction_unloaded_peak_clearance_m_per_foot"][side][foot] for row in selected])
                    event[f"{role}_high_swing_samples"] = sum(trials)
                    event[f"{role}_lift_success_rate"] = ratio(
                        [row["direction_lift_success_rate_per_foot"][side][foot] * trial for row, trial in zip(selected, trials)], trials)
                    event[f"{role}_unloaded_rate"] = ratio(
                        [row["direction_unloaded_swing_samples_per_foot"][side][foot] for row in selected], trials)
                case["directions"][side_name] = event
        cases[name] = case
    return {"stage": 1, "episodes": len(rows), "cases": cases,
            "survival_rate": mean([float(row["survived"]) for row in rows])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=481031)
    parser.add_argument("--protect-support-roll-yaw", action="store_true",
                        help="Also bound support hip roll/yaw and ankle roll; default protects pitch joints only")
    parser.add_argument("--body-ik-variants", action="store_true",
                        help="Compare baseline, swing velocity FF, body-aware swing IK and their combination; no support caps")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if args.body_ik_variants and args.protect_support_roll_yaw:
        parser.error("Support caps are not part of --body-ik-variants")
    args.output = args.output.resolve()
    target_path = args.output.with_name(args.output.stem + "_targets.npz")
    error_path = args.output.with_name(args.output.stem + "_error.json")
    if any(path.exists() for path in (args.output, target_path, error_path)):
        parser.error("Refusing to overwrite an existing diagnostic, target trace or error report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = env = None
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from leg_humanoid_lab.lift_env import LiftEnvCfg, LiftLegEnv, lift_contract
        from leg_humanoid_lab.lift_objective import control_targets
        from leg_humanoid_lab.lift_evaluation import STAGE_CASES, check_lift_gates, evaluate_lift
        from leg_humanoid_lab.robot import JOINTS, ROOT
        from runtime_compat import load_checkpoint, runner_cfg

        variants = BODY_IK_VARIANTS if args.body_ik_variants else VARIANTS
        support_caps = {} if args.body_ik_variants else {**PITCH_CAPS, **(OTHER_CAPS if args.protect_support_roll_yaw else {})}
        body_ik = None
        if args.body_ik_variants:
            from leg_humanoid_lab.swing_body_ik import BodyAwareSwingIK
            body_ik = BodyAwareSwingIK(iterations=3)
        block = len(STAGE_CASES[1]) * args.replicates
        count = len(variants) * block

        class AblationEnv(LiftLegEnv):
            def _pre_physics_step(self, actions):
                previous_ref = self.refq.clone()
                previous_mode = self.mode.clone()
                super()._pre_physics_step(actions)
                variant = torch.arange(self.num_envs, device=self.device) // block
                protected = ((variant == 1) | (variant == 3)) & (not args.body_ik_variants)
                feedforward = ((variant == 1) | (variant == 3) if args.body_ik_variants
                               else (variant == 2) | (variant == 3))
                swing = ~self.reference_contact.bool() & (self.reference_height > .008)
                scales = self.action_scales.clone()
                for foot, side in enumerate(("left", "right")):
                    support = protected & self.reference_contact[:, foot] & swing[:, 1-foot]
                    for part, cap in support_caps.items():
                        index = JOINTS.index(f"{side}_{part}_joint")
                        scales[:, index] = torch.where(support, cap, scales[:, index])
                # Rebuild before joint-limit clipping, keeping the exact shared
                # IMU correction and swing caps. Only residual amplitude changes.
                adjusted = self.actions * (scales / self.action_scales)
                data = self.robot.data
                control_reference = self.refq.clone()
                if body_ik is not None:
                    selected = (variant == 2) | (variant == 3)
                    height = data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
                    control_reference[selected] = body_ik(
                        self.refq[selected], data.root_quat_w[selected], height[selected],
                        self.reference_height[selected], self.reference_contact[selected])
                # Keep self.refq nominal: the checkpoint sees the exact original
                # reference-conditioned observations in every ablation variant.
                self.target, _ = control_targets(
                    control_reference, self.stand_q, adjusted, data.projected_gravity_b,
                    data.root_ang_vel_b, self.reference_contact, self.reference_height,
                    self.lower, self.upper)
                self.action_scales = scales
                if args.body_ik_variants:
                    previous_ref = getattr(self, "diagnostic_previous_control_reference", previous_ref)
                qdot = ((control_reference - previous_ref) / self.step_dt).clamp(-10., 10.)
                self.diagnostic_previous_control_reference = control_reference.clone()
                # Zero on initial/reset or mode-switch frames; subsequent values
                # always use adjacent references, never accumulated differences.
                continuous = (self.mode == previous_mode) & (self.episode_length_buf > 0)
                self.diagnostic_velocity_target = torch.zeros_like(self.target)
                for foot, side in enumerate(("left", "right")):
                    ids = [JOINTS.index(f"{side}_{part}_joint") for part in LEG_PARTS]
                    enabled = feedforward & swing[:, foot] & continuous
                    self.diagnostic_velocity_target[:, ids] = torch.where(enabled[:, None], qdot[:, ids], 0.)
                self.diagnostic_command = {
                    "reference_q": control_reference.clone(),
                    "nominal_reference_q": self.refq.clone(),
                    "compensated_reference_q": control_reference.clone(),
                    "residual_actions": self.actions.clone(),
                    "residual_scales_rad": scales.clone(), "mode": self.mode.clone(),
                    "reference_contact": self.reference_contact.clone(),
                    "reference_height_m": self.reference_height.clone(),
                }

            def _apply_action(self):
                super()._apply_action()
                self.robot.set_joint_velocity_target(self.diagnostic_velocity_target, joint_ids=self.jids)
                # Retain the buffers at the actual setter call. An automatic
                # reset after a terminal physics step may overwrite live data.
                self.diagnostic_position_applied = self.robot.data.joint_pos_target[:, self.jids].clone()
                self.diagnostic_velocity_applied = self.robot.data.joint_vel_target[:, self.jids].clone()

        cfg = LiftEnvCfg()
        cfg.scene.num_envs = count
        cfg.training_mode = False
        cfg.curriculum_stage = 1
        cfg.episode_length_s = 20.5
        cfg.seed = args.seed
        cfg.sim.device = args.device
        env = AblationEnv(cfg)
        if body_ik is not None:
            body_ik = body_ik.to(env.device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
        contract = lift_contract()
        saved = json.loads((args.checkpoint.parent / "contract.json").read_text())
        keys = ("robot", "schema", "urdf_sha256", "gait_sha256", "control_sha256", "control_profile",
                "motor", "joint_names", "action_size", "observation_size", "proxy")
        if any(saved.get(key) != contract.get(key) for key in keys):
            raise ValueError("Checkpoint does not match the unmodified native v5 control contract")
        load_checkpoint(runner, args.checkpoint, load_optimizer=False, load_iteration=True)

        class FirstEpisodeTrace:
            def __init__(self, transport):
                self.transport = transport
                self.records = []

            def __getattr__(self, name):
                return getattr(self.transport, name)

            def reset(self):
                result = self.transport.reset()
                self.active = torch.ones(count, device=env.device, dtype=torch.bool)
                self.length = env.episode_length_buf.clone()
                self.unloaded_peak = torch.zeros((count, 2), device=env.device)
                self.side_peak = torch.zeros((count, 2, 2), device=env.device)
                self.side_trials = self.side_peak.clone()
                self.side_unloaded = self.side_peak.clone()
                return result

            def step(self, actions):
                result = self.transport.step(actions)
                done = result[2].reshape(-1).bool()
                implicit_reset = (self.length > 0) & (env.episode_length_buf <= self.length)
                self.length = env.episode_length_buf.clone()
                height, speed, loaded, _ = env.foot_step_metrics()
                data = env.robot.data
                finite = (torch.isfinite(height).all(-1) & torch.isfinite(data.root_pos_w).all(-1)
                          & torch.isfinite(data.root_lin_vel_b).all(-1)
                          & torch.isfinite(data.projected_gravity_b).all(-1)
                          & torch.isfinite(speed).all(-1)
                          & torch.isfinite(actions).all(-1))
                self.active &= ~done & ~implicit_reset & finite
                command = env.diagnostic_command
                mode = command["mode"]
                moving = self.active & ((mode == 2) | (mode == 3))
                unloaded_height = torch.where(moving[:, None] & ~loaded, height.clamp_min(0.), 0.)
                self.unloaded_peak = torch.maximum(self.unloaded_peak, unloaded_height)
                for side, mode_id in enumerate((2, 3)):
                    relevant = self.active & (mode == mode_id)
                    self.side_peak[:, side] = torch.maximum(self.side_peak[:, side], torch.where(relevant[:, None], unloaded_height, 0.))
                    high = relevant[:, None] & ~command["reference_contact"] & (command["reference_height_m"] > .022)
                    self.side_trials[:, side] += high
                    self.side_unloaded[:, side] += high & ~loaded
                record = {**command, "first_episode_valid": self.active,
                          # These are the articulation buffers written by the
                          # setters and held throughout all decimation substeps.
                          "joint_position_target_rad": env.diagnostic_position_applied,
                          "joint_velocity_target_rad_s": env.diagnostic_velocity_applied,
                          "actual_joint_q_rad": data.joint_pos[:, env.jids],
                          "actual_sole_height_m": height, "actual_loaded": loaded,
                          "foot_vertical_force_n": env.contacts.data.net_forces_w[:, env.feet, 2],
                          "root_position_w_m": data.root_pos_w,
                          "root_quaternion_wxyz": data.root_quat_w}
                self.records.append({key: value.detach().cpu().numpy().copy() for key, value in record.items()})
                return result

        trace = FirstEpisodeTrace(wrapped)
        evaluated = evaluate_lift(runner, trace, env, episodes=count, seed=args.seed, stage=1)
        to_list = lambda value: value.detach().cpu().tolist()
        for row in evaluated["per_episode"]:
            index = row["env_index"]
            row.update(variant=variants[index // block],
                       unloaded_peak_clearance_m_per_foot=to_list(trace.unloaded_peak[index]),
                       direction_unloaded_peak_clearance_m_per_foot=to_list(trace.side_peak[index]),
                       direction_high_swing_samples_per_foot=to_list(trace.side_trials[index]),
                       direction_unloaded_swing_samples_per_foot=to_list(trace.side_unloaded[index]))
        summaries = {}
        for name in variants:
            rows = [row for row in evaluated["per_episode"] if row["variant"] == name]
            summary = summarize(rows, STAGE_CASES[1])
            checks = check_lift_gates(summary)
            summary.update(diagnostic_gate_preview=checks, diagnostic_metrics_gate_passed=all(checks.values()),
                           diagnostic_score=sum(checks.values()) / len(checks), per_episode=rows)
            summaries[name] = summary
        arrays = {key: np.stack([row[key] for row in trace.records]) for key in trace.records[0]}
        arrays["time_s"] = np.arange(1, len(trace.records) + 1) * env.step_dt
        arrays["variant_index"] = np.arange(count) // block
        arrays["case_index"] = np.arange(count) % len(STAGE_CASES[1])
        with target_path.open("xb") as output:
            np.savez_compressed(output, **arrays)
        overrides = {"support_caps_rad": support_caps, "support_condition": "planned contact with opposite active swing >.008 m",
                     "velocity_ff": "Adjacent blended reference q finite difference / control dt; swing leg joints only; +/-10 rad/s; zero on reset/mode change",
                     "variants": list(variants), "all_contact_residual_cap_rad": .35,
                     "reference_q_semantics": "Joint reference actually supplied to control_targets and used for adjacent-reference FF; excludes residual and IMU corrections",
                     "nominal_reference_q_semantics": "Original LiftEnv blended reference; remains in actor observations",
                     "compensated_reference_q_semantics": "Body-aware joint reference in selected IK variants; otherwise equals nominal_reference_q",
                     "unchanged": "IMU feedback, swing residual caps, motor/effort/velocity limits, root dynamics, contacts, collisions and episode termination"}
        sources = [Path(__file__), ROOT/"config/lift_control_v5.json", ROOT/contract["gait_file"]]
        sources += [ROOT/"source/leg_humanoid_lab/leg_humanoid_lab"/name for name in
                    ("lift_env.py", "lift_objective.py", "lift_evaluation.py", "interactive_env.py", "isaac_env.py", "robot.py")]
        if args.body_ik_variants:
            ik_file = ROOT/"source/leg_humanoid_lab/leg_humanoid_lab/swing_body_ik.py"
            sources.append(ik_file)
            overrides["body_ik"] = {
                "iterations": 3, "module_sha256": digest(ik_file),
                "inputs": "Nominal joint reference, measured root wxyz quaternion and ground-relative height, nominal target sole height/contact",
                "effect": "Swing joint references only; no root state write; original actor observations retained",
            }
            overrides["velocity_ff"] = "Adjacent compensated reference q finite difference / control dt; swing leg joints only; +/-10 rad/s; zero on reset/mode change"
        report = {
            "schema": 1, "kind": "native_checkpoint_control_ablation", "backend": "isaaclab", "proxy": False,
            "ready": False, "production_acceptance": False, "diagnostic_control_override": True,
            "scope": "Fixed-policy diagnostic only; control overrides are not trained and cannot promote or unlock keyboard controls",
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
            "completed_iterations": int(runner.current_learning_iteration), "contract": contract,
            "seed": args.seed, "envs": count, "replicates_per_case_variant": args.replicates,
            "stage": 1, "duration_s": 20., "first_episode_only": True,
            "exclude_done_and_all_post_reset_samples": True, "reset_between_transitions": False,
            "control_override": overrides,
            "control_override_sha256": hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest(),
            "source_sha256": {str(path.resolve().relative_to(ROOT.resolve())): digest(path) for path in sources},
            "target_trace": str(target_path), "target_trace_sha256": digest(target_path),
            "target_trace_steps": len(trace.records), "control_dt_s": env.step_dt,
            "physics_decimation": cfg.decimation, "joint_names": list(JOINTS),
            "joint_ids_in_articulation": list(env.jids),
            "overall": evaluated, "variants": summaries,
        }
        write_fresh_json(args.output, report)
        brief = {"output": str(args.output), "target_trace": str(target_path),
                 "checkpoint_sha256": report["checkpoint_sha256"],
                 "variants": {name: {key: value for key, value in summary.items()
                              if key not in ("per_episode", "diagnostic_gate_preview")}
                              for name, summary in summaries.items()}}
        print("LIFT_CONTROL_ABLATION " + json.dumps(brief), flush=True)
    except BaseException as error:
        traceback.print_exc()
        failure = {"kind": "native_checkpoint_control_ablation_error", "error": str(error), "type": type(error).__name__,
                   "production_acceptance": False, "ready": False}
        if not error_path.exists():
            write_fresh_json(error_path, failure)
        print("LIFT_CONTROL_ABLATION_ERROR " + json.dumps(failure), flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
