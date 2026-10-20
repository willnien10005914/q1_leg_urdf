"""Native, finite command-and-sequence evaluation without publishing checkpoints.

Every environment contributes only its first episode. Auto-reset states are
excluded and all controller changes preserve the actual simulator state.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import torch

from .robot import ROOT, URDF


CASE_NAMES = ("stand", "w", "s", "a", "d", "yaw_left", "yaw_right", "sequence")
CASE_COMMANDS = ((0., 0., 0.), (.18, 0., 0.), (-.15, 0., 0.),
                 (0., .14, 0.), (0., -.14, 0.), (0., 0., .4),
                 (0., 0., -.4), (0., 0., 0.))
GATES = {
    "survival_rate_min": .875,
    "upright_min": .97,
    "stop_speed_m_s_max": .07,
    "walk_directional_displacement_m_min": .35,
    "walk_velocity_rmse_m_s_max": .12,
    "foot_clearance_m_min": .025,
    "foot_lift_success_min": .4,
    "yaw_displacement_rad_min": .5,
    "yaw_rmse_rad_s_max": .20,
    "lead_forward_m_min": .02,
    "lead_lateral_m_min": .025,
    "hand_cross_success_min": .5,
    "root_forward_m_min": .015,
    "root_lateral_m_min": .015,
    "push_proxy_success_min": .25,
    "stance_slip_m_s_max": .10,
}


def check_interactive_gates(report: dict) -> dict:
    """Return every required gate; missing/non-finite data always fails closed."""
    checks: dict[str, bool] = {}

    def finite(value: object) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    def minimum(key: str, value: object, threshold: float) -> None:
        checks[key] = finite(value) and value >= threshold

    def maximum(key: str, value: object, threshold: float) -> None:
        checks[key] = finite(value) and value < threshold

    cases = report.get("cases", {})
    for name in CASE_NAMES:
        case = cases.get(name, {})
        checks[f"{name}.sample_count"] = case.get("episodes", 0) >= 1
        minimum(f"{name}.survival", case.get("survival_rate"), GATES["survival_rate_min"])
        minimum(f"{name}.upright", case.get("upright"), GATES["upright_min"])
        maximum(f"{name}.stop", case.get("final_stand_speed_m_s"), GATES["stop_speed_m_s_max"])
        maximum(f"{name}.stance_slip", case.get("stance_slip_m_s"), GATES["stance_slip_m_s_max"])
        if name in ("w", "s", "a", "d", "yaw_left", "yaw_right"):
            if name in ("w", "s", "a", "d"):
                minimum(f"{name}.direction", case.get("directional_displacement_m"),
                        GATES["walk_directional_displacement_m_min"])
                maximum(f"{name}.velocity", case.get("velocity_rmse_m_s"),
                        GATES["walk_velocity_rmse_m_s_max"])
            else:
                minimum(f"{name}.yaw_direction", case.get("signed_yaw_displacement_rad"),
                        GATES["yaw_displacement_rad_min"])
                maximum(f"{name}.yaw_tracking", case.get("yaw_rmse_rad_s"),
                        GATES["yaw_rmse_rad_s_max"])
            for foot in (0, 1):
                peak = case.get("foot_peak_clearance_m_per_foot", [])
                rate = case.get("foot_lift_success_rate_per_foot", [])
                minimum(f"{name}.foot_{foot}.clearance", peak[foot] if len(peak) > foot else None,
                        GATES["foot_clearance_m_min"])
                minimum(f"{name}.foot_{foot}.lift", rate[foot] if len(rate) > foot else None,
                        GATES["foot_lift_success_min"])
        if name == "sequence":
            for side in ("left", "right"):
                event = case.get("directions", {}).get(side, {})
                for field, threshold in (
                    ("lead_forward_m", GATES["lead_forward_m_min"]),
                    ("lead_signed_lateral_m", GATES["lead_lateral_m_min"]),
                    ("lead_peak_clearance_m", GATES["foot_clearance_m_min"]),
                    ("lead_lift_success_rate", GATES["foot_lift_success_min"]),
                    ("trailing_peak_clearance_m", GATES["foot_clearance_m_min"]),
                    ("trailing_lift_success_rate", GATES["foot_lift_success_min"]),
                    ("hand_cross_success_rate", GATES["hand_cross_success_min"]),
                    ("root_forward_m", GATES["root_forward_m_min"]),
                    ("root_signed_lateral_m", GATES["root_lateral_m_min"]),
                    ("push_support_forward_proxy_rate", GATES["push_proxy_success_min"]),
                ):
                    minimum(f"sequence.{side}.{field}", event.get(field), threshold)
                minimum(f"sequence.{side}.landing", event.get("lead_landing_rate"), .875)
            checks["sequence.finite_completion"] = bool(case.get("finite_sequence_completed", False))
    return checks


def _quat_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into a body frame; quaternions use Isaac's wxyz."""
    unit = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scalar, axis = unit[..., :1], unit[..., 1:]
    return ((2 * scalar.square() - 1) * vector
            - 2 * scalar * torch.cross(axis, vector, dim=-1)
            + 2 * axis * (axis * vector).sum(-1, keepdim=True))


def _yaw(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def _tensor_json(value: torch.Tensor) -> object:
    return value.detach().cpu().tolist()


@torch.inference_mode()
def evaluate_interactive(runner, wrapped, env, episodes: int = 32, seed: int = 70000) -> dict:
    """Evaluate eight equally sampled cases for 20 s, with no intermediate reset.

    ``env`` must already be configured with an episode horizon >=20 s. The
    caller owns publication, checkpoint hashes, multi-seed promotion and video.
    """
    if episodes < len(CASE_NAMES) or episodes % len(CASE_NAMES):
        raise ValueError("Interactive evaluation needs a positive multiple of eight episodes")
    if env.num_envs < episodes:
        raise ValueError(f"Need at least {episodes} native environments, got {env.num_envs}")
    if env.cfg.allow_proxy:
        raise ValueError("Interactive acceptance requires the native CAD robot")
    duration = 20.
    step_dt = float(env.step_dt)
    if env.max_episode_length * step_dt < duration - step_dt:
        raise ValueError("Interactive evaluation requires a 20-second episode horizon")
    steps = round(duration / step_dt)
    device, count, total = env.device, episodes, env.num_envs
    sidestep_duration = float(env.library.get("sidestep_duration_s", 1.6))
    right_start, recover_start = 1. + sidestep_duration, 1. + 2 * sidestep_duration
    recover_end = recover_start + 1.
    if recover_end >= duration - 2:
        raise ValueError("Finite sequence must leave at least two seconds for standing")

    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)
    previous_training_mode = env.cfg.training_mode
    previous_manual_mode = getattr(env, "manual_mode", False)
    env.cfg.training_mode = False
    env.manual_mode = True
    ids = torch.arange(count, device=device)
    task = ids % len(CASE_NAMES)
    sequence = task == 7
    moving = (task >= 1) & (task <= 6)
    commands = torch.tensor(CASE_COMMANDS, device=device)[task]
    full_mode = torch.zeros(total, device=device, dtype=torch.long)
    full_commands = torch.zeros((total, 3), device=device)
    full_phase = torch.zeros(total, device=device)
    policy = runner.get_inference_policy(device=device)

    # All accumulators stay on the simulator device until the final report.
    active = torch.ones(count, device=device, dtype=torch.bool)
    survived = torch.zeros_like(active)
    samples = torch.zeros(count, device=device)
    elapsed = torch.zeros_like(samples)
    upright_sum = torch.zeros_like(samples)
    vel_error_sum = torch.zeros_like(samples)
    yaw_error_sum = torch.zeros_like(samples)
    moving_samples = torch.zeros_like(samples)
    directional_displacement = torch.zeros_like(samples)
    signed_yaw = torch.zeros_like(samples)
    stop_speed_sum = torch.zeros_like(samples)
    stop_samples = torch.zeros_like(samples)
    stop_drift = torch.zeros_like(samples)
    foot_peak = torch.zeros((count, 2), device=device)
    lift_hits = torch.zeros_like(foot_peak)
    lift_trials = torch.zeros_like(foot_peak)
    contact_hits = torch.zeros_like(samples)
    contact_trials = torch.zeros_like(samples)
    slip_sum = torch.zeros_like(samples)
    slip_square_sum = torch.zeros_like(samples)
    slip_trials = torch.zeros_like(samples)
    seq_peak = torch.zeros((count, 2), device=device)
    seq_hits = torch.zeros_like(seq_peak)
    seq_trials = torch.zeros_like(seq_peak)
    trailing_peak = torch.zeros_like(seq_peak)
    trailing_hits = torch.zeros_like(seq_peak)
    trailing_trials = torch.zeros_like(seq_peak)
    hand_hits = torch.zeros_like(seq_peak)
    hand_trials = torch.zeros_like(seq_peak)
    push_hits = torch.zeros_like(seq_peak)
    push_trials = torch.zeros_like(seq_peak)
    lead_displacement = torch.zeros((count, 2, 3), device=device)
    root_displacement = torch.zeros_like(lead_displacement)
    landed = torch.zeros((count, 2), device=device, dtype=torch.bool)
    lifted_before = torch.zeros_like(landed)
    anchor_set = torch.zeros_like(landed)
    anchor_foot = torch.zeros_like(lead_displacement)
    anchor_root = torch.zeros_like(lead_displacement)
    anchor_quat = torch.zeros((count, 2, 4), device=device)
    anchor_quat[..., 0] = 1.
    push_sensor_available = hasattr(env, "contacts") and hasattr(env, "feet")

    try:
        env.set_manual_control(full_mode, full_commands, full_phase)
        wrapped.reset()
        data = env.robot.data
        move_anchor_pos = data.root_pos_w[:count].clone()
        move_anchor_quat = data.root_quat_w[:count].clone()
        move_anchor_set = torch.zeros(count, device=device, dtype=torch.bool)
        stop_anchor = data.root_pos_w[:count, :2].clone()
        stop_anchor_set = torch.zeros_like(move_anchor_set)
        previous_yaw = _yaw(data.root_quat_w[:count]).clone()
        hand_bodies, _ = env.robot.find_bodies(
            ["right_gripper_link", "left_gripper_link"], preserve_order=True)
        if len(hand_bodies) != 2:
            raise ValueError("Native gripper bodies are required for cross-body acceptance")

        for step in range(steps):
            time = step * step_dt
            before = active.clone()
            full_mode.zero_()
            full_commands.zero_()
            full_phase.zero_()
            command_window = 1. <= time < 8.
            if command_window:
                full_mode[:count][moving] = 1
                full_commands[:count][moving] = commands[moving]
                full_phase[:count][moving] = ((time - 1.) / float(env.library.get("walk_cycle_s", .9))) % 1.
            side = 0 if 1. <= time < right_start else 1 if right_start <= time < recover_start else None
            phase = 0.
            if side is not None:
                phase = (time - (1. if side == 0 else right_start)) / sidestep_duration
                full_mode[:count][sequence] = 2 + side
                full_phase[:count][sequence] = phase
            elif recover_start <= time < recover_end:
                full_mode[:count][sequence] = 4
                full_phase[:count][sequence] = (time - recover_start) / (recover_end - recover_start)
            env.set_manual_control(full_mode, full_commands, full_phase)
            # The command is part of the policy observation for this action.
            obs = wrapped.get_observations()
            data = env.robot.data
            current_pos = data.root_pos_w[:count]
            current_quat = data.root_quat_w[:count]
            velocity = data.root_lin_vel_b[:count]
            finite = (torch.isfinite(current_pos).all(-1) & torch.isfinite(velocity).all(-1)
                      & torch.isfinite(current_quat).all(-1)
                      & torch.isfinite(data.projected_gravity_b[:count]).all(-1))
            active &= finite
            alive = active.float()
            upright_sum += (-data.projected_gravity_b[:count, 2]).nan_to_num() * alive
            samples += alive
            elapsed[active] = (step + 1) * step_dt

            if command_window:
                move_mask = active & moving
                new_anchor = move_mask & ~move_anchor_set
                move_anchor_pos[new_anchor] = current_pos[new_anchor]
                move_anchor_quat[new_anchor] = current_quat[new_anchor]
                move_anchor_set |= new_anchor
                vel_error_sum += (velocity[:, :2] - commands[:, :2]).square().sum(-1).nan_to_num() * move_mask
                yaw_error_sum += (data.root_ang_vel_b[:count, 2] - commands[:, 2]).square().nan_to_num() * move_mask
                moving_samples += move_mask
                delta = _quat_inverse(move_anchor_quat, current_pos - move_anchor_pos)
                direction = commands[:, :2] / commands[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                directional_displacement[move_mask] = (delta[:, :2] * direction).sum(-1)[move_mask]
                yaw_now = _yaw(current_quat)
                yaw_delta = torch.atan2(torch.sin(yaw_now - previous_yaw), torch.cos(yaw_now - previous_yaw))
                signed_yaw += yaw_delta.nan_to_num() * commands[:, 2].sign() * move_mask
            previous_yaw = _yaw(current_quat).clone()

            if time >= 18.:
                first_stop = active & ~stop_anchor_set
                stop_anchor[first_stop] = current_pos[first_stop, :2]
                stop_anchor_set |= first_stop
                stop_speed_sum += velocity[:, :2].norm(dim=-1).nan_to_num() * alive
                stop_samples += alive
                stop_drift[active] = (current_pos[:, :2] - stop_anchor).norm(dim=-1)[active]

            clearance, foot_speed, loaded, _ = env.foot_step_metrics()
            clearance = clearance[:count].nan_to_num()
            loaded = loaded[:count].bool()
            reference_height = env.reference_height[:count]
            reference_contact = env.reference_contact[:count].bool()
            relevant = active & ((moving & command_window) | (sequence & (side is not None)))
            foot_peak = torch.maximum(foot_peak, torch.where(relevant[:, None], clearance, 0.))
            peak_trial = relevant[:, None] & ~reference_contact & (reference_height >= .022)
            lift_trials += peak_trial
            lift_hits += peak_trial & (clearance >= .025) & ~loaded
            stance = active[:, None] & reference_contact & loaded
            slip_sum += (foot_speed[:count].nan_to_num() * stance).sum(-1)
            slip_square_sum += (foot_speed[:count].nan_to_num().square() * stance).sum(-1)
            slip_trials += stance.sum(-1)
            contact_hits += ((loaded == reference_contact) & active[:, None]).sum(-1)
            contact_trials += active.float() * 2

            if side is not None:
                event_mask = active & sequence
                lead, trailing, sign = (0, 1, 1.) if side == 0 else (1, 0, -1.)
                new_anchor = event_mask & ~anchor_set[:, side]
                anchor_foot[new_anchor, side] = data.body_pos_w[:count, env.foot_body_ids[lead]][new_anchor]
                anchor_root[new_anchor, side] = current_pos[new_anchor]
                anchor_quat[new_anchor, side] = current_quat[new_anchor]
                anchor_set[:, side] |= new_anchor
                seq_peak[:, side] = torch.maximum(seq_peak[:, side], torch.where(event_mask, clearance[:, lead], 0.))
                lead_trial = event_mask & ~reference_contact[:, lead] & (reference_height[:, lead] >= .022)
                seq_trials[:, side] += lead_trial
                seq_hits[:, side] += lead_trial & (clearance[:, lead] >= .025) & ~loaded[:, lead]
                if phase >= .60:
                    trailing_peak[:, side] = torch.maximum(trailing_peak[:, side],
                        torch.where(event_mask, clearance[:, trailing], 0.))
                    trailing_trial = (event_mask & ~reference_contact[:, trailing]
                                      & (reference_height[:, trailing] >= .022))
                    trailing_trials[:, side] += trailing_trial
                    trailing_hits[:, side] += trailing_trial & (clearance[:, trailing] >= .025) & ~loaded[:, trailing]
                lifted_before[:, side] |= event_mask & ~loaded[:, lead] & (clearance[:, lead] >= .012)
                if .35 <= phase <= .68:
                    touchdown = event_mask & lifted_before[:, side] & loaded[:, lead] & ~landed[:, side]
                    delta = _quat_inverse(anchor_quat[:, side],
                                          data.body_pos_w[:count, env.foot_body_ids[lead]] - anchor_foot[:, side])
                    lead_displacement[touchdown, side] = delta[touchdown]
                    landed[:, side] |= touchdown
                root_delta = _quat_inverse(anchor_quat[:, side], current_pos - anchor_root[:, side])
                root_displacement[event_mask, side] = root_delta[event_mask]
                if .25 <= phase <= .70:
                    hand = _quat_inverse(current_quat,
                                          data.body_pos_w[:count, hand_bodies[side]] - current_pos)
                    hand_trials[:, side] += event_mask
                    hand_hits[:, side] += event_mask & (hand[:, 0] > .04) & (hand[:, 1] * sign > .02)
                if .45 <= phase <= .65:
                    push_trials[:, side] += event_mask
                    if push_sensor_available:
                        vertical_force = env.contacts.data.net_forces_w[:count, env.feet[trailing], 2]
                        push_hits[:, side] += event_mask & (vertical_force > 60.) & (velocity[:, 0] > .02)

            _, _, done, _ = wrapped.step(policy(obs))
            ended = done[:count].bool() & before
            failed = env.reset_terminated[:count].bool()
            completed_horizon = (step + 1) * step_dt >= duration - step_dt
            survived[ended] = ~failed[ended] & completed_horizon
            active[ended] = False
            if not bool(active.any()):
                break
        survived[active] = True

        cases = {}
        for task_id, name in enumerate(CASE_NAMES):
            mask = task == task_id

            def mean(value: torch.Tensor) -> float:
                return float(value[mask].nan_to_num().mean().item())

            case = {
                "episodes": int(mask.sum().item()),
                "completed_episodes": int((survived & mask).sum().item()),
                "survival_rate": mean(survived.float()),
                "duration_s": mean(elapsed),
                "upright": mean(upright_sum / samples.clamp(min=1)),
                "velocity_rmse_m_s": mean((vel_error_sum / moving_samples.clamp(min=1)).sqrt()),
                "yaw_rmse_rad_s": mean((yaw_error_sum / moving_samples.clamp(min=1)).sqrt()),
                "directional_displacement_m": mean(directional_displacement),
                "signed_yaw_displacement_rad": mean(signed_yaw),
                "final_stand_speed_m_s": mean(torch.where(stop_samples > 0,
                    stop_speed_sum / stop_samples.clamp(min=1), torch.full_like(stop_samples, 1e6))),
                "final_stand_drift_m": mean(stop_drift),
                "foot_peak_clearance_m_per_foot": _tensor_json(foot_peak[mask].mean(0)),
                "foot_lift_success_rate_per_foot": _tensor_json(
                    lift_hits[mask].sum(0) / lift_trials[mask].sum(0).clamp(min=1)),
                "planned_contact_match": mean(contact_hits / contact_trials.clamp(min=1)),
                "loaded_foot_speed_m_s": mean(slip_sum / slip_trials.clamp(min=1)),
                "stance_slip_m_s": mean((slip_square_sum / slip_trials.clamp(min=1)).sqrt()),
            }
            if name == "sequence":
                case["finite_sequence_completed"] = bool(
                    (survived[mask] & anchor_set[mask].all(-1)).float().mean() >= GATES["survival_rate_min"])
                case["directions"] = {}
                for index, side_name in enumerate(("left", "right")):
                    sign = 1 if index == 0 else -1
                    case["directions"][side_name] = {
                        "lead_forward_m": mean(lead_displacement[:, index, 0]),
                        "lead_signed_lateral_m": mean(lead_displacement[:, index, 1] * sign),
                        "lead_landing_rate": mean(landed[:, index].float()),
                        "lead_peak_clearance_m": mean(seq_peak[:, index]),
                        "lead_lift_success_rate": float(seq_hits[mask, index].sum()
                            / seq_trials[mask, index].sum().clamp(min=1)),
                        "trailing_peak_clearance_m": mean(trailing_peak[:, index]),
                        "trailing_lift_success_rate": float(trailing_hits[mask, index].sum()
                            / trailing_trials[mask, index].sum().clamp(min=1)),
                        "hand_cross_success_rate": float(hand_hits[mask, index].sum()
                            / hand_trials[mask, index].sum().clamp(min=1)),
                        "push_support_forward_proxy_rate": float(push_hits[mask, index].sum()
                            / push_trials[mask, index].sum().clamp(min=1)),
                        "root_forward_m": mean(root_displacement[:, index, 0]),
                        "root_signed_lateral_m": mean(root_displacement[:, index, 1] * sign),
                    }
            cases[name] = case

        gait_file = ROOT / "config/v4_gaits.json"
        report = {
            "schema": 1, "skill": "interactive", "backend": "isaaclab", "proxy": False,
            "episodes": count, "evaluation_seed": seed, "duration_s": duration,
            "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
            "gait_sha256": hashlib.sha256(gait_file.read_bytes()).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment_sha256": hashlib.sha256(Path(__file__).with_name("interactive_env.py").read_bytes()).hexdigest(),
            "contract": env.library.get("contract"),
            "source": "native_physx_ppo_first_episode_finite_command_suite",
            "scope": "Native CAD simulation; real hardware unvalidated",
            "push_measurement": "Proxy: trailing-foot vertical force >60 N and body-forward velocity >0.02 m/s; no tangential impulse measured",
            "push_sensor_available": push_sensor_available,
            "hand_measurement": "Native right/left_gripper_link body origin in the current robot body frame",
            "reset_between_transitions": False,
            "gates": GATES.copy(), "cases": cases,
            "survival_rate": float(survived.float().mean()),
        }
        # Keep first-episode diagnostics so a failed seed/environment can be
        # reproduced; do not substitute a visually pleasing rollout for them.
        report["per_episode"] = []
        for index in range(count):
            item = {
                "env_index": index,
                "case": CASE_NAMES[int(task[index])],
                "survived": bool(survived[index]),
                "duration_s": float(elapsed[index]),
                "upright": float(upright_sum[index] / samples[index].clamp(min=1)),
                "directional_displacement_m": float(directional_displacement[index]),
                "signed_yaw_displacement_rad": float(signed_yaw[index]),
                "velocity_rmse_m_s": float((vel_error_sum[index] / moving_samples[index].clamp(min=1)).sqrt()),
                "foot_peak_clearance_m_per_foot": _tensor_json(foot_peak[index]),
                "foot_lift_success_rate_per_foot": _tensor_json(lift_hits[index] / lift_trials[index].clamp(min=1)),
                "stance_slip_m_s": float((slip_square_sum[index] / slip_trials[index].clamp(min=1)).sqrt()),
                "final_stand_speed_m_s": float(stop_speed_sum[index] / stop_samples[index].clamp(min=1))
                    if bool(stop_samples[index] > 0) else 1e6,
            }
            if bool(sequence[index]):
                item["lead_landing_body_delta_m"] = _tensor_json(lead_displacement[index])
                item["lead_landed"] = _tensor_json(landed[index])
                item["root_body_delta_m"] = _tensor_json(root_displacement[index])
                item["hand_cross_success_rate"] = _tensor_json(hand_hits[index] / hand_trials[index].clamp(min=1))
                item["trailing_peak_clearance_m"] = _tensor_json(trailing_peak[index])
                item["trailing_lift_success_rate"] = _tensor_json(trailing_hits[index] / trailing_trials[index].clamp(min=1))
            report["per_episode"].append(item)
        checks = check_interactive_gates(report)
        report["checks"] = checks
        report["failed_gates"] = [name for name, passed in checks.items() if not passed]
        report["score"] = sum(checks.values()) / max(len(checks), 1)
        report["passed"] = all(checks.values())
        # Ensure a caller can persist this with strict JSON before returning.
        json.dumps(report, allow_nan=False)
        return report
    finally:
        full_mode.zero_()
        full_commands.zero_()
        full_phase.zero_()
        env.set_manual_control(full_mode, full_commands, full_phase)
        env.manual_mode = previous_manual_mode
        env.cfg.training_mode = previous_training_mode
