"""V6 upper-body acceptance layered over the unchanged native leg suites.

The proxy samples the real environment after each physics/control step. It
never changes a command, action, reset, lower-body metric, or lower-body gate.
Only an articulation's first episode contributes; reset poses cannot improve
the measured return of a hand or conceal a lost torso.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .interactive_evaluation import CASE_NAMES, evaluate_interactive
from .lift_evaluation import STAGE_CASES, evaluate_lift


DEFAULT_GATES = {
    "chest_peak_max_deg": 25.,
    "chest_peak_p95_max_deg": 20.,
    "chest_rms_max_deg": 8.,
    "prep_chest_peak_max_deg": 15.,
    "prep_chest_peak_p95_max_deg": 15.,
    "dynamic_chest_angular_xy_rms_max_rad_s": .7,
    "gesture_arm_reference_rmse_max_rad": .08,
    "early_return_chest_peak_max_deg": 8.,
    "early_return_arm_reference_rmse_max_rad": .08,
    "early_return_hand_distance_rms_max_m": .02,
    "early_return_arm_speed_rms_max_rad_s": .6,
    "tail_hand_distance_rms_max_m": .015,
    "tail_chest_angular_xy_rms_max_rad_s": .35,
    "tail_arm_speed_rms_max_rad_s": .4,
    "observed_episode_rate_min": .875,
}

SCALAR_MEASUREMENTS = (
    "chest_tilt_rad", "chest_angular_xy_rad_s", "chest_angular_xyz_rad_s",
    "arm_reference_mse_rad2", "arm_velocity_error_mse_rad2_s2",
    "waist_reference_mse_rad2", "arm_joint_speed_mse_rad2_s2",
    "waist_joint_speed_mse_rad2_s2",
)


def _window_masks(stage, names, count, time, maneuver, device):
    """Use the existing suites' case ordering and published command timeline."""
    case_ids = torch.arange(count, device=device) % len(names)
    dynamic = torch.zeros(count, device=device, dtype=torch.bool)
    gesture = torch.zeros_like(dynamic)
    early = torch.zeros_like(dynamic)
    directions = torch.zeros((count, 2), device=device, dtype=torch.bool)
    returns = torch.zeros_like(directions)
    for case_id, name in enumerate(names):
        selected = case_ids == case_id
        if stage == 0:
            dynamic |= selected & (name == "inplace") & (1. <= time < 8.)
            continue
        if name in ("left", "right", "pair", "sequence"):
            pair = name in ("pair", "sequence")
            end = 1. + maneuver * (2 if pair else 1)
            dynamic |= selected & (1. <= time < end + 1.)
            gesture |= selected & (1. <= time < end + 1.)
            early |= selected & (end + .8 <= time < end + 1.5)
            for side, side_name in enumerate(("left", "right")):
                if not pair and name != side_name:
                    continue
                start = 1. + maneuver if pair and side == 1 else 1.
                directions[:, side] |= selected & (start <= time < start + maneuver)
                # The final 250 ms must follow the moving reference back. A
                # pair's left arm is assessed before the right gesture starts;
                # stand-FK hand distance is never used during that next gesture.
                returns[:, side] |= selected & (start + maneuver - .25 <= time < start + maneuver)
            if stage == 2:
                dynamic |= selected & ((0. <= time < .5) | (6.5 <= time < 8.))
        elif stage == 2 and name != "stand":
            dynamic |= selected & (1. <= time < 8.)
    return {"all": torch.ones(count, device=device, dtype=torch.bool),
            "dynamic": dynamic, "gesture": gesture, "early_return": early,
            "tail": torch.full((count,), 18. <= time < 20., device=device),
            "directions": directions, "direction_return": returns}


def _safe_number(value):
    result = float(value)
    return result if math.isfinite(result) else None


class UpperBodyTracker:
    """Device-side first-episode measurements; independent of Isaac imports."""

    def __init__(self, env, count, stage):
        self.env, self.count, self.stage = env, count, stage
        self.names = STAGE_CASES[stage] if stage in STAGE_CASES else CASE_NAMES
        self.device, self.dt = env.device, float(env.step_dt)
        self.steps = round(20. / self.dt)
        self.maneuver = float(env.library.get("sidestep_duration_s", 1.6))
        self.active = torch.ones(count, device=self.device, dtype=torch.bool)
        self.invalid = torch.zeros_like(self.active)
        self.failed = torch.zeros_like(self.active)
        self.observed_steps = torch.zeros(count, device=self.device)
        self.previous_length = None
        self.errors = []
        self.step_index = 0
        self.sums, self.peaks, self.counts, self.expected = {}, {}, {}, {}
        for window in ("all", "dynamic", "gesture", "early_return", "tail"):
            self.counts[window] = torch.zeros(count, device=self.device)
            self.expected[window] = torch.zeros(count, device=self.device)
            self.peaks[window] = torch.zeros(count, device=self.device)
            for metric in (*SCALAR_MEASUREMENTS, "hands_return_distance_m"):
                shape = (count, 2) if metric == "hands_return_distance_m" else (count,)
                self.sums[window, metric] = torch.zeros(shape, device=self.device)
        self.direction_count = torch.zeros((count, 2), device=self.device)
        self.direction_expected = torch.zeros_like(self.direction_count)
        self.direction_peak = torch.zeros_like(self.direction_count)
        self.direction_tilt_sq = torch.zeros_like(self.direction_count)
        self.direction_angular_sq = torch.zeros_like(self.direction_count)
        self.direction_angular_xyz_sq = torch.zeros_like(self.direction_count)
        self.direction_arm_mse = torch.zeros_like(self.direction_count)
        self.direction_return_count = torch.zeros_like(self.direction_count)
        self.direction_return_expected = torch.zeros_like(self.direction_count)
        self.direction_return_arm_mse = torch.zeros_like(self.direction_count)
        # Coverage is determined against the entire prescribed suite, even if
        # all environments die and the lower evaluator exits its loop early.
        for step in range(self.steps):
            masks = self.masks(step * self.dt)
            for window in self.expected:
                self.expected[window] += masks[window]
            self.direction_expected += masks["directions"]
            self.direction_return_expected += masks["direction_return"]

    def masks(self, time):
        return _window_masks(self.stage, self.names, self.count, time,
                             self.maneuver, self.device)

    def reset(self):
        if self.step_index:
            raise RuntimeError("Upper-body acceptance cannot reset an episode mid-suite")
        if hasattr(self.env, "episode_length_buf"):
            self.previous_length = self.env.episode_length_buf[:self.count].clone()

    def sample(self, done, actions):
        env, count = self.env, self.count
        masks = self.masks(self.step_index * self.dt)
        self.step_index += 1
        before = self.active.clone()
        done = torch.as_tensor(done, device=self.device).reshape(-1)[:count].bool()
        implicit_reset = torch.zeros_like(before)
        if self.previous_length is not None:
            length = env.episode_length_buf[:count]
            implicit_reset = (self.previous_length > 0) & (length <= self.previous_length)
            self.previous_length = length.clone()
        failed = env.reset_terminated[:count].bool()
        completed_horizon = self.step_index * self.dt >= 20. - self.dt
        self.failed |= before & ((implicit_reset & ~done) | (done & (failed | ~torch.full_like(done, completed_horizon))))
        self.active &= ~(done | implicit_reset)
        if not bool(self.active.any()):
            return
        try:
            measured = env.upper_body_measurements()
            values = {}
            for name in (*SCALAR_MEASUREMENTS, "hands_return_distance_m"):
                value = measured[name]
                expected_shape = (env.num_envs, 2) if name == "hands_return_distance_m" else (env.num_envs,)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(f"{name} must be a tensor with shape {expected_shape}")
                values[name] = value[:count].to(device=self.device)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            self.errors.append(f"step {self.step_index}: {type(error).__name__}: {error}")
            self.invalid |= self.active
            self.active.zero_()
            return
        finite = torch.isfinite(actions[:count]).reshape(count, -1).all(-1)
        for name, value in values.items():
            valid = torch.isfinite(value) & (value >= 0)
            finite &= valid.all(-1) if value.ndim == 2 else valid
        self.invalid |= self.active & ~finite
        self.active &= finite
        valid = self.active
        self.observed_steps += valid
        # Nonfinite values must not poison other environments' sums. Their
        # episode is permanently invalid and has a separate failing gate.
        values = {key: value.nan_to_num() for key, value in values.items()}
        tilt = values["chest_tilt_rad"]
        for window in self.counts:
            selected = valid & masks[window]
            self.counts[window] += selected
            self.peaks[window] = torch.maximum(self.peaks[window], torch.where(selected, tilt, 0.))
            for metric, value in values.items():
                # Tilt, angular speed and per-hand distance are scalar
                # magnitudes. The other inputs already contain joint means
                # of squared errors or squared native joint speeds.
                squared = value.square() if metric in ("chest_tilt_rad", "chest_angular_xy_rad_s", "chest_angular_xyz_rad_s", "hands_return_distance_m") else value
                mask = selected[:, None] if value.ndim == 2 else selected
                self.sums[window, metric] += squared * mask
        selected = valid[:, None] & masks["directions"]
        self.direction_count += selected
        self.direction_peak = torch.maximum(self.direction_peak, torch.where(selected, tilt[:, None], 0.))
        self.direction_tilt_sq += tilt[:, None].square() * selected
        self.direction_angular_sq += values["chest_angular_xy_rad_s"][:, None].square() * selected
        self.direction_angular_xyz_sq += values["chest_angular_xyz_rad_s"][:, None].square() * selected
        self.direction_arm_mse += values["arm_reference_mse_rad2"][:, None] * selected
        selected = valid[:, None] & masks["direction_return"]
        self.direction_return_count += selected
        self.direction_return_arm_mse += values["arm_reference_mse_rad2"][:, None] * selected

    def _window(self, window, index):
        count = self.counts[window][index]
        expected = self.expected[window][index]
        observed = bool(expected > 0 and count >= max(1., float(expected) - 1.)
                        and not self.invalid[index] and not self.failed[index])
        result = {"samples": int(count), "expected_samples": int(expected), "observed": observed,
                  "chest_peak_deg": _safe_number(torch.rad2deg(self.peaks[window][index])) if count > 0 else None}
        for metric in (*SCALAR_MEASUREMENTS, "hands_return_distance_m"):
            value = (self.sums[window, metric][index] / count.clamp(min=1)).sqrt()
            key = {
                "chest_tilt_rad": "chest_rms_deg",
                "chest_angular_xy_rad_s": "chest_angular_xy_rms_rad_s",
                "chest_angular_xyz_rad_s": "chest_angular_xyz_rms_rad_s",
                "arm_reference_mse_rad2": "arm_reference_rmse_rad",
                "arm_velocity_error_mse_rad2_s2": "arm_velocity_error_rmse_rad_s",
                "waist_reference_mse_rad2": "waist_reference_rmse_rad",
                "arm_joint_speed_mse_rad2_s2": "arm_speed_rms_rad_s",
                "waist_joint_speed_mse_rad2_s2": "waist_speed_rms_rad_s",
                "hands_return_distance_m": "hands_return_distance_rms_m_per_hand",
            }[metric]
            if metric == "chest_tilt_rad":
                value = torch.rad2deg(value)
            result[key] = ([_safe_number(item) for item in value] if value.ndim else _safe_number(value)) if count > 0 else None
        return result

    def report(self):
        rows = []
        for index in range(self.count):
            item = {"env_index": index, "case": self.names[index % len(self.names)],
                    "finite_measurements": not bool(self.invalid[index]),
                    "first_episode_failed": bool(self.failed[index]),
                    "first_episode_samples": int(self.observed_steps[index]),
                    **{window: self._window(window, index) for window in self.counts},
                    "directions": {}}
            for side, name in enumerate(("left", "right")):
                expected = self.direction_expected[index, side]
                if expected <= 0:
                    continue
                count = self.direction_count[index, side]
                returns = self.direction_return_count[index, side]
                clean = not bool(self.invalid[index] | self.failed[index])
                item["directions"][name] = {
                    "samples": int(count), "expected_samples": int(expected),
                    "observed": bool(count >= max(1., float(expected) - 1.) and clean),
                    "chest_peak_deg": _safe_number(torch.rad2deg(self.direction_peak[index, side])) if count > 0 else None,
                    "chest_rms_deg": _safe_number(torch.rad2deg((self.direction_tilt_sq[index, side] / count.clamp(min=1)).sqrt())) if count > 0 else None,
                    "chest_angular_xy_rms_rad_s": _safe_number((self.direction_angular_sq[index, side] / count.clamp(min=1)).sqrt()) if count > 0 else None,
                    "chest_angular_xyz_rms_rad_s": _safe_number((self.direction_angular_xyz_sq[index, side] / count.clamp(min=1)).sqrt()) if count > 0 else None,
                    "arm_reference_rmse_rad": _safe_number((self.direction_arm_mse[index, side] / count.clamp(min=1)).sqrt()) if count > 0 else None,
                    "return_samples": int(returns),
                    "return_expected_samples": int(self.direction_return_expected[index, side]),
                    "return_observed": bool(returns >= max(1., float(self.direction_return_expected[index, side]) - 1.) and clean),
                    "return_arm_reference_rmse_rad": _safe_number((self.direction_return_arm_mse[index, side] / returns.clamp(min=1)).sqrt()) if returns > 0 else None,
                }
            rows.append(item)
        cases = {}
        for name in self.names:
            selected = [item for item in rows if item["case"] == name]
            cases[name] = {"episodes": len(selected), "finite_measurements": all(item["finite_measurements"] for item in selected), "directions": {}}
            for window in self.counts:
                cases[name][window] = _aggregate_windows([item[window] for item in selected])
            for side in ("left", "right"):
                events = [item["directions"][side] for item in selected if side in item["directions"]]
                if events:
                    cases[name]["directions"][side] = _aggregate_windows(events)
        return {"cases": cases, "per_episode": rows, "measurement_errors": self.errors,
                "step_calls": self.step_index, "expected_step_calls": self.steps}


def _aggregate_windows(items):
    """Equal episode weighting; absent values remain absent, never zero."""
    result = {"episodes": len(items), "observed_episodes": sum(bool(item["observed"]) for item in items),
              "observed_episode_rate": sum(bool(item["observed"]) for item in items) / len(items),
              "samples": sum(item["samples"] for item in items),
              "expected_samples": sum(item["expected_samples"] for item in items)}
    for key in ("chest_rms_deg", "chest_angular_xy_rms_rad_s", "chest_angular_xyz_rms_rad_s", "arm_reference_rmse_rad",
                "arm_velocity_error_rmse_rad_s", "waist_reference_rmse_rad", "arm_speed_rms_rad_s",
                "waist_speed_rms_rad_s", "return_arm_reference_rmse_rad"):
        values = [item[key] for item in items if item.get(key) is not None]
        result[key] = math.sqrt(sum(value * value for value in values) / len(values)) if values else None
    peaks = [item["chest_peak_deg"] for item in items if item.get("chest_peak_deg") is not None]
    result["chest_peak_max_deg"] = max(peaks) if peaks else None
    result["chest_peak_p95_deg"] = float(torch.quantile(torch.tensor(peaks, dtype=torch.float64), .95)) if peaks else None
    hands = [item["hands_return_distance_rms_m_per_hand"] for item in items if item.get("hands_return_distance_rms_m_per_hand") is not None]
    result["hands_return_distance_rms_m_per_hand"] = [math.sqrt(sum(value[side] ** 2 for value in hands) / len(hands)) for side in (0, 1)] if hands else None
    if any("return_observed" in item for item in items):
        result["return_observed_episode_rate"] = sum(bool(item.get("return_observed")) for item in items) / len(items)
    return result


def check_upper_body_gates(report):
    """Require every case and direction, finite data, and complete windows."""
    stage = report.get("stage")
    checks = {"stage.valid": type(stage) is int and stage in (0, 1, 2)}
    if not checks["stage.valid"]:
        return checks
    gates = report.get("gates", DEFAULT_GATES)
    names = STAGE_CASES[stage] if stage in STAGE_CASES else CASE_NAMES

    def bound(key, value, gate, minimum=False):
        threshold = gates.get(gate)
        finite = lambda number: isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
        checks[key] = bool(finite(value) and finite(threshold) and (value >= threshold if minimum else value <= threshold))

    checks["measurement_schema"] = not bool(report.get("measurement_errors", ["missing"]))
    for name in names:
        item = report.get("cases", {}).get(name, {})
        checks[f"{name}.finite_measurements"] = item.get("finite_measurements") is True
        checks[f"{name}.sample_count"] = isinstance(item.get("episodes"), int) and item["episodes"] > 0
        for window in ("all", "tail"):
            bound(f"{name}.{window}.coverage", item.get(window, {}).get("observed_episode_rate"), "observed_episode_rate_min", True)
        all_window = item.get("all", {})
        bound(f"{name}.chest_peak", all_window.get("chest_peak_max_deg"), "prep_chest_peak_max_deg" if stage == 0 else "chest_peak_max_deg")
        bound(f"{name}.chest_peak_p95", all_window.get("chest_peak_p95_deg"), "prep_chest_peak_p95_max_deg" if stage == 0 else "chest_peak_p95_max_deg")
        bound(f"{name}.chest_rms", all_window.get("chest_rms_deg"), "chest_rms_max_deg")
        tail = item.get("tail", {})
        bound(f"{name}.tail.angular_xy", tail.get("chest_angular_xy_rms_rad_s"), "tail_chest_angular_xy_rms_max_rad_s")
        bound(f"{name}.tail.arm_speed", tail.get("arm_speed_rms_rad_s"), "tail_arm_speed_rms_max_rad_s")
        for side in (0, 1):
            hands = tail.get("hands_return_distance_rms_m_per_hand")
            bound(f"{name}.tail.hand_{side}", hands[side] if isinstance(hands, list) and len(hands) == 2 else None, "tail_hand_distance_rms_max_m")
        if name != "stand":
            dynamic = item.get("dynamic", {})
            bound(f"{name}.dynamic.coverage", dynamic.get("observed_episode_rate"), "observed_episode_rate_min", True)
            bound(f"{name}.dynamic.angular_xy", dynamic.get("chest_angular_xy_rms_rad_s"), "dynamic_chest_angular_xy_rms_max_rad_s")
        if name in ("left", "right", "pair", "sequence"):
            gesture = item.get("gesture", {})
            bound(f"{name}.gesture.coverage", gesture.get("observed_episode_rate"), "observed_episode_rate_min", True)
            bound(f"{name}.gesture.arm_reference", gesture.get("arm_reference_rmse_rad"), "gesture_arm_reference_rmse_max_rad")
            early = item.get("early_return", {})
            bound(f"{name}.early_return.coverage", early.get("observed_episode_rate"), "observed_episode_rate_min", True)
            bound(f"{name}.early_return.chest_peak", early.get("chest_peak_max_deg"), "early_return_chest_peak_max_deg")
            bound(f"{name}.early_return.arm_reference", early.get("arm_reference_rmse_rad"), "early_return_arm_reference_rmse_max_rad")
            bound(f"{name}.early_return.arm_speed", early.get("arm_speed_rms_rad_s"), "early_return_arm_speed_rms_max_rad_s")
            for side in (0, 1):
                hands = early.get("hands_return_distance_rms_m_per_hand")
                bound(f"{name}.early_return.hand_{side}", hands[side] if isinstance(hands, list) and len(hands) == 2 else None, "early_return_hand_distance_rms_max_m")
            for side in (("left", "right") if name in ("pair", "sequence") else (name,)):
                event = item.get("directions", {}).get(side, {})
                prefix = f"{name}.{side}"
                bound(f"{prefix}.coverage", event.get("observed_episode_rate"), "observed_episode_rate_min", True)
                bound(f"{prefix}.chest_peak", event.get("chest_peak_max_deg"), "chest_peak_max_deg")
                bound(f"{prefix}.chest_peak_p95", event.get("chest_peak_p95_deg"), "chest_peak_p95_max_deg")
                bound(f"{prefix}.chest_rms", event.get("chest_rms_deg"), "chest_rms_max_deg")
                bound(f"{prefix}.angular_xy", event.get("chest_angular_xy_rms_rad_s"), "dynamic_chest_angular_xy_rms_max_rad_s")
                bound(f"{prefix}.arm_reference", event.get("arm_reference_rmse_rad"), "gesture_arm_reference_rmse_max_rad")
                bound(f"{prefix}.controlled_return.coverage", event.get("return_observed_episode_rate"), "observed_episode_rate_min", True)
                bound(f"{prefix}.controlled_return.arm_reference", event.get("return_arm_reference_rmse_rad"), "gesture_arm_reference_rmse_max_rad")
    return checks


class _TrackingWrapper:
    def __init__(self, wrapped, tracker):
        self.wrapped, self.tracker = wrapped, tracker

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def reset(self, *args, **kwargs):
        result = self.wrapped.reset(*args, **kwargs)
        self.tracker.reset()
        return result

    def step(self, actions):
        result = self.wrapped.step(actions)
        self.tracker.sample(result[2], actions)
        return result


@torch.inference_mode()
def evaluate_upper_body(runner, wrapped, env, episodes=32, seed=70000, stage=0):
    """Run an original lower suite once and append independent upper gates."""
    if type(stage) is not int or stage not in (0, 1, 2):
        raise ValueError("Upper-body acceptance supports stages 0, 1 and 2 only")
    names = STAGE_CASES[stage] if stage in STAGE_CASES else CASE_NAMES
    if episodes < len(names) or episodes % len(names) or env.num_envs < episodes:
        raise ValueError("Upper-body acceptance needs balanced native case coverage")
    tracker = UpperBodyTracker(env, episodes, stage)
    tracking = _TrackingWrapper(wrapped, tracker)
    lower = (evaluate_lift(runner, tracking, env, episodes=episodes, seed=seed, stage=stage)
             if stage in STAGE_CASES else evaluate_interactive(runner, tracking, env, episodes=episodes, seed=seed))
    profile = getattr(env, "upper_body_profile", {})
    declared = profile.get("evaluation_gates", {})
    unknown = set(declared) - set(DEFAULT_GATES)
    if unknown:
        raise ValueError(f"Unknown upper-body gate names: {sorted(unknown)}")
    gates = {**DEFAULT_GATES, **declared}
    for key, value in gates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid upper-body threshold {key}")
    upper = tracker.report()
    upper.update(schema=1, stage=stage, gates=gates,
                 evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 control_sha256=getattr(env, "upper_body_control_sha256", None),
                 control_contract=getattr(env, "upper_body_contract", profile.get("contract")),
                 measurement={
                     "timing": "After native wrapped.step; exclude done, implicit reset and all subsequent first-episode samples",
                     "chest_tilt": "Native torso up-axis inclination from world vertical; yaw invariant",
                     "angular_speed": "Native torso local roll/pitch angular speed norm; yaw excluded; differs from older XYZ diagnostic",
                     "arm_reference": "RMS joint position error against the current moving reference; gesture and recover only",
                     "controlled_return": "Arm reference tracking during the final 250 ms of each left/right gesture; not a stand-hand comparison during the next gesture",
                     "early_return": "Both hands versus stand-FK in pelvis coordinates, gesture end +0.8 to +1.5 s, before interactive WASD resume",
                     "tail": "18–20 s standing; RMS Euclidean distance per hand, not per-coordinate RMS",
                     "coverage": "At least all prescribed window samples except one terminal frame per counted episode; failed first episodes cannot count as observed",
                     "aggregation": "Equal episode mean-square weighting; peaks include partial first episodes; P95 over per-episode peaks; missing values remain null"})
    upper_checks = check_upper_body_gates(upper)
    original_checks = dict(lower["checks"])
    checks = {**original_checks, **{f"upper_body.{key}": value for key, value in upper_checks.items()}}
    lower.update(stage=stage, upper_body=upper, upper_body_checks=upper_checks,
                 lower_body_passed=bool(lower["passed"]), lower_body_checks=original_checks,
                 checks=checks, failed_gates=[key for key, value in checks.items() if not value],
                 passed=all(checks.values()), score=sum(checks.values()) / len(checks),
                 upper_body_control_contract=upper["control_contract"],
                 upper_body_control_sha256=upper["control_sha256"],
                 upper_body_evaluator_sha256=upper["evaluator_sha256"])
    json.dumps(lower, allow_nan=False)
    return lower
