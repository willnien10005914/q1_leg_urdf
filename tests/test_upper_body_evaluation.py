"""Acceptance transport regressions, not synthetic evidence of robot skill."""
import copy
import json
import math

import torch

from leg_humanoid_lab.upper_body_evaluation import (
    DEFAULT_GATES, UpperBodyTracker, check_upper_body_gates, evaluate_upper_body,
)
from leg_humanoid_lab.lift_evaluation import evaluate_lift
from test_lift_gates import transport_fixture


def fixture(*, modifier=None, reset_index=None, report_done=True, count=4):
    env, wrapped, runner, _, _ = transport_fixture(
        stage=1, count=count, reset_index=reset_index, report_done=report_done)
    env.upper_body_profile = {"evaluation_gates": DEFAULT_GATES.copy()}
    env.upper_body_contract = {"name": "synthetic_v6_transport"}
    env.upper_body_control_sha256 = "c" * 64

    def measure():
        # A pre-step collector would see an obviously failing value. Every
        # accepted sample must come from the completed control/physics step.
        tilt = math.radians(4.) if wrapped.steps else math.radians(80.)
        values = {
            "chest_tilt_rad": torch.full((count,), tilt),
            "chest_angular_xy_rad_s": torch.full((count,), .1),
            "chest_angular_xyz_rad_s": torch.full((count,), .15),
            "arm_reference_mse_rad2": torch.full((count,), .02 ** 2),
            "arm_velocity_error_mse_rad2_s2": torch.full((count,), .1 ** 2),
            "waist_reference_mse_rad2": torch.full((count,), .01 ** 2),
            "arm_joint_speed_mse_rad2_s2": torch.full((count,), .1 ** 2),
            "waist_joint_speed_mse_rad2_s2": torch.full((count,), .05 ** 2),
            "hands_return_distance_m": torch.tensor([[.012, .009]] * count),
        }
        if modifier:
            modifier(values, (wrapped.steps - 1) * env.step_dt, wrapped.steps)
        return values

    env.upper_body_measurements = measure
    return env, wrapped, runner


def test_layers_native_lower_checks_unchanged_and_collects_after_physics():
    native_env, native_wrapped, runner = fixture()
    original = evaluate_lift(runner, native_wrapped, native_env, episodes=4, stage=1)
    env, wrapped, runner = fixture()
    report = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1, seed=91)
    assert report["passed"] and report["lower_body_passed"]
    assert report["lower_body_checks"] == original["checks"]
    assert all(report["checks"][key] == value for key, value in original["checks"].items())
    assert report["cases"] == original["cases"]
    assert wrapped.resets == 1 and wrapped.steps == 1000
    assert env.cfg.training_mode is True and not hasattr(env, "manual_mode")
    pair = report["upper_body"]["cases"]["pair"]
    assert abs(pair["all"]["chest_peak_max_deg"] - 4.) < 1e-5
    assert abs(pair["gesture"]["arm_reference_rmse_rad"] - .02) < 1e-5
    assert abs(pair["tail"]["hands_return_distance_rms_m_per_hand"][0] - .012) < 1e-5
    assert pair["early_return"]["observed_episode_rate"] == 1.
    assert set(pair["directions"]) == {"left", "right"}
    assert report["upper_body_control_sha256"] == "c" * 64
    assert len(report["upper_body_evaluator_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_late_standing_cannot_conceal_loose_immediate_hand_return():
    def loose(values, time, step):
        if 5. <= time < 5.7:  # pair gesture end4.2 +.8…1.5 s
            values["hands_return_distance_m"][3] = .08
            values["arm_reference_mse_rad2"][3] = .25 ** 2
            values["arm_joint_speed_mse_rad2_s2"][3] = 1.2 ** 2

    env, wrapped, runner = fixture(modifier=loose)
    report = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1)
    assert report["lower_body_passed"] and not report["passed"]
    checks = report["upper_body_checks"]
    assert not checks["pair.early_return.hand_0"]
    assert not checks["pair.early_return.arm_reference"]
    assert not checks["pair.early_return.arm_speed"]
    assert checks["pair.tail.hand_0"] and checks["pair.tail.arm_speed"]
    assert checks["pair.left.arm_reference"] and checks["pair.right.arm_reference"]


def test_each_direction_has_its_own_controlled_return_gate():
    def bad_right(values, time, step):
        if 3.95 <= time < 4.2:
            values["arm_reference_mse_rad2"][3] = .4 ** 2

    env, wrapped, runner = fixture(modifier=bad_right)
    report = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1)
    checks = report["upper_body_checks"]
    assert checks["pair.left.controlled_return.arm_reference"]
    assert not checks["pair.right.controlled_return.arm_reference"]
    assert checks["pair.early_return.arm_reference"]
    assert report["lower_body_passed"] and not report["passed"]


def test_intended_yaw_is_reported_separately_without_failing_roll_pitch_gate():
    def turning(values, time, step):
        values["chest_angular_xyz_rad_s"][:] = 1.5

    env, wrapped, runner = fixture(modifier=turning)
    report = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1)
    assert report["passed"]
    right = report["upper_body"]["cases"]["pair"]["directions"]["right"]
    assert abs(right["chest_angular_xy_rms_rad_s"] - .1) < 1e-5
    assert abs(right["chest_angular_xyz_rms_rad_s"] - 1.5) < 1e-5


def test_done_and_unreported_reset_poses_never_contribute_or_become_zero_error():
    for report_done in (True, False):
        def attractive_reset(values, time, step):
            if step >= 10:
                values["chest_tilt_rad"][1] = 1.5
                values["hands_return_distance_m"][1] = 0.

        env, wrapped, runner = fixture(modifier=attractive_reset, reset_index=1, report_done=report_done)
        report = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1)
        row = report["upper_body"]["per_episode"][1]
        assert row["first_episode_samples"] == 9
        assert row["first_episode_failed"]
        assert abs(row["all"]["chest_peak_deg"] - 4.) < 1e-5
        assert row["tail"]["hands_return_distance_rms_m_per_hand"] is None
        assert not report["upper_body_checks"]["left.all.coverage"]
        assert not report["upper_body_checks"]["left.tail.hand_0"]
        assert not report["passed"]
        json.dumps(report, allow_nan=False)


def test_nan_or_missing_measurements_fail_closed_without_poisoning_json():
    for missing in (False, True):
        def invalid(values, time, step):
            if step == 80:
                if missing:
                    del values["chest_tilt_rad"]
                else:
                    values["chest_tilt_rad"][3] = float("nan")

        env, wrapped, runner = fixture(modifier=invalid)
        report = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1)
        assert not report["passed"] and report["lower_body_passed"]
        assert not report["upper_body_checks"]["pair.finite_measurements"]
        assert report["upper_body"]["per_episode"][3]["first_episode_samples"] == 79
        assert report["upper_body"]["per_episode"][3]["tail"]["arm_reference_rmse_rad"] is None
        assert bool(report["upper_body"]["measurement_errors"]) == missing
        json.dumps(report, allow_nan=False)


def test_p95_and_maximum_do_not_average_away_a_single_large_chest_swing():
    def one_swing(values, time, step):
        if step == 100:
            values["chest_tilt_rad"][3] = math.radians(28.)

    env, wrapped, runner = fixture(modifier=one_swing, count=32)
    report = evaluate_upper_body(runner, wrapped, env, episodes=32, stage=1)
    pair = report["upper_body"]["cases"]["pair"]["all"]
    assert pair["chest_rms_deg"] < 8.
    assert pair["chest_peak_max_deg"] > 27.9
    assert pair["chest_peak_p95_deg"] > 19.
    assert not report["upper_body_checks"]["pair.chest_peak"]
    assert report["upper_body_checks"]["pair.chest_rms"]


def test_missing_case_direction_and_nonfinite_aggregates_cannot_pass():
    env, wrapped, runner = fixture()
    upper = evaluate_upper_body(runner, wrapped, env, episodes=4, stage=1)["upper_body"]
    for damage in ("case", "direction", "nan", "coverage"):
        invalid = copy.deepcopy(upper)
        if damage == "case":
            del invalid["cases"]["right"]
        elif damage == "direction":
            del invalid["cases"]["pair"]["directions"]["right"]
        elif damage == "nan":
            invalid["cases"]["stand"]["all"]["chest_rms_deg"] = float("nan")
        else:
            invalid["cases"]["pair"]["early_return"]["observed_episode_rate"] = .5
        assert not all(check_upper_body_gates(invalid).values())


def test_a_safe_terminal_horizon_reset_keeps_coverage_except_excluded_last_frame():
    env, wrapped, _ = fixture()
    tracker = UpperBodyTracker(env, 4, 1)
    wrapped.reset()
    tracker.reset()
    for step in range(1000):
        actions = torch.zeros(4, 23)
        result = wrapped.step(actions)
        done = result[2]
        if step == 999:
            done[:] = True
            env.episode_length_buf.zero_()
        tracker.sample(done, actions)
    report = tracker.report()
    assert all(row["all"]["observed"] for row in report["per_episode"])
    assert all(row["tail"]["observed"] for row in report["per_episode"])
    assert all(row["first_episode_samples"] == 999 for row in report["per_episode"])
    assert not any(row["first_episode_failed"] for row in report["per_episode"])
