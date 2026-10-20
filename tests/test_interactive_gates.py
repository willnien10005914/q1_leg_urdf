"""Acceptance regressions: a good average must not hide a missing direction."""
import torch
from types import SimpleNamespace

from leg_humanoid_lab.interactive_evaluation import (
    CASE_NAMES, _quat_inverse, check_interactive_gates, evaluate_interactive,
)


def complete_report():
    cases = {}
    for name in CASE_NAMES:
        cases[name] = {
            "episodes": 4, "survival_rate": 1., "upright": .995,
            "final_stand_speed_m_s": .01, "directional_displacement_m": .7,
            "velocity_rmse_m_s": .04, "signed_yaw_displacement_rad": 1.2,
            "yaw_rmse_rad_s": .04, "foot_peak_clearance_m_per_foot": [.035, .035],
            "foot_lift_success_rate_per_foot": [.8, .8],
            "stance_slip_m_s": .04,
        }
    cases["sequence"].update({"finite_sequence_completed": True, "directions": {
        side: {"lead_forward_m": .06, "lead_signed_lateral_m": .08,
               "lead_peak_clearance_m": .04, "lead_lift_success_rate": .8,
               "trailing_peak_clearance_m": .035, "trailing_lift_success_rate": .8,
               "lead_landing_rate": 1., "hand_cross_success_rate": .8,
               "root_forward_m": .04, "root_signed_lateral_m": .04,
               "push_support_forward_proxy_rate": .7}
        for side in ("left", "right")}})
    return {"cases": cases}


def test_complete_report_passes_and_missing_tasks_fail_closed():
    report = complete_report()
    assert all(check_interactive_gates(report).values())
    del report["cases"]["s"]
    assert not all(check_interactive_gates(report).values())


def test_stationary_skating_one_sided_and_wrong_diagonal_cannot_pass():
    stationary = complete_report()
    stationary["cases"]["w"]["directional_displacement_m"] = 0.
    assert not all(check_interactive_gates(stationary).values())
    skating = complete_report()
    skating["cases"]["a"]["foot_lift_success_rate_per_foot"] = [0., 0.]
    assert not all(check_interactive_gates(skating).values())
    one_sided = complete_report()
    one_sided["cases"]["sequence"]["directions"]["right"]["lead_peak_clearance_m"] = .004
    assert not all(check_interactive_gates(one_sided).values())
    backward = complete_report()
    backward["cases"]["sequence"]["directions"]["left"]["lead_forward_m"] = -.04
    assert not all(check_interactive_gates(backward).values())
    dragging = complete_report()
    dragging["cases"]["sequence"]["directions"]["left"]["trailing_peak_clearance_m"] = .002
    assert not all(check_interactive_gates(dragging).values())
    slipping = complete_report()
    slipping["cases"]["sequence"]["stance_slip_m_s"] = .2
    assert not all(check_interactive_gates(slipping).values())


def test_handoff_cross_arm_stop_and_nonfinite_are_required():
    for category, field, value in (
        ("sequence", "finite_sequence_completed", False),
        ("d", "final_stand_speed_m_s", .2),
        ("stand", "upright", float("nan")),
    ):
        report = complete_report()
        report["cases"][category][field] = value
        assert not all(check_interactive_gates(report).values())
    report = complete_report()
    report["cases"]["sequence"]["directions"]["right"]["hand_cross_success_rate"] = 0.
    assert not all(check_interactive_gates(report).values())


def test_wxyz_body_frame_rotation_preserves_robot_relative_directions():
    angle = torch.tensor(torch.pi / 2)
    quaternion = torch.tensor([[torch.cos(angle / 2), 0., 0., torch.sin(angle / 2)]])
    # A robot facing world +Y has world +Y as its own forward.
    forward = _quat_inverse(quaternion, torch.tensor([[0., 1., 0.]]))
    left = _quat_inverse(quaternion, torch.tensor([[-1., 0., 0.]]))
    torch.testing.assert_close(forward, torch.tensor([[1., 0., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(left, torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=0)


def test_finite_evaluator_rejects_static_policy_and_excludes_auto_reset_history():
    """Exercise the full collection loop without Isaac imports or fake success."""
    count = 8
    root_pos = torch.zeros(count, 3)
    root_pos[:, 2] = .86
    quaternion = torch.zeros(count, 4)
    quaternion[:, 0] = 1.
    gravity = torch.zeros(count, 3)
    gravity[:, 2] = -1.
    data = SimpleNamespace(
        root_pos_w=root_pos, root_quat_w=quaternion,
        root_lin_vel_b=torch.zeros(count, 3), root_ang_vel_b=torch.zeros(count, 3),
        projected_gravity_b=gravity, body_pos_w=torch.zeros(count, 4, 3),
    )
    mode_history = []

    def set_manual(mode, command, phase):
        mode_history.append(mode.clone())

    env = SimpleNamespace(
        num_envs=count, device="cpu", step_dt=.02, max_episode_length=1010,
        cfg=SimpleNamespace(allow_proxy=False, training_mode=True),
        library={"sidestep_duration_s": 1.6, "walk_cycle_s": .9, "contract": "test"},
        robot=SimpleNamespace(data=data, find_bodies=lambda names, preserve_order: ([2, 3], names)),
        reference_height=torch.zeros(count, 2), reference_contact=torch.ones(count, 2, dtype=torch.bool),
        foot_body_ids=[0, 1], reset_terminated=torch.zeros(count, dtype=torch.bool),
        feet=[0, 1], contacts=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(count, 4, 3))),
        set_manual_control=set_manual,
        foot_step_metrics=lambda: (torch.zeros(count, 2), torch.zeros(count, 2),
                                   torch.ones(count, 2, dtype=torch.bool), None),
    )

    class Wrapped:
        resets = 0
        steps = 0

        def reset(self):
            self.resets += 1
            return torch.zeros(count, 1), {}

        def get_observations(self):
            return torch.zeros(count, 1)

        def step(self, action):
            self.steps += 1
            done = torch.zeros(count, dtype=torch.bool)
            env.reset_terminated.zero_()
            if self.steps == 10:
                # A simulator auto-reset must not make the first episode alive.
                done[0] = True
                env.reset_terminated[0] = True
            return self.get_observations(), torch.zeros(count), done, {}

    wrapped = Wrapped()
    runner = SimpleNamespace(get_inference_policy=lambda device: lambda obs: torch.zeros(count, 23))
    report = evaluate_interactive(runner, wrapped, env, episodes=count)
    assert wrapped.resets == 1 and wrapped.steps == 1000
    assert not report["passed"]
    assert report["cases"]["stand"]["survival_rate"] == 0.
    assert abs(report["cases"]["stand"]["duration_s"] - .2) < 1e-6
    assert report["cases"]["w"]["directional_displacement_m"] == 0.
    assert report["cases"]["sequence"]["directions"]["right"]["trailing_peak_clearance_m"] == 0.
    changes = []
    for mode in mode_history:
        value = int(mode[7])
        if not changes or changes[-1] != value:
            changes.append(value)
    assert changes == [0, 2, 3, 4, 0]
    assert env.cfg.training_mode is True
