#!/usr/bin/env python3
"""Build bounded native-axis foot IK templates for command-driven Direct PPO.

This is an offline kinematic reference, not a claim of dynamic feasibility or a
trained policy. Original MediaPipe data and v3 checkpoints are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/leg_humanoid_lab"))
from leg_humanoid_lab.robot import DEFAULT, FOOT_VERTICES, JMAP, JOINTS, LOWER, UPPER, URDF, fk, vector

PHASE = np.linspace(0., 1., 61)
HEIGHT = .86
SUPPORT = .003
CYCLE = .9
GESTURE_DURATION = 1.6
COMMANDS = {
    "stand": [0., 0., 0.], "forward": [.22, 0., 0.],
    "backward": [-.22, 0., 0.], "left": [0., .18, 0.],
    "right": [0., -.18, 0.], "yaw_left": [0., 0., .6],
    "yaw_right": [0., 0., -.6],
}


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3. - 2. * t)


def profile(phase, knots, values):
    """C1 interpolation with zero velocity at explicit trajectory knots."""
    for i in range(len(knots) - 1):
        if phase <= knots[i + 1]:
            alpha = smooth((phase - knots[i]) / (knots[i + 1] - knots[i]))
            return np.asarray(values[i]) * (1. - alpha) + np.asarray(values[i + 1]) * alpha
    return np.asarray(values[-1])


class Limb:
    def __init__(self, side, parts):
        self.names = [f"{side}_{part}_joint" for part in parts]
        self.ids = np.asarray([JOINTS.index(name) for name in self.names])
        self.origins = []
        self.axes = []
        for name in self.names:
            joint = JMAP[name]
            origin = np.eye(4)
            origin[:3, 3] = vector(joint.find("origin"))
            origin[:3, :3] = Rotation.from_euler("xyz", vector(joint.find("origin"), "rpy")).as_matrix()
            self.origins.append(origin)
            self.axes.append(vector(joint.find("axis")))
        self.axes = np.asarray(self.axes)
        self.lower = LOWER[self.ids] + 1e-6
        self.upper = UPPER[self.ids] - 1e-6

    def transform(self, values, base=None):
        transform = np.eye(4) if base is None else base.copy()
        for origin, axis, angle in zip(self.origins, self.axes, values):
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ turn
        return transform

    def solve_leg(self, position, rotation, seed, neutral):
        def residual(values):
            actual = self.transform(values)
            return np.r_[
                (actual[:3, 3] - position) * 70.,
                Rotation.from_matrix(rotation.T @ actual[:3, :3]).as_rotvec() * 8.,
                (values - neutral) * .015,
            ]
        result = least_squares(residual, np.clip(seed, self.lower, self.upper),
                               bounds=(self.lower, self.upper), max_nfev=80,
                               xtol=1e-9, ftol=1e-9, gtol=1e-9)
        return result.x


LEGS = {side: Limb(side, ["hip_Pitch", "hip_Roll", "hip_Yaw", "knee_Pitch", "ankle_Pitch", "ankle_Roll"])
        for side in ("left", "right")}
ARMS = {side: Limb(side, ["shoulder_Pitch", "shoulder_Roll", "shoulder_Yaw", "elbow_Pitch"])
        for side in ("left", "right")}


def body_foot_position(side, xy, clearance, root, rotation):
    """Place the lowest *rotated source CAD vertex* at the desired clearance."""
    lowest = np.min(FOOT_VERTICES[side] @ rotation[2])
    return np.r_[xy, clearance - lowest] - root


def build():
    home = fk(DEFAULT)
    xy = {side: home[f"{side}_foot_link"][:2, 3].copy() for side in LEGS}
    neutral = DEFAULT.copy()
    for side, leg in LEGS.items():
        target = body_foot_position(side, xy[side], SUPPORT, np.array([0., 0., HEIGHT]), np.eye(3))
        neutral[leg.ids] = leg.solve_leg(target, np.eye(3), DEFAULT[leg.ids], DEFAULT[leg.ids])
    plans = {}
    diagnostics = {}
    gesture_arm_peak = {}
    for sign, side in ((1,"right"),(-1,"left")):
        arm = ARMS[side]
        peak_pose = neutral.copy();peak_pose[0]=sign*.16;peak_pose[2]=.055
        base = fk(peak_pose)["center_torso_link"]
        wrist_offset = vector(JMAP[f"{side}_wrist_Yaw_joint"].find("origin"))
        target = np.array([.23,sign*.085,.035])
        def residual(values):
            transform = arm.transform(values,base)
            point = transform[:3,3]+transform[:3,:3]@wrist_offset
            return np.r_[(point-target)*35.,(values-neutral[arm.ids])*.05]
        candidates=[]
        for yaw in (0.,-.8,-1.4):
            seed=neutral[arm.ids].copy();seed[2]=yaw
            result=least_squares(residual,np.clip(seed,arm.lower,arm.upper),bounds=(arm.lower,arm.upper),max_nfev=150)
            candidates.append(result)
        gesture_arm_peak[sign]=min(candidates,key=lambda r:np.linalg.norm(r.fun)).x
    for name in list(COMMANDS) + ["sidestep_left", "sidestep_right"]:
        gesture = name.startswith("sidestep_")
        sign = 1 if name in ("left", "sidestep_left") else -1
        command = np.asarray(COMMANDS.get(name, [0., 0., 0.]))
        qs, roots, foot_positions, foot_heights, contacts, foot_rotations, hands = [], [], [], [], [], [], []
        previous = neutral.copy()
        for phase in PHASE:
            q = neutral.copy()
            rotations = [np.eye(3), np.eye(3)]
            desired_xy = [xy[side].copy() for side in LEGS]
            clearance = np.full(2, SUPPORT)
            loaded = np.ones(2, dtype=bool)
            root = np.array([0., 0., HEIGHT])
            if gesture:
                leading = 0 if sign == 1 else 1
                trailing = 1 - leading
                displacement = np.array([.085, sign * .085])
                # Transfer the trunk before each single-support period. A PPO
                # policy must realize the resulting balance and ground forces.
                root[:2] = profile(phase, [0., .14, .28, .44, .64, .83, 1.],
                    [[0., 0.], [-.006, -sign*.03], [-.006, -sign*.03],
                     [.015, 0.], [.078, sign*.11], [.085, sign*.12], displacement])
                root[2] -= float(profile(phase, [0., .18, .4, .65, .88, 1.], [0., .012, .012, .012, .006, 0.]))
                for index, start, end in [(leading, .18, .44), (trailing, .66, .9)]:
                    progress = np.clip((phase - start) / (end - start), 0., 1.)
                    desired_xy[index] += displacement * smooth(progress)
                    clearance[index] += .04 * np.sin(np.pi * progress) ** 2
                    loaded[index] = not (start < phase < end)
                    if index == trailing:
                        # Toe-down swing avoids demanding ankle angles beyond
                        # the native +/-0.5 rad range during the recovery lift.
                        rotations[index] = Rotation.from_euler("y", .10*np.sin(np.pi*progress)**2).as_matrix()
                # The trailing heel rises about 1 cm while the toe remains on
                # the floor. Horizontal movement of a loaded whole foot is not
                # used as a substitute for push-off.
                push = float(profile(phase, [0., .44, .55, .64, 1.], [0., 0., .085, 0., 0.]))
                rotations[trailing] = Rotation.from_euler("y", push).as_matrix() @ rotations[trailing]
                toe = np.array([FOOT_VERTICES["right" if trailing else "left"][:, 0].max(), 0., -.048])
                desired_xy[trailing] += (toe - rotations[trailing] @ toe)[:2]
                arm_envelope = float(profile(phase, [0., .14, .4, .64, .9, 1.], [0., .12, 1., 1., 0., 0.]))
                q[0] = sign * .16 * arm_envelope
                q[2] = .055 * arm_envelope
            elif name != "stand":
                root[1] = .045 * np.sin(2 * np.pi * phase)
                root[2] += .003 * (1. - np.cos(4 * np.pi * phase)) / 2
                duty = .62
                for index, side in enumerate(LEGS):
                    foot_phase = (phase + .5 * index) % 1.
                    foot_velocity = command[:2] + command[2] * np.array([-xy[side][1], xy[side][0]])
                    half = duty * CYCLE / 2
                    if foot_phase < duty:
                        horizontal = foot_velocity * (half - CYCLE * foot_phase)
                    else:
                        swing = (foot_phase - duty) / (1 - duty)
                        # Cubic Hermite recovery matches stance velocities at
                        # takeoff and landing, avoiding a velocity seam.
                        duration = CYCLE * (1 - duty)
                        h00, h10 = 2*swing**3-3*swing**2+1, swing**3-2*swing**2+swing
                        h01, h11 = -2*swing**3+3*swing**2, swing**3-swing**2
                        horizontal = foot_velocity * (-half*h00-duration*h10+half*h01-duration*h11)
                        clearance[index] += .04 * np.sin(np.pi * swing) ** 2
                        loaded[index] = False
                    desired_xy[index] += horizontal
                    # Foot yaw in body coordinates compensates commanded body
                    # rotation during stance; the foot realigns while airborne.
                    theta = command[2] * (half - CYCLE*foot_phase) if foot_phase < duty else command[2] * horizontal.dot(foot_velocity) / max(foot_velocity.dot(foot_velocity), 1e-8)
                    rotations[index] = Rotation.from_euler("z", theta).as_matrix()
                swing = np.sin(2*np.pi*phase) * command[0] / .22
                q[JOINTS.index("left_shoulder_Pitch_joint")] = .10*swing
                q[JOINTS.index("right_shoulder_Pitch_joint")] = .10*swing
            positions = []
            for index, (side, leg) in enumerate(LEGS.items()):
                target = body_foot_position(side, desired_xy[index], clearance[index], root, rotations[index])
                q[leg.ids] = leg.solve_leg(target, rotations[index], previous[leg.ids], neutral[leg.ids])
                positions.append(target)
            if gesture:
                active_side = "right" if sign == 1 else "left"
                arm = ARMS[active_side]
                start_hand = fk(neutral)[f"{active_side}_gripper_link"][:3, 3]
                goal_hand = np.array([.23, sign*.085, .035])
                target_hand = start_hand*(1-arm_envelope)+goal_hand*arm_envelope
                # Interpolate one IK branch in joint space. Re-solving at each
                # frame can switch redundant shoulder-yaw branches abruptly.
                q[arm.ids] = neutral[arm.ids]+arm_envelope*(gesture_arm_peak[sign]-neutral[arm.ids])
                if phase in (0., 1.):
                    q = neutral.copy()
                hands.append(target_hand.tolist())
            previous = q.copy()
            qs.append(q.tolist()); roots.append((root-np.array([0.,0.,HEIGHT])).tolist())
            foot_positions.append(np.asarray(positions).tolist()); foot_heights.append(clearance.tolist())
            contacts.append(loaded.tolist()); foot_rotations.append(np.asarray(rotations).tolist())
        data = dict(phase=PHASE.tolist(), q=qs, root_offset_body=roots, foot_position_body=foot_positions,
                    foot_height=foot_heights, contact=contacts, foot_rotation_body=foot_rotations,
                    command=command.tolist(), duration_s=GESTURE_DURATION if gesture else CYCLE,
                    loop=not gesture)
        if gesture:
            data["active_hand_target_body"] = hands
        plans[name] = data
        q = np.asarray(qs)
        max_position_error = 0.
        max_rotation_error = 0.
        fk_clearance = []
        for index, pose in enumerate(q):
            transforms = fk(pose)
            heights = []
            for foot, side in enumerate(LEGS):
                actual = transforms[f"{side}_foot_link"]
                max_position_error = max(max_position_error, np.linalg.norm(actual[:3, 3] - foot_positions[index][foot]))
                max_rotation_error = max(max_rotation_error, np.linalg.norm(Rotation.from_matrix(np.asarray(foot_rotations[index][foot]).T @ actual[:3,:3]).as_rotvec()))
                heights.append(float(np.min(FOOT_VERTICES[side] @ actual[2, :3] + actual[2, 3]) + HEIGHT + roots[index][2]))
            fk_clearance.append(heights)
        speed = np.abs(np.diff(q, axis=0))/(data["duration_s"]/(len(PHASE)-1))
        diagnostics[name] = dict(max_foot_position_error_m=max_position_error,
            max_foot_rotation_error_rad=max_rotation_error,
            max_joint_speed_rad_s=float(speed.max()),
            joint_max_speed_name=JOINTS[int(np.argmax(speed.max(axis=0)))],
            minimum_fk_clearance_m=np.min(fk_clearance,axis=0).tolist(),
            maximum_fk_clearance_m=np.max(fk_clearance,axis=0).tolist(),
            max_clearance_error_m=float(np.max(np.abs(np.asarray(fk_clearance)-np.asarray(foot_heights)))),
            endpoint_joint_difference_rad=float(np.max(np.abs(q[0]-q[-1]))),
            within_native_joint_limits=bool(np.all(q>=LOWER-1e-8) and np.all(q<=UPPER+1e-8)))
    return dict(schema=1, contract="q1_leg_command_v4", source="native_axis_bounded_foot_ik",
                reference_scope="Kinematic templates; dynamic and force objectives require PPO validation",
                robot_urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                joint_names=list(JOINTS), root_height=HEIGHT, support_clearance_m=SUPPORT,
                walk_cycle_s=CYCLE, sidestep_duration_s=GESTURE_DURATION, default=neutral.tolist(),
                basis_command=COMMANDS, plans=plans, diagnostics=diagnostics,
                coordinates="body x forward, y left, z up; left/right from robot perspective",
                video_reference="KKhF516rSyo first seven seconds; geometry/timing adapted, not an exact reconstruction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"config/v4_gaits.json")
    args = parser.parse_args()
    data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps({"path":str(args.output), "root_height":HEIGHT,"diagnostics":data["diagnostics"]}, indent=2))
    if max(x["max_foot_position_error_m"] for x in data["diagnostics"].values()) > .008:
        raise SystemExit("IK error exceeds 8 mm; inspect reference before training")
    if max(x["max_joint_speed_rad_s"] for x in data["diagnostics"].values()) > 10.:
        raise SystemExit("Template exceeds native 10 rad/s actuator cap")


if __name__ == "__main__":
    main()
