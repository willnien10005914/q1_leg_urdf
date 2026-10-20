"""CPU lookup and validation for the command-driven native v4 gait contract.

Training may copy these arrays to CUDA once and use its own batched torch
interpolation; this module intentionally does not import Isaac Sim or torch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .robot import ROOT, URDF, JOINTS, LOWER, UPPER

WALK_BASES = ("stand", "forward", "backward", "left", "right", "yaw_left", "yaw_right")
SIDESTEP_PLANS = ("sidestep_left", "sidestep_right")
ARRAY_FIELDS = ("q", "root_offset_body", "foot_position_body", "foot_height", "contact")


def load_gaits(path: Path | None = None, *, verify: bool = True):
    path = Path(path or ROOT / "config/v4_gaits.json")
    data = json.loads(path.read_text())
    if verify:
        if data.get("contract") != "q1_leg_command_v4" or data.get("schema") != 1:
            raise ValueError("Unexpected v4 reference contract")
        if data.get("robot_urdf_sha256") != hashlib.sha256(URDF.read_bytes()).hexdigest():
            raise ValueError("v4 gait reference belongs to a different URDF; rerun build_v4_gaits.py")
        if data.get("joint_names") != list(JOINTS):
            raise ValueError("v4 reference joint order mismatch")
        for name in WALK_BASES + SIDESTEP_PLANS:
            plan = data["plans"][name]
            phase = np.asarray(plan["phase"])
            q = np.asarray(plan["q"])
            if len(phase) < 3 or phase[0] != 0 or phase[-1] != 1 or not np.all(np.diff(phase) > 0):
                raise ValueError(f"Invalid phase samples in {name}")
            if q.shape != (len(phase), len(JOINTS)) or not np.isfinite(q).all():
                raise ValueError(f"Invalid joint samples in {name}")
            if np.any(q < LOWER-1e-7) or np.any(q > UPPER+1e-7):
                raise ValueError(f"Reference exceeds native limits in {name}")
            if not np.allclose(q[0], q[-1], atol=1e-5):
                raise ValueError(f"Reference joint endpoint discontinuity in {name}")
            for field in ARRAY_FIELDS:
                values = np.asarray(plan[field])
                if len(values) != len(phase) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {field} in {name}")
    return data


class V4Reference:
    def __init__(self, path: Path | None = None):
        self.data = load_gaits(path)
        self.root_height = float(self.data["root_height"])
        self.default = np.asarray(self.data["default"], dtype=np.float64)
        self.walk_cycle_s = float(self.data["walk_cycle_s"])
        self.sidestep_duration_s = float(self.data["sidestep_duration_s"])

    def sample_plan(self, name: str, phase: float):
        plan = self.data["plans"][name]
        value = float(phase) % 1. if plan["loop"] else np.clip(float(phase), 0., 1.)
        phases = np.asarray(plan["phase"])
        output = {}
        for field in ARRAY_FIELDS:
            array = np.asarray(plan[field], dtype=np.float64)
            flat = array.reshape(len(phases), -1)
            output[field] = np.asarray([np.interp(value, phases, flat[:, i]) for i in range(flat.shape[1])]).reshape(array.shape[1:])
        # Contact is a confidence in [0,1] near schedule boundaries.
        return output

    def walk_weights(self, command):
        """Convex basis weights; diagonal/yaw requests share the gait budget.

        Users should apply a command ramp in the environment. The returned
        effective command is the exact convex combination for reward tracking.
        """
        vx, vy, wz = np.asarray(command, dtype=np.float64)
        raw = np.array([max(vx,0)/.22,max(-vx,0)/.22,max(vy,0)/.18,max(-vy,0)/.18,max(wz,0)/.6,max(-wz,0)/.6])
        raw /= max(1., raw.sum())
        weights = np.r_[max(0., 1.-raw.sum()), raw]
        commands = np.asarray([self.data["basis_command"][name] for name in WALK_BASES])
        return weights, weights @ commands

    def sample_walk(self, command, phase: float):
        weights, effective_command = self.walk_weights(command)
        samples = [self.sample_plan(name, phase) for name in WALK_BASES]
        result = {field:sum(weight*sample[field] for weight,sample in zip(weights,samples)) for field in ARRAY_FIELDS}
        result["effective_command"] = effective_command
        return result

    def sample_sidestep(self, side: str | int, phase: float):
        name = "sidestep_left" if side in ("left",1) else "sidestep_right" if side in ("right",-1) else None
        if name is None:
            raise ValueError("side must be left/right or +1/-1")
        return self.sample_plan(name, phase)
