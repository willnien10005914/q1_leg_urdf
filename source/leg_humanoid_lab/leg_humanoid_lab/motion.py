"""Shared, continuous skill references. No fabricated MediaPipe fallback."""
import json
import hashlib
import numpy as np
from .robot import ROOT, URDF, JOINTS, DEFAULT, LOWER, UPPER, HEIGHT, SKILLS

class Motion:
    def __init__(self, skill, reference=None):
        if skill not in SKILLS:
            raise ValueError(skill)
        self.skill = skill
        self.duration = 1.0
        self.data = None
        if skill == "sidestep":
            path = reference or ROOT / "docs/reference/sidestep_motion.json"
            self.data = json.loads(path.read_text())
            if self.data["joint_names"] != list(JOINTS):
                raise ValueError("Wrong reference contract: joint_names mismatch")
            # mediapipe = Store geometric mimic (root XY often zero).
            # hhtools / beyondmimic / roboparty = RoboParty-style retarget with root trajectory.
            allowed = {"mediapipe", "hhtools", "beyondmimic", "roboparty"}
            if self.data.get("source") not in allowed:
                raise ValueError("Wrong reference contract or unsupported motion source")
            if self.data.get('robot_urdf_sha256')!=hashlib.sha256(URDF.read_bytes()).hexdigest():
                raise ValueError('Reference belongs to another robot revision; rerun prepare_motion.sh')
            self.t = np.array(self.data["time"])
            self.q = np.array(self.data["q"])
            self.root = np.array(self.data["root"])
            self.duration = float(self.t[-1])
            if self.duration <= 0 or not np.all(np.diff(self.t) > 0):
                raise ValueError("Motion times must increase")

    def sample(self, time):
        # A finite seven-second gesture: hold the final pose, never teleport at a wrap.
        if self.data is not None:
            t = np.clip(float(time), 0, self.duration)
            q = np.array([np.interp(t, self.t, self.q[:, k]) for k in range(len(JOINTS))])
            root = np.array([np.interp(t, self.t, self.root[:, k]) for k in range(3)])
            ahead = min(t + .02, self.duration)
            behind = max(t - .02, 0)
            v = (np.array([np.interp(ahead, self.t, self.root[:, k]) for k in range(2)]) - np.array([np.interp(behind, self.t, self.root[:, k]) for k in range(2)])) / max(ahead - behind, 1e-6)
            return q, root, np.r_[v, 0]
        q = DEFAULT.copy()
        if self.skill == "walk":
            p = 2 * np.pi * time / .9
            for side, sign, offset in (("left", 1, 0), ("right", -1, np.pi)):
                swing = np.sin(p + offset)
                lift = max(0., swing)
                for part, value in (("hip_Pitch", .15 + .18 * swing), ("knee_Pitch", .3 + .25 * lift), ("ankle_Pitch", -.15 + .18 * swing - .25 * lift), ("shoulder_Pitch", .12 * swing)):
                    q[JOINTS.index(f"{side}_{part}_joint")] = sign * value
            return np.clip(q, LOWER, UPPER), np.array([.2 * time, 0., HEIGHT]), np.array([.2, 0., 0.])
        return q, np.array([0., 0., HEIGHT]), np.zeros(3)
