"""Native CAD contract, independent of either physics engine."""
from pathlib import Path
import os
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(os.environ.get("Q1_LEG_ROOT", Path(__file__).resolve().parents[3]))
URDF = ROOT / "urdf/q1_inertia/quanta_robot_legged.urdf"
SKILLS = ("stand", "walk", "sidestep")
JOINTS = ("center_waist_Yaw_joint", "center_waist_Roll_joint", "center_waist_Pitch_joint") + tuple(
    f"{side}_{part}_joint" for side in ("left", "right") for part in (
        "hip_Pitch", "hip_Roll", "hip_Yaw", "knee_Pitch", "ankle_Pitch", "ankle_Roll",
        "shoulder_Pitch", "shoulder_Roll", "shoulder_Yaw", "elbow_Pitch"))
TREE = ET.parse(URDF).getroot()
LINKS = {x.get("name"): x for x in TREE.findall("link")}
ALL_JOINTS = TREE.findall("joint")
JMAP = {j.get("name"): j for j in ALL_JOINTS}
LOWER = np.array([float(JMAP[n].find("limit").get("lower")) for n in JOINTS])
UPPER = np.array([float(JMAP[n].find("limit").get("upper")) for n in JOINTS])
EFFORT = np.array([float(JMAP[n].find("limit").get("effort")) for n in JOINTS])
VELOCITY = np.array([float(JMAP[n].find("limit").get("velocity")) for n in JOINTS])
DEFAULT = np.zeros(len(JOINTS))
for side, sign in (("left", 1), ("right", -1)):
    for part, value in (("hip_Pitch", .15), ("knee_Pitch", .3), ("ankle_Pitch", -.15), ("elbow_Pitch", -.15)):
        DEFAULT[JOINTS.index(f"{side}_{part}_joint")] = sign * value

def vector(element, key="xyz", default="0 0 0"):
    return np.fromstring(element.get(key, default) if element is not None else default, sep=" ")

def fk(q):
    """Link transforms in pelvis coordinates, honoring native signed axes."""
    values = dict(zip(JOINTS, q))
    transforms = {"center_hip_link": np.eye(4)}
    pending = list(ALL_JOINTS)
    while pending:
        progressed = False
        for j in pending[:]:
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent not in transforms:
                continue
            origin = j.find("origin")
            t = np.eye(4)
            t[:3, 3] = vector(origin)
            t[:3, :3] = Rotation.from_euler("xyz", vector(origin, "rpy")).as_matrix()
            a = np.eye(4)
            a[:3, :3] = Rotation.from_rotvec(vector(j.find("axis")) * values.get(j.get("name"), 0)).as_matrix()
            transforms[child] = transforms[parent] @ t @ a
            pending.remove(j)
            progressed = True
        if not progressed:
            raise ValueError("URDF is not a connected tree")
    return transforms

# Source STL is authoritative for native ground height. The proxy retains its
# own explicitly assumed box dimensions for optional regression work.
FOOT_HALF = np.array([.11, .05, .02])
FOOT_CENTER = np.array([.025, 0, -.025])
def mesh_vertices(mesh):
    path=(URDF.parent/mesh.get('filename')).resolve()
    data=path.read_bytes()
    count=int.from_bytes(data[80:84],'little')
    dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
    vertices=np.frombuffer(data,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3).astype(float)
    return np.unique(vertices,axis=0)*vector(mesh,'scale','1 1 1')

FOOT_VERTICES={s:mesh_vertices(LINKS[f'{s}_foot_link'].find('collision/geometry/mesh')) for s in ('left','right')}
def grounded_height(q=DEFAULT):
    transforms = fk(q)
    return -min(np.min(FOOT_VERTICES[s]@transforms[f'{s}_foot_link'][2,:3]+transforms[f'{s}_foot_link'][2,3]) for s in ('left','right'))

HEIGHT = grounded_height()
OBSERVATION_SIZE = 9 + 3 * len(JOINTS) + 3 + 3 + 2 + 1 + len(JOINTS)

def contract():
    return {"schema": 1, "robot": "q1_leg_native_v3", "joint_names": list(JOINTS),
            "urdf_sha256":hashlib.sha256(URDF.read_bytes()).hexdigest(),
            "action_size": len(JOINTS), "observation_size": OBSERVATION_SIZE,
            "observation_order": ["body_lin_vel", "body_ang_vel", "projected_gravity", "q-default", "qvel*0.1", "previous_action", "velocity_command", "skill_onehot", "phase_sin_cos", "height", "reference-default"],
            "action": "joint position target = reference + ankle attitude feedback + 0.20 * clipped action; native limits enforced",
            "ankle_feedback": "pitch/roll + 0.15 * body angular velocity, correction clipped to +/-0.15 rad; native left/right pitch signs respected",
            "skills": list(SKILLS), "default": DEFAULT.tolist(), "root_height": HEIGHT,
            "limits_lower": LOWER.tolist(), "limits_upper": UPPER.tolist(),
            "locked_joints": [n for n in JMAP if n not in JOINTS],
            "wheel_checkpoints_compatible": False}
