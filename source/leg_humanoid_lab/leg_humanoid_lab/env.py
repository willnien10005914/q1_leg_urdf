"""MuJoCo CPU verification backend; the Isaac backend uses the same contract.

Native CAD is the default; simplified boxes require explicit proxy opt-in.
"""
import json
import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .robot import ROOT, JOINTS, DEFAULT, LOWER, UPPER, EFFORT, VELOCITY, HEIGHT, SKILLS, OBSERVATION_SIZE
from .motion import Motion

class LegEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    def __init__(self, skill="stand", *, allow_proxy=False, render_mode=None, randomize=True, episode_seconds=None):
        super().__init__()
        self.proxy=allow_proxy
        self.skill = skill
        self.motion = Motion(skill)
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "urdf" / ("leg_humanoid_proxy.xml" if allow_proxy else "leg_humanoid.xml")))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.vadr = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.feet = [self.model.body(f"{s}_foot_link").id for s in ("left", "right")]
        self.foot_geoms = [self.model.geom(f"{s}_foot_link_collision").id for s in ("left", "right")]
        self.ground = self.model.geom("ground").id
        self.dt = .02
        self.randomize = randomize
        self.max_steps = round((episode_seconds or (7.8 if skill=="sidestep" else 10.))/self.dt)
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = gym.spaces.Box(-1,1,(len(JOINTS),),np.float32)
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(OBSERVATION_SIZE,),np.float32)
        self.last_action = np.zeros(len(JOINTS))
        self.kp = np.array([160 if any(k in n for k in ("hip_","knee_","ankle_")) else 70 for n in JOINTS])
        self.kd = np.array([8 if "hip_" in n or "knee_" in n or "ankle_" in n else 4 for n in JOINTS])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self.steps = 0
        self.last_action = np.zeros(len(JOINTS))
        q,root,_ = self.motion.sample(0)
        self.data.qpos[self.qadr] = np.clip(q + (self.np_random.normal(0,.008,len(q)) if self.randomize else 0),LOWER,UPPER)
        self.data.qpos[:3] = root + [0,0,.003]
        self.data.qpos[3:7] = [1,0,0,0]
        self.start_xy = root[:2].copy()
        if self.randomize:
            self.model.geom_friction[:,0] = self.np_random.uniform(.7,1.1)
        mujoco.mj_forward(self.model,self.data)
        return self._obs(), {"proxy":self.proxy,"skill":self.skill}

    def _rotation(self):
        return Rotation.from_quat(self.data.qpos[[4,5,6,3]]).as_matrix()

    def _obs(self):
        ref,root,command = self.motion.sample(self.steps*self.dt)
        rot = self._rotation()
        phase = min(self.steps*self.dt/self.motion.duration,1) if self.skill=="sidestep" else self.steps*self.dt/.9
        obs = np.r_[rot.T@self.data.qvel[:3],self.data.qvel[3:6],rot.T@[0,0,-1],
            self.data.qpos[self.qadr]-DEFAULT,self.data.qvel[self.vadr]*.1,self.last_action,
            command,np.eye(3)[SKILLS.index(self.skill)],np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),self.data.qpos[2],ref-DEFAULT]
        return obs.astype(np.float32)

    def step(self,action):
        action = np.clip(np.asarray(action),-1,1)
        ref,root,command = self.motion.sample(self.steps*self.dt)
        # Physical low-level stabilization, using the same IMU observations as PPO.
        rotation=self._rotation()
        gravity=rotation.T@np.array([0,0,-1])
        # Free-joint angular qvel is already in body coordinates in MuJoCo.
        angular=self.data.qvel[3:6]
        roll=np.arctan2(-gravity[1],-gravity[2])
        pitch=np.arctan2(gravity[0],-gravity[2])
        target=ref+.20*action
        for side,sign in (("left",1),("right",-1)):
            target[JOINTS.index(f"{side}_ankle_Pitch_joint")]+=sign*np.clip(pitch+.15*angular[1],-.15,.15)
            target[JOINTS.index(f"{side}_ankle_Roll_joint")]+=np.clip(roll+.15*angular[0],-.15,.15)
        target = np.clip(target,LOWER,UPPER)
        for _ in range(4):
            torque = self.kp*(target-self.data.qpos[self.qadr])-self.kd*self.data.qvel[self.vadr]
            # No acceleration past the native no-load speed in the same direction.
            torque = np.where((abs(self.data.qvel[self.vadr])>VELOCITY)&(torque*self.data.qvel[self.vadr]>0),0,torque)
            self.data.ctrl[:] = np.clip(torque,-EFFORT,EFFORT)
            mujoco.mj_step(self.model,self.data)
        self.steps+=1
        ref,root,command = self.motion.sample(self.steps*self.dt)
        rot=self._rotation()
        upright=float(rot[2,2])
        height_error=abs(self.data.qpos[2]-root[2])
        pose_error=float(np.mean((self.data.qpos[self.qadr]-ref)**2))
        body_velocity=rot.T@self.data.qvel[:3]
        velocity_error=float(np.linalg.norm(body_velocity[:2]-command[:2]))
        root_error=float(np.linalg.norm(self.data.qpos[:2]-root[:2]))
        contacts=np.zeros(2,dtype=bool)
        bad_contact=False
        for c in self.data.contact:
            pair={c.geom1,c.geom2}
            if self.ground not in pair:
                continue
            for k,g in enumerate(self.foot_geoms):
                contacts[k] |= g in pair
            if not any(g in pair for g in self.foot_geoms):
                bad_contact=True
        slip=0.
        for k,b in enumerate(self.feet):
            if contacts[k]:
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,b,velocity,0)
                slip+=float(np.sum(velocity[3:5]**2))
        terminated=bool(self.data.qpos[2]<.56 or upright<.65 or bad_contact or not np.isfinite(self.data.qpos).all())
        truncated=self.steps>=self.max_steps
        reward=(1. + 2.*max(upright,0)**4 + 2.*np.exp(-height_error**2/.012)
            + 2.*np.exp(-pose_error/.10) + 2.*np.exp(-velocity_error**2/.16)
            + 2.*np.exp(-root_error**2/.09)
            -.04*np.mean((action-self.last_action)**2)-.02*np.mean(action**2)-.08*slip
            -.00001*np.mean(self.data.ctrl**2) - .3*velocity_error**2 - 10.*terminated) / 10.
        if self.skill=='walk':
            phase=self.steps*self.dt/.9*2*np.pi
            expected=np.sin([phase,phase+np.pi])<=0
            reward+=(1.5*np.mean(contacts==expected)+.5*np.exp(-self.data.qvel[5]**2/.25))/10.
        if self.skill=='sidestep':
            reward+=(4*np.exp(-root_error**2/.04)+4*np.exp(-pose_error/.03)-2*np.exp(-pose_error/.10)+4*np.exp(-velocity_error**2/.09)-2*np.exp(-velocity_error**2/.16))/10.
        self.last_action=action.copy()
        info={"proxy":self.proxy,"upright":upright,"height_error":height_error,"pose_rmse":pose_error**.5,
            "velocity_error":velocity_error,"root_error":root_error,"contacts":contacts.tolist(),
            "time":self.steps*self.dt,"displacement":float(np.linalg.norm(self.data.qpos[:2]-self.start_xy)),"fallen":terminated}
        return self._obs(),float(reward),terminated,truncated,info

    def frame(self):
        return {"q":self.data.qpos[self.qadr].tolist(),"root":self.data.qpos[:3].tolist(),"quaternion":self.data.qpos[3:7].tolist()}

    def render(self):
        if self.renderer is None:
            self.renderer=mujoco.Renderer(self.model,480,640)
        cam=mujoco.MjvCamera()
        cam.lookat[:]=[0,0,.8]
        cam.distance=2.7
        cam.azimuth=140
        cam.elevation=-15
        self.renderer.update_scene(self.data,camera=cam)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
