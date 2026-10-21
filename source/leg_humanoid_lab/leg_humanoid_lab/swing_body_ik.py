"""Experimental bounded swing-foot IK; independent of simulator state writes.

The requested foot pose stays in the pelvis-yaw frame while measured pelvis
tilt and height are compensated by joint targets. Native source-CAD geometry
defines the requested sole height. Joint limits may make the pose infeasible;
the achieved clearance must still be checked in the physical simulation.

This module is an ablation candidate, not part of the frozen v5 controller.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .robot import FOOT_VERTICES, JMAP, JOINTS, LOWER, UPPER, vector


PARTS = ("hip_Pitch", "hip_Roll", "hip_Yaw", "knee_Pitch", "ankle_Pitch", "ankle_Roll")


def quaternion_matrix(q):
    """Isaac wxyz quaternion to rotation; normalization is explicit."""
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def _skew(v):
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)


class BodyAwareSwingIK(torch.nn.Module):
    """Batched native six-axis bounded damped-least-squares pose correction.

    Only planned swing feet whose requested height exceeds 8 mm are changed.
    ``reference_q`` is the nominal 23-joint target before this compensation.
    ``root_height`` is actual pelvis height above the local ground. The method
    produces joint targets only and has no access to a simulator or root setter.
    """
    def __init__(self,iterations=5,damping=.015,orientation_weight=.114,
                 max_iteration_step=.20):
        super().__init__()
        self.iterations=int(iterations);self.damping=float(damping)
        self.orientation_weight=float(orientation_weight)
        self.max_iteration_step=float(max_iteration_step)
        ids=[];origins=[];rotations=[];axes=[]
        for side in ('left','right'):
            ns=[f'{side}_{part}_joint' for part in PARTS]
            ids.append([JOINTS.index(n) for n in ns])
            origins.append([vector(JMAP[n].find('origin')) for n in ns])
            rotations.append([Rotation.from_euler('xyz',vector(JMAP[n].find('origin'),'rpy')).as_matrix() for n in ns])
            axes.append([vector(JMAP[n].find('axis')) for n in ns])
        ids=np.asarray(ids)
        self.register_buffer('joint_ids',torch.tensor(ids,dtype=torch.long))
        self.register_buffer('origins',torch.tensor(np.asarray(origins),dtype=torch.float32))
        self.register_buffer('origin_rotations',torch.tensor(np.asarray(rotations),dtype=torch.float32))
        self.register_buffer('axes',torch.tensor(np.asarray(axes),dtype=torch.float32))
        self.register_buffer('lower',torch.tensor(LOWER[ids],dtype=torch.float32))
        self.register_buffer('upper',torch.tensor(UPPER[ids],dtype=torch.float32))
        # Binary STL can have different vertex counts on the two native feet.
        for side in ('left','right'):
            self.register_buffer(side+'_vertices',torch.tensor(FOOT_VERTICES[side],dtype=torch.float32))

    def leg_fk(self,q,with_jacobian=False):
        """Return body-frame foot origins/orientations for (..., 2, 6) q."""
        shape=q.shape[:-1]
        eye=torch.eye(3,device=q.device,dtype=q.dtype).expand(*shape,3,3)
        r=eye.clone();p=torch.zeros((*shape,3),device=q.device,dtype=q.dtype)
        positions=[];directions=[]
        for i in range(6):
            p=p+(r@self.origins[:,i,:,None]).squeeze(-1)
            r=r@self.origin_rotations[:,i]
            positions.append(p);directions.append((r@self.axes[:,i,:,None]).squeeze(-1))
            k=_skew(self.axes[:,i]);angle=q[...,i,None,None]
            r=r@(eye+angle.sin()*k+(1-angle.cos())*(k@k))
        if not with_jacobian:return p,r
        axis=torch.stack(directions,-2);joint_p=torch.stack(positions,-2)
        linear=torch.linalg.cross(axis,p.unsqueeze(-2)-joint_p,dim=-1).transpose(-1,-2)
        angular=axis.transpose(-1,-2)
        return p,r,linear,angular

    def sole_height(self,q,root_quaternion,root_height):
        p,r=self.leg_fk(q)
        world=quaternion_matrix(root_quaternion).unsqueeze(-3)
        world_r=world@r;world_p=(world@p.unsqueeze(-1)).squeeze(-1)
        heights=[]
        for side,vertices in enumerate((self.left_vertices,self.right_vertices)):
            lowest=(world_r[...,side,2,:]@vertices.T).amin(-1)
            heights.append(root_height+world_p[...,side,2]+lowest)
        return torch.stack(heights,-1)

    @torch.no_grad()
    def forward(self,reference_q,root_quaternion,root_height,
                target_clearance,reference_contact):
        active=(~reference_contact.bool())&(target_clearance>.008)
        q=reference_q[...,self.joint_ids].clone()
        p_ref,r_ref=self.leg_fk(q)
        world=quaternion_matrix(root_quaternion)
        yaw=torch.atan2(world[...,1,0],world[...,0,0])
        c,s=yaw.cos(),yaw.sin();zero=torch.zeros_like(c);one=torch.ones_like(c)
        yaw_r=torch.stack((c,-s,zero,s,c,zero,zero,zero,one),-1).reshape(*yaw.shape,3,3)
        desired_world_r=yaw_r.unsqueeze(-3)@r_ref
        desired_world_p=(yaw_r.unsqueeze(-3)@p_ref.unsqueeze(-1)).squeeze(-1)
        desired_z=[]
        for side,vertices in enumerate((self.left_vertices,self.right_vertices)):
            lowest=(desired_world_r[...,side,2,:]@vertices.T).amin(-1)
            desired_z.append(target_clearance[...,side]-root_height-lowest)
        desired_world_p=desired_world_p.clone()
        desired_world_p[...,2]=torch.stack(desired_z,-1)
        inverse=world.transpose(-1,-2).unsqueeze(-3)
        desired_p=(inverse@desired_world_p.unsqueeze(-1)).squeeze(-1)
        desired_r=inverse@desired_world_r
        weight=self.orientation_weight
        identity=torch.eye(6,device=q.device,dtype=q.dtype)
        for _ in range(self.iterations):
            p,r,linear,angular=self.leg_fk(q,True)
            relative=desired_r@r.transpose(-1,-2)
            # vee(R-R.T)/2 is a well-conditioned body-frame rotation error for
            # the small (<0.3 rad) compensation range tested by this ablation.
            rotation_error=torch.stack((relative[...,2,1]-relative[...,1,2],
                                        relative[...,0,2]-relative[...,2,0],
                                        relative[...,1,0]-relative[...,0,1]),-1)*.5
            error=torch.cat((desired_p-p,rotation_error*weight),-1)
            jacobian=torch.cat((linear,angular*weight),-2)
            free=torch.ones_like(q,dtype=torch.bool)
            # Re-solve once after identifying outward steps at joint limits.
            # Clamping without this step leaves other joints stalled against
            # a saturated ankle when the full requested pose is infeasible.
            for _ in range(2):
                j=jacobian*free.unsqueeze(-2)
                update=(j.transpose(-1,-2)@torch.linalg.solve(j@j.transpose(-1,-2)+self.damping**2*identity,error.unsqueeze(-1))).squeeze(-1)
                outward=((q<=self.lower+1e-6)&(update<0))|((q>=self.upper-1e-6)&(update>0))
                free=free&~outward
            update=update.clamp(-self.max_iteration_step,self.max_iteration_step)
            q=(q+update).clamp(self.lower,self.upper)
        result=reference_q.clone()
        result[...,self.joint_ids]=torch.where(active.unsqueeze(-1),q,reference_q[...,self.joint_ids])
        return result


class SwingClearanceFloor(torch.nn.Module):
    """Experimental smooth, raise-only wrapper around bounded body-aware IK.

    A healthy nominal swing is preserved even when it clears more than 4 cm.
    Corrections fade with the planned swing height and with the actual-root
    geometry deficit. An update that lowers the predicted CAD sole is rejected
    per foot. This remains a joint-target ablation, not physical acceptance.
    """
    def __init__(self,iterations=3,**ik_kwargs):
        super().__init__()
        self.ik=BodyAwareSwingIK(iterations=iterations,**ik_kwargs)

    @property
    def joint_ids(self):
        return self.ik.joint_ids

    def sole_height(self,q,root_quaternion,root_height):
        return self.ik.sole_height(q,root_quaternion,root_height)

    @staticmethod
    def _smoothstep(x):
        x=x.clamp(0.,1.)
        return x*x*(3.-2.*x)

    @torch.no_grad()
    def forward(self,reference_q,root_quaternion,root_height,
                target_clearance,reference_contact):
        legs=reference_q[...,self.joint_ids]
        predicted=self.ik.sole_height(legs,root_quaternion,root_height)
        desired=target_clearance.clamp(max=.04)
        active=(~reference_contact.bool())&(target_clearance>.008)
        phase=self._smoothstep((target_clearance-.008)/(.022-.008))
        deficit=self._smoothstep((desired-predicted)/.02)
        blend=phase*deficit*active
        projected=self.ik(reference_q,root_quaternion,root_height,desired,reference_contact)
        corrected=(legs+blend.unsqueeze(-1)*(projected[...,self.joint_ids]-legs)).clamp(self.ik.lower,self.ik.upper)
        achieved=self.ik.sole_height(corrected,root_quaternion,root_height)
        accept=active&(blend>0)&torch.isfinite(achieved)&(achieved>=predicted)
        result=reference_q.clone()
        result[...,self.joint_ids]=torch.where(accept.unsqueeze(-1),corrected,legs)
        return result
