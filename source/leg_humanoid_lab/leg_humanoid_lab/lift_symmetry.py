"""Experimental native left/right kinematic transforms for Q1 policy data.

No simulator, reward, policy or training loop is changed by this module.
FK reflection is exact, but the native dynamics are only approximately
symmetric. Both shoulder-Pitch limits are [-1.1345, 2.1817], so their required
sign reversal is NOT globally limit invariant. Position augmentation must use
the feasibility mask; clamping an invalid mirror would destroy the FK proof.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .robot import DEFAULT, JMAP, JOINTS, LOWER, UPPER, vector


def _other(name):
    if name.startswith('left_'):return 'right_'+name[5:]
    if name.startswith('right_'):return 'left_'+name[6:]
    return name


def _derive_native_transform():
    """Derive joint signs from reflected axial vectors, not part-name guesses."""
    reflection=np.diag([1.,-1.,1.]);indices=[];signs=[]
    for name in JOINTS:
        source=JMAP[_other(name)];destination=JMAP[name]
        source_axis=vector(source.find('axis'))
        destination_axis=vector(destination.find('axis'))
        # Rotation axes are axial vectors: det(S)*S*a = -S*a.
        reflected_axis=-reflection@source_axis
        sign=float(destination_axis@reflected_axis)
        if not np.isclose(abs(sign),1.,atol=1e-10):
            raise ValueError(f'Native joint axes do not support a signed mirror: {name}')
        source_origin=source.find('origin');destination_origin=destination.find('origin')
        if not np.allclose(reflection@vector(source_origin),vector(destination_origin),atol=1e-12):
            raise ValueError(f'Native joint translations are not mirrored: {name}')
        sr=Rotation.from_euler('xyz',vector(source_origin,'rpy')).as_matrix()
        dr=Rotation.from_euler('xyz',vector(destination_origin,'rpy')).as_matrix()
        if not np.allclose(reflection@sr@reflection,dr,atol=1e-12):
            raise ValueError(f'Native joint origin rotations are not mirrored: {name}')
        indices.append(JOINTS.index(_other(name)));signs.append(round(sign))
    return tuple(indices),tuple(signs)


MIRROR_INDICES,MIRROR_SIGNS=_derive_native_transform()
_indices=np.asarray(MIRROR_INDICES);_signs=np.asarray(MIRROR_SIGNS)
MIRRORED_LOWER=np.minimum(LOWER[_indices]*_signs,UPPER[_indices]*_signs)
MIRRORED_UPPER=np.maximum(LOWER[_indices]*_signs,UPPER[_indices]*_signs)
LIMIT_EXCEPTIONS=tuple(name for i,name in enumerate(JOINTS)
                       if not np.allclose([MIRRORED_LOWER[i],MIRRORED_UPPER[i]],
                                          [LOWER[i],UPPER[i]],atol=1e-10))
SYMMETRIC_LOWER=np.maximum(LOWER,MIRRORED_LOWER)
SYMMETRIC_UPPER=np.minimum(UPPER,MIRRORED_UPPER)


class NativeMirror(torch.nn.Module):
    """Device-resident involutions for 23-joint tensors and flat 110-D obs."""
    def __init__(self):
        super().__init__()
        self.register_buffer('indices',torch.tensor(MIRROR_INDICES,dtype=torch.long))
        self.register_buffer('signs',torch.tensor(MIRROR_SIGNS,dtype=torch.float32))
        self.register_buffer('lower',torch.tensor(LOWER,dtype=torch.float32))
        self.register_buffer('upper',torch.tensor(UPPER,dtype=torch.float32))
        self.register_buffer('symmetric_lower',torch.tensor(SYMMETRIC_LOWER,dtype=torch.float32))
        self.register_buffer('symmetric_upper',torch.tensor(SYMMETRIC_UPPER,dtype=torch.float32))

    def joints(self,values):
        """Exact signed permutation for q, qvel, residual actions or q deltas.

        It does not clamp angles. Call ``positions_feasible`` for positions.
        A delta about neutral transforms identically only when that neutral is
        mirror invariant, as tested for both DEFAULT and the V5 stand pose.
        """
        if values.shape[-1]!=23:raise ValueError('Native mirror requires 23 native joint values')
        return values[...,self.indices]*self.signs

    def actions(self,values):
        return self.joints(values)

    def observations(self,values):
        """Reflect the exact unnormalized InteractiveLegEnv 110-D layout.

        Walk phase advances half a cycle, negating both sin and cos. Gesture
        phase is preserved while its explicit lateral command changes sign.
        Stand/recover phase is unchanged. Skill channels remain unchanged.
        """
        if values.shape[-1]!=110:raise ValueError('Native mirror requires flat 110-D policy observations')
        result=values.clone()
        result[...,0:3]=values[...,0:3]*values.new_tensor([1.,-1.,1.])
        result[...,3:6]=values[...,3:6]*values.new_tensor([-1.,1.,-1.])
        result[...,6:9]=values[...,6:9]*values.new_tensor([1.,-1.,1.])
        for start in (9,32,55,87):
            result[...,start:start+23]=self.joints(values[...,start:start+23])
        result[...,78:81]=values[...,78:81]*values.new_tensor([1.,-1.,-1.])
        walk=values[...,82]>.5
        result[...,84:86]=torch.where(walk.unsqueeze(-1),-values[...,84:86],values[...,84:86])
        return result

    def positions_feasible(self,positions,tolerance=1e-6):
        """Whether BOTH source q and exact mirrored q obey native limits."""
        mirrored=self.joints(positions)
        return (torch.isfinite(positions).all(-1)&torch.isfinite(mirrored).all(-1)
                &(positions>=self.lower-tolerance).all(-1)&(positions<=self.upper+tolerance).all(-1)
                &(mirrored>=self.lower-tolerance).all(-1)&(mirrored<=self.upper+tolerance).all(-1))

    def observations_feasible(self,observations,stand_q,tolerance=1e-6):
        """Physical q and reference q must both have a legal exact mirror.

        The caller supplies the actual environment neutral (not a substituted
        pose). A noninvariant neutral is rejected instead of silently shifting
        the meaning of q-default and reference-default observation channels.
        """
        if observations.shape[-1]!=110:raise ValueError('Expected 110-D native observations')
        stand=torch.as_tensor(stand_q,device=observations.device,dtype=observations.dtype)
        if not torch.allclose(self.joints(stand),stand,atol=1e-6,rtol=0.):
            raise ValueError('Observation mirror requires a mirror-invariant native neutral')
        return (torch.isfinite(observations).all(-1)
                &self.positions_feasible(observations[...,9:32]+stand,tolerance)
                &self.positions_feasible(observations[...,87:110]+stand,tolerance))


def augment_policy(env,obs=None,actions=None):
    """RSL-RL 5 callback for weak mirror consistency, NOT PPO augmentation.

    Expected configuration is ``use_data_augmentation=False`` and a small
    mirror-loss coefficient, because source inertias are not exactly symmetric.
    RSL first calls with observations, then with ``obs=None`` and original
    actor means. The exact same cached feasibility mask must govern both calls.

    Invalid rows are duplicated unchanged, so their predicted and requested
    second-copy actor means are identical and contribute zero mirror loss.
    Observations may be a TensorDict or a plain dictionary of 110-D tensors.
    """
    if env is None:raise ValueError('Mirror callback requires its environment for feasibility/cache state')
    if obs is None and actions is None:raise ValueError('Mirror callback needs observations or actions')
    augmented_obs=None;augmented_actions=None
    if obs is not None:
        if 'policy' not in obs:raise ValueError('Mirror callback requires the native policy observation group')
        policy=obs['policy']
        if policy.ndim!=2 or policy.shape[-1]!=110:
            raise ValueError('Mirror callback requires a flat N x 110 native policy batch')
        mirror=getattr(env,'_q1_native_mirror',None)
        if mirror is None:
            mirror=NativeMirror();env._q1_native_mirror=mirror
        mirror=mirror.to(device=policy.device,dtype=policy.dtype)
        base=getattr(env,'unwrapped',env)
        if not hasattr(base,'stand_q'):raise ValueError('Mirror callback requires the actual native stand_q')
        stand=torch.as_tensor(base.stand_q,device=policy.device,dtype=policy.dtype)
        feasible=mirror.observations_feasible(policy,stand)
        shoulders=[JOINTS.index(side+'_shoulder_Pitch_joint') for side in ('left','right')]
        actual=(policy[...,9:32]+stand)[...,shoulders]
        reference=(policy[...,87:110]+stand)[...,shoulders]
        feasible=(feasible&(actual.abs()<=1.1345+1e-6).all(-1)
                  &(reference.abs()<=1.1345-.35+1e-6).all(-1))
        env._q1_mirror_feasible=feasible.detach().clone()
        mirrored={}
        for key,value in obs.items():
            if value.ndim!=2 or value.shape!=(len(policy),110):
                raise ValueError('Every mirrored observation group must share the native N x 110 layout')
            transformed=mirror.observations(value)
            mirrored[key]=torch.where(feasible.unsqueeze(-1),transformed,value)
        if isinstance(obs,dict):
            augmented_obs={key:torch.cat((value,mirrored[key]),dim=0) for key,value in obs.items()}
        else:
            # TensorDict preserves batch_size and device metadata through cat.
            second=obs.clone()
            for key,value in mirrored.items():second[key]=value
            augmented_obs=torch.cat((obs,second),dim=0)
    if actions is not None:
        if actions.ndim!=2 or actions.shape[-1]!=23:
            raise ValueError('Mirror callback requires an N x 23 native action batch')
        feasible=getattr(env,'_q1_mirror_feasible',None)
        mirror=getattr(env,'_q1_native_mirror',None)
        if feasible is None or mirror is None:
            raise ValueError('Action mirror requires a feasibility mask from the preceding observation call')
        if len(feasible)!=len(actions) or feasible.device!=actions.device:
            raise ValueError('Cached mirror mask does not match the action batch size/device')
        transformed=mirror.actions(actions)
        mirrored_actions=torch.where(feasible.unsqueeze(-1),transformed,actions)
        augmented_actions=torch.cat((actions,mirrored_actions),dim=0)
    return augmented_obs,augmented_actions
