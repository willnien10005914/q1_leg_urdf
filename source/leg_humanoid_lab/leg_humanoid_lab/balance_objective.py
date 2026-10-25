"""Measured balance reward proxies; never a substitute for native gates."""
import torch

def support_capture_error(com,velocity,feet,loaded):
    count=loaded.sum(-1)
    a=torch.where(loaded[:,0,None],feet[:,0,:2],feet[:,1,:2])
    b=torch.where(loaded[:,1,None],feet[:,1,:2],feet[:,0,:2])
    support_z=(feet[:,:,2]*loaded).sum(-1)/count.clamp_min(1)
    omega=(9.81/(com[:,2]-support_z).clamp(.2,1.5)).sqrt()
    capture=com[:,:2]+velocity[:,:2]/omega[:,None]
    segment=b-a
    f=((capture-a)*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-8)
    closest=a+f.clamp(0,1)[:,None]*segment
    excess=((capture-closest).norm(dim=-1)-.045).clamp_min(0)
    return torch.where(count>0,excess,torch.full_like(excess,.20))

def graded_cost(error,scale):
    v=(error/scale).clamp(0,6)
    return torch.where(v<=1,.5*v.square(),v-.5)
