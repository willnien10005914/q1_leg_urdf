#!/usr/bin/env python3
"""Sanity-check the inertia page of each workbook against the meshes.

    python3 check_inertia.py [wheeled|legged|all] [--plot]   (default: all)

Flags, per link:
  - no data at all
  - mass that cannot be right for the part's volume (density outside 200..12000 kg/m3)
  - centre of mass outside the mesh bounding box
  - inertia tensor that is not physically possible (non-positive diagonal,
    triangle inequality violated, or products of inertia too large)
  - a tensor far from what a solid box of that mass and bounding box would give
  - left/right pairs whose mass or tensor do not match
--plot writes output/inertia/<link>.png: the mesh in three views with the centre
of mass marked, so you can see whether it sits inside the part.
Exit code is 1 if anything is flagged, so it can be used as a gate.
"""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from common import out_dir, load_stl, read_model, read_inertia, VARIANTS

VIEWS=[((0,2),1,'side  (X-Z)'),((1,2),0,'front  (Y-Z)'),((0,1),2,'top  (X-Y)')]
def plot_link(name,t,e,flags):
    """Mesh + centre of mass, one figure per link."""
    c=np.array(e['cog']); v=t.reshape(-1,3); mn,mx=v.min(0),v.max(0)
    fig,axs=plt.subplots(1,3,figsize=(15,5.6))
    for ax,((a,b),d,ttl) in zip(axs,VIEWS):
        o=np.argsort(t[:,:,d].mean(1))
        ax.add_collection(PolyCollection(t[o][:,:,[a,b]],facecolors='#8fb3cc',edgecolors='none',alpha=.85))
        ax.add_patch(plt.Rectangle((mn[a],mn[b]),mx[a]-mn[a],mx[b]-mn[b],fill=False,ec='navy',ls='--',lw=1))
        ax.plot(0,0,'k+',ms=13,mew=2,zorder=5)
        ax.annotate('link origin',(0,0),xytext=(6,-13),textcoords='offset points',fontsize=8)
        inside=all(mn<=c)and all(c<=mx)
        ax.plot(c[a],c[b],marker='*',ms=19,color='seagreen' if inside else 'crimson',mec='k',mew=.8,zorder=6)
        ax.annotate(f'CoM ({c[a]:+.4f}, {c[b]:+.4f})',(c[a],c[b]),xytext=(8,8),textcoords='offset points',
                    fontsize=8,color='seagreen' if inside else 'crimson',
                    bbox=dict(fc='white',ec='seagreen' if inside else 'crimson',lw=.6,alpha=.9))
        ax.set_aspect('equal'); ax.grid(True,lw=.3); ax.autoscale()
        ax.set_title(ttl,fontsize=10); ax.set_xlabel('XYZ'[a]+' (m)'); ax.set_ylabel('XYZ'[b]+' (m)')
    out=np.maximum(np.maximum(mn-c,c-mx),0)*1000
    head=f'{name}   mass {e["mass"]:.3f} kg   CoM {np.round(c,4)}'
    head+='   INSIDE the mesh' if out.max()<=0 else f'   OUTSIDE the mesh by {np.round(out,1)} mm (x,y,z)'
    if flags: head+='\n'+'  |  '.join(flags)
    fig.suptitle(head,fontsize=12,color='0.1' if out.max()<=0 and not flags else 'crimson')
    fig.tight_layout(rect=[0,0,1,.93])
    f=os.path.join(out_dir('inertia'),f'{name}.png'); fig.savefig(f,dpi=80); plt.close(fig)
    return f

DENS=(200,12000)        # kg/m3, loosest plausible range for a built-up assembly
def check(variant,plot=False):
    links,_=read_model(variant); I=read_inertia(variant)
    bad=[]
    vals={}; tris={}; per={}
    for l,d in sorted(links.items()):
        t=load_stl(d['mesh'])*d['scale']
        v=t.reshape(-1,3); mn,mx=v.min(0),v.max(0); box=mx-mn
        # closed-mesh volume via the divergence theorem
        vol=abs(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
        e=I.get(l)
        tris[l]=t
        if e is None: bad.append((l,'no inertia data')); continue
        m=e['mass']; c=np.array(e['cog']); ixx,ixy,ixz,iyy,iyz,izz=e['I']
        vals[l]=(m,c,np.array(e['I']))
        if vol>1e-9:
            rho=m/vol
            if not DENS[0]<rho<DENS[1]:
                bad.append((l,f'implied density {rho:8.0f} kg/m3 (mass {m:.3f} kg, mesh volume {vol*1e6:.0f} cm3)'))
        if not(all(mn-0.02<=c) and all(c<=mx+0.02)):
            bad.append((l,f'CoM {np.round(c,4)} outside the mesh box x{np.round([mn[0],mx[0]],3)} '
                          f'y{np.round([mn[1],mx[1]],3)} z{np.round([mn[2],mx[2]],3)}'))
        if min(ixx,iyy,izz)<=0: bad.append((l,'non-positive moment of inertia'))
        elif ixx+iyy<izz or ixx+izz<iyy or iyy+izz<ixx: bad.append((l,'triangle inequality violated'))
        elif min(np.linalg.eigvalsh(np.array([[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]])))<=0:
            bad.append((l,'tensor not positive definite'))
        else:
            box_I=np.array([m/12*(box[1]**2+box[2]**2),m/12*(box[0]**2+box[2]**2),m/12*(box[0]**2+box[1]**2)])
            r=np.array([ixx,iyy,izz])/box_I
            if r.max()>1.6 or r.min()<0.04:
                bad.append((l,f'tensor far from a solid box of the same size: Ixx/Iyy/Izz are '
                              f'{np.round(r,2)} x the box value'))
    for l in sorted(vals):
        if not l.startswith('left_'): continue
        r='right_'+l[5:]
        if r not in vals: continue
        (m1,c1,I1),(m2,c2,I2)=vals[l],vals[r]
        if abs(m1-m2)>0.02*max(m1,m2)+1e-6:
            bad.append((l,f'mass differs from {r}: {m1:.3f} vs {m2:.3f} kg'))
        if np.abs(c1*[1,-1,1]-c2).max()>2e-3:
            bad.append((l,f'CoM is not the mirror of {r}: {np.round(c1,4)} vs {np.round(c2,4)}'))
        d=np.abs(I1[[0,3,5]]-I2[[0,3,5]])/np.maximum(np.abs(I1[[0,3,5]]),1e-12)
        if d.max()>0.05:
            bad.append((l,f'diagonal tensor differs from {r} by {d.max()*100:.0f} %'))
    print(f'=== {variant}: {len(links)} links, {sum(1 for l in links if I.get(l))} with data')
    for l,msg in bad:
        print(f'  {l:26s} {msg}'); per.setdefault(l,[]).append(msg)
    if not bad: print('  all checks pass')
    if plot:
        for l in sorted(tris):
            if I.get(l): plot_link(l,tris[l],I[l],per.get(l,[]))
        print(f'  plots in {out_dir("inertia")}')
    return len(bad)

if __name__=='__main__':
    args=[a for a in sys.argv[1:] if not a.startswith('--')]
    want=args[0] if args else 'all'
    plot='--plot' in sys.argv
    n=sum(check(v,plot) for v in (VARIANTS if want=='all' else [want]))
    sys.exit(1 if n else 0)
