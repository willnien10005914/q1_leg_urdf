#!/usr/bin/env python3
"""Check that every left STL is the mirror of its right counterpart.

    python3 check_symmetry.py [--plot]

Mirrors each left mesh in Y and measures the surface-to-surface distance to the
right one. Prints a table sorted by the area that disagrees by more than 1 mm.
--plot writes output/symmetry/lr_diff_<link>.png for every pair, so the
real differences can be told from the export tessellation by eye.
"""
import sys, os, glob, numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from common import out_dir, load_stl, read_model, VARIANTS

def sample(t,step=8e-4,cap=150000):
    a=0.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    k=np.clip((a/(step*step)).astype(int)+1,1,400)
    if k.sum()>cap: k=np.maximum(1,(k*cap/k.sum()).astype(int))
    i=np.repeat(np.arange(len(t)),k); r1,r2=np.random.rand(len(i)),np.random.rand(len(i))
    m=r1+r2>1; r1[m],r2[m]=1-r1[m],1-r2[m]; T=t[i]
    return T[:,0]+(T[:,1]-T[:,0])*r1[:,None]+(T[:,2]-T[:,0])*r2[:,None]

def _closest(p,t):
    a,b,c=t[:,0],t[:,1],t[:,2]; ab,ac=b-a,c-a; ap=p-a
    d1=(ab*ap).sum(1); d2=(ac*ap).sum(1); bp=p-b; d3=(ab*bp).sum(1); d4=(ac*bp).sum(1)
    cp=p-c; d5=(ab*cp).sum(1); d6=(ac*cp).sum(1)
    vc=d1*d4-d3*d2; vb=d5*d2-d1*d6; va=d3*d6-d5*d4
    den=np.where(np.abs(va+vb+vc)<1e-20,1e-20,va+vb+vc)
    q=a+ab*(vb/den)[:,None]+ac*(vc/den)[:,None]
    e=np.clip(d1/np.where(d1-d3==0,1e-20,d1-d3),0,1); q=np.where(((vc<=0)&(d1>=0)&(d3<=0))[:,None],a+ab*e[:,None],q)
    e=np.clip(d2/np.where(d2-d6==0,1e-20,d2-d6),0,1); q=np.where(((vb<=0)&(d2>=0)&(d6<=0))[:,None],a+ac*e[:,None],q)
    e=np.clip((d4-d3)/np.where((d4-d3)+(d5-d6)==0,1e-20,(d4-d3)+(d5-d6)),0,1)
    q=np.where(((va<=0)&(d4-d3>=0)&(d5-d6>=0))[:,None],b+(c-b)*e[:,None],q)
    q=np.where(((d1<=0)&(d2<=0))[:,None],a,q); q=np.where(((d3>=0)&(d4<=d3))[:,None],b,q)
    q=np.where(((d6>=0)&(d5<=d6))[:,None],c,q)
    return np.linalg.norm(p-q,axis=1)

def dist_to_mesh(P,T,k=24):
    tree=cKDTree(T.mean(1)); out=np.empty(len(P))
    for i in range(0,len(P),20000):
        p=P[i:i+20000]; _,idx=tree.query(p,k=min(k,len(T)))
        out[i:i+20000]=np.stack([_closest(p,T[idx[:,j]]) for j in range(idx.shape[1])],1).min(1)
    return out

def pairs():
    seen={}
    for v in VARIANTS:
        links,_=read_model(v)
        for n,d in links.items():
            if n.startswith('left_'): seen.setdefault(n,(d,links.get('right_'+n[5:])))
    return [(n,a,b) for n,(a,b) in seen.items() if b]

VIEW=[((1,2),0,'front  (Y-Z)'),((0,2),1,'side  (X-Z)'),((0,1),2,'top  (X-Y)')]
def main(plot=False):
    rows=[]
    for n,ld,rd in pairs():
        L=load_stl(ld['mesh'])*ld['scale']; R=load_stl(rd['mesh'])*rd['scale']; L[:,:,1]*=-1
        ar=lambda t:0.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
        dL=dist_to_mesh(L.mean(1),R)*1000; dR=dist_to_mesh(R.mean(1),L)*1000
        aL,aR=ar(L),ar(R)
        rows.append(dict(n=n[5:],L=L,R=R,dL=dL,mx=max(dL.max(),dR.max()),
                         A=(aL[dL>1].sum()+aR[dR>1].sum())/2*1e4,
                         pc=np.concatenate([aL[dL>1],aR[dR>1]]).sum()/(aL.sum()+aR.sum())*100))
    rows.sort(key=lambda r:-r['A'])
    print(f'{"link pair":26s}{"max dev":>10s}{"area >1mm":>12s}{"of surface":>12s}')
    for r in rows: print(f'  {r["n"]:24s}{r["mx"]:8.2f}mm{r["A"]:10.1f}cm2{r["pc"]:10.1f} %')
    if not plot: return
    for r in rows:
        L,R=r['L'],r['R']; fig,axs=plt.subplots(1,4,figsize=(23,7))
        tri=np.vstack([L,R]); col=np.array([(0.55,0.05,0.08,.8)]*len(L)+[(0.15,0.45,0.80,.8)]*len(R))
        for ax,((a,b),c,ttl) in zip(axs[:3],VIEW):
            o=np.argsort(tri[:,:,c].mean(1))
            ax.add_collection(PolyCollection(tri[o][:,:,[a,b]],facecolors=col[o],edgecolors='none'))
            ax.autoscale(); ax.set_aspect('equal'); ax.grid(True,lw=.3); ax.set_title(ttl,fontsize=10)
        ax=axs[3]; (a,b),c,_=VIEW[0]; o=np.argsort(L[:,:,c].mean(1)); dc=r['dL']
        fc=np.where((dc>1)[:,None],plt.cm.inferno(np.clip(dc/r['mx'],0,1))[:,:4],np.array([.8,.8,.8,1.]))
        ax.add_collection(PolyCollection(L[o][:,:,[a,b]],facecolors=fc[o],edgecolors='none'))
        ax.autoscale(); ax.set_aspect('equal'); ax.grid(True,lw=.3)
        ax.set_title(f'left surface, distance to right (grey <1 mm, colour -> {r["mx"]:.1f} mm)',fontsize=10)
        fig.suptitle(f'{r["n"]}:  mirrored LEFT = red, RIGHT = blue   |   max {r["mx"]:.1f} mm, '
                     f'{r["A"]:.1f} cm2 ({r["pc"]:.1f} %) off by >1 mm',fontsize=13)
        fig.tight_layout(); f=os.path.join(out_dir('symmetry'),f'lr_diff_{r["n"]}.png'); fig.savefig(f,dpi=70); plt.close(fig)
        print('  wrote',f)

if __name__=='__main__': main('--plot' in sys.argv)
