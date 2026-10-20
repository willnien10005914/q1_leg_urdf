#!/usr/bin/env python3
"""Draw the robot from the spreadsheets and the STL files - the URDF is not used.

    python3 generate_pdf.py [wheeled|legged|all]     (default: all)

Writes output/pdf/quanta_robot_<variant>.pdf: whole robot, each leg,
each arm and the torso, in front / side / top view. Every link gets its name on
the link, a marker at its origin, and the parent->child offset split per axis
along a stepped line.
"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from common import out_dir, load_stl, read_model, fk, VARIANTS

VIEWS=[((1,2),0,'FRONT   (Y–Z)',-1),((0,2),1,'SIDE   (X–Z)',+1),((0,1),2,'TOP   (X–Y)',+1)]

def panel(ax,sel,view,tri,pos,jof,col,order,fs=7):
    (a,b),c,title,sgn=view
    for l in order:
        if l not in sel: continue
        t=tri[l]; o=np.argsort(sgn*t[:,:,c].mean(1))
        ax.add_collection(PolyCollection(t[o][:,:,[a,b]],facecolors=[col[l]]*len(t),
                          edgecolors='none',alpha=.9,rasterized=True))
    spots=[]
    for l in sel:
        t=tri[l][:,:,[a,b]]
        w=np.abs((t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0]))
        cc=t.mean(1)
        if w.sum()<=0: w=np.ones(len(t))
        spots.append([l,(cc[:,0]*w).sum()/w.sum(),(cc[:,1]*w).sum()/w.sum()])
    xs=[p[1] for p in spots]+[pos[l][a] for l in sel]; ys=[p[2] for p in spots]+[pos[l][b] for l in sel]
    xr=max(max(xs)-min(xs),1e-6); yr=max(max(ys)-min(ys),1e-6)
    step=yr*.05; near=xr*.25
    spots.sort(key=lambda p:-p[2])
    for i in range(1,len(spots)):
        for k in range(i):
            if abs(spots[i][2]-spots[k][2])<step and abs(spots[i][1]-spots[k][1])<near:
                spots[i][2]=spots[k][2]-step
    taken=[(p[1],p[2],xr*.34,yr*.045) for p in spots]
    free=lambda x,y: min(max(abs(x-t)/w,abs(y-u)/v) for t,u,w,v in taken) if taken else 9
    for child,(parent,off) in jof.items():
        if child not in sel or parent not in sel: continue
        p,g=pos[parent],pos[child]
        ax.plot([p[a],g[a]],[p[b],p[b]],'-',color='k',lw=.7,alpha=.55,zorder=4)
        ax.plot([g[a],g[a]],[p[b],g[b]],'-',color='k',lw=.7,alpha=.55,zorder=4)
        for along,(x0,x1,y0,y1) in ((a,(p[a],g[a],p[b],p[b])),(b,(g[a],g[a],p[b],g[b]))):
            d=(x1-x0) if along==a else (y1-y0)
            if abs(d)<2e-3: continue
            cand=[(x0+(x1-x0)*f,y0+(y1-y0)*f) for f in (.5,.42,.58,.34,.66,.26,.74,.18,.82,.12,.88)]
            bx,by=max(cand,key=lambda q:free(*q))
            ax.annotate(f'Δ{"XYZ"[along].lower()} {d:+.4f}',(bx,by),fontsize=fs-1.4,color='#8B1A1A',
                        ha='center',va='center',zorder=8,
                        bbox=dict(fc='white',ec='#E0B4B4',lw=.4,alpha=.92,boxstyle='round,pad=0.2'))
            taken.append((bx,by,xr*.20,yr*.040))
    for l in sel:
        g=pos[l]
        ax.plot(g[a],g[b],marker='o',ms=3.4,mfc='w',mec='k',mew=1.2,zorder=6)
        ax.plot(g[a],g[b],marker='+',ms=6.5,color='k',mew=1.0,zorder=6)
    for l,x,y in spots:
        ax.text(x,y,l,fontsize=fs,ha='center',va='center',zorder=7,color='0.1',
                bbox=dict(fc='white',ec=col[l],lw=1.0,alpha=.9,boxstyle='round,pad=0.25'))
    ax.set_aspect('equal'); ax.grid(True,lw=.3,alpha=.5); ax.set_title(title,fontsize=11,weight='bold')
    ax.set_xlabel('XYZ'[a]+'  (m)',fontsize=8); ax.set_ylabel('XYZ'[b]+'  (m)',fontsize=8)
    ax.tick_params(labelsize=7); ax.autoscale()

def build(variant):
    links,joints=read_model(variant); pos,root=fk(links,joints)
    tri={l:load_stl(d['mesh'])*d['scale']+pos[l] for l,d in links.items()}
    jof={j['child']:(j['parent'],j['xyz']) for j in joints.values()}
    cmap=plt.get_cmap('tab20'); col={l:cmap(i%20) for i,l in enumerate(sorted(links))}
    order=sorted(links,key=lambda l:-pos[l][2])
    ARMK=('shoulder','elbow','wrist','gripper','finger')
    arm=lambda s:[l for l in links if l.startswith(s) and any(k in l for k in ARMK)]
    leg=lambda s:[l for l in links if l.startswith(s) and not any(k in l for k in ARMK)]
    body=[l for l in links if l.startswith('center')]
    pages=[('Whole robot',list(links),6),('Left leg',body[:1]+leg('left'),9),('Right leg',body[:1]+leg('right'),9),
           ('Left arm',[l for l in body if 'torso' in l]+arm('left'),9),
           ('Right arm',[l for l in body if 'torso' in l]+arm('right'),9),('Torso / waist',body,9)]
    out=os.path.join(out_dir('pdf'),f'quanta_robot_{variant}.pdf')
    with PdfPages(out) as pdf:
        for name,sel,fs in pages:
            fig,axs=plt.subplots(1,3,figsize=(19,11))
            for ax,v in zip(axs,VIEWS): panel(ax,set(sel),v,tri,pos,jof,col,order,fs)
            fig.suptitle(f'quanta_robot  —  {variant.upper()}  —  {name}\n'
                 'built from link_dim_common + link_dim_%s and the STL files   •   '
                 'marker = link origin   •   stepped red line = parent → child offset, split per axis'%variant,
                 fontsize=12)
            fig.tight_layout(rect=[0,0,1,.95]); pdf.savefig(fig,dpi=200); plt.close(fig)
        pdf.infodict()['Title']=f'quanta_robot {variant} (from spreadsheet)'
    print(f'{out}   {len(pages)} pages, {len(links)} links, {len(joints)} joints')

if __name__=='__main__':
    want=sys.argv[1] if len(sys.argv)>1 else 'all'
    for v in (VARIANTS if want=='all' else [want]): build(v)
