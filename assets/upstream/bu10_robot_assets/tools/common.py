"""Shared helpers: project paths, STL loading, spreadsheet reading."""
import os, re, struct, numpy as np, openpyxl

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))   # the folder holding meshes/ urdf/ ME_data/ tools/
def path(*p): return os.path.join(ROOT,*p)
def book(variant): return path('ME_data',f'link_dim_{variant}.xlsx')
def out_dir(tool):
    """Each tool writes into output/<tool>/."""
    d=path('output',tool); os.makedirs(d,exist_ok=True); return d
VARIANTS=('wheeled','legged')

def load_stl(p):
    """Return the triangles of a binary STL as (n,3,3)."""
    d=open(p,'rb').read(); n=struct.unpack('<I',d[80:84])[0]
    a=np.frombuffer(d[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]))
    return a['v'].reshape(-1,3,3).astype(float)

def _rows(ws,hdr_col1='Part'):
    hr=[r for r in range(1,12) if ws.cell(r,1).value==hdr_col1][0]
    head=[ws.cell(hr,c).value for c in range(1,ws.max_column+1)]
    out=[]
    for r in range(hr+1,ws.max_row+1):
        v=dict(zip(head,[ws.cell(r,c).value for c in range(1,ws.max_column+1)]))
        if any(x not in (None,'') for x in v.values()): out.append(v)
    return out

def read_model(variant):
    """Build the kinematic model for one variant from the common + variant workbooks.

    Returns (links, joints) where links[name] = {'mesh','scale','part'} and
    joints[name] = {'parent','child','xyz','axis'}."""
    links={}; joints={}
    for key in ('common',variant):
        ws=openpyxl.load_workbook(book(key),data_only=True)['link_dim']
        parent=None
        for v in _rows(ws):
            if v['Link']: parent=v['Link']
            if v.get('mesh file'):
                links[v['Link']]={'mesh':path(v['mesh file']),
                                  'scale':float(str(v.get('scale') or '1').split()[0]),
                                  'part':v['Part']}
            j=v['Joint it carries']
            if j and not j.startswith('('):
                joints[j]={'parent':parent,'child':v['child link'],
                           'xyz':np.array([float(v['X']),float(v['Y']),float(v['Z'])]),
                           'axis':v['axis'] or ''}
    return links,joints

def fk(links,joints):
    """World position of every link origin, root first."""
    kids={}
    for n,j in joints.items(): kids.setdefault(j['parent'],[]).append(n)
    children={j['child'] for j in joints.values()}
    root=[l for l in links if l not in children]
    assert len(root)==1, f'expected one root link, got {root}'
    pos={root[0]:np.zeros(3)}
    def walk(l):
        for n in kids.get(l,[]):
            c=joints[n]['child']; pos[c]=pos[l]+joints[n]['xyz']; walk(c)
    walk(root[0])
    return pos,root[0]

def read_inertia(variant):
    """inertia pages of the common + variant workbooks: name -> dict or None."""
    out={}
    for key in ('common',variant):
        ws=openpyxl.load_workbook(book(key),data_only=True)['inertia']
        hr=[r for r in range(1,12) if ws.cell(r,1).value=='Part'][0]
        for r in range(hr+1,ws.max_row+1):
            l=ws.cell(r,2).value
            if not l: continue
            m=ws.cell(r,3).value
            if m in (None,''): out[l]=None; continue
            out[l]=dict(mass=float(m),cog=[float(ws.cell(r,c).value) for c in (4,5,6)],
                        I=[float(ws.cell(r,c).value) for c in range(7,13)],
                        source=ws.cell(r,14).value,note=ws.cell(r,15).value)
    return out
