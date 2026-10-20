#!/usr/bin/env python3
"""Build the URDF from the xacro.

    python3 build_urdf.py [wheeled|legged|all]      (default: all)

Writes urdf/quanta_robot_<variant>.urdf, comments stripped.

Then checks that every link puts its mesh at 0,0,0: the STL origin has to be the
joint's centre of rotation, so a visual or collision <origin> other than zero
would mean the part is being shifted to hide a bad export. Exits 1 if any is.
"""
import sys, re, shutil, subprocess
import xml.etree.ElementTree as ET
from common import path, VARIANTS

def check_origins(urdf):
    """Every visual/collision origin must be 0,0,0 with no rotation."""
    bad=[]
    for l in ET.parse(urdf).getroot().findall('link'):
        for tag in ('visual','collision'):
            e=l.find(tag)
            if e is None: continue
            o=e.find('origin')
            if o is None: continue
            xyz=[float(v) for v in (o.get('xyz') or '0 0 0').split()]
            rpy=[float(v) for v in (o.get('rpy') or '0 0 0').split()]
            if any(abs(v)>1e-12 for v in xyz+rpy):
                bad.append(f'{l.get("name")} {tag} origin xyz="{o.get("xyz")}" rpy="{o.get("rpy")}"')
    return bad

def build(variant):
    src=path('urdf','quanta_robot.xacro')
    out=path('urdf',f'quanta_robot_{variant}.urdf')
    cmd=['xacro'] if shutil.which('xacro') else [sys.executable,'-m','xacro']
    r=subprocess.run(cmd+[src,f'lower_body:={variant}'],capture_output=True,text=True)
    if r.returncode:
        sys.exit(f'xacro failed for {variant}. Is it installed?  pip install xacro\n{r.stderr}')
    t=re.sub(r'[ \t]*<!--.*?-->[ \t]*\n?','',r.stdout,flags=re.S)
    open(out,'w').write(re.sub(r'\n{3,}','\n\n',t))
    n=t.count('<link name='); j=t.count('<joint name=')
    print(f'{out}   {n} links, {j} joints')
    bad=check_origins(out)
    for b in bad: print(f'  link origin not at 0,0,0: {b}')
    print(f'  link origins: all {n} meshes sit at 0,0,0' if not bad else f'  {len(bad)} bad link origins')
    return len(bad)

if __name__=='__main__':
    want=sys.argv[1] if len(sys.argv)>1 else 'all'
    sys.exit(1 if sum(build(v) for v in (VARIANTS if want=='all' else [want])) else 0)
