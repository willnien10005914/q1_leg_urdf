#!/usr/bin/env python3
"""Import user-supplied BU10 archive; expand legged xacro and record provenance.

Native layout matches wheel: urdf/q1_inertia and meshes/q1_inertia.
No joint, limit, mass, CoM, inertia or mesh scale is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import stat
import xml.etree.ElementTree as E
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    args=p.parse_args()
    destination=ROOT/'assets/upstream'
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        for entry in z.infolist():
            name=PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(entry.external_attr>>16):
                raise ValueError(f'Unsafe archive member: {entry.filename}')
        z.extractall(destination)
    candidates=list(destination.glob('*/urdf/quanta_robot.xacro'))
    if len(candidates)!=1:
        raise ValueError('Expected exactly one upstream quanta_robot.xacro')
    upstream=candidates[0].parents[1]
    import xacro
    expanded=xacro.process_file(str(candidates[0]),mappings={'lower_body':'legged'}).toprettyxml(indent='  ')
    source=upstream/'urdf/quanta_robot_legged.urdf'
    source.write_text(expanded)
    shutil.copytree(upstream/'meshes',ROOT/'meshes/q1_inertia',dirs_exist_ok=True)
    target=ROOT/'urdf/q1_inertia/quanta_robot_legged.urdf'
    target.parent.mkdir(parents=True,exist_ok=True)
    tree=E.ElementTree(E.fromstring(expanded))
    mesh_files=[]
    for mesh in tree.getroot().findall('.//mesh'):
        old=mesh.get('filename')
        original=(source.parent/old).resolve()
        rel=original.relative_to((upstream/'meshes').resolve())
        copied=ROOT/'meshes/q1_inertia'/rel
        if not copied.is_file():raise FileNotFoundError(copied)
        mesh.set('filename','../../meshes/q1_inertia/'+rel.as_posix())
        mesh_files.append({'path':copied.relative_to(ROOT).as_posix(),'sha256':sha(copied),'scale':mesh.get('scale','1 1 1')})
    E.indent(tree)
    tree.write(target,encoding='utf-8',xml_declaration=True)
    # Root URDF compatibility entry point, like wheel/urdf/wheel_humanoid.urdf.
    for mesh in tree.getroot().findall('.//mesh'):
        mesh.set('filename',mesh.get('filename').replace('../../meshes/','../meshes/',1))
    tree.write(ROOT/'urdf/leg_humanoid.urdf',encoding='utf-8',xml_declaration=True)
    tree.write(ROOT/'urdf/quanta_robot_legged.urdf',encoding='utf-8',xml_declaration=True)
    for mesh in tree.getroot().findall('.//mesh'):
        mesh.set('filename',mesh.get('filename').removeprefix('../'))
    tree.write(ROOT/'quanta_robot_legged.urdf',encoding='utf-8',xml_declaration=True)
    report={'schema':1,'archive':args.archive.name,'archive_sha256':sha(args.archive),
        'source_xacro':candidates[0].relative_to(ROOT).as_posix(),'source_urdf':source.relative_to(ROOT).as_posix(),
        'source_urdf_sha256':sha(source),'integrated_urdf':target.relative_to(ROOT).as_posix(),'integrated_urdf_sha256':sha(target),
        'variant':'legged','physical_changes':'None. Mesh filenames relocated only; source axes, limits, masses, CoM, inertias and scales retained.',
        'upstream_known_limits':(upstream/'TODO.md').read_text(),'mesh_files':list({m['path']:m for m in mesh_files}.values())}
    (ROOT/'config/asset_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('mesh_files','upstream_known_limits')},indent=2))
    print('Unique meshes:',len(report['mesh_files']))
if __name__=='__main__':main()
