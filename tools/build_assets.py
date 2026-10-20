#!/usr/bin/env python3
"""Validate CAD, build an explicit proxy URDF and a MuJoCo verification model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/leg_humanoid_lab"))
import argparse
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as E
import numpy as np
from scipy.spatial import ConvexHull
from leg_humanoid_lab.robot import ROOT, URDF, TREE, JOINTS, ALL_JOINTS, LINKS, HEIGHT, vector, contract, FOOT_HALF, FOOT_CENTER, FOOT_VERTICES, mesh_vertices

def convex_stl(mesh,path):
    """Exact source convex envelope. Avoid MuJoCo's 200k-face STL decoder limit."""
    vertices=mesh_vertices(mesh)
    hull=ConvexHull(vertices)
    triangles=vertices[hull.simplices].copy()
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    flip=(normals*hull.equations[:,:3]).sum(axis=1)<0
    triangles[flip]=triangles[flip][:,[0,2,1]]
    normals=hull.equations[:,:3]
    dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
    out=np.zeros(len(triangles),dtype=dtype)
    out['vertices']=triangles
    out['normal']=normals
    path.write_bytes(b'Q1 native source convex hull'.ljust(80,b'\0')+len(out).to_bytes(4,'little')+out.tobytes())

def geometry(name):
    if "foot" in name:
        return FOOT_CENTER, FOOT_HALF
    if name == "center_hip_link":
        return np.array([0, 0, -.075]), np.array([.09, .085, .075])
    if name == "center_torso_link":
        return np.array([0, 0, .14]), np.array([.10, .12, .15])
    children = [j for j in ALL_JOINTS if j.find("parent").get("link") == name]
    if children:
        end = max((vector(j.find("origin")) for j in children), key=np.linalg.norm)
        center = end / 2
        size = np.maximum(abs(end) / 2, [.027, .027, .027])
        return center, size
    return vector(LINKS[name].find("inertial/origin")), np.array([.025, .025, .03])

def build():
    missing = sorted({m.get("filename") for m in TREE.findall(".//mesh") if not (URDF.parent / m.get("filename")).resolve().is_file()})
    proxy = copy.deepcopy(TREE)
    for link in proxy.findall("link"):
        center, size = geometry(link.get("name"))
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                mesh = el.find("geometry/mesh")
                if tag == "visual" and mesh is not None and (URDF.parent / mesh.get("filename")).resolve().is_file():
                    mesh.set('filename',os.path.relpath((URDF.parent/mesh.get('filename')).resolve(),ROOT/'urdf'))
                    continue
                link.remove(el)
            # Collision uses boxes for all links, visual boxes only when missing.
            if tag == "collision" or not link.findall(tag):
                el = E.SubElement(link, tag)
                E.SubElement(el, "origin", xyz=" ".join(map(str, center)))
                E.SubElement(E.SubElement(el, "geometry"), "box", size=" ".join(map(str, 2 * size)))
                if tag == "visual":
                    E.SubElement(E.SubElement(el, "material", name="proxy_orange"), "color", rgba="0.95 0.49 0.12 1")
    E.indent(proxy)
    E.ElementTree(proxy).write(ROOT / "urdf/leg_humanoid_proxy.urdf", encoding="utf-8", xml_declaration=True)

    model = E.Element("mujoco", model="q1_leg_proxy")
    E.SubElement(model, "compiler", angle="radian", autolimits="true", fusestatic="false")
    E.SubElement(model, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast", iterations="30")
    defaults = E.SubElement(model, "default")
    E.SubElement(defaults, "joint", damping="1", armature=".04")
    E.SubElement(defaults, "geom", friction="0.9 0.02 0.002", condim="3", solref="0.01 1")
    world = E.SubElement(model, "worldbody")
    E.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    E.SubElement(world, "geom", name="ground", type="plane", size="10 10 .1", rgba=".17 .20 .23 1", contype="1", conaffinity="2")
    def body(name, parent, origin=None, joint=None):
        b = E.SubElement(parent, "body", name=name, pos=" ".join(map(str, vector(origin))))
        if origin is not None:
            b.set("euler", " ".join(map(str, vector(origin, "rpy"))))
        if name == "center_hip_link":
            b.set("pos", f"0 0 {HEIGHT + .003}")
            E.SubElement(b, "freejoint", name="root")
        elif joint.get("name") in JOINTS:
            limit = joint.find("limit")
            E.SubElement(b, "joint", name=joint.get("name"), type="hinge", axis=joint.find("axis").get("xyz"), range=f'{limit.get("lower")} {limit.get("upper")}')
        inertial = LINKS[name].find("inertial")
        i = inertial.find("inertia")
        E.SubElement(b, "inertial", pos=inertial.find("origin").get("xyz"), mass=inertial.find("mass").get("value"), fullinertia=" ".join(i.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        center, size = geometry(name)
        E.SubElement(b, "geom", name=name + "_collision", type="box", pos=" ".join(map(str, center)), size=" ".join(map(str, size)), rgba=".86 .55 .23 1" if "foot" in name else ".65 .73 .80 1", contype="2", conaffinity="1")
        for j in ALL_JOINTS:
            if j.find("parent").get("link") == name:
                body(j.find("child").get("link"), b, j.find("origin"), j)
    body("center_hip_link", world)
    actuators = E.SubElement(model, "actuator")
    for name in JOINTS:
        E.SubElement(actuators, "motor", name=name, joint=name, ctrllimited="true", ctrlrange="-100 100")
    E.indent(model)
    E.ElementTree(model).write(ROOT / "urdf/leg_humanoid_proxy.xml", encoding="utf-8", xml_declaration=True)
    if not missing:
        native=copy.deepcopy(model)
        native.set('model','q1_leg_native')
        asset=E.SubElement(native,'asset')
        collision_dir=ROOT/'meshes/collision_native'
        collision_dir.mkdir(parents=True,exist_ok=True)
        for name,link in LINKS.items():
            mesh=link.find('collision/geometry/mesh')
            hull=collision_dir/(name+'.stl')
            convex_stl(mesh,hull)
            path=os.path.relpath(hull,ROOT/'urdf')
            E.SubElement(asset,'mesh',name=name+'_mesh',file=path,scale='1 1 1')
            geom=native.find(f".//geom[@name='{name}_collision']")
            geom.attrib.pop('size')
            geom.set('type','mesh')
            geom.set('mesh',name+'_mesh')
            geom.set('pos','0 0 0')
            geom.set('rgba','.55 .66 .76 1')
        E.indent(native)
        E.ElementTree(native).write(ROOT/'urdf/leg_humanoid.xml',encoding='utf-8',xml_declaration=True)
    manifest = {**contract(), "source_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(), "missing_meshes": missing,
                "native_cad_ready": not missing, "proxy": False,
                "link_count":len(LINKS),"movable_joint_count":len(ALL_JOINTS),"total_mass_kg":sum(float(l.find('inertial/mass').get('value')) for l in LINKS.values()),
                "native_collision": "Source STL convex hulls; no self collision. Source masses, CoM, inertia and mesh scales retained.",
                "foot_bounds":{s:{'min':FOOT_VERTICES[s].min(axis=0).tolist(),'max':FOOT_VERTICES[s].max(axis=0).tolist()} for s in ('left','right')},
                "proxy_notes": "Optional proxy only: native axes and inertias with box collisions; foot size 0.22 x 0.10 x 0.04m assumed."}
    (ROOT / "config/robot_contract.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"actions": len(JOINTS), "height": HEIGHT, "missing_meshes": len(missing), "native_cad_ready": not missing}, indent=2))
    return manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--require-native", action="store_true")
    args = p.parse_args()
    report = build()
    if args.require_native and report["missing_meshes"]:
        raise SystemExit("Missing CAD meshes: place LOWERBODY_LEGGED/MESH under meshes/ first.")
