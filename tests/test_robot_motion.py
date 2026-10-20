import json
import numpy as np
import mujoco
from leg_humanoid_lab.robot import ROOT,URDF,JOINTS,LOWER,UPPER,DEFAULT,HEIGHT,TREE,fk
from leg_humanoid_lab.motion import Motion
from leg_humanoid_lab.env import LegEnv

def test_native_axes_and_inertias_survive_model_conversion():
    model=mujoco.MjModel.from_xml_path(str(ROOT/"urdf/leg_humanoid.xml"))
    for joint in TREE.findall('joint'):
        if joint.get('name') in JOINTS:
            np.testing.assert_allclose(model.joint(joint.get('name')).axis,np.fromstring(joint.find('axis').get('xyz'),sep=' '))
    assert model.nu==len(JOINTS)
    for link in TREE.findall('link'):
        mass=float(link.find('inertial/mass').get('value'))
        np.testing.assert_allclose(model.body(link.get('name')).mass,[mass])
    left=JOINTS.index('left_hip_Pitch_joint')
    right=JOINTS.index('right_hip_Pitch_joint')
    q=np.zeros(len(JOINTS));q[left]=.2;q[right]=-.2
    t=fk(q)
    assert t['left_foot_link'][0,3]>0 and t['right_foot_link'][0,3]>0

def test_extracted_reference_has_traceable_source_and_native_limits():
    motion=Motion('sidestep')
    assert motion.data['quality']['coverage']>=.65
    assert motion.data['quality']['max_gap_s']<=.25
    assert motion.data['clip_start']==0 and motion.data['clip_end']==7
    assert len(motion.data['model_sha256'])==64
    assert np.isfinite(motion.q).all()
    assert np.all(motion.q>=LOWER-1e-6) and np.all(motion.q<=UPPER+1e-6)
    speed=np.abs(np.diff(motion.q,axis=0)/np.diff(motion.t)[:,None])
    assert speed.max()<=10.00001
    q1,r1,_=motion.sample(motion.duration)
    q2,r2,_=motion.sample(motion.duration+10)
    np.testing.assert_allclose(q1,q2)
    np.testing.assert_allclose(r1,r2)

def test_shared_observations_actions_and_termination():
    spaces=[]
    for skill in ('stand','walk','sidestep'):
        env=LegEnv(skill,randomize=False)
        obs,info=env.reset(seed=123)
        spaces.append((env.action_space.shape,env.observation_space.shape))
        assert env.observation_space.contains(obs)
        obs,reward,done,truncated,info=env.step(np.zeros(len(JOINTS)))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        env.data.qpos[2]=.1
        _,_,done,_,info=env.step(np.zeros(len(JOINTS)))
        assert done and info['fallen']
        env.close()
    assert len(set(spaces))==1

def test_native_archive_meshes_resolve_and_are_unmodified():
    import hashlib
    data=json.loads((ROOT/'config/asset_manifest.json').read_text())
    assert data['variant']=='legged' and len(data['mesh_files'])==30
    for entry in data['mesh_files']:
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    for mesh in TREE.findall('.//mesh'):
        assert (URDF.parent/mesh.get('filename')).resolve().is_file()
    contract=json.loads((ROOT/'config/robot_contract.json').read_text())
    assert contract['native_cad_ready'] and not contract['missing_meshes']
    assert HEIGHT>.88 and HEIGHT<.89
    import xml.etree.ElementTree as E
    for path in (ROOT/'quanta_robot_legged.urdf',ROOT/'urdf/leg_humanoid.urdf',URDF):
        for mesh in E.parse(path).getroot().findall('.//mesh'):
            assert (path.parent/mesh.get('filename')).resolve().is_file()

def test_staying_still_does_not_pass_sidestep_evaluation():
    from leg_humanoid_lab.evaluation import passes
    report={'skill':'sidestep','survival_rate':1.,'upright':1.,'height_error':.01,
            'pose_rmse':.10,'root_error':.10,'reference_lateral_range':.4,'lateral_range':0.}
    assert not passes(report)
    report['lateral_range']=.3
    assert passes(report)
