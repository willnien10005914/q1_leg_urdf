#!/usr/bin/env python3
"""Offline native-CAD evaluation of swing IK against captured root poses.

No simulator is started. Results are geometry predictions conditional on the
captured pelvis pose, not physical clearance or learned-policy acceptance.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/leg_humanoid_lab'))

import torch

from leg_humanoid_lab.swing_body_ik import BodyAwareSwingIK
from leg_humanoid_lab.robot import LOWER,UPPER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--iterations',type=int,default=3)
    args=p.parse_args()
    source=json.loads(args.trace.read_text())
    rows=[row for row in source['trace'] if row.get('mode') in (1,2,3)
          and 'root_pose_world' in row and 'reference_joint_position_rad' in row]
    if not rows:raise ValueError('The trace needs full joint and root poses during motion')
    q=torch.tensor([row['reference_joint_position_rad'] for row in rows])
    quaternion=torch.tensor([row['root_pose_world'][3:] for row in rows])
    height=torch.tensor([row['root_height_m'] for row in rows])
    target=torch.tensor([row['target_clearance_m_per_foot'] for row in rows])
    contact=torch.tensor([row['planned_contact_per_foot'] for row in rows])
    ik=BodyAwareSwingIK(iterations=args.iterations)
    result=ik(q,quaternion,height,target,contact)
    original=ik.sole_height(q[...,ik.joint_ids],quaternion,height)
    corrected=ik.sole_height(result[...,ik.joint_ids],quaternion,height)
    active=(~contact)&(target>.022)
    predictions=[]
    for index,row in enumerate(rows):
        if not bool(active[index].any()):continue
        predictions.append({'env_index':row['env_index'],'case':row['case'],
            'time_s':row['time_s'],'gesture_phase':row.get('gesture_phase'),
            'actual_native_clearance_m':row['sole_clearance_m_per_foot'],
            'reference_under_actual_root_cad_m':original[index].tolist(),
            'ik_under_actual_root_cad_m':corrected[index].tolist(),
            'requested_clearance_m':target[index].tolist(),
            'planned_contact':contact[index].tolist(),
            'maximum_joint_correction_rad':float((result[index]-q[index]).abs().max())})
    report={'schema':1,'diagnostic':True,'accepted':False,
        'scope':'Offline kinematic prediction under captured root pose; not a physical rollout or policy acceptance',
        'source_trace':str(args.trace.resolve()),'checkpoint_sha256':source.get('checkpoint_sha256'),
        'iterations':args.iterations,'input_frames':len(rows),'high_swing_samples':int(active.sum()),
        'finite':bool(torch.isfinite(result).all()),
        'native_joint_limits_respected':bool(((result>=torch.tensor(LOWER)-1e-6)&(result<=torch.tensor(UPPER)+1e-6)).all()),
        'high_swing_geometry_reaching_25mm_fraction':float((corrected[active]>=.025).float().mean()) if bool(active.any()) else None,
        'predictions':predictions}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))


if __name__=='__main__':main()
