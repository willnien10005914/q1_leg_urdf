#!/usr/bin/env python3
"""Export raw native poses for a labelled, read-only web diagnostic replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=7.2)
    args = parser.parse_args()
    if args.output.exists() or args.duration <= 0:
        parser.error('Use a fresh output and positive duration')
    report = json.loads(args.report.read_text())
    index, row = next((i, r) for i, r in enumerate(report['rows'])
                      if r['variant'] == args.variant and r['case'] == 'pair')
    if not row['survived'] or not report['experimental'] or report['accepted']:
        raise ValueError('Only surviving, explicitly experimental diagnostics may be exported')
    trace = args.report.with_suffix('.npz')
    with np.load(trace, allow_pickle=False) as data:
        step_dt = report['duration_s'] / len(data['valid'])
        count = round(args.duration / step_dt)
        if count > len(data['valid']) or not data['valid'][:count, index].all():
            raise ValueError('Missing frames or invalid native episode')
        frames = []
        for i in range(count):
            q, root, quat = (data[name][i, index] for name in
                              ('q', 'root_position', 'root_quaternion'))
            if q.shape != (23,) or not all(np.isfinite(x).all() for x in (q, root, quat)):
                raise ValueError('Invalid native pose')
            frames.append({'time': round((i + 1) * step_dt, 6), 'q': q.tolist(),
                           'root': root.tolist(), 'quaternion': quat.tolist()})
    payload = {'schema': 1, 'experimental': True, 'accepted': False, 'passed': False,
               'retrained': False, 'proxy': False, 'source': 'native_physx_recorded_poses',
               'variant': args.variant,
               'label': {'baseline': '原設定',
                         'upper_residual_guard': '限制腰臂 PPO 修正幅度，保留原 PD',
                         'upper_pd': '實驗腰臂 PD',
                         'waist_pd_guard': '限制腰臂 PPO 修正幅度，只加強腰部阻尼，保留手臂原 PD',
                         'upper_pd_guard': '實驗腰臂 PD 與修正幅度限制'}[args.variant],
               'checkpoint_updates': 13115, 'checkpoint_sha256': report['checkpoint_sha256'],
               'profile': report['profile'], 'profile_module_sha256': report['profile_module_sha256'],
               'report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
               'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
               'root_forces_or_constraints': False, 'rendered_pose_filtering': False,
               'metrics': {key: row[key] for key in ('chest_tilt_peak_deg', 'arm_reference_rmse_rad')},
               'frames': frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    print(f'{args.output}: {len(frames)} raw native poses')


if __name__ == '__main__':
    main()
