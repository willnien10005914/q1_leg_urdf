#!/usr/bin/env python3
"""Resume each stage until its gate passes; downstream work is never mislabeled."""
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'config/curriculum.json')
    p.add_argument('--allow-proxy',action='store_true')
    p.add_argument('--attempts',type=int,default=3)
    args=p.parse_args()
    if args.attempts<1:p.error('--attempts must be at least 1')
    config=json.loads(args.config.read_text())
    checkpoint=None
    for stage in config['stages']:
        skill=stage['skill']
        folder=ROOT/'checkpoints/isaac'/skill
        contract=folder/'contract.json'
        expected=json.loads((ROOT/'config/robot_contract.json').read_text())
        saved=json.loads(contract.read_text()) if contract.exists() else {}
        same_geometry=saved.get('proxy')==args.allow_proxy and saved.get('robot')==config['contract'] and saved.get('urdf_sha256')==expected['urdf_sha256']
        own=folder/'latest.pt'
        if own.exists() and same_geometry:
            if stage.get('warm_start')!='previous_stage' or checkpoint is None:
                checkpoint=own
            evaluation=folder/'evaluation.json'
            report=json.loads(evaluation.read_text()) if evaluation.exists() else {}
            if not config.get('force_train',False) and report.get('passed') and report.get('urdf_sha256')==expected['urdf_sha256'] and report.get('proxy')==args.allow_proxy and (skill!='sidestep' or report.get('episodes',0)>=32):
                print(f'{skill}: existing native contract and evaluation passed; continuing to next stage.',flush=True)
                continue
        for attempt in range(args.attempts):
            # Keep every attempt's snapshots; RSL-RL iteration names restart at zero.
            snapshots=list(folder.glob('model_*.pt'))
            if snapshots:
                history=folder/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                history.mkdir(parents=True,exist_ok=True)
                for snapshot in snapshots:shutil.move(str(snapshot),str(history/snapshot.name))
                for name in ('contract.json','evaluation.json','latest.pt'):
                    if (folder/name).exists():shutil.copyfile(folder/name,history/name)
            status={'state':'training','skill':skill,'attempt':attempt+1,'backend':'isaaclab','proxy':args.allow_proxy,'envs':config['num_envs'],'iterations':stage['iterations'],'config':str(args.config.resolve()),'warm_start':str(checkpoint) if checkpoint else None}
            (ROOT/'outputs/training_status.json').write_text(json.dumps(status,indent=2))
            command=['bash',str(ROOT/'train_native.sh'),'--skill',skill,'--num_envs',str(config['num_envs']),'--max_iterations',str(stage['iterations']),'--headless']
            if args.allow_proxy:command.append('--allow-proxy')
            if checkpoint:command.extend(['--checkpoint',str(checkpoint)])
            for key,flag in (('learning_rate','--learning-rate'),('schedule','--schedule'),('exploration_std','--exploration-std'),('entropy_coef','--entropy-coef')):
                if key in stage:command.extend([flag,str(stage[key])])
            print('Training command: '+ ' '.join(command),flush=True)
            evaluation=folder/'evaluation.json'
            previous_mtime=evaluation.stat().st_mtime_ns if evaluation.exists() else None
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,'Q1_LEG_CURRICULUM_CONFIG':str(args.config.resolve())})
            if result.returncode or not evaluation.exists() or evaluation.stat().st_mtime_ns==previous_mtime:
                status['state']='runtime_error'
                (ROOT/'outputs/training_status.json').write_text(json.dumps(status,indent=2))
                return result.returncode or 1
            report=json.loads(evaluation.read_text())
            status.update(state='passed' if report['passed'] else 'gate_failed',evaluation=report)
            (ROOT/'outputs/training_status.json').write_text(json.dumps(status,indent=2))
            checkpoint=folder/'latest.pt'
            if report['passed']:break
        if skill=='sidestep' or not report['passed'] or config.get('confirmation_episodes',0)>8:
            status.update(state='validating_checkpoints')
            (ROOT/'outputs/training_status.json').write_text(json.dumps(status,indent=2))
            selection=['bash',str(ROOT/'select_policy.sh'),'--skill',skill,'--episodes',str(config.get('confirmation_episodes',32)),'--confirmation-seed',str(config.get('confirmation_seed',20000))]
            if config.get('select_current_run_only',False):selection.append('--current-run-only')
            previous_mtime=(folder/'evaluation.json').stat().st_mtime_ns
            selected=subprocess.run(selection,cwd=ROOT,env={**os.environ,'Q1_LEG_CURRICULUM_CONFIG':str(args.config.resolve())})
            if selected.returncode or (folder/'evaluation.json').stat().st_mtime_ns==previous_mtime:
                status.update(state='runtime_error')
                (ROOT/'outputs/training_status.json').write_text(json.dumps(status,indent=2))
                return selected.returncode or 1
            report=json.loads((folder/'evaluation.json').read_text())
            status.update(state='passed' if report['passed'] else 'gate_failed',evaluation=report)
            (ROOT/'outputs/training_status.json').write_text(json.dumps(status,indent=2))
            if not report['passed']:
                print(f'{skill}: candidate confirmation failed; retained checkpoints. Curriculum stopped.',flush=True)
                return 2
            checkpoint=folder/'latest.pt'
    return 0
if __name__=='__main__':raise SystemExit(main())
