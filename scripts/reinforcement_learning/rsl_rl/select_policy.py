#!/usr/bin/env python3
"""Select a saved policy on validation episodes, then confirm on fresh seeds."""
import argparse
import sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--skill',choices=['stand','walk','sidestep'],default='sidestep')
p.add_argument('--episodes',type=int,default=32)
p.add_argument('--confirmation-seed',type=int,default=20000)
p.add_argument('--current-run-only',action='store_true',help='Exclude archived runs when selecting this training run')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
launcher=AppLauncher(args)
app=launcher.app

import json
import os
import shutil
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from leg_humanoid_lab.robot import ROOT,contract
from leg_humanoid_lab.isaac_env import LegEnvCfg,IsaacLegEnv
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.isaac_evaluation import evaluate
from runtime_compat import runner_cfg,load_checkpoint

env=None
try:
    folder=ROOT/'checkpoints/isaac'/args.skill
    saved=json.loads((folder/'contract.json').read_text())
    expected=contract()
    if saved.get('proxy') is not False or any(saved.get(k)!=expected[k] for k in ('robot','urdf_sha256','joint_names','observation_size')):
        raise ValueError('Native checkpoint contract mismatch')
    cfg=LegEnvCfg();cfg.skill=args.skill;cfg.scene.num_envs=max(32,args.episodes);cfg.seed=42;cfg.sim.device=args.device
    env=IsaacLegEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=args.device)
    candidates=[]
    for path in (folder.glob('model_*.pt') if args.current_run_only else folder.rglob('model_*.pt')):
        metadata=path.parent/'contract.json'
        if metadata.exists():
            candidate_contract=json.loads(metadata.read_text())
            if candidate_contract.get('proxy') is False and all(candidate_contract.get(k)==expected[k] for k in ('robot','urdf_sha256','joint_names','observation_size')):
                candidates.append(path)
    candidates.sort(key=lambda x:(str(x.parent),int(x.stem.split('_')[-1])))
    candidates.append(folder/'latest.pt')
    results=[]
    for checkpoint in candidates:
        load_checkpoint(runner,checkpoint,load_optimizer=False,load_iteration=False)
        report=evaluate(runner,wrapped,env,episodes=8,checkpoint=checkpoint,publish=False)
        results.append({'file':str(checkpoint),'evaluation':report})
    def rank(item):
        r=item['evaluation']
        excursion=abs(r.get('lateral_range',0)-r.get('reference_lateral_range',0))
        return (r['passed'],r['survival_rate'],-r['root_error']-excursion,-r['pose_rmse'])
    best=max(results,key=rank)
    from pathlib import Path
    checkpoint=Path(best['file'])
    load_checkpoint(runner,checkpoint,load_optimizer=False,load_iteration=False)
    report=evaluate(runner,wrapped,env,episodes=args.episodes,checkpoint=checkpoint,seed=args.confirmation_seed,publish=True)
    if report['passed'] and checkpoint.name!='latest.pt':
        shutil.copyfile(checkpoint,folder/'selected.tmp')
        os.replace(folder/'selected.tmp',folder/'latest.pt')
        report['selected_from']=checkpoint.name
        report['checkpoint']='latest.pt'
        (folder/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'outputs'/f'{args.skill}_checkpoint_selection.json').write_text(json.dumps({'validation':results,'selected':str(checkpoint),'confirmation':report},indent=2)+'\n')
    (ROOT/'outputs/training_status.json').write_text(json.dumps({'state':'passed' if report['passed'] else 'gate_failed','skill':args.skill,'backend':'isaaclab','proxy':False,'evaluation':report},indent=2)+'\n')
    print('Fresh-seed confirmation:',json.dumps(report,indent=2),flush=True)
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush()
finally:
    if env:env.close()
    app.close()
