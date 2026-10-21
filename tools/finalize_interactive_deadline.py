#!/usr/bin/env python3
"""Export an explicitly unaccepted deadline candidate and its real diagnostic."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from finalize_interactive_v4 import replace_readme

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'outputs/interactive_deadline_finalization.json'
TERMINAL={'deadline_reached','budget_exhausted','runtime_error','interrupted','complete','artifact_error'}

def digest(path,algorithm='sha256'):
    return hashlib.new(algorithm,Path(path).read_bytes()).hexdigest()

def progress(stage,**values):
    old=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    old.update(stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),**values)
    STATUS.parent.mkdir(exist_ok=True)
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(old,indent=2)+'\n');temp.replace(STATUS)
    print('DEADLINE_ARTIFACT '+json.dumps({'stage':stage,**values}),flush=True)

def run(args,timeout=180,private=False,env=None):
    result=subprocess.run([str(x) for x in args],cwd=ROOT,timeout=timeout,env=env,
                          text=True,capture_output=private,umask=0o077 if private else -1)
    if result.returncode:raise RuntimeError(f'{Path(str(args[0])).name} failed with exit {result.returncode}')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--local-only',action='store_true',help='Keep all candidate media local; no network or Git publication')
    mode.add_argument('--publish-unaccepted',action='store_true',help='Publish diagnostic media only after explicit user authorization')
    options=parser.parse_args()
    progress('waiting_for_training')
    cutoff=datetime.fromisoformat('2026-10-21T00:55:00+00:00')
    while True:
        state=json.loads((ROOT/'outputs/interactive_training_status.json').read_text())
        active=subprocess.run(['systemctl','--user','is-active','--quiet','q1-leg-training']).returncode==0
        if state.get('state') in TERMINAL and not active:break
        if datetime.now(timezone.utc)>=cutoff:raise RuntimeError('Training did not finish within the reserved artifact window')
        time.sleep(3)
    if state.get('ready'):
        progress('handled_by_confirmed_policy_finalizer',accepted=True)
        return
    folder=Path(state['run'])
    report=json.loads(Path(state['last_evaluation']['report']).read_text())
    sha=digest(folder/'resume.pt')
    if report.get('checkpoint_sha256')!=sha or report.get('completed_iterations')!=state['completed_iterations']:
        raise RuntimeError('The latest evaluation does not match the retained candidate; refusing to publish mismatched results')
    candidate=ROOT/'checkpoints/isaac/interactive_v4/candidates/deadline_20261021'
    candidate.mkdir(parents=True,exist_ok=True)
    shutil.copy2(folder/'resume.pt',candidate/'model.pt')
    shutil.copy2(folder/'contract.json',candidate/'contract.json')
    public_report={**report,'accepted':False,'ready':False,'candidate':True,
                   'training_profile':state['profile'],'source_sha256':state['source_sha256'],
                   'checkpoint':'model.pt','checkpoint_sha256':sha}
    (candidate/'evaluation.json').write_text(json.dumps(public_report,indent=2)+'\n')
    progress('summarizing',completed_iterations=state['completed_iterations'],checkpoint_sha256=sha,accepted=False)
    run([ROOT/'.venv/bin/python','tools/summarize_interactive_deadline.py'])
    # The training wrapper has restored the old viewer. Release it for RTX capture.
    restore=subprocess.run(['systemctl','--user','is-active','--quiet','q1-leg-live']).returncode==0
    if restore:run(['systemctl','--user','stop','q1-leg-live'])
    video='docs/results/q1_leg_interactive_deadline_diagnostic.mp4'
    gif='docs/images/interactive-deadline-diagnostic.gif'
    try:
        progress('recording_unaccepted_native_candidate')
        metadata_path=(ROOT/video).with_suffix('.json')
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        reusable=((ROOT/video).exists() and metadata.get('checkpoint_sha256')==sha
                  and metadata.get('video_sha256')==digest(ROOT/video) and metadata.get('gate_passed') is False)
        if not reusable:
            env=os.environ.copy();env['OUT']=video
            run(['bash','record_interactive_diagnostic.sh','--checkpoint',candidate/'model.pt','--sequence-only','--seed','42'],timeout=180,env=env)
            metadata=json.loads(metadata_path.read_text())
        if metadata.get('gate_passed') is not False or metadata.get('checkpoint_sha256')!=sha:
            raise RuntimeError('Diagnostic identity or acceptance marker mismatch')
        progress('converting_diagnostic_gif',duration=metadata['actual_duration_s'],failure=metadata['failure'])
        run([ROOT/'.venv/bin/python','tools/video_to_gif.py',video,gif,'--fps','10','--max-width','640','--colors','96','--force'],timeout=90)
    finally:
        if restore:run(['bash','start_native_web.sh'],timeout=30)
    if not options.publish_unaccepted:
        progress('complete_local',accepted=False,video=video,gif=gif,completed_iterations=state['completed_iterations'],publication='Awaiting explicit user approval for public diagnostic media')
        return
    progress('uploading_diagnostic')
    response=ROOT/'outputs/interactive_deadline_upload.gofile.json'
    run([ROOT/'.venv/bin/python','tools/upload_gofile.py',video,response],timeout=90,private=True)
    os.chmod(response,0o600)
    uploaded=json.loads(response.read_text());data=uploaded.get('data',{})
    url=data.get('downloadPage','')
    if uploaded.get('status')!='ok' or data.get('md5')!=digest(ROOT/video,'md5') or not url.startswith('https://gofile.io/d/'):
        raise RuntimeError('GoFile did not verify the diagnostic file and public URL')
    block=f'''## V4：限時訓練已結束，候選尚未合格

依 2026-10-21 台北 **09:00 前完成**要求，本次使用 **{state['num_envs']:,} envs**，累計保存 **{state['completed_iterations']:,} iterations**（從 501 次接續；限時上限 801 次，原始上限 15,000 次）。採單一 DirectRLEnv／PPO，WASD／QE、停止與左右閃身共用同一物理狀態。

**目前未通過完整操作驗收與獨立 128 回合確認，不能稱為已學會。** 最新 32 回合通過的檢查項目比例為 **{report['score']:.1%}**，這不是任務成功率。完整結果與各方向抬腳數值見 [限時訓練報告](docs/DEADLINE_20261021.md)。正式鍵盤策略仍保持停用。

![未合格候選的原生 PhysX 診斷，並非成功示範]({gif})

上圖是此候選的真實原生 PhysX／PPO 錄影，seed 42、{metadata['actual_duration_s']:.2f} 秒，畫面標有 **EXPERIMENTAL / NOT ACCEPTED**；本回合{'發生終止，立即停止且未重置' if metadata['failure'] else '完成有限序列，仍不能替代整批驗收'}。[GoFile 診斷影片]({url}) · [專案 MP4]({video})。

[未合格候選權重](checkpoints/isaac/interactive_v4/candidates/deadline_20261021/model.pt) 僅供重現診斷，並未取代正式權重。重新錄製可執行：

```bash
OUT=outputs/my_diagnostic.mp4 bash record_interactive_diagnostic.sh --checkpoint checkpoints/isaac/interactive_v4/candidates/deadline_20261021/model.pt --sequence-only --seed 42
bash start_native_web.sh
```

網頁：http://127.0.0.1:8876/web/ 。未合格時仍可檢視 V3 物理預覽與參考動作；新版控制及驗收設計見 [V4 架構](docs/V4_ARCHITECTURE.md)。
'''
    readme=ROOT/'README.md';readme.write_text(replace_readme(readme.read_text(),block))
    paths=['README.md','config/interactive_v4_deadline.json','tools/finalize_interactive_deadline.py',
           'tools/summarize_interactive_deadline.py','tests/test_interactive_deadline_summary.py',
           'tests/test_interactive_diagnostic.py','record_interactive_diagnostic.sh',
           'scripts/reinforcement_learning/rsl_rl/record_interactive_diagnostic.py',
           'docs/DEADLINE_20261021.md','docs/results/interactive_v4_deadline_20261021.json',
           'docs/images/interactive_deadline_metrics.svg',video,str(Path(video).with_suffix('.json')),gif]
    paths += [str((candidate/name).relative_to(ROOT)) for name in ('model.pt','contract.json','evaluation.json')]
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip():
        raise RuntimeError('Unrelated staged changes exist; media retained without changing the index')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip()
    if remote!='git@github.com:willnien10005914/q1_leg_urdf.git':raise RuntimeError('Expected the authorized SSH origin')
    if any((ROOT/name).stat().st_size>95*1024*1024 for name in paths):raise RuntimeError('Publication file exceeds GitHub size limit')
    progress('committing_diagnostic')
    run(['git','add','-f','--',*paths]);run(['git','diff','--cached','--check'])
    run(['git','commit','-m','Publish deadline PPO candidate diagnostics without claiming acceptance'])
    env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -o BatchMode=yes -o ConnectTimeout=15';env['GIT_TERMINAL_PROMPT']='0'
    run(['git','push','origin','main'],timeout=45,env=env)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote_head=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True,env=env,timeout=30).split()[0]
    if remote_head!=commit:raise RuntimeError('Remote main does not match the diagnostic commit')
    progress('complete',accepted=False,commit=commit,url=url,video=video,gif=gif,completed_iterations=state['completed_iterations'])

if __name__=='__main__':
    try:main()
    except Exception as error:
        progress('failed',error=str(error));raise
