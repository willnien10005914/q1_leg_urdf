#!/usr/bin/env python3
"""Keep a terminal unsuccessful run honest in README; never publish fake media."""
import json
from pathlib import Path
import subprocess
from finalize_interactive_v4 import replace_readme

ROOT=Path(__file__).resolve().parents[1]

def status_block(state):
    if state.get('ready'):raise ValueError('Confirmed artifacts belong to the media finalizer')
    count=state.get('completed_iterations',0)
    profile=state.get('profile') or {}
    envs=state.get('num_envs',profile.get('num_envs',10240))
    maximum=state.get('max_iterations',profile.get('max_iterations',15000))
    outcome={'budget_exhausted':'已達本次預算，尚未通過完整操作驗收','deadline_reached':'已依早上九點期限結束，尚未通過完整操作驗收','runtime_error':'執行異常，已保留權重與診斷','interrupted':'已停止，保留完成次數供接續'}.get(state.get('state'),'尚未通過操作驗收')
    last=state.get('last_evaluation') or {}
    initial=state.get('adopted_worker_iterations',0)
    progress=f'本次接續自 **{initial:,} iterations**；最新完整保存進度為 **{count:,} iterations**。'
    timestamps=[]
    for label,key in (('本次啟動','started_at'),('最後更新','updated_at')):
        if state.get(key):timestamps.append(f'{label}：`{state[key]}`')
    if timestamps:progress+=' '+'；'.join(timestamps)+'。'
    config='config/interactive_v4_deadline.json' if profile.get('training_deadline_utc') else 'config/interactive_v4.json'
    deadline=''
    if state.get('state')=='deadline_reached' or profile.get('training_stop_at') or profile.get('training_deadline_at') or profile.get('training_deadline_utc'):
        deadline='\n\n依使用者要求，以**台北時間早上 09:00 前結束本次訓練與整理結果**為期限，調整本次更新預算並預留驗證時間。期限到達只代表停止訓練，不代表動作已學會。'
    return f'''## V4：鍵盤控制與左右閃身 — {outcome}

本次使用 **{envs:,} envs**，已完成 **{count:,}／{maximum:,} iterations**，reward revision **{state.get('reward_revision',0)}**。{progress}{deadline}

最新 {last.get('episodes',32)} 回合驗證的通過檢查比例為 **{last.get('score',0):.1%}**；這是檢查項目比例，不是任務成功率。

**尚未通過獨立 128 回合確認，因此鍵盤策略保持停用，未發布新版成功 GIF／影片。** 舊版 V3 成果保留於下方。程式碼包含 WASD／QE／Space、有限左右序列、完整 optimizer 續訓、停滯診斷與達標後自動錄影／發布。

```bash
cat outputs/interactive_training_status.json
# 接續剩餘預算，請代入狀態檔的 run 路徑：
bash start_interactive_training.sh --config {config} --resume-run /path/to/the/saved/run
```

請參閱 [V4 架構與驗收](docs/V4_ARCHITECTURE.md)。完整原始評估與失敗項目保留在該次 run 中，不能把達到預算或曲線停滯稱作學會動作。
'''

def main():
    path=ROOT/'outputs/interactive_training_status.json'
    if not path.exists():return
    state=json.loads(path.read_text())
    if state.get('ready') or state.get('state') not in ('budget_exhausted','deadline_reached','runtime_error','interrupted'):return
    readme=ROOT/'README.md'
    readme.write_text(replace_readme(readme.read_text(),status_block(state)))
    # Commit only this generated status block, never another staged change.
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).decode().splitlines()
    if staged:raise RuntimeError('Unrelated staged work exists; README updated locally without committing it')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()
    if remote!='git@github.com:willnien10005914/q1_leg_urdf.git':raise RuntimeError('Expected project SSH origin')
    subprocess.run(['git','add','--','README.md'],cwd=ROOT,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode
    if changed:subprocess.run(['git','commit','-m','Document interactive PPO run status and remaining acceptance failures'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True,env=__import__('os').environ|{'GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=20'})

if __name__=='__main__':main()
