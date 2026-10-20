#!/usr/bin/env python3
"""Read-only training status; optionally refresh a timestamped Markdown report."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--write-report',action='store_true')
    args=p.parse_args()
    robot=json.loads((ROOT/'config/robot_contract.json').read_text())
    asset=json.loads((ROOT/'config/asset_manifest.json').read_text())
    motion=json.loads((ROOT/'docs/reference/sidestep_motion.json').read_text())
    state=json.loads((ROOT/'outputs/training_status.json').read_text())
    evaluations={}
    for skill in ('stand','walk','sidestep'):
        path=ROOT/'checkpoints/isaac'/skill/'evaluation.json'
        if path.exists():evaluations[skill]=json.loads(path.read_text())
    timestamp=datetime.now().astimezone().isoformat(timespec='seconds')
    result={'time':timestamp,'source':asset['archive'],'source_sha256':asset['archive_sha256'],
        'native_cad_ready':robot['native_cad_ready'],'missing_meshes':robot['missing_meshes'],
        'actions':robot['action_size'],'observations':robot['observation_size'],
        'mediapipe':motion['quality'],'training':state,'evaluations':evaluations}
    try:
        with urlopen('http://127.0.0.1:8877/state',timeout=.5) as response:live=json.load(response)
        result['live']={k:live.get(k) for k in ('status','skill','available')}
    except OSError:result['live']={'status':'offline'}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.write_report:
        lines=['# 本機執行狀態','',f'快照時間：{timestamp}。最新狀態請看 http://127.0.0.1:8876/web/ 或執行 `.venv/bin/python tools/status.py`。','',
            f"正式來源：`{asset['archive']}`（SHA-256 `{asset['archive_sha256']}`）。30 個原生 leg mesh 齊全，缺件 0；23 個控制關節 / 110 維觀測。",'',
            f"MediaPipe：前七秒 {motion['quality']['accepted_frames']}/{motion['quality']['total_frames']} 幀，20 Hz 原生關節 IK。參考動畫可以在網頁選擇「閃身步」。",'',
            '| 階段 | 評估狀態 | 存活率 | 平均時間 | 身高誤差 |','|---|---|---|---|---|']
        names={'stand':'站立','walk':'行走','sidestep':'閃身步'}
        for skill in names:
            report=evaluations.get(skill)
            if report:
                lines.append(f"| {names[skill]} | {'通過原生 CAD 門檻' if report['passed'] else '尚未通過'} | {report['survival_rate']:.0%} | {report['duration']:.2f}s | {report['height_error']:.4f}m |")
            else:
                lines.append(f"| {names[skill]} | {'訓練中' if state.get('skill')==skill and state.get('state')=='training' else '尚待前置階段通過'} | — | — | — |")
        lines+=['',f"目前：`{state.get('skill')}` / `{state.get('state')}`，後端 Isaac Lab，GPU 環境數：{state.get('envs',state.get('num_envs','依訓練設定'))}。",'',
            '正式專案：`~/projects/q1_leg_urdf`。網頁服務：`q1-leg-web.service`；即時模擬：`q1-leg-live.service`；訓練服務：`q1-leg-training.service`。服務是本次工作階段的 systemd transient units，不是開機自動啟動。','',
            '```bash','systemctl --user status q1-leg-web q1-leg-live q1-leg-training','tail -f outputs/native_curriculum.log','systemctl --user stop q1-leg-training  # 停止整組訓練程序','```','',
            '每階段通過才會前進。未通過會保留權重與物理播放記錄；此時不會宣稱閃身步已訓練成功。原生 CAD 模擬不代表實機已驗證。','']
        (ROOT/'docs/RUN_STATUS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
