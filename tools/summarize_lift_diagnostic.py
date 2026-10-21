#!/usr/bin/env python3
"""Write an explicitly unaccepted V5 diagnostic report using CPU JSON only.

No simulator, network, checkpoint modification, README mutation or publication.
Only allowlisted numeric fields and source identities enter the public report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
ROBOT='q1_leg_lift_v5'
STAGE_CASES={0:('stand','inplace'),1:('stand','left','right','pair')}
LABELS={'stand':'站立','inplace':'原地抬腳','left':'左閃身','right':'右閃身','pair':'左右各一下'}
ENV_SOURCE='source/leg_humanoid_lab/leg_humanoid_lab/lift_env.py'
CONTROL_SOURCE='config/lift_control_v5.json'
SHA=re.compile(r'[a-fA-F0-9]{64}\Z')
SAFE_LABEL=re.compile(r'[A-Za-z0-9_.-]{1,160}\Z')


def finite(value,description,minimum=None,maximum=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{description} must be a finite number')
    if (minimum is not None and value<minimum) or (maximum is not None and value>maximum):
        raise ValueError(f'{description} is outside its physical/statistical bounds')
    return float(value)


def sha(value,description):
    if not isinstance(value,str) or not SHA.fullmatch(value):
        raise ValueError(f'{description} must contain a full SHA256 identity')
    return value.lower()


def pair(values,description,minimum=0.,maximum=None):
    if not isinstance(values,list) or len(values)!=2:
        raise ValueError(f'{description} must contain left/right measurements')
    return [finite(value,description,minimum,maximum) for value in values]


def public_gofile_url(value):
    if value is None:return None
    parsed=urlsplit(value)
    if (parsed.scheme!='https' or parsed.netloc!='gofile.io'
            or not re.fullmatch(r'/d/[A-Za-z0-9_-]+',parsed.path)
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('Only a public https://gofile.io/d/<id> URL without credentials or query parameters is allowed')
    return value


def validate(evaluation,metadata):
    """Bind one stage-specific native evaluation to its actual diagnostic clip."""
    if evaluation.get('backend')!='isaaclab' or evaluation.get('proxy') is not False:
        raise ValueError('Evaluation must be native Isaac Lab, never proxy')
    if metadata.get('robot')!=ROBOT or metadata.get('proxy') is not False:
        raise ValueError('Footage must identify the native lift-v5 robot, never proxy')
    if evaluation.get('robot',ROBOT)!=ROBOT:
        raise ValueError('Evaluation robot is not lift-v5')
    contract=evaluation.get('contract')
    if contract!='q1_leg_command_v5' and not (isinstance(contract,dict) and contract.get('robot')==ROBOT):
        raise ValueError('Evaluation must carry its independent v5 gait or robot contract')
    stage=evaluation.get('stage')
    if type(stage) is not int or stage not in STAGE_CASES:
        raise ValueError('This foundation diagnostic supports stages 0 and 1 only; stage 2 needs full interactive acceptance')
    if (metadata.get('diagnostic') is not True or metadata.get('gate_passed') is not False
            or metadata.get('accepted') is not False
            or metadata.get('source')!='native Isaac Sim PhysX PPO'
            or metadata.get('overlay')!='EXPERIMENTAL / NOT ACCEPTED'
            or metadata.get('technical_error') is not None
            or metadata.get('reset_count_after_start')!=0):
        raise ValueError('Footage must remain an explicitly unaccepted native diagnostic without resets or technical errors')
    task='inplace_curriculum' if stage==0 else 'finite_sidestep_sequence'
    if metadata.get('task')!=task:
        raise ValueError('Footage task must match the evaluated curriculum stage')
    identity=sha(evaluation.get('checkpoint_sha256'),'Evaluation checkpoint')
    if sha(metadata.get('checkpoint_sha256'),'Footage checkpoint')!=identity:
        raise ValueError('Evaluation and footage checkpoint SHA256 do not match')
    updates=evaluation.get('completed_iterations')
    if type(updates) is not int or updates<0:
        raise ValueError('Evaluation must state an exact nonnegative completed update count')
    if 'completed_iterations' in metadata and metadata['completed_iterations']!=updates:
        raise ValueError('Footage and evaluation update counts disagree')
    native=metadata.get('contract',{})
    if (native.get('robot')!=ROBOT or native.get('observation_size')!=110
            or len(native.get('joint_names',[]))!=23):
        raise ValueError('Footage must contain the 23-action/110-observation native lift-v5 contract')
    for name in ('urdf_sha256','gait_sha256'):
        if sha(evaluation.get(name),f'Evaluation {name}')!=sha(native.get(name),f'Footage {name}'):
            raise ValueError(f'Native source identity does not match: {name}')
    manifest=metadata.get('source_sha256',{})
    if sha(evaluation.get('environment_sha256'),'Evaluation environment')!=sha(manifest.get(ENV_SOURCE),'Footage environment'):
        raise ValueError('Evaluation and footage environment source SHA256 do not match')
    control_hash=sha(native.get('control_sha256'),'Native control')
    if sha(manifest.get(CONTROL_SOURCE),'Footage control source')!=control_hash:
        raise ValueError('Footage control source SHA256 does not match its contract')
    profile=metadata.get('control_profile')
    if (not isinstance(profile,str) or not SAFE_LABEL.fullmatch(profile)
            or native.get('control_profile')!=profile):
        raise ValueError('Native control profile must be explicit and consistent')
    sha(metadata.get('video_sha256'),'Footage bytes')
    frames=metadata.get('frames');fps=metadata.get('fps',50)
    if type(frames) is not int or frames<=0 or fps!=50:
        raise ValueError('Footage must contain actual native 50 FPS frames')
    duration=finite(metadata.get('actual_duration_s'),'Footage duration',0.)
    if abs(duration-frames/50.)>1e-7:
        raise ValueError('Footage duration does not match its recorded frame count')
    if type(metadata.get('failure')) is not bool:
        raise ValueError('Footage must state whether the native episode failed')
    episodes=evaluation.get('episodes')
    if type(episodes) is not int or episodes<1:
        raise ValueError('Evaluation episode count is required')
    checks=evaluation.get('checks',{})
    if not checks or not all(type(value) is bool for value in checks.values()):
        raise ValueError('Evaluation must contain explicit boolean acceptance checks')
    score=finite(evaluation.get('score'),'Evaluation score',0.,1.)
    if abs(score-sum(checks.values())/len(checks))>1e-8 or evaluation.get('passed') is not all(checks.values()):
        raise ValueError('Evaluation pass/score contradicts its checks')
    for gate in checks:
        if not isinstance(gate,str) or not SAFE_LABEL.fullmatch(gate):
            raise ValueError('Unexpected evaluation gate label; raw paths/secret fields cannot enter a public report')
    cases=evaluation.get('cases',{})
    counted=0
    for name in STAGE_CASES[stage]:
        item=cases.get(name,{})
        count=item.get('episodes')
        if type(count) is not int or count<1:
            raise ValueError(f'Missing evaluated case: {name}')
        counted+=count
        finite(item.get('survival_rate'),f'{name} survival',0.,1.)
        finite(item.get('upright'),f'{name} upright',-1.,1.)
        finite(item.get('stance_slip_m_s'),f'{name} stance slip',0.)
        pair(item.get('foot_peak_clearance_m_per_foot'),f'{name} foot peaks')
        pair(item.get('foot_lift_success_rate_per_foot'),f'{name} foot success',maximum=1.)
        pair(item.get('high_swing_samples_per_foot'),f'{name} high-swing samples')
    if counted!=episodes:
        raise ValueError('Case episode counts do not match the complete evaluation')
    metrics=metadata.get('foot_metrics',{})
    measured=metrics.get('valid_first_episode_frames')
    if type(measured) is not int or not 0<=measured<=frames:
        raise ValueError('Actual foot measurements must be limited to recorded first-episode frames')
    for name in ('unloaded_peak_clearance_m_per_foot','foot_world_peak_displacement_m_per_foot','foot_world_path_length_m_per_foot'):
        pair(metrics.get(name),name)
    for name in ('unloaded_samples_per_foot','planned_high_swing_samples_per_foot'):
        counts=metrics.get(name)
        if not isinstance(counts,list) or len(counts)!=2 or not all(type(count) is int and 0<=count<=measured for count in counts):
            raise ValueError(f'{name} must contain actual per-foot frame counts')
    rates=metrics.get('planned_high_swing_success_rate_per_foot')
    if not isinstance(rates,list) or len(rates)!=2:
        raise ValueError('Footage requires per-foot measured lift success')
    for count,rate in zip(metrics['planned_high_swing_samples_per_foot'],rates):
        if not count and rate is not None:
            raise ValueError('Absent planned swings must be reported as unobserved')
        if count:finite(rate,'Measured high-swing success',0.,1.)
    displacement=metrics.get('foot_world_displacement_xyz_m_per_foot')
    if measured:
        if not isinstance(displacement,list) or len(displacement)!=2:
            raise ValueError('Footage requires actual left/right world displacement vectors')
        for vector in displacement:
            if not isinstance(vector,list) or len(vector)!=3:raise ValueError('World displacement must be XYZ')
            for value in vector:finite(value,'World displacement')
    return stage


def render(evaluation,metadata,gofile_url=None):
    stage=validate(evaluation,metadata)
    gofile_url=public_gofile_url(gofile_url)
    updates=evaluation['completed_iterations'];metrics=metadata['foot_metrics']
    stage_name='原地抬腳' if stage==0 else '有限左右閃身'
    status='通過此階段門檻' if evaluation['passed'] else '未通過此階段門檻'
    lines=[f'# V5 {stage_name}診斷：NOT ACCEPTED / 尚未正式驗收','',
           f'本報告比較**同一份 checkpoint** 的原生 PhysX／PPO 評估與實際錄影，對應 **{updates:,} 次新 V5 PPO 更新**。{status}，仍不能替代 Stage 2 的 WASD／轉向／閃身後續走，以及獨立 128 回合確認。','',
           '6 cm 是抬腳命令；經原生關節限位與 CAD／FK 計算後，參考峰值約 5–6 cm。以下數值才是物理模擬中的實際量測，幾何參考不表示 PPO 已學會。全程為原生接觸與關節驅動，不固定機身或施加根部外力。','',
           f"評估：Stage {stage}（{stage_name}）、{evaluation['episodes']} 回合；檢查項目通過比例 **{evaluation['score']:.1%}**，這不是任務成功率。只計第一回合，跌倒／重置之後的姿態不納入。",'',
           '| 類別 | 回合 | 存活 | 直立指標 | 支撐滑動 RMS m/s | 腳底峰值 cm（左／右） | 卸載抬腳成功率（左／右） |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in STAGE_CASES[stage]:
        item=evaluation['cases'][name]
        peaks=' / '.join(f'{v*100:.2f}' for v in item['foot_peak_clearance_m_per_foot'])
        rates=['未要求' if count==0 else f'{value:.1%}' for value,count in zip(item['foot_lift_success_rate_per_foot'],item['high_swing_samples_per_foot'])]
        slip='未完整觀測' if item['stance_slip_m_s']>=1e5 else f"{item['stance_slip_m_s']:.3f}"
        lines.append(f"| {LABELS[name]} | {item['episodes']} | {item['survival_rate']:.1%} | {item['upright']:.3f} | {slip} | {peaks} | {' / '.join(rates)} |")
    lines.extend(['','評估峰值是各回合腳底峰值的平均，與單一錄影的最高卸載高度不同。抬腳成功要求規劃擺腿、參考高度 >2.2 cm、實際離地 ≥2.5 cm 且腳已卸載；支撐滑動使用規劃支撐且承重時的速度。','',
                  '## 同 checkpoint 實際影片','',
                  f"錄影任務：{'原地抬腳課程，不是鍵盤行走測試' if stage==0 else '左閃、右閃各一下，再回穩站立'}。**{metadata['actual_duration_s']:.2f} 秒／{metadata['frames']} 幀／50 FPS**；{'發生原生回合終止，立即停止' if metadata['failure'] else '未發生原生回合終止'}，開始後未重置。畫面標有 `EXPERIMENTAL / NOT ACCEPTED`。",'',
                  '| 腳 | 最高實際卸載離地 cm | 卸載樣本 | 規劃高擺腿樣本 | 實際卸載抬腳成功率 | 世界最大位移 cm | 世界累計路徑 cm |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for foot,name in enumerate(('左腳','右腳')):
        count=metrics['planned_high_swing_samples_per_foot'][foot]
        rate=metrics['planned_high_swing_success_rate_per_foot'][foot]
        lift='未觀測' if count==0 else f'{rate:.1%}'
        lines.append(f"| {name} | {metrics['unloaded_peak_clearance_m_per_foot'][foot]*100:.2f} | {metrics['unloaded_samples_per_foot'][foot]} | {count} | {lift} | {metrics['foot_world_peak_displacement_m_per_foot'][foot]*100:.2f} | {metrics['foot_world_path_length_m_per_foot'][foot]*100:.2f} |")
    displacement=metrics.get('foot_world_displacement_xyz_m_per_foot')
    if metrics['valid_first_episode_frames']:
        vectors=['('+'，'.join(f'{value*100:+.2f}' for value in vector)+')' for vector in displacement]
        lines.extend(['',f'腳連桿原點相對首個有效影格的最終世界位移 XYZ（cm）：左 {vectors[0]}；右 {vectors[1]}。世界最大位移／路徑包含水平及垂直運動，不等於向前行走距離。'])
    if gofile_url:lines.extend(['',f'[觀看這段未合格診斷影片（GoFile）]({gofile_url})。'])
    lines.extend(['','## 身分與未通過項目','',f"- 原生機器人：`{ROBOT}`；控制設定：`{metadata['control_profile']}`。",
                  f"- Checkpoint SHA256：`{evaluation['checkpoint_sha256']}`。",
                  f"- 原生 URDF SHA256：`{evaluation['urdf_sha256']}`。",
                  f"- Gait SHA256：`{evaluation['gait_sha256']}`。",
                  f"- Control SHA256：`{metadata['contract']['control_sha256']}`。",
                  f"- 影片 SHA256：`{metadata['video_sha256']}`。",''])
    failed=[name for name,passed in evaluation['checks'].items() if not passed]
    lines.extend([f'- `{name}`' for name in failed] or ['本階段沒有未通過項目；正式整合操作仍須另行驗收。'])
    lines.extend(['','本報告只描述這個候選與單段錄影，不發布策略、不改驗收條件，也不將片段視為整批成功證明。',''])
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/LIFT_V5_DIAGNOSTIC.md')
    parser.add_argument('--gofile-url')
    args=parser.parse_args(argv)
    evaluation=json.loads(args.evaluation.read_text())
    metadata=json.loads(args.metadata.read_text())
    content=render(evaluation,metadata,args.gofile_url)
    video=args.metadata.with_suffix('.mp4')
    if video.is_file() and hashlib.sha256(video.read_bytes()).hexdigest()!=metadata['video_sha256'].lower():
        raise ValueError('Adjacent diagnostic MP4 bytes do not match metadata SHA256')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(content);temporary.replace(args.output)
    print(f'Wrote native stage-{evaluation["stage"]} NOT ACCEPTED report: {args.output}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
