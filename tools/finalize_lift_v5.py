#!/usr/bin/env python3
"""Prepare or explicitly publish confirmed lift-v5 media, outside training.

Default preparation uses no network and leaves README/the Git index untouched.
--publish enables the already-authorized GoFile and SSH publication pipeline.
No checkpoint or unaccepted recording is ever in this tool's publish allowlist.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(os.environ.get('Q1_LEG_ROOT',Path(__file__).resolve().parents[1]))
START='<!-- V5_STATUS_START -->';END='<!-- V5_STATUS_END -->'
VIDEO='docs/results/q1_leg_lift_v5.mp4'
METADATA='docs/results/q1_leg_lift_v5.json'
GIF='docs/images/lift-v5.gif'
EVALUATION='docs/results/lift_v5_confirmed_evaluation.json'
REPORT='docs/LIFT_V5_ACCEPTED.md'
PUBLISH_PATHS=frozenset(('README.md',VIDEO,METADATA,GIF,EVALUATION,REPORT))
SSH_REMOTES=frozenset(('git@github.com:willnien10005914/q1_leg_urdf.git',
                      'ssh://git@github.com/willnien10005914/q1_leg_urdf.git'))


class Pending(RuntimeError):
    """The accepted policy/GPU queue is not yet ready; no bypass is permitted."""


def digest(path,algorithm='sha256'):
    value=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default


def private_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.fchmod(fd,0o600)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    temporary.replace(path);os.chmod(path,0o600)


def checked_url(value):
    if not isinstance(value,str) or not re.fullmatch(r'https://gofile\.io/d/[A-Za-z0-9_-]+',value):
        raise ValueError('GoFile did not return a credential-free public download URL')
    return value


def upload_url(receipt,video):
    if not isinstance(receipt,dict) or receipt.get('status')!='ok':return None
    data=receipt.get('data',{})
    if not isinstance(data,dict) or data.get('md5')!=digest(video,'md5'):return None
    try:return checked_url(data.get('downloadPage'))
    except ValueError:return None


def readme_parts(text):
    if text.count(START)!=1 or text.count(END)!=1:raise ValueError('README needs one V5 status marker pair')
    before,tail=text.split(START,1);block,after=tail.split(END,1)
    return before,block.strip(),after


def replace_readme(text,block):
    before,_,after=readme_parts(text)
    return before+START+'\n'+block.strip()+'\n'+END+after


def job_identity(state,gate,root):
    run=Path(state['run']).resolve();root=Path(root).resolve()
    if not run.is_relative_to(root/'checkpoints/isaac/lift_v5/runs'):
        raise ValueError('Supervisor run must stay inside this project\'s lift-v5 runs')
    identity={'run':str(run.relative_to(root)),'checkpoint_sha256':gate['checkpoint_sha256']}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def supervisor_ready(state,run_state,gate,root):
    """Both monitor copies must identify the completed, currently gated run."""
    if not gate.get('ready'):raise Pending(gate.get('error','Full v5 gate is not ready'))
    for item in (state,run_state):
        if not (item.get('state')=='complete' and item.get('ready') is True
                and item.get('robot')=='q1_leg_lift_v5' and item.get('backend')=='isaaclab'
                and item.get('proxy') is False and item.get('curriculum_stage')==2):
            raise Pending('Lift supervisor has not completed the full interactive curriculum')
        evaluation=item.get('evaluation',{})
        if (evaluation.get('checkpoint_sha256')!=gate['checkpoint_sha256']
                or evaluation.get('confirmation_episodes')!=gate['confirmation_episodes']
                or evaluation.get('episodes')!=gate['confirmation_episodes']
                or evaluation.get('stage')!=2 or evaluation.get('ready') is not True
                or evaluation.get('passed') is not True
                or evaluation.get('completed_iterations')!=item.get('completed_iterations')
                or evaluation.get('training_profile')!=item.get('profile')
                or evaluation.get('source_sha256')!=gate['source_sha256']):
            raise Pending('Supervisor completion does not identify this confirmed checkpoint/source')
        published=Path(item.get('published_checkpoint','')).resolve()
        if published!=Path(gate['checkpoint']).resolve():raise Pending('Supervisor publication pointer is inconsistent')
    if (Path(state['run']).resolve()!=Path(run_state['run']).resolve()
            or state.get('completed_iterations')!=run_state.get('completed_iterations')
            or state.get('profile')!=run_state.get('profile')):
        raise Pending('Supervisor monitor and run completion marker disagree')
    return job_identity(state,gate,root)


def recorder_module(root):
    directory=Path(root)/'scripts/reinforcement_learning/rsl_rl'
    sys.path.insert(0,str(directory))
    try:
        spec=importlib.util.spec_from_file_location('q1_lift_artifact_recorder',directory/'record_lift_v5.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    finally:sys.path.remove(str(directory))


def public_evaluation(evaluation,metadata,url,video_sha,gif_sha):
    names=('robot','backend','proxy','ready','passed','stage','completed_iterations','confirmation_episodes',
           'episodes','evaluation_seed','duration_s','reset_between_transitions','survival_rate','measurement',
           'cases','gates','checks','failed_gates','score','checkpoint_sha256','urdf_sha256','gait_sha256',
           'environment_sha256','base_environment_sha256','evaluator_sha256','source_sha256','training_profile')
    result={name:evaluation[name] for name in names if name in evaluation}
    result['media']={'video':VIDEO,'metadata':METADATA,'gif':GIF,'video_sha256':video_sha,'gif_sha256':gif_sha,
                     'gofile_url':checked_url(url),'recording_seed':metadata['seed'],
                     'recording_duration_s':metadata['actual_duration_s'],'directions':metadata['directions']}
    return result


def accepted_report(evaluation,metadata,url):
    lines=['# V5 正式策略與原生錄影驗證','',
        f"實際完成 **{evaluation['completed_iterations']:,} PPO iterations**；通過完整 Stage 2、**{evaluation['confirmation_episodes']} 回合獨立確認**。這是該批門檻驗收通過，不表示每個回合皆成功。",'',
        f"同一 checkpoint 的原生 PhysX 錄影：seed **{metadata['seed']}**，**{metadata['actual_duration_s']:.2f} 秒／{metadata['frames']} 幀／50 FPS**。W 前進、停止、左右各一下閃身、回站立，全程沒有重置。",'',
        '| 閃身方向 | 角色 | 最高實際卸載離地 cm | 規劃高擺腿卸載成功率 | 高擺腿樣本 |',
        '|---|---|---:|---:|---:|']
    for side,label in (('left','左'),('right','右')):
        event=metadata['directions'][side]
        for role,role_label in (('lead','前導腳'),('trailing','後隨腳')):
            lines.append(f"| {label} | {role_label} | {100*event[role+'_peak_clearance_m']:.2f} | {event[role+'_lift_success_rate']:.1%} | {event[role+'_high_swing_samples']} |")
    lines.extend(['',f"前進投影位移 **{metadata['walk_forward_displacement_m']:.3f} m**；最後站立速度 **{metadata['final_stand_speed_m_s']:.3f} m/s**。",'',
        f'[完整影片（GoFile）]({checked_url(url)}) · [MP4](results/q1_leg_lift_v5.mp4) · [逐幀量測](results/q1_leg_lift_v5.json) · [完整確認評估](results/lift_v5_confirmed_evaluation.json)。','',
        '![正式 V5 原生 PPO 錄影](images/lift-v5.gif)','',
        f"Checkpoint SHA256：`{metadata['checkpoint_sha256']}`。",'',
        '單段錄影另行要求左右前導／後隨腳各自卸載 ≥2.5 cm 且高擺腿成功率 ≥40%；全部由逐幀 CAD／接觸量測重算，未以整段峰值替代失敗跨步。','',
        '原生 CAD 模擬結果；實機尚未驗證。',''])
    return '\n'.join(lines)


def readme_block(evaluation,metadata,url,previous,job):
    profile=evaluation['training_profile']
    block=f'''<!-- V5_ACCEPTED_JOB: {job} -->
## V5：完整鍵盤控制與左右閃身已通過驗收

使用 **{profile['num_envs']:,} envs**，實際完成 **{evaluation['completed_iterations']:,} PPO iterations**（上限 {profile['max_iterations']:,}），採 Isaac Lab DirectRLEnv／指令條件 PPO；通過 **{evaluation['confirmation_episodes']} 回合獨立完整確認**。涵蓋 WASD、QE 轉向、停止、左右閃身與序列後恢復行走；不是宣稱全部回合都成功。

![正式 V5 原生 PPO：前進、停止、左右閃身、站立]({GIF})

此片為同一 checkpoint 在原生 Isaac Sim／PhysX 執行，seed **{metadata['seed']}**、**{metadata['actual_duration_s']:.2f} 秒／50 FPS**。影片本身也通過四組方向／角色的實際卸載抬腳門檻，全程未重置。[GoFile 完整影片]({checked_url(url)}) · [MP4]({VIDEO}) · [逐方向量測與完整結果]({REPORT})。

```bash
bash start_native_web.sh
# http://127.0.0.1:8876/web/
bash record_lift_v5.sh
```

選擇即時 PPO 後，W／S 前後、A／D 左右、Q／E 轉向；放開按鍵停止。按「閃身步 · 左右各一次 → 站立」執行有限序列，結束後可繼續鍵盤控制。網頁顯示背景 PhysX 的物理狀態；本結果尚未經實機驗證。

確認權重保留在已訓練的本機，未隨本次 GitHub 媒體提交；上述操作指令適用於該本機 checkout。新 clone 須在本機訓練並通過相同確認，才會啟用 V5。
'''
    if previous:
        block+='\n<details>\n<summary>保留先前 V5 訓練與未合格診斷歷史</summary>\n\n以下為先前版本當時的紀錄，非目前策略狀態。\n\n'+previous.strip()+'\n\n</details>\n'
    return block


class Finalizer:
    def __init__(self,root,runner=None,recorder=None):
        self.root=Path(root).resolve();self.runner=runner or subprocess.run
        self.recorder=recorder or recorder_module(self.root)
        self.output=self.root/'outputs/lift_v5_finalization.json'
        self.state={};self.job=None;self.gate=None;self.training=None
        self.logfile=self.root/'outputs/lift_v5_finalization.log'
        self.logfile.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.logfile,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600);os.close(fd);os.chmod(self.logfile,0o600)

    def progress(self,phase,**values):
        self.state.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**values)
        if self.job is not None:private_json(self.job/'state.json',self.state)
        private_json(self.output,self.state)
        print('LIFT_ARTIFACT '+json.dumps({'phase':phase,**{key:self.state[key] for key in ('job_id','commit','url') if key in self.state}}),flush=True)

    def run(self,command,label,*,check=True,private=False,timeout=120,env=None):
        environment=os.environ.copy();environment['Q1_LEG_ROOT']=str(self.root)
        if env:environment.update(env)
        result=self.runner([str(x) for x in command],cwd=self.root,text=True,capture_output=True,
                           env=environment,timeout=timeout,umask=0o077)
        if not private:
            with self.logfile.open('a') as stream:stream.write(label+'\n'+(result.stdout or '')+(result.stderr or ''))
        if check and result.returncode:raise RuntimeError(f'{label} failed (exit {result.returncode}); inspect the private log')
        return result

    def git(self,*arguments,**kwargs):return self.run(['git',*arguments],'git '+str(arguments[0]),**kwargs)

    def ready(self):
        training=read_json(self.root/'outputs/lift_training_status.json')
        try:gate=self.recorder.preflight_recording(self.root)
        except (ValueError,OSError,KeyError) as error:raise Pending(str(error)) from error
        if not training.get('run'):raise Pending('No completed supervisor run is recorded')
        run=Path(training['run']).resolve()
        if not run.is_relative_to(self.root/'checkpoints/isaac/lift_v5/runs'):
            raise Pending('Supervisor run pointer is outside the project')
        run_state=read_json(run/'supervisor_state.json')
        identity=supervisor_ready(training,run_state,gate,self.root)
        if self.gate is not None and gate!=self.gate:raise RuntimeError('Confirmed checkpoint/source changed during artifact finalization')
        self.gate=gate;self.training=training
        if self.job is None:
            self.job=self.root/'outputs/lift_v5_artifacts'/identity
            self.job.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(self.job,0o700)
            self.state=read_json(self.job/'state.json')
            if self.state and self.state.get('job_id')!=identity:raise RuntimeError('Artifact job identity mismatch')
            self.state.update(job_id=identity,checkpoint_sha256=gate['checkpoint_sha256'],run=str(run.relative_to(self.root)))
        return gate

    def gpu_idle(self):
        for unit in ('q1-leg-lift-training','q1-leg-training'):
            status=self.run(['systemctl','--user','is-active',unit],'Check training queue',check=False).stdout.strip()
            if status not in ('inactive','failed','unknown'):raise Pending(f'{unit} is active or its state is unavailable')
        result=self.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                        'Check GPU compute queue',check=False)
        if result.returncode:raise Pending('Cannot verify an idle GPU compute queue')
        # Display/Xorg memory is not a compute application and does not block.
        if result.stdout.strip():raise Pending('A GPU compute/Isaac process is still active')
        processes=self.run(['ps','-eo','comm=,args='],'Check Isaac startup processes',private=True)
        for line in processes.stdout.splitlines():
            fields=line.strip().split(None,1)
            if not fields:continue
            command=fields[0];arguments=fields[1] if len(fields)>1 else ''
            if (command in ('kit','isaac-sim','isaac-sim.sh')
                    or re.search(r'(?:/|^)(?:train_lift|train_interactive|train_native|play_interactive_web|play_native_web|record_lift_v5|record_interactive_diagnostic)\.py(?:\s|$)',arguments)):
                raise Pending('An Isaac worker is still starting or running')

    @contextmanager
    def gpu_lease(self):
        handles=[]
        try:
            for relative in ('outputs/lift_training.lock','outputs/interactive_training.lock'):
                handle=(self.root/relative).open('a');handles.append(handle)
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as error:raise Pending('A training supervisor still owns its queue lock') from error
            self.gpu_idle();self.ready();yield
        finally:
            for handle in handles:handle.close()

    def capture(self):
        video=self.job/'q1_leg_lift_v5.mp4'
        if video.exists():
            metadata=self.recorder.verify_artifact(video,root=self.root)
            if self.state.get('video_sha256') not in (None,metadata['video_sha256']):raise RuntimeError('Retained recording bytes changed')
            self.progress('recorded',video_sha256=metadata['video_sha256']);return video,metadata
        if self.state.get('video_sha256'):raise RuntimeError('Retained accepted recording is missing; refusing duplicate publication')
        attempts=self.state.get('recording_attempts',[])
        for seed in (42,43,44):
            if any(item['seed']==seed for item in attempts):continue
            with self.gpu_lease():
                attempt={'seed':seed,'started_at':datetime.now(timezone.utc).isoformat(),'accepted':False}
                attempts.append(attempt);self.progress('recording',recording_attempts=attempts)
                process=self.run(['bash','record_lift_v5.sh','--seed',seed],'Record confirmed native v5',
                    check=False,timeout=1800,env={'OUT':str(video)})
            attempt['exit_code']=process.returncode
            if process.returncode==0:
                metadata=self.recorder.verify_artifact(video,root=self.root)
                attempt['accepted']=True;attempt['recording_run_id']=metadata['recording_run_id']
                self.progress('recorded',video_sha256=metadata['video_sha256'],recording_attempts=attempts)
                return video,metadata
            self.progress('recording_failed',recording_attempts=attempts)
        raise RuntimeError('No accepted capture for seeds 42/43/44; failed candidates stay private, no publication')

    def gif(self,video):
        gif=self.job/'lift-v5.gif';sha=digest(video)
        if (gif.exists() and 1024<gif.stat().st_size<20*1024*1024
                and self.state.get('gif_video_sha256')==sha and self.state.get('gif_sha256')==digest(gif)):return gif
        temporary=self.job/'lift-v5.pending.gif'
        for width,colors in ((640,128),(540,96),(480,64)):
            self.run([self.root/'.venv/bin/python','tools/video_to_gif.py',video,temporary,
                      '--fps',10,'--max-width',width,'--colors',colors,'--force'],'Convert accepted recording',timeout=300)
            if 1024<temporary.stat().st_size<20*1024*1024:
                temporary.replace(gif);self.progress('gif_ready',gif_video_sha256=sha,gif_sha256=digest(gif));return gif
        raise RuntimeError('Accepted GIF exceeds its publication size bound')

    def upload(self,video):
        receipt=self.job/'upload.gofile.json';sha=digest(video)
        if receipt.exists():os.chmod(receipt,0o600)
        url=upload_url(read_json(receipt),video)
        if url:
            self.progress('uploaded',url=url,uploaded_video_sha256=sha);return url
        if self.state.get('upload_attempted'):
            raise RuntimeError('Prior upload has no verified matching receipt; reconcile it before retrying to avoid duplicate upload')
        self.progress('uploading',upload_attempted=True,upload_video_sha256=sha)
        self.run([self.root/'.venv/bin/python','tools/upload_gofile.py',video,receipt],
                 'Upload accepted v5 recording',private=True,timeout=420)
        if receipt.exists():os.chmod(receipt,0o600)
        url=upload_url(read_json(receipt),video)
        if not url:raise RuntimeError('GoFile did not confirm matching MD5 and public URL; README remains unchanged')
        self.progress('uploaded',url=url,uploaded_video_sha256=sha);return url

    def git_preflight(self):
        for relative in PUBLISH_PATHS:
            path=self.root/relative
            if not path.resolve().is_relative_to(self.root):raise ValueError('Artifact path leaves the project boundary')
            if any((self.root/Path(*Path(relative).parts[:i])).is_symlink() for i in range(1,len(Path(relative).parts)+1)):
                raise ValueError('Artifact publication paths must be real project files, not symlinks')
        if self.git('branch','--show-current').stdout.strip()!='main':raise ValueError('Artifact publication requires main')
        remote=self.git('remote','get-url','--push','origin',private=True).stdout.strip()
        if remote not in SSH_REMOTES:raise ValueError('origin must use the approved q1_leg_urdf SSH URL')
        staged=[p for p in self.git('diff','--cached','--name-only','-z').stdout.split('\0') if p]
        if any(p not in PUBLISH_PATHS for p in staged):raise ValueError('Unrelated staged files exist; index left unchanged')
        if staged and not self.state.get('git_stage_intent'):raise ValueError('Pre-existing staged artifact changes require review; index left unchanged')
        readme_parts((self.root/'README.md').read_text())
        head=self.git('rev-parse','HEAD').stdout.strip()
        if not self.state.get('base_commit'):self.progress('git_preflight',base_commit=head)
        return head

    def remote_head(self):
        result=self.git('ls-remote','origin','refs/heads/main',env={'GIT_TERMINAL_PROMPT':'0',
            'GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=20'})
        fields=result.stdout.split()
        if len(fields)!=2 or not re.fullmatch(r'[0-9a-f]{40}',fields[0]):raise RuntimeError('Cannot verify remote main')
        return fields[0]

    def publish_files(self,video,gif,metadata,url):
        if self.state.get('prepared_hashes'):
            if any(not (self.root/name).is_file() or digest(self.root/name)!=sha for name,sha in self.state['prepared_hashes'].items()):
                raise RuntimeError('Prepared publication files changed; refusing to overwrite reviewable changes')
            return
        self.ready();self.recorder.verify_artifact(video,root=self.root)
        evaluation=read_json(self.root/'checkpoints/isaac/lift_v5/evaluation.json')
        text=(self.root/'README.md').read_text();_,old,_=readme_parts(text)
        previous=self.state.get('previous_v5_block',old)
        marker='<!-- V5_ACCEPTED_JOB: '+self.state['job_id']+' -->'
        if old!=previous and marker not in old:raise RuntimeError('V5 README block changed during publication')
        self.progress('preparing_public_files',previous_v5_block=previous)
        for source,relative in ((video,VIDEO),(gif,GIF)):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        public_metadata={**metadata,'checkpoint':'checkpoints/isaac/lift_v5/latest.pt'}
        for name,value in ((METADATA,public_metadata),(EVALUATION,public_evaluation(evaluation,metadata,url,digest(video),digest(gif)))):
            target=self.root/name;target.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        (self.root/REPORT).write_text(accepted_report(evaluation,metadata,url))
        (self.root/'README.md').write_text(replace_readme(text,readme_block(evaluation,metadata,url,previous,self.state['job_id'])))
        self.recorder.verify_artifact(self.root/VIDEO,root=self.root)
        if digest(self.root/VIDEO)!=digest(video) or digest(self.root/GIF)!=digest(gif):
            raise RuntimeError('Public media copy does not match its verified private source')
        hashes={name:digest(self.root/name) for name in PUBLISH_PATHS}
        self.progress('public_files_prepared',prepared_hashes=hashes)

    def commit_push(self):
        self.ready();head=self.git_preflight();base=self.state['base_commit']
        if any(digest(self.root/name)!=sha for name,sha in self.state['prepared_hashes'].items()):
            raise RuntimeError('Publication files changed after verification')
        self.recorder.verify_artifact(self.root/VIDEO,root=self.root)
        message='Publish confirmed lift-v5 media '+self.state['checkpoint_sha256'][:12]+' ('+self.state['job_id'][:12]+')'
        commit=self.state.get('commit')
        if commit is None and head!=base:
            subject=self.git('show','-s','--format=%s',head).stdout.strip()
            parent=self.git('rev-parse',head+'^').stdout.strip()
            paths=set(self.git('diff-tree','--no-commit-id','--name-only','-r',head).stdout.splitlines())
            if not self.state.get('git_commit_intent') or subject!=message or parent!=base or not paths.issubset(PUBLISH_PATHS):
                raise RuntimeError('HEAD changed outside this artifact job; refusing to push unrelated commits')
            commit=head;self.progress('committed',commit=commit)
        if commit is None:
            self.progress('staging',git_stage_intent=True)
            self.git('add','-f','--',*sorted(PUBLISH_PATHS))
            staged=set(p for p in self.git('diff','--cached','--name-only','-z').stdout.split('\0') if p)
            if not staged or not staged.issubset(PUBLISH_PATHS):raise RuntimeError('Artifact commit has missing or unrelated staged paths')
            for path in staged:
                if self.git('rev-parse',':'+path).stdout.strip()!=self.git('hash-object','--',path).stdout.strip():
                    raise RuntimeError('Staged artifact differs from its verified working file')
            self.git('diff','--cached','--check');self.progress('committing',git_commit_intent=True)
            body=self.job/'commit-message.txt';body.write_text(message+'\n\nVerified native full-stage confirmation, per-direction unloaded foot gates and complete recording.\n')
            self.git('commit','--file',body);commit=self.git('rev-parse','HEAD').stdout.strip()
            self.progress('committed',commit=commit)
        if self.git('rev-parse','HEAD').stdout.strip()!=commit:raise RuntimeError('HEAD no longer identifies the retained artifact commit')
        remote=self.remote_head()
        if remote!=commit:
            if remote!=base:raise RuntimeError('Remote main changed; no force push is permitted')
            self.progress('pushing')
            self.git('push','origin',commit+':refs/heads/main',env={'GIT_TERMINAL_PROMPT':'0',
                'GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=20'},timeout=300)
        if self.remote_head()!=commit:raise RuntimeError('Remote main did not confirm the exact artifact commit')
        self.progress('complete',published=True)

    def execute(self,publish=False):
        self.ready();self.gpu_idle()
        if publish:
            self.git_preflight();remote=self.remote_head()
            if remote not in (self.state['base_commit'],self.state.get('commit')):
                raise RuntimeError('Publish code commits first; artifact finalizer never pushes unrelated local history')
        video,metadata=self.capture();gif=self.gif(video)
        if video.stat().st_size>=95*1024*1024:raise RuntimeError('Accepted video exceeds the GitHub publication size bound')
        self.ready();self.recorder.verify_artifact(video,root=self.root)
        if not publish:
            self.progress('prepared',published=False,video_sha256=digest(video),gif_sha256=digest(gif));return self.state
        url=self.upload(video)
        self.ready();self.publish_files(video,gif,metadata,url);self.commit_push();return self.state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--wait',action='store_true');mode.add_argument('--once',action='store_true')
    parser.add_argument('--publish',action='store_true',help='Upload verified media, update README and commit/push over approved SSH')
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--wait-timeout',type=float,default=0.,help='Zero waits indefinitely; otherwise a wall-clock limit')
    args=parser.parse_args(argv)
    if (not math.isfinite(args.poll_seconds) or not 1<=args.poll_seconds<=60
            or not math.isfinite(args.wait_timeout) or args.wait_timeout<0):
        parser.error('Poll interval must be 1–60 s and timeout finite/nonnegative')
    root=args.project_root.resolve();(root/'outputs').mkdir(parents=True,exist_ok=True)
    with (root/'outputs/lift_v5_finalize.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.exit(1,'Another lift-v5 artifact finalizer is already running\n')
        finalizer=Finalizer(root);started=time.monotonic()
        while True:
            try:finalizer.execute(publish=args.publish);return 0
            except Pending as error:
                finalizer.progress('waiting',reason=str(error),published=False)
                if not args.wait:return 2
                if args.wait_timeout and time.monotonic()-started>=args.wait_timeout:return 2
                time.sleep(args.poll_seconds)
            except (Exception,KeyboardInterrupt) as error:
                finalizer.progress('failed',error=str(error),published=False)
                print('Lift-v5 finalization stopped: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
