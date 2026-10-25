#!/usr/bin/env python3
"""Publish the explicitly requested 20000-update snapshot, never promote acceptance."""
import fcntl, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terminal_v6_publication.json'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(s):
    tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(OUT)
def run(args,timeout=120):
    r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'{args[0]} failed: {r.stderr[-1500:]}')
    return r.stdout.strip()
def main():
    s=read(OUT) if OUT.exists() else {}
    if s.get('published'):return
    while True:
        state=read(ROOT/'outputs/upper_body_training_status.json')
        if state['completed_iterations']==20000 and state['state'] in ('budget_exhausted','complete'):
            if subprocess.run(['systemctl','--user','is-active','--quiet','q1-leg-upper-training']).returncode!=0:break
        s.update(phase='waiting_for_20000',saved_updates=state['completed_iterations']);save(s);time.sleep(30)
    # Serialize with both existing Git publishers. No frozen training files change.
    with (ROOT/'outputs/upper_body_publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if s.get('commit'):
            if run(['git','rev-parse','HEAD'])!=s['commit']:raise RuntimeError('HEAD changed after publication commit')
            run(['git','push','origin','main'],timeout=120)
            if run(['git','ls-remote','origin','refs/heads/main']).split()[0]!=s['commit']:raise RuntimeError('Remote mismatch')
            s.update(phase='published',published=True);save(s);return
        from run_upper_body_training import source_manifest
        if source_manifest(Path(state['profile_path']))!=state['source_sha256']:raise RuntimeError('Source mismatch')
        folder=Path(state['run']);result=read(folder/'training_result.json')
        if result['completed_iterations']!=20000 or digest(folder/'resume.pt')!=result['checkpoint_sha256']:raise RuntimeError('Checkpoint mismatch')
        dest=ROOT/'checkpoints/isaac/upper_body_v6_20000';dest.mkdir(exist_ok=True)
        for src,name in [('resume.pt','latest.pt'),('contract.json','contract.json'),('training_result.json','training_result.json')]:shutil.copyfile(folder/src,dest/name)
        video=ROOT/'docs/results/q1_leg_v6_20000_diagnostic.mp4'
        if not video.with_suffix('.json').exists():
            if run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):raise RuntimeError('GPU compute process active; refusing overlapping recording')
            s.update(phase='recording');save(s)
            cmd='source scripts/isaac_env.sh\n"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/record_terminal_v6_diagnostic.py --checkpoint checkpoints/isaac/upper_body_v6_20000/latest.pt --video docs/results/q1_leg_v6_20000_diagnostic.mp4 --sequence-only --headless'
            # Full native episode, no fallback checkpoint, no hidden reset.
            run(['bash','-c',cmd],timeout=900)
        meta=read(video.with_suffix('.json'))
        if meta.get('technical_error') or meta.get('frames',0)<1 or meta.get('accepted') is not False or meta.get('video_sha256')!=digest(video) or meta.get('checkpoint_sha256')!=digest(dest/'latest.pt'):raise RuntimeError('Invalid diagnostic recording')
        run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],timeout=180)
        receipt=ROOT/'outputs/terminal_v6.gofile.json'
        if not receipt.exists():
            if s.get('upload_started'):raise RuntimeError('Upload outcome unknown; inspect before retry')
            s.update(phase='uploading',upload_started=True);save(s)
            run([str(ROOT/'.venv/bin/python'),'tools/upload_gofile.py',str(video),str(receipt)],timeout=420)
            os.chmod(receipt,0o600)
        upload=read(receipt);data=upload['data'];url=data['downloadPage']
        if upload.get('status')!='ok' or not url.startswith('https://gofile.io/d/') or data.get('md5')!=hashlib.md5(video.read_bytes()).hexdigest():raise RuntimeError('Upload identity mismatch')
        report={'iterations':20000,'num_envs':10240,'accepted':False,'scope':'Training budget completed; this snapshot and video are diagnostic, not a qualified controller.', 'checkpoint_sha256':digest(dest/'latest.pt'),'video_sha256':digest(video),'gofile':url,'latest_validation':state.get('latest_validation'),'failure_in_video':meta['failure']}
        # Include the actual latest full evaluation rather than claiming acceptance.
        if state.get('latest_validation') is not None:report['latest_validation']=state['latest_validation']
        validations=[e for e in state['evaluations'] if e['kind']=='validation']
        if validations:shutil.copyfile(validations[-1]['report'],dest/'evaluation.json')
        (dest/'publication.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        text=(ROOT/'README.md').read_text();marker='<!-- V6_20000_DIAGNOSTIC -->'
        if marker not in text:
            text+=f'\n{marker}\n\n### V6 累計 20,000 次 PPO 終點紀錄\n\n使用 10,240 envs；包含先前分支更新。此權重尚未通過完整動作驗收，影片為原生 PhysX 診斷，保留失敗、不重設、不代表已學會閃身步。\n\n[觀看 GoFile 影片]({url}) · [PPO 權重與驗測紀錄](checkpoints/isaac/upper_body_v6_20000/)\n'
            (ROOT/'README.md').write_text(text)
        paths=['README.md','scripts/reinforcement_learning/rsl_rl/record_terminal_v6_diagnostic.py','tools/publish_terminal_v6.py',str(dest.relative_to(ROOT)),str(video.relative_to(ROOT)),str(video.with_suffix('.json').relative_to(ROOT))]
        if run(['git','remote','get-url','origin'])!='git@github.com:willnien10005914/q1_leg_urdf.git':raise RuntimeError('Unexpected origin')
        if run(['git','branch','--show-current'])!='main':raise RuntimeError('Unexpected branch')
        run(['git','fetch','origin','main'],timeout=90)
        # Never push unrelated unpublished commits.
        if not s.get('commit') and run(['git','rev-parse','HEAD'])!=run(['git','rev-parse','origin/main']):raise RuntimeError('Unpublished commits require review')
        run(['git','add','-f','--',*paths]);run(['git','commit','--only','-m','Publish 20000-update V6 diagnostic checkpoint and native video','--',*paths])
        s.update(phase='pushing',commit=run(['git','rev-parse','HEAD']),url=url);save(s)
        run(['git','push','origin','main'],timeout=120)
        if run(['git','ls-remote','origin','refs/heads/main']).split()[0]!=s['commit']:raise RuntimeError('Remote mismatch')
        s.update(phase='published',published=True);save(s)
if __name__=='__main__':
    try:main()
    except Exception as e:
        state=read(OUT) if OUT.exists() else {};state.update(phase='error',error=str(e));save(state);raise
