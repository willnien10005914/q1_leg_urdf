#!/usr/bin/env python3
"""Bounded balance retraining with native four-step audits and frozen sources."""
import fcntl,hashlib,json,subprocess,time,zipfile,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'checkpoints/isaac/balance_v7/runs/balance_20261025'
STATE=ROOT/'outputs/balance_v7_status.json'
WORKER='scripts/reinforcement_learning/rsl_rl/train_balance_v7.py'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sources():
 old=read(ROOT/'outputs/upper_body_training_status.json')['source_sha256']
 paths=list(old)+[WORKER,'tools/run_balance_v7.py','source/leg_humanoid_lab/leg_humanoid_lab/balance_env.py','source/leg_humanoid_lab/leg_humanoid_lab/balance_objective.py']
 return {p:sha(ROOT/p) for p in paths}
def wait_gpu(s):
 while True:
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if not busy:return
  s.update(state='waiting_for_gpu',waiting_compute_pids=busy.splitlines(),updated_at=time.time());write(STATE,s)
  time.sleep(30)

def worker(s,checkpoint,out,*,evaluate=False,initialize=False,steps=100,seed=42,full=False):
 wait_gpu(s)
 if sources()!=s['source_sha256']:raise RuntimeError('Frozen balance source changed')
 args=['--checkpoint',str(checkpoint),'--output',str(out),'--completed',str(s['completed_iterations']),'--envs',str(64 if evaluate else 10240),'--seed',str(seed),'--iterations',str(steps),'--headless']
 if evaluate:args+=['--evaluate']
 if initialize:args+=['--initialize']
 if full:args+=['--full-gates']
 # Positional parameters preserve shell quoting; no interpolated commands.
 command=['bash','-c','source scripts/isaac_env.sh\n"$ISAAC_PYTHON" -u "$@"','balance-v7',WORKER,*args]
 s.update(state='evaluating' if evaluate else 'training',worker_output=str(out),updated_at=time.time());write(STATE,s)
 out.mkdir(parents=True,exist_ok=True)
 with (out/'native.log').open('a') as log:
  child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  s['child_pid']=child.pid;write(STATE,s)
  # Chunk startup may occasionally deadlock. Fail closed, bounded 12 minutes,
  # no automatic retry and no repeating updates if a snapshot appeared.
  try:code=child.wait(timeout=720)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise RuntimeError('Native worker timed out; checkpoint retained for review')
 if sources()!=s['source_sha256']:raise RuntimeError('Source changed during worker')
 target=out/('evaluation.json' if evaluate else 'training_result.json')
 if code or not target.exists() or (out/'worker_error.json').exists():raise RuntimeError('Native worker failed; inspect '+str(out/'native.log'))
 result=read(target)
 if not evaluate:
  if result['completed_iterations']!=s['completed_iterations']+steps or sha(out/'resume.pt')!=result['checkpoint_sha256']:raise RuntimeError('Serialized checkpoint counter/hash mismatch')
  with zipfile.ZipFile(out/'resume.pt') as z:
   if z.testzip():raise RuntimeError('Corrupt checkpoint')
 return result

def main():
 RUN.mkdir(parents=True,exist_ok=True)
 if STATE.exists():raise RuntimeError('Existing balance run requires explicit review, no silent reset')
 s={'experiment':'balance_v7','completed_iterations':0,'max_new_iterations':20000,'prior_v6_updates':20000,'num_envs':10240,'accepted':False,'run':str(RUN),'source_sha256':sources(),'evaluations':[]}
 write(RUN/'source_manifest.json',s['source_sha256'])
 # Preserve all source bytes required to reproduce this new experiment.
 for relative in s['source_sha256']:
  dest=RUN/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
 base=ROOT/'checkpoints/isaac/lift_v5/runs/20261022_0338_recovery20/chunks/0013115_s2_r2/resume.pt'
 terminal=ROOT/'checkpoints/isaac/upper_body_v6_20000/latest.pt'
 baseline=[]
 for name,path in [('original13115',base),('terminal20000',terminal)]:
  r=worker(s,path,RUN/('baseline_'+name),evaluate=True)
  baseline.append((r['four_step_survival'],r['survival_rate'],name,path,r))
 selected=max(baseline,key=lambda x:(x[0],x[1],x[2]=='original13115'))
 s['baseline_comparison']=[{'name':x[2],'checkpoint_sha256':sha(x[3]),'report':x[4]} for x in baseline]
 s['initial_actor']=selected[2];s['initial_checkpoint_sha256']=sha(selected[3]);checkpoint=selected[3]
 s['initialization']='Complete actor transfer only; fresh critic/Adam/std; prior V6 budget retained separately'
 write(STATE,s)
 # First chunk is only ten updates to validate the full 10240-env training path.
 while s['completed_iterations']<s['max_new_iterations']:
  steps=10 if s['completed_iterations']==0 else min(100 if s['completed_iterations']<510 else 500,s['max_new_iterations']-s['completed_iterations'])
  out=RUN/f"chunk_{s['completed_iterations']+steps:07d}"
  result=worker(s,checkpoint,out,initialize=s['completed_iterations']==0,steps=steps)
  s['completed_iterations']=result['completed_iterations'];checkpoint=out/'resume.pt';s['checkpoint']=str(checkpoint);s['checkpoint_sha256']=result['checkpoint_sha256'];write(STATE,s)
  r=worker(s,checkpoint,out/'four_step_eval',evaluate=True,seed=1042+len(s['evaluations']))
  s['evaluations'].append({'completed_iterations':s['completed_iterations'],'report':str(out/'four_step_eval/evaluation.json'),**r});write(STATE,s)
  if r['four_step_survival']>=.98 and r['survival_rate']>=.98:
   full=worker(s,checkpoint,out/'full_gate_eval',evaluate=True,seed=2142+len(s['evaluations']),full=True)
   s['latest_full_gate_report']=str(out/'full_gate_eval/evaluation.json')
   if full.get('passed'):
    # Passing this report is only a candidate; no accepted contract/media
    # publication without repeated full gates and independent confirmation.
    s.update(state='candidate_requires_independent_confirmation',accepted=False);write(STATE,s);return
  recent=s['evaluations'][-5:]
  if s['completed_iterations']>=2000 and len(recent)==5 and max(x['four_step_survival'] for x in recent)-min(x['four_step_survival'] for x in recent)<.02:
   s.update(state='plateau_requires_review',accepted=False);write(STATE,s);return
 s.update(state='budget_exhausted',accepted=False);write(STATE,s)
if __name__=='__main__':
 with (ROOT/'outputs/upper_body_training.lock').open('a') as lock:
  write(ROOT/'outputs/balance_v7_queue.json',{'state':'waiting_for_shared_gpu_queue','prior_v6_updates':20000,'new_updates':0,'num_envs':10240,'max_new_iterations':20000,'accepted':False})
  fcntl.flock(lock,fcntl.LOCK_EX)
  write(ROOT/'outputs/balance_v7_queue.json',{'state':'shared_queue_acquired','accepted':False})
  try:main()
  except BaseException as e:
   s=read(STATE) if STATE.exists() else {};s.update(state='runtime_error',error=str(e),accepted=False);write(STATE,s);raise
