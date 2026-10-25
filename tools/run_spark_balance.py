#!/usr/bin/env python3
"""Spark-only bounded PPO queue, frozen sources and auditable progress."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,re,shutil,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKER="scripts/reinforcement_learning/rsl_rl/train_spark_balance.py"
STATE=ROOT/"outputs/spark_balance_status.json"
LOCK=ROOT.parent/".spark-training-gpu.lock"
STOP=False

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)

def next_chunk(completed,maximum,chunk=250):
    if not 0<=completed<maximum:raise ValueError("No authorized updates remain")
    return min(10 if completed==0 else chunk-completed%chunk,maximum-completed)

def validate_config(config):
    if config["num_envs"]!=6144 or config["max_new_iterations"]!=15000:
        raise ValueError("Authorized experiment requires exactly 6144 envs and 15000 new iterations")
    for key in ("chunk_iterations","validation_every_iterations","worker_timeout_s"):
        if config[key]<=0:raise ValueError(key+" must be positive")
    if config["validation_episodes"]<4 or config["validation_episodes"]%4:
        raise ValueError("Validation must cover all four cases equally")
    if config["backend"]!="physx":raise ValueError("Q1 Newton migration must pass separate validation")
    bootstrap=ROOT/config["bootstrap"]
    if not bootstrap.is_file() or sha(bootstrap)!=config["bootstrap_sha256"]:
        raise ValueError("Bootstrap checkpoint hash mismatch")

def source_paths():
    names=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0")
    prefixes=("source/","scripts/","tools/","config/","urdf/","meshes/","assets/","docs/reference/","checkpoints/isaac/upper_body_v6_20000/")
    return sorted(n for n in names if n and (n.startswith(prefixes) or n in ("pyproject.toml","quanta_robot_legged.urdf")))

def runtime_sources(config):
    directory=Path(config["isaaclab_path"])/"source"
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.suffix in (".py",".toml")}

def snapshot(run,config):
    destination=run/"source_snapshot";manifest={}
    for relative in source_paths():
        original=ROOT/relative;target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target);manifest[relative]=sha(target)
    if WORKER not in manifest:raise ValueError("Stage/commit Spark worker before freezing run")
    write(run/"source_manifest.json",manifest);write(run/"config.json",config)
    write(run/"runtime_source_manifest.json",runtime_sources(config))
    return destination,manifest

def verify_snapshot(run):
    runtime=run/"runtime_source_manifest.json"
    if runtime.exists() and read(runtime)!=runtime_sources(read(run/"config.json")):
        raise ValueError("Shared Isaac Lab runtime source changed")
    for relative,digest in read(run/"source_manifest.json").items():
        if sha(run/"source_snapshot"/relative)!=digest:raise ValueError("Frozen source changed: "+relative)

def event(run,kind,**details):
    with (run/"events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time":time.time(),"kind":kind,**details},allow_nan=False)+"\n")

def publish(run,record,**updates):
    record.update(updated_at=time.time(),**updates);write(run/"status.json",record);write(STATE,record)

def other_compute():
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],
                          capture_output=True,text=True,check=True)
    busy=[]
    for value in result.stdout.splitlines():
        if not value.strip().isdigit():continue
        pid=int(value)
        try:
            command=(Path("/proc")/str(pid)/"cmdline").read_bytes().replace(b"\0",b" ").decode()
            cwd=(Path("/proc")/str(pid)/"cwd").resolve()
        except FileNotFoundError:continue
        except (PermissionError,ProcessLookupError):
            busy.append({"pid":pid,"command":"unreadable compute process"});continue
        # This existing native viewer was verified alongside 6144-env training.
        if "play_native_web.py" in command and cwd==ROOT.parent/"q1_wheel_urdf":continue
        busy.append({"pid":pid,"command":command[:300]})
    return busy

def stop_requested(signum,frame):
    global STOP
    STOP=True

def terminate(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def worker(run,state,config,checkpoint,output,*,evaluate=False,initialize=False,full=False,steps=0,episodes=None):
    verify_snapshot(run)
    while True:
        if STOP:raise InterruptedError("Stop requested before worker")
        busy=other_compute()
        if not busy:break
        publish(run,state,state="waiting_for_other_compute",waiting_compute=busy);time.sleep(10)
    frozen=run/"source_snapshot"
    count=(episodes or config["validation_episodes"]) if evaluate else config["num_envs"]
    arguments=["--checkpoint",str(checkpoint),"--output",str(output),
        "--completed",str(state["completed_iterations"]),"--envs",str(count),
        "--seed",str(config["seed"]+state["completed_iterations"]+(100000 if evaluate else 0)),
        "--iterations",str(1 if evaluate else steps),"--max-new-iterations",str(config["max_new_iterations"]),"--headless"]
    if evaluate:arguments.append("--evaluate")
    if initialize:arguments.append("--initialize")
    if full:arguments.append("--full-gates")
    command=["bash","-c",'source scripts/isaac_env.sh\nexec "$ISAAC_PYTHON" -u "$@"',"spark-balance",WORKER,*arguments]
    output.mkdir(parents=True,exist_ok=False);env=os.environ.copy()
    env.update(Q1_LEG_ROOT=str(frozen),ISAACLAB_PATH=config["isaaclab_path"],ISAAC_PYTHON=config["isaac_python"])
    publish(run,state,state="evaluating" if evaluate else "training",worker_output=str(output),
            worker_started_at=time.time(),current_chunk_iterations=steps,worker_log_iteration=None,
            saved_iterations=state["completed_iterations"])
    start=time.monotonic()
    with (output/"native.log").open("w") as log:
        child=subprocess.Popen(command,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        publish(run,state,child_pid=child.pid)
        event(run,"worker_started",pid=child.pid,output=str(output),evaluate=evaluate,updates=steps)
        try:
            while child.poll() is None:
                if STOP:raise InterruptedError("Stop requested; last committed chunk retained")
                if time.monotonic()-start>config["worker_timeout_s"]:
                    raise TimeoutError("Worker timeout; no automatic replay of possibly spent updates")
                text=(output/"native.log").read_text(errors="replace")[-20000:]
                clean=re.sub(r"\x1b\[[0-9;]*m","",text)
                matches=re.findall(r"Learning iteration\s+(\d+)/(\d+)",clean)
                rates=re.findall(r"Steps per second:\s+([0-9.]+)",clean)
                publish(run,state,worker_elapsed_s=round(time.monotonic()-start,1),
                        worker_log_iteration=int(matches[-1][0]) if matches else None,
                        latest_steps_per_second=float(rates[-1]) if rates else None)
                time.sleep(5)
        finally:terminate(child)
    verify_snapshot(run)
    result_path=output/("evaluation.json" if evaluate else "training_result.json")
    if child.returncode or not result_path.is_file() or (output/"worker_error.json").exists():
        raise RuntimeError(f"Worker exited {child.returncode}; inspect {output/'native.log'}")
    result=read(result_path)
    if not evaluate:
        if result["completed_iterations"]!=state["completed_iterations"]+steps or result["num_envs"]!=config["num_envs"]:
            raise ValueError("Saved PPO counter/env count differs from scheduled chunk")
        if result["checkpoint_sha256"]!=sha(output/"resume.pt"):raise ValueError("Checkpoint hash mismatch")
    event(run,"worker_completed",output=str(output),completed=result.get("completed_iterations"))
    return result

def evaluation_attempt(base):
    if not base.exists():return base
    attempt=2
    while base.with_name(base.name+f"_attempt_{attempt:03d}").exists():attempt+=1
    return base.with_name(base.name+f"_attempt_{attempt:03d}")

def evaluate_checkpoint(run,state,config,checkpoint,chunk):
    target=state["completed_iterations"]
    report_dir=evaluation_attempt(chunk/"audit")
    report=worker(run,state,config,checkpoint,report_dir,evaluate=True)
    item={"completed_iterations":target,"report":str(report_dir/"evaluation.json"),**report}
    qualifies=report.get("completed_four_step_rate",0)>=.98 and report["survival_rate"]>=.98
    if qualifies or target==15000:
        full_dir=evaluation_attempt(chunk/"full_gates")
        full=worker(run,state,config,checkpoint,full_dir,evaluate=True,full=True)
        item["full_gates_passed"]=bool(full.get("passed"))
        item["full_gates_report"]=str(full_dir/"evaluation.json")
    state["evaluations"]=[v for v in state["evaluations"] if v["completed_iterations"]!=target]+[item]
    publish(run,state,latest_evaluation=item,pending_evaluation=False)

def execute(run,config,resume):
    if resume:
        state=read(run/"status.json")
        if read(run/"config.json")!=config:raise ValueError("Resume config differs from frozen run")
        if state["state"] not in ("stopped","runtime_error"):raise ValueError("Resume requires stopped/error state")
        if state.get("unsafe_to_auto_resume"):raise ValueError("Interrupted training chunk needs manual budget reconciliation")
        verify_snapshot(run)
    else:
        if run.exists():raise ValueError("Run already exists; choose explicit resume")
        run.mkdir(parents=True);frozen,manifest=snapshot(run,config)
        state={"experiment":config["experiment"],"backend":"physx","accepted":False,
            "completed_iterations":0,"saved_iterations":0,"max_new_iterations":15000,
            "prior_v6_updates":20000,"num_envs":6144,"run":str(run),
            "source_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "source_files":len(manifest),"evaluations":[],"started_at":time.time()}
        publish(run,state,state="prepared")
    checkpoint=Path(state.get("checkpoint",run/"source_snapshot"/config["bootstrap"]))
    if state["completed_iterations"]==0 and "baseline" not in state:
        baseline=worker(run,state,config,checkpoint,evaluation_attempt(run/"baseline"),evaluate=True)
        publish(run,state,baseline=baseline)
    if state.get("pending_evaluation"):
        evaluate_checkpoint(run,state,config,checkpoint,checkpoint.parent)
    while state["completed_iterations"]<15000:
        steps=next_chunk(state["completed_iterations"],15000,config["chunk_iterations"])
        target=state["completed_iterations"]+steps;chunk=run/"chunks"/f"{target:07d}"
        result=worker(run,state,config,checkpoint,chunk,steps=steps,initialize=state["completed_iterations"]==0)
        checkpoint=chunk/"resume.pt"
        publish(run,state,completed_iterations=target,saved_iterations=target,checkpoint=str(checkpoint),
                checkpoint_sha256=result["checkpoint_sha256"],transitions=target*config["num_envs"]*24,
                unsafe_to_auto_resume=False)
        if target==10 or target%config["validation_every_iterations"]==0 or target==15000:
            publish(run,state,pending_evaluation=True)
            evaluate_checkpoint(run,state,config,checkpoint,chunk)
    final=state["evaluations"][-1]
    if len(state["evaluations"])>=3 and all(v.get("full_gates_passed") for v in state["evaluations"][-3:]):
        state["confirmation"]=worker(run,state,config,checkpoint,evaluation_attempt(run/"confirmation_128"),evaluate=True,full=True,episodes=128)
    publish(run,state,state="budget_exhausted",child_pid=None,accepted=False,finished_at=time.time(),
            validation_status="candidate_requires_review" if final.get("full_gates_passed") else "failed_gate")
    event(run,"budget_exhausted",completed_iterations=15000,accepted=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/spark_balance_6144_15000.json")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--status",action="store_true")
    parser.add_argument("--dry-run",action="store_true");args=parser.parse_args()
    if args.status:
        print(json.dumps(read(STATE) if STATE.exists() else {"state":"not_started"},indent=2));return
    config=read(args.config);validate_config(config);run=(ROOT/config["run"]).resolve()
    if not run.is_relative_to(ROOT/"checkpoints/isaac/spark_balance/runs"):
        raise ValueError("Run must use dedicated Spark checkpoint tree")
    if args.dry_run:
        print(json.dumps({"config":config,"run":str(run),"other_compute":other_compute(),
                          "source_files":len(source_paths()),"valid":True},indent=2));return
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_requested)
    with LOCK.open("a") as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                write(STATE,{"state":"queued","run":str(run),"accepted":False,"max_new_iterations":15000,
                             "num_envs":6144,"updated_at":time.time()})
                if STOP:return
                time.sleep(5)
        try:execute(run,config,args.resume)
        except BaseException as error:
            state=read(run/"status.json") if (run/"status.json").exists() else {}
            was_training=state.get("state")=="training"
            publish(run,state,state="stopped" if isinstance(error,InterruptedError) else "runtime_error",
                    error=str(error),unsafe_to_auto_resume=was_training,child_pid=None,accepted=False)
            event(run,"stopped",error=str(error),unsafe_to_auto_resume=was_training);raise

if __name__=="__main__":main()
