#!/usr/bin/env python3
"""Read-only Motion Store backed by real Spark state."""
import argparse,hashlib,json,time
import xml.etree.ElementTree as ET
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
from serve import Handler as ViewerHandler
ROOT=Path(__file__).resolve().parents[1]
PROGRESS_KEYS=("state","completed_iterations","saved_iterations","max_new_iterations","num_envs",
 "worker_log_iteration","latest_steps_per_second","updated_at","started_at","finished_at",
 "transitions","accepted","error","backend","source_git_commit")
EVAL_KEYS=("completed_iterations","episodes","survival_rate","case_survival","case_names",
 "four_step_survival","completed_four_step_rate","full_gates_passed","seed","duration_s")
def robot_specs(root=ROOT):
    """Read the source model, not a marketing estimate of real hardware."""
    root=Path(root)
    try:
        urdf=root/"urdf/q1_inertia/quanta_robot_legged.urdf"
        source=urdf.read_bytes();model=ET.fromstring(source)
        contract=json.loads((root/"config/robot_contract.json").read_text())
        joints=model.findall("joint");names={joint.get("name") for joint in joints}
        controlled=contract["joint_names"]
        if len(controlled)!=len(set(controlled)) or not set(controlled)<=names:
            raise ValueError("Control joint order does not match URDF")
        mass=sum(float(value.get("value")) for value in model.findall("link/inertial/mass"))
        return {"available":True,"name":"Q1 Native Biped","model_only":True,
            "model_uri":"/urdf/q1_inertia/quanta_robot_legged.urdf",
            "contract_uri":"/config/robot_contract.json",
            "urdf_sha256":hashlib.sha256(source).hexdigest(),"mass_kg":round(mass,9),
            "active_joints":len(controlled),"source_movable_joints":sum(
                joint.get("type") not in ("fixed","floating") for joint in joints),
            "source_links":len(model.findall("link")),"controlled_joint_names":controlled,
            "locked_joints":contract.get("locked_joints",[]),
            "battery_capacity_wh":None,"endurance_minutes":None,
            "maximum_measured_speed_m_s":None,"verified_workspace_m":None,
            "scope":"URDF 模型資料；非整機秤重、外形尺寸或已驗收的實機性能。"}
    except (OSError,ValueError,KeyError,ET.ParseError):
        return {"available":False,"model_only":True,"scope":"原生機型資料暫時無法讀取。"}

def store_payload(root=ROOT,now=None):
    root=Path(root);now=time.time() if now is None else now
    catalog=json.loads((root/"web/store/catalog.json").read_text())
    progress={"state":"not_started","accepted":False,"evaluations":[],"stale":False}
    path=root/"outputs/spark_balance_status.json"
    if path.exists():
        try:
            raw=json.loads(path.read_text());progress.update({k:raw[k] for k in PROGRESS_KEYS if k in raw})
            progress["evaluations"]=[{k:v[k] for k in EVAL_KEYS if k in v} for v in raw.get("evaluations",[])]
            progress["stale"]=now-raw.get("updated_at",0)>120 and raw.get("state") not in ("budget_exhausted","stopped","runtime_error")
            saved=raw.get("saved_iterations",raw.get("completed_iterations",0))
            progress["saved_iterations"]=saved
            progress["live_iteration"]=max(saved,raw.get("worker_log_iteration") or 0) if raw.get("state")=="training" else saved
            elapsed=now-raw.get("started_at",now)
            if progress["live_iteration"]>0 and elapsed>0:
                progress["estimated_remaining_s"]=max(0,(raw.get("max_new_iterations",15000)-progress["live_iteration"])*elapsed/progress["live_iteration"])
        except (OSError,ValueError,TypeError):
            progress.update(state="updating",stale=True)
    return {"catalog":catalog,"robot":robot_specs(root),"training":progress,"generated_at":now,
            "features":{"youtube_ingest":False,"upload":False,"new_clip_training":False}}
class StoreHandler(ViewerHandler):
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path in ("/","/store","/store/"):
            self.send_response(302);self.send_header("Location","/web/store/");self.end_headers();return
        if path=="/api/store":
            try:content=json.dumps(store_payload(),ensure_ascii=False,allow_nan=False).encode()
            except (OSError,ValueError):self.send_error(503);return
            self.send_response(200);self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(content)));self.end_headers();self.wfile.write(content);return
        if path=="/web/store/":self.path="/web/store/index.html"
        if path.startswith("/outputs/"):
            permitted={f"/outputs/{prefix}{skill}_policy_rollout.json" for prefix in ("","isaac_") for skill in ("stand","walk","sidestep")}
            if path not in permitted:self.send_error(404);return
        super().do_GET()
    def do_POST(self):self.send_error(405,"Store is currently read-only")
def main():
    p=argparse.ArgumentParser();p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8876);args=p.parse_args()
    with ThreadingHTTPServer((args.host,args.port),StoreHandler) as server:
        print(f"Q1 Motion Store: http://{args.host}:{args.port}/web/store/",flush=True);server.serve_forever()
if __name__=="__main__":main()
