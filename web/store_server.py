#!/usr/bin/env python3
"""Read-only Motion Store backed by real Spark state."""
import argparse,json,time
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
from serve import Handler as ViewerHandler
ROOT=Path(__file__).resolve().parents[1]
PROGRESS_KEYS=("state","completed_iterations","saved_iterations","max_new_iterations","num_envs",
 "worker_log_iteration","latest_steps_per_second","updated_at","started_at","finished_at",
 "transitions","accepted","error","backend","source_git_commit")
EVAL_KEYS=("completed_iterations","episodes","survival_rate","case_survival","case_names",
 "four_step_survival","completed_four_step_rate","full_gates_passed","seed","duration_s")
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
    return {"catalog":catalog,"training":progress,"generated_at":now,
            "features":{"youtube_ingest":False,"upload":False,"new_clip_training":False}}
class StoreHandler(ViewerHandler):
    def do_GET(self):
        path=urlsplit(self.path).path
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
