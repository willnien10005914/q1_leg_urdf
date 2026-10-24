#!/usr/bin/env python3
"""Local read-only viewer; never exposes checkpoint, environment, or .git files."""
import argparse
import importlib.util
import json
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
from urllib.request import Request,urlopen
from urllib.error import URLError,HTTPError

ROOT=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,".js":"application/javascript",".urdf":"application/xml",".stl":"model/stl"}
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT),**kwargs)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/live':
            self.live_proxy('/state');return
        if path=="/":
            self.send_response(302)
            self.send_header("Location","/web/")
            self.end_headers()
            return
        if path=="/api/status":
            result={"robot":json.loads((ROOT/"config/robot_contract.json").read_text()),"skills":{}}
            policy_status=ROOT/"config/web_policy_status.json"
            if policy_status.exists():
                result["policy_status"]=json.loads(policy_status.read_text())
            accepted=ROOT/'checkpoints/isaac/upper_body_v6/evaluation.json'
            if accepted.exists():
                spec=importlib.util.spec_from_file_location('web_v6_policy_gate',ROOT/'tools/v6_policy_gate.py')
                gate_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate_module)
                if gate_module.v6_policy_gate(ROOT).get('ready') is True:
                    result.setdefault('policy_status',{}).update(v6_live_accepted=True,
                        notice='本機 V6 已通過完整上／下半身驗收與獨立128回合確認，可啟動即時控制。13115回放保留作舊版比較。')
            for skill in ("stand","walk","sidestep"):
                file=ROOT/"checkpoints"/skill/"evaluation.json"
                native=ROOT/"checkpoints/isaac"/skill/"evaluation.json"
                if native.exists():
                    result["skills"][skill]={**json.loads(native.read_text()),"rollout":f"/outputs/isaac_{skill}_policy_rollout.json"}
                else:
                    result["skills"][skill]={**json.loads(file.read_text()),"rollout":f"/outputs/{skill}_policy_rollout.json"} if file.exists() else {"passed":False,"state":"not_trained"}
            training=ROOT/"outputs/training_status.json"
            if training.exists():
                try:
                    result["training"]=json.loads(training.read_text())
                except json.JSONDecodeError:
                    result["training"]={"state":"updating"}
            content=json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        allowed={"web","meshes","urdf","docs","config","outputs"}
        parts=Path(path.lstrip("/")).parts
        resolved=(ROOT/path.lstrip("/")).resolve()
        if not parts or parts[0] not in allowed or any(p.startswith(".") or p==".." or p.endswith(('.gofile.json','.private.json')) for p in parts) or not resolved.is_relative_to(ROOT) or (resolved.is_dir() and path!="/web/"):
            self.send_error(404)
            return
        super().do_GET()
    def do_POST(self):
        if urlsplit(self.path).path!='/api/live/command':
            self.send_error(404);return
        origin=self.headers.get('Origin')
        if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
            self.send_error(403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>4096:raise ValueError()
            body=self.rfile.read(length)
            json.loads(body)
        except (ValueError,UnicodeDecodeError):
            self.send_error(400);return
        self.live_proxy('/command',body)
    def live_proxy(self,path,body=None):
        try:
            req=Request('http://127.0.0.1:8877'+path,data=body,headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=1) as response:
                content=response.read();status=response.status
        except HTTPError as error:
            content=error.read();status=error.code
        except (URLError,TimeoutError):
            content=json.dumps({'error':'即時 PPO 模擬尚未啟動，請使用參考動作或評估記錄。'}).encode();status=503
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(content)))
        self.end_headers();self.wfile.write(content)
    def end_headers(self):
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        super().end_headers()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=8876)
    p.add_argument("--host",default="127.0.0.1")
    args=p.parse_args()
    with ThreadingHTTPServer((args.host,args.port),Handler) as server:
        print(f"Q1 Leg: http://{args.host}:{args.port}/web/",flush=True)
        server.serve_forever()
if __name__=="__main__":
    main()
