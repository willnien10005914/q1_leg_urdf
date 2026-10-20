#!/usr/bin/env python3
"""Native PhysX PPO player with a local-only live state and command API.

Switches reset simulation; they are not claims of learned skill transitions.
Only policies with a matching native contract and a passed evaluation can run.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--port',type=int,default=8877)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
launcher=AppLauncher(args)
app=launcher.app

import json
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import queue
import signal
import threading
import time
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from leg_humanoid_lab.isaac_env import IsaacLegEnv,LegEnvCfg
from leg_humanoid_lab.agent_cfg import PPOCfg
from leg_humanoid_lab.robot import ROOT,JOINTS,SKILLS,contract
from runtime_compat import runner_cfg,load_checkpoint

commands=queue.Queue(maxsize=32)
lock=threading.Lock()
state={'status':'starting','skill':'stand','backend':'isaaclab','proxy':False,'frame':None}
stopping=threading.Event()
signal.signal(signal.SIGTERM,lambda *_:stopping.set())
signal.signal(signal.SIGINT,lambda *_:stopping.set())
def available():
    out={}
    expected=contract()
    for skill in SKILLS:
        folder=ROOT/'checkpoints/isaac'/skill
        try:
            saved=json.loads((folder/'contract.json').read_text())
            evaluation=json.loads((folder/'evaluation.json').read_text())
            checkpoint=folder/'latest.pt'
            out[skill]=bool(evaluation.get('passed') and (skill!='sidestep' or evaluation.get('episodes',0)>=32) and evaluation.get('proxy') is False and evaluation.get('urdf_sha256')==expected['urdf_sha256'] and checkpoint.is_file() and evaluation.get('checkpoint_sha256')==hashlib.sha256(checkpoint.read_bytes()).hexdigest() and all(saved.get(k)==expected[k] for k in ('robot','urdf_sha256','joint_names','observation_size')))
        except (OSError,ValueError):
            out[skill]=False
    return out

class Handler(BaseHTTPRequestHandler):
    def send_json(self,value,status=200):
        content=json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    def do_GET(self):
        if self.path!='/state':
            self.send_json({'error':'unknown endpoint'},404);return
        with lock:
            value=dict(state)
        value['available']=available()
        self.send_json(value)
    def do_POST(self):
        if self.path!='/command':
            self.send_json({'error':'unknown endpoint'},404);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>4096:raise ValueError('invalid body length')
            body=json.loads(self.rfile.read(length))
            skill=body.get('skill','stand')
            action=body.get('action','restart')
            if skill not in SKILLS or action not in ('restart','pause','play'):raise ValueError('invalid command')
            if not available().get(skill):
                self.send_json({'error':f'{skill} 尚未通過原生 CAD PPO 評估'},409);return
            commands.put_nowait((skill,action))
        except (ValueError,queue.Full):
            self.send_json({'error':'invalid or busy command'},400);return
        self.send_json({'accepted':True,'skill':skill,'action':action})
    def log_message(self,*args):pass

env=None
server=None
try:
    cfg=LegEnvCfg()
    cfg.scene.num_envs=1
    cfg.sim.device=args.device
    cfg.seed=42
    env=IsaacLegEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=args.device)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    current=None
    paused=True
    obs=None
    policy=None
    if available().get('stand'):commands.put(('stand','restart'))
    print(f'Native live PPO API: http://127.0.0.1:{args.port}/state',flush=True)
    with torch.inference_mode():
        while not stopping.is_set() and app.is_running():
            start=time.monotonic()
            try:
                skill,action=commands.get_nowait()
                if current!=skill or action=='restart' or state.get('status')=='finished':
                    load_checkpoint(runner,ROOT/'checkpoints/isaac'/skill/'latest.pt',load_optimizer=False,load_iteration=False)
                    env.set_skill(skill)
                    obs=wrapped.reset()[0]
                    policy=runner.get_inference_policy(device=env.device)
                    current=skill
                paused=action=='pause'
                with lock:state.update(status='paused' if paused else 'running',skill=current,error=None)
            except queue.Empty:pass
            if current is not None and not paused:
                if env.episode_length_buf[0]>=env.max_episode_length-2:
                    paused=True
                    with lock:state['status']='finished'
                else:
                    obs,_,done,_=wrapped.step(policy(obs))
                    if bool(done[0]):
                        paused=True
                        with lock:state.update(status='failed',error='物理回合終止，請重播或選擇其他動作')
                    else:
                        frame={'time':float(env.episode_length_buf[0]*env.step_dt),
                            'q':env.robot.data.joint_pos[0,env.jids].cpu().tolist(),
                            'root':(env.robot.data.root_pos_w[0]-env.scene.env_origins[0]).cpu().tolist(),
                            'quaternion':env.robot.data.root_quat_w[0].cpu().tolist()}
                        with lock:state.update(frame=frame,duration=env.max_episode_length_s,joint_names=list(JOINTS))
            time.sleep(max(0,.02-(time.monotonic()-start)))
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush()
finally:
    if server:server.shutdown()
    if env:env.close()
    app.close()
