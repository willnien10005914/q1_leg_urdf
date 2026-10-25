#!/usr/bin/env python3
"""Independent V7 worker: native balance training or reset-free four-step audit."""
import argparse,json,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=100);p.add_argument('--completed',type=int,default=0);p.add_argument('--envs',type=int,default=10240);p.add_argument('--seed',type=int,default=42);p.add_argument('--evaluate',action='store_true');p.add_argument('--initialize',action='store_true');p.add_argument('--full-gates',action='store_true');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from leg_humanoid_lab.balance_env import BalanceLegEnv,UpperBodyEnvCfg,upper_body_contract
from leg_humanoid_lab.agent_cfg import PPOCfg
from runtime_compat import runner_cfg,load_checkpoint,checkpoint_for_runtime,policy_module,reset_noise_std

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def audit(runner,wrapped,env):
    wrapped.reset();active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    cases=torch.arange(env.num_envs,device=env.device)%4
    modes=torch.zeros_like(cases);phases=torch.zeros(env.num_envs,device=env.device)
    first_fail=[];modes_seen=set();capture_max=torch.zeros(env.num_envs,device=env.device)
    peak=torch.zeros((env.num_envs,2),device=env.device);chest_peak=phases.clone();completed_steps=torch.zeros_like(cases)
    policy=runner.get_inference_policy(device=env.device);duration=float(env.library['sidestep_duration_s']);pair=.5+2*duration+1.
    with torch.inference_mode():
      for step in range(1000):
        t=step*env.step_dt;modes.zero_();phases.zero_()
        for start in (1.,1.+pair):
          local=t-start-.5
          if 0<=local<2*duration:
            which=0 if local<duration else 1;mask=cases<2
            right=(cases==1) if which==0 else (cases==0)
            modes[mask]=torch.where(right[mask],3,2);phases[mask]=(local%duration)/duration
          elif 2*duration<=local<2*duration+1.:modes[cases<2]=4
        # Separate command-walk/in-place cases test that balance cannot be
        # achieved simply by freezing; active task gates remain independent.
        walk=(cases==2)&(t>=1)&(t<7)
        modes[walk]=1
        command=torch.zeros((env.num_envs,3),device=env.device);command[walk,0]=.15
        env.set_manual_control(modes,command,phases)
        prior=active.clone();_,_,done,_=wrapped.step(policy(wrapped.get_observations()))
        failed=prior&env.reset_terminated.bool()
        for idx in torch.where(failed)[0].cpu().tolist():
          event=env.last_termination.get(idx,{})
          first_fail.append({'env':idx,'case':int(cases[idx]),'time_s':(step+1)*env.step_dt,**event})
        active&=~done.bool();valid=active
        error,_,_=env.balance_measurements();capture_max=torch.maximum(capture_max,torch.where(valid,error,0))
        height,_,loaded,_=env.foot_step_metrics();peak=torch.maximum(peak,torch.where(valid[:,None]&~loaded,height,0))
        measured=env.upper_body_measurements();chest_peak=torch.maximum(chest_peak,torch.where(valid,measured['chest_tilt_rad'],0))
        for mode in modes[prior].unique().cpu().tolist():modes_seen.add(mode)
    alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    for event in first_fail:alive[event['env']]=False
    return {'diagnostic':True,'accepted':False,'episodes':env.num_envs,'duration_s':20,'seed':args.seed,
      'survival_rate':float(alive.float().mean()),'case_survival':{str(k):float(alive[cases==k].float().mean()) for k in range(4)},
      'case_names':['LRLR','RLRL','forward_walk_stop','stand'],'four_step_survival':float(alive[cases<2].float().mean()),
      'first_failure_events':first_fail,'modes_seen':sorted(modes_seen),'peak_unloaded_sole_m_mean_per_foot':peak.mean(0).cpu().tolist(),
      'chest_peak_deg_max':float(chest_peak.max()*180/torch.pi),'capture_excess_m_max':float(capture_max.max()),
      'no_post_reset_measurements':True,'checkpoint_sha256':sha(args.checkpoint)}

env=None
try:
 args.output.mkdir(parents=True,exist_ok=True)
 class MeasuredBalanceEnv(BalanceLegEnv):
    def _get_dones(self):
      done,timeout=super()._get_dones();self.last_termination={}
      if self.cfg.training_mode:return done,timeout
      for idx in torch.where(done)[0].cpu().tolist():
       d=self.robot.data;f=self.contacts.data.net_forces_w[idx]
       self.last_termination[idx]={'root_height_m':float(d.root_pos_w[idx,2]-self.scene.env_origins[idx,2]),'upright_cosine':float(-d.projected_gravity_b[idx,2]),'nonfoot_contacts':[{'body':self.contacts.body_names[j],'force_n':float(f[j].norm())} for j in self.other_bodies if float(f[j].norm())>25]}
      return done,timeout
 cfg=UpperBodyEnvCfg();cfg.scene.num_envs=args.envs;cfg.training_mode=not args.evaluate;cfg.curriculum_stage=2;cfg.reward_revision=2;cfg.seed=args.seed;cfg.episode_length_s=20.5 if args.evaluate else 20.;cfg.sim.device=args.device
 env=MeasuredBalanceEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
 agent=PPOCfg();agent.seed=args.seed;agent.algorithm.learning_rate=5e-5;agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=.0005;agent.algorithm.num_mini_batches=8;agent.save_interval=100
 config=runner_cfg(agent)
 runner=OnPolicyRunner(wrapped,config,log_dir=None if args.evaluate else str(args.output),device=args.device)
 expected=upper_body_contract();saved=json.loads((args.checkpoint.parent/'contract.json').read_text())
 for key in ('urdf_sha256','gait_sha256','joint_names','action_size','observation_size'):
  if expected[key]!=saved.get(key):raise ValueError('Native interface mismatch: '+key)
 if args.initialize:
  if args.completed:raise ValueError('New balance experiment must start at zero, separately from spent V6 updates')
  payload=torch.load(checkpoint_for_runtime(args.checkpoint),map_location=env.device,weights_only=False)
  if 'actor_state_dict' in payload:source={k.removeprefix('mlp.'):v for k,v in payload['actor_state_dict'].items() if k.startswith('mlp.')}
  else:source={k.removeprefix('actor.'):v for k,v in payload['model_state_dict'].items() if k.startswith('actor.')}
  model=policy_module(runner);current=model.state_dict();count=0
  for key in current:
    if key.startswith(('mlp.','actor.')):
      incoming=source[key.split('.',1)[1]]
      if incoming.shape!=current[key].shape or not torch.isfinite(incoming).all():raise ValueError('Actor shape/finite mismatch')
      current[key]=incoming;count+=1
  if count!=6:raise ValueError('Incomplete actor MLP')
  model.load_state_dict(current);reset_noise_std(runner,.04)
 elif args.evaluate:
  load_checkpoint(runner,args.checkpoint,load_optimizer=False,load_iteration=False)
 else:
  if saved.get('robot')!='q1_leg_balance_v7':raise ValueError('Resume must be V7')
  load_checkpoint(runner,args.checkpoint,load_optimizer=True,load_iteration=True)
  if runner.current_learning_iteration!=args.completed:raise ValueError('Counter mismatch')
 if args.evaluate:
  if args.full_gates:
   from leg_humanoid_lab.upper_body_evaluation import evaluate_upper_body
   report=evaluate_upper_body(runner,wrapped,env,episodes=args.envs,seed=args.seed,stage=2)
   report.update(checkpoint_sha256=sha(args.checkpoint),balance_experiment=True,accepted=False)
  else:report=audit(runner,wrapped,env)
  write(args.output/'evaluation.json',report);print('BALANCE_EVALUATION '+json.dumps(report),flush=True)
 else:
  contract={**expected,'robot':'q1_leg_balance_v7','schema':7,'balance_scope':'Native COM capture proxy, contact costs and mixed four-step/command curriculum; not accepted','balance_source_sha256':{name:sha(Path(__file__).resolve().parents[3]/name) for name in ('source/leg_humanoid_lab/leg_humanoid_lab/balance_env.py','source/leg_humanoid_lab/leg_humanoid_lab/balance_objective.py')}}
  write(args.output/'contract.json',contract)
  runner.current_learning_iteration=args.completed
  runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
  completed=args.completed+args.iterations;runner.current_learning_iteration=completed
  runner.save(str(args.output/'resume.tmp.pt'),infos={'completed_iterations':completed,'prior_v6_updates':20000,'experiment':'balance_v7','accepted':False})
  (args.output/'resume.tmp.pt').replace(args.output/'resume.pt')
  write(args.output/'training_result.json',{'completed_iterations':completed,'prior_v6_updates':20000,'checkpoint_sha256':sha(args.output/'resume.pt'),'num_envs':args.envs,'accepted':False})
except BaseException as error:
 import traceback;traceback.print_exc();args.output.mkdir(parents=True,exist_ok=True);write(args.output/'worker_error.json',{'error':str(error)});raise
finally:
 if env:env.close()
 app.close()
