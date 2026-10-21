"""Independent first-episode acceptance for lifting curriculum stages 0 and 1.

Passing this suite proves only native simulated lift and recovery at the tested
stage. It never substitutes for stage 2 keyboard and directional acceptance.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import random

import torch

from .robot import URDF


STAGE_CASES = {0: ('stand', 'inplace'), 1: ('stand', 'left', 'right', 'pair')}
GATES = {
    'survival_rate_min': .875, 'upright_min': .97,
    'stop_speed_m_s_max': .07, 'stop_observed_rate_min': .875,
    'stance_slip_rms_m_s_max': .10,
    'foot_clearance_m_min': .025, 'high_swing_target_height_m_min': .022,
    'unloaded_lift_success_rate_min': .4, 'unloaded_force_threshold_n': 5.,
}


def check_lift_gates(report: dict) -> dict[str, bool]:
    """Missing metrics, absent high-swing samples and nonfinite values fail."""
    stage=report.get('stage')
    checks={'stage.valid': type(stage) is int and stage in STAGE_CASES}
    if not checks['stage.valid']:
        return checks

    def bound(key, value, threshold, minimum=True):
        finite=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        checks[key]=bool(finite and (value>=threshold if minimum else value<threshold))

    def feet(prefix, item):
        for index in (0,1):
            for field, threshold in (('foot_peak_clearance_m_per_foot',.025),
                                     ('foot_lift_success_rate_per_foot',.4),
                                     ('high_swing_samples_per_foot',1)):
                values=item.get(field,[])
                value=values[index] if isinstance(values,(list,tuple)) and len(values)>index else None
                bound(f'{prefix}.foot_{index}.{field}',value,threshold)

    for name in STAGE_CASES[stage]:
        item=report.get('cases',{}).get(name,{})
        bound(f'{name}.sample_count',item.get('episodes'),1)
        bound(f'{name}.survival',item.get('survival_rate'),.875)
        bound(f'{name}.upright',item.get('upright'),.97)
        bound(f'{name}.stop_observed',item.get('stop_observed_rate'),.875)
        bound(f'{name}.stop',item.get('final_stand_speed_m_s'),.07,False)
        bound(f'{name}.stance_slip',item.get('stance_slip_m_s'),.10,False)
        if name!='stand':
            feet(name,item)
        if stage==1 and name!='stand':
            for side in (('left','right') if name=='pair' else (name,)):
                event=item.get('directions',{}).get(side,{})
                for role in ('lead','trailing'):
                    for suffix, threshold in (('peak_clearance_m',.025),('lift_success_rate',.4),('high_swing_samples',1)):
                        bound(f'{name}.{side}.{role}_{suffix}',event.get(f'{role}_{suffix}'),threshold)
    return checks


def _tensor(value):
    return value.detach().cpu().tolist()


@torch.inference_mode()
def evaluate_lift(runner, wrapped, env, episodes: int=32, seed: int=70000, stage: int=0) -> dict:
    """Run a balanced 20-second suite with exactly one initial explicit reset.

    Samples after any environment's first done/reset are permanently excluded,
    including the done frame because vector wrappers may already return a reset
    pose. Commands never reset or teleport an articulation.
    """
    if type(stage) is not int or stage not in STAGE_CASES:
        raise ValueError('Lift acceptance supports stages 0 and 1 only')
    names=STAGE_CASES[stage]
    if episodes<len(names) or episodes%len(names):
        raise ValueError('Episodes must be a positive multiple of the stage case count')
    if env.num_envs<episodes:
        raise ValueError('Insufficient native environments for requested episodes')
    if env.cfg.allow_proxy:
        raise ValueError('Native CAD is required for lift acceptance')
    duration=20.;dt=float(env.step_dt);steps=round(duration/dt)
    if env.max_episode_length*dt<duration-dt:
        raise ValueError('Lift acceptance requires a 20-second episode horizon')
    maneuver=float(env.library.get('sidestep_duration_s',1.6))
    if abs(maneuver-1.6)>1e-6:
        raise ValueError('Stage acceptance uses the reviewed 1.6-second maneuver')
    count,total,device=episodes,env.num_envs,env.device
    task=torch.arange(count,device=device)%len(names)
    mode=torch.zeros(total,device=device,dtype=torch.long)
    command=torch.zeros((total,3),device=device)
    phase=torch.zeros(total,device=device)
    active=torch.ones(count,device=device,dtype=torch.bool)
    survived=torch.zeros_like(active)
    sums=torch.zeros(count,device=device)
    elapsed=sums.clone();samples=sums.clone();up_sum=sums.clone();stop_sum=sums.clone();stop_n=sums.clone()
    slip_sq=sums.clone();slip_n=sums.clone()
    peak=torch.zeros((count,2),device=device);hits=peak.clone();trials=peak.clone()
    side_peak=torch.zeros((count,2,2),device=device);side_hits=side_peak.clone();side_trials=side_peak.clone()
    torch.manual_seed(seed);random.seed(seed)
    if hasattr(env,'seed'): env.seed(seed)
    old_training=env.cfg.training_mode
    had_manual=hasattr(env,'manual_mode');old_manual=getattr(env,'manual_mode',False)
    env.cfg.training_mode=False;env.manual_mode=True
    policy=runner.get_inference_policy(device=device)
    try:
        env.set_manual_control(mode,command,phase)
        wrapped.reset()
        previous_episode_length=(env.episode_length_buf[:count].clone()
                                 if hasattr(env,'episode_length_buf') else None)
        for step in range(steps):
            time=step*dt
            mode.zero_();phase.zero_();command.zero_()
            if stage==0:
                moving=(task==1)&(1.<=time<8.)
                mode[:count][moving]=1
                phase[:count][moving]=((time-1.)/float(env.library.get('walk_cycle_s',.9)))%1.
            else:
                for case_id,name in enumerate(names):
                    selected=task==case_id
                    if name=='stand': continue
                    pair=name=='pair';end=1.+maneuver*(2 if pair else 1)
                    if 1.<=time<end:
                        right=(name=='right') or (pair and time>=1.+maneuver)
                        mode[:count][selected]=3 if right else 2
                        start=1.+maneuver if pair and right else 1.
                        phase[:count][selected]=(time-start)/maneuver
                    elif end<=time<end+1.:
                        mode[:count][selected]=4
                        phase[:count][selected]=time-end
            env.set_manual_control(mode,command,phase)
            obs=wrapped.get_observations()
            actions=policy(obs)
            finite_action=torch.isfinite(actions[:count]).all(-1)
            _,_,done,_=wrapped.step(actions.nan_to_num())
            done=done.reshape(-1)[:count].bool()
            data=env.robot.data
            clearance,foot_speed,loaded,_=env.foot_step_metrics()
            clearance=clearance[:count];foot_speed=foot_speed[:count];loaded=loaded[:count].bool()
            reference_height=env.reference_height[:count]
            reference_contact=env.reference_contact[:count].bool()
            finite=(torch.isfinite(data.root_pos_w[:count]).all(-1)
                    &torch.isfinite(data.root_lin_vel_b[:count]).all(-1)
                    &torch.isfinite(data.projected_gravity_b[:count]).all(-1)
                    &torch.isfinite(clearance).all(-1)&torch.isfinite(foot_speed).all(-1)
                    &torch.isfinite(reference_height).all(-1)&finite_action)
            implicit_reset=torch.zeros_like(active)
            if previous_episode_length is not None:
                length=env.episode_length_buf[:count]
                implicit_reset=(previous_episode_length>0)&(length<=previous_episode_length)
                previous_episode_length=length.clone()
            before=active.clone()
            ended=before&(done|implicit_reset|~finite)
            elapsed[before]=(step+1)*dt
            failed=env.reset_terminated[:count].bool()
            horizon=(step+1)*dt>=duration-dt
            survived[ended]=done[ended]&~failed[ended]&finite[ended]&horizon
            active[ended]=False
            valid=active
            samples+=valid
            up_sum+=(-data.projected_gravity_b[:count,2]).nan_to_num()*valid
            if time>=18.:
                stop_sum+=data.root_lin_vel_b[:count,:2].norm(dim=-1).nan_to_num()*valid
                stop_n+=valid
            moving=valid&((mode[:count]>=1)&(mode[:count]<=3))
            clean_clearance=clearance.nan_to_num()
            peak=torch.maximum(peak,torch.where(moving[:,None],clean_clearance,0.))
            swing=(moving[:,None]&~reference_contact&(reference_height>.022))
            lifted=(clean_clearance>=.025)&~loaded
            trials+=swing;hits+=swing&lifted
            stance=valid[:,None]&reference_contact&loaded
            slip_sq+=(foot_speed.nan_to_num().square()*stance).sum(-1)
            slip_n+=stance.sum(-1)
            for side,side_mode in enumerate((2,3)):
                relevant=valid&(mode[:count]==side_mode)
                side_peak[:,side]=torch.maximum(side_peak[:,side],torch.where(relevant[:,None],clean_clearance,0.))
                side_swing=relevant[:,None]&~reference_contact&(reference_height>.022)
                side_trials[:,side]+=side_swing
                side_hits[:,side]+=side_swing&lifted
            if not bool(active.any()): break
        survived[active]=True
        upright=up_sum/samples.clamp(min=1)
        stop=torch.where(stop_n>0,stop_sum/stop_n.clamp(min=1),torch.full_like(stop_n,1e6))
        # An episode must cover the final two-second window, allowing one
        # excluded terminal frame. A death does not become zero velocity or a
        # fabricated 1e6 m/s in the case average: missing coverage has its own
        # explicit gate aligned with the survival threshold.
        required_stop_samples=max(1,round(2./dt)-1)
        stop_observed=stop_n>=required_stop_samples
        slip=torch.where(slip_n>0,(slip_sq/slip_n.clamp(min=1)).sqrt(),torch.full_like(slip_n,1e6))
        rate=hits/trials.clamp(min=1)
        cases={}
        per_episode=[]
        for case_id,name in enumerate(names):
            selected=task==case_id
            observed=selected&stop_observed
            item={'episodes':int(selected.sum()),'completed_episodes':int((selected&survived).sum()),
                  'survival_rate':float(survived[selected].float().mean()),'duration_s':float(elapsed[selected].mean()),
                  'upright':float(upright[selected].mean()),
                  'stop_observed_rate':float(stop_observed[selected].float().mean()),
                  'stop_observed_episodes':int(observed.sum()),
                  'final_stand_speed_m_s':float(stop[observed].mean()) if bool(observed.any()) else 1e6,
                  'stance_slip_m_s':float(slip[selected].mean()),
                  'foot_peak_clearance_m_per_foot':_tensor(peak[selected].mean(0)),
                  'foot_lift_success_rate_per_foot':_tensor(hits[selected].sum(0)/trials[selected].sum(0).clamp(min=1)),
                  'high_swing_samples_per_foot':_tensor(trials[selected].sum(0))}
            if stage==1 and name!='stand':
                item['directions']={}
                for side_name in (('left','right') if name=='pair' else (name,)):
                    side=0 if side_name=='left' else 1
                    event={}
                    for role,foot in (('lead',side),('trailing',1-side)):
                        event[f'{role}_peak_clearance_m']=float(side_peak[selected,side,foot].mean())
                        event[f'{role}_lift_success_rate']=float(side_hits[selected,side,foot].sum()/side_trials[selected,side,foot].sum().clamp(min=1))
                        event[f'{role}_high_swing_samples']=float(side_trials[selected,side,foot].sum())
                    item['directions'][side_name]=event
            cases[name]=item
        for index in range(count):
            row={'env_index':index,'case':names[int(task[index])],'survived':bool(survived[index]),
                 'duration_s':float(elapsed[index]),'upright':float(upright[index]),
                 'final_stand_speed_m_s':float(stop[index]),
                 'stop_observed':bool(stop_observed[index]),'stop_window_samples':float(stop_n[index]),
                 'stance_slip_m_s':float(slip[index]),
                 'foot_peak_clearance_m_per_foot':_tensor(peak[index]),
                 'foot_lift_success_rate_per_foot':_tensor(rate[index]),
                 'high_swing_samples_per_foot':_tensor(trials[index])}
            if stage==1:
                row['direction_peak_clearance_m_per_foot']=_tensor(side_peak[index])
                row['direction_lift_success_rate_per_foot']=_tensor(side_hits[index]/side_trials[index].clamp(min=1))
            per_episode.append(row)
        environment_file=Path(inspect.getfile(type(env))) if type(env).__module__!='types' else None
        gait_file=getattr(env,'gait_path',None)
        if gait_file is None:
            # New environments expose their loaded path; serialized contents
            # remain an unambiguous fallback for synthetic transport tests.
            gait_sha=hashlib.sha256(json.dumps(env.library,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            gait_sha_source='canonical_loaded_library_json'
        else:
            gait_sha=hashlib.sha256(Path(gait_file).read_bytes()).hexdigest();gait_sha_source=str(gait_file)
        report={'schema':1,'skill':'lift_foundation','backend':'isaaclab','proxy':False,
                'stage':stage,'episodes':count,'evaluation_seed':seed,'duration_s':duration,
                'contract':env.library.get('contract'),'source':'native_physx_ppo_first_episode_lift_suite',
                'scope':'Native CAD simulated lift foundation only; keyboard and full directional acceptance require stage 2.',
                'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),
                'gait_sha256':gait_sha,'gait_sha_source':gait_sha_source,
                'environment_sha256':hashlib.sha256(environment_file.read_bytes()).hexdigest() if environment_file and environment_file.is_file() else None,
                'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'measurement':{'clearance':'minimum native source-CAD sole world height minus environment origin',
                               'unloaded':'vertical foot force <=5 N, from native foot_step_metrics',
                               'high_swing':'reference_height >.022 m and reference_contact is false',
                               'slip':'RMS horizontal speed of loaded feet during planned stance',
                               'stop':'Mean speed over observed 18–20 s windows only; each counted episode must cover the window except at most one excluded terminal frame; missing coverage fails a separate rate gate'},
                'explicit_initial_resets':1,'reset_between_transitions':False,
                'exclude_done_and_all_post_reset_samples':True,
                'timeline':{'stand':'stand 0–20 s','inplace':'stand 0–1; walk with zero command 1–8; stand 8–20',
                            'left_right':'stand 0–1; maneuver 1–2.6; recover 2.6–3.6; stand 3.6–20',
                            'pair':'stand 0–1; left 1–2.6; right 2.6–4.2; recover 4.2–5.2; stand 5.2–20'},
                'gates':GATES.copy(),'cases':cases,'per_episode':per_episode,
                'survival_rate':float(survived.float().mean())}
        checks=check_lift_gates(report)
        report.update(checks=checks,failed_gates=[key for key,value in checks.items() if not value],
                      passed=all(checks.values()),score=sum(checks.values())/len(checks))
        return report
    finally:
        mode.zero_();command.zero_();phase.zero_()
        env.set_manual_control(mode,command,phase)
        env.cfg.training_mode=old_training
        if had_manual: env.manual_mode=old_manual
        else: delattr(env,'manual_mode')
