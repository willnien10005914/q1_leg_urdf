#!/usr/bin/env python3
"""Record only the formally confirmed native stage-2 upper-body-v6 policy.

The simulator-free preflight runs before importing AppLauncher. A failed or
incomplete recording stays in a unique candidate file and never replaces the
accepted movie. This tool performs no upload or policy promotion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from play_interactive_web import ROOT,policy_environment,selected_policy,sha256
from record_interactive_diagnostic import FootDiagnosticMetrics


RECORDER_SOURCES=(
    'scripts/reinforcement_learning/rsl_rl/record_upper_body_v6.py',
    'scripts/reinforcement_learning/rsl_rl/record_interactive_diagnostic.py',
    'scripts/reinforcement_learning/rsl_rl/play_interactive_web.py',
    'tools/v6_policy_gate.py',
)


def upper_summary(observations, plan):
    """Actual PhysX torso and hand metrics, independently recomputed for film."""
    def window(rows):
        if not rows:return None
        rms=lambda key:math.sqrt(sum(row[key]**2 for row in rows)/len(rows))
        return {'samples':len(rows),'chest_peak_deg':max(row['chest_tilt_deg'] for row in rows),
            'chest_rms_deg':rms('chest_tilt_deg'),
            'chest_angular_xy_rms_rad_s':rms('chest_angular_xy_rad_s'),
            'chest_angular_xyz_rms_rad_s':rms('chest_angular_xyz_rad_s'),
            'arm_reference_rmse_rad':math.sqrt(sum(row['arm_reference_mse_rad2'] for row in rows)/len(rows)),
            'arm_speed_rms_rad_s':math.sqrt(sum(row['arm_joint_speed_mse_rad2_s2'] for row in rows)/len(rows)),
            'hands_return_distance_rms_m_per_hand':[
                math.sqrt(sum(row['hands_return_distance_m'][side]**2 for row in rows)/len(rows)) for side in (0,1)]}
    valid=[row for row in observations if row.get('valid_first_episode') is True]
    sequence=next(item for item in plan if item['name']=='sidestep_sequence')
    # The controller spends 0.5 s preparing, then left/right, then 1 s recover.
    end=sequence['end']-1.
    result={'all':window(valid),
        'dynamic':window([row for row in valid if row['mode'] in ('walk','sidestep_left','sidestep_right')]),
        'early_return':window([row for row in valid if end+.8<=row['time']<end+1.5]),
        'tail':window([row for row in valid if plan[-1]['end']-.5<=row['time']<=plan[-1]['end']]),
        'directions':{}}
    for side in ('left','right'):
        result['directions'][side]=window([row for row in valid if row['mode']=='sidestep_'+side])
        result['directions'][side+'_return']=window([row for row in valid
            if row['mode']=='sidestep_'+side and row['gesture_phase']>=.85])
    return result


def validate_upper_recording(report):
    rows=report.get('upper_observations',[]);poses=report.get('poses',[])
    if len(rows)!=report.get('frames') or len(poses)!=report.get('frames'):return False
    for i,row in enumerate(rows):
        if (row.get('valid_first_episode') is not True or row.get('mode')!=report['foot_observations'][i]['mode']
                or row.get('gesture_phase')!=report['foot_observations'][i]['gesture_phase']
                or abs(row.get('time',-1)-(i+1)/50.)>1e-6):return False
        for key in ('chest_tilt_deg','chest_angular_xy_rad_s','chest_angular_xyz_rad_s',
                    'arm_reference_mse_rad2','arm_joint_speed_mse_rad2_s2'):
            if type(row.get(key)) not in (int,float) or not math.isfinite(row[key]) or row[key]<0:return False
        hands=row.get('hands_return_distance_m')
        if not isinstance(hands,list) or len(hands)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in hands):return False
        pose=poses[i]
        if abs(pose.get('time',-1)-(i+1)/50.)>1e-6:return False
        for key,n in (('q',23),('root',3),('quaternion',4)):
            values=pose.get(key)
            if not isinstance(values,list) or len(values)!=n or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):return False
    summary=upper_summary(rows,report['plan'])
    if summary!=report.get('upper_metrics'):return False
    gates=report.get('contract',{}).get('upper_body_gates',{})
    required=('chest_peak_max_deg','chest_peak_p95_max_deg','chest_rms_max_deg',
              'dynamic_chest_angular_xy_rms_max_rad_s','gesture_arm_reference_rmse_max_rad',
              'early_return_chest_peak_max_deg','early_return_arm_reference_rmse_max_rad',
              'early_return_hand_distance_rms_max_m','early_return_arm_speed_rms_max_rad_s',
              'tail_hand_distance_rms_max_m','tail_chest_angular_xy_rms_max_rad_s','tail_arm_speed_rms_max_rad_s')
    if any(type(gates.get(k)) not in (int,float) or not math.isfinite(gates[k]) or gates[k]<=0 for k in required):return False
    for name in ('all','dynamic','early_return','tail'):
        if not summary[name] or summary[name]['samples']<10:return False
    if summary['all']['chest_peak_deg']>gates['chest_peak_max_deg'] or summary['all']['chest_rms_deg']>gates['chest_rms_max_deg']:return False
    if summary['dynamic']['chest_angular_xy_rms_rad_s']>gates['dynamic_chest_angular_xy_rms_max_rad_s']:return False
    for side in ('left','right'):
        event=summary['directions'][side];returned=summary['directions'][side+'_return']
        if not event or not returned or returned['samples']<5:return False
        if (event['chest_peak_deg']>gates['chest_peak_p95_max_deg']
                or event['chest_rms_deg']>gates['chest_rms_max_deg']
                or event['chest_angular_xy_rms_rad_s']>gates['dynamic_chest_angular_xy_rms_max_rad_s']
                or event['arm_reference_rmse_rad']>gates['gesture_arm_reference_rmse_max_rad']
                or returned['arm_reference_rmse_rad']>gates['gesture_arm_reference_rmse_max_rad']):return False
    early=summary['early_return'];tail=summary['tail']
    if (early['chest_peak_deg']>gates['early_return_chest_peak_max_deg']
            or early['arm_reference_rmse_rad']>gates['early_return_arm_reference_rmse_max_rad']
            or early['arm_speed_rms_rad_s']>gates['early_return_arm_speed_rms_max_rad_s']
            or any(x>gates['early_return_hand_distance_rms_max_m'] for x in early['hands_return_distance_rms_m_per_hand'])
            or any(x>gates['tail_hand_distance_rms_max_m'] for x in tail['hands_return_distance_rms_m_per_hand'])
            or tail['chest_angular_xy_rms_rad_s']>gates['tail_chest_angular_xy_rms_max_rad_s']
            or tail['arm_speed_rms_rad_s']>gates['tail_arm_speed_rms_max_rad_s']):return False
    return report.get('root_forces_or_constraints') is False and report.get('rendered_pose_filtering') is False


def recording_plan(sidestep_duration=1.6):
    if not math.isfinite(sidestep_duration) or sidestep_duration<=0:
        raise ValueError('Sidestep duration must be finite and positive')
    end=4.+.5+2*sidestep_duration+1.
    return [
        {'name':'stand','start':0.,'end':1.,'command':None},
        {'name':'W_forward','start':1.,'end':3.,'command':[.22,0.,0.]},
        {'name':'stop','start':3.,'end':4.,'command':[0.,0.,0.]},
        {'name':'sidestep_sequence','start':4.,'end':end,'command':None},
        {'name':'final_stand','start':end,'end':end+1.5,'command':None},
    ]


def preflight_recording(root=ROOT,sidestep_duration=1.6):
    """No candidate folder, fallback version or stage-0/1 bypass is accepted."""
    root=Path(root);folder=root/'checkpoints/isaac/upper_body_v6'
    version,selected,gate=selected_policy(root,folder,version='v6')
    if version!='v6' or not gate.get('ready'):
        raise ValueError(gate.get('error','Confirmed native upper-body-v6 policy is unavailable'))
    if selected.resolve()!=folder.resolve() or Path(gate['checkpoint']).resolve()!=(folder/'latest.pt').resolve():
        raise ValueError('Accepted recording requires the published upper-body-v6/latest.pt')
    if gate.get('confirmation_episodes',0)<128 or gate.get('contract',{}).get('robot')!='q1_leg_upper_body_v6':
        raise ValueError('Accepted recording requires the full native 128-episode v6 confirmation')
    if not math.isfinite(sidestep_duration) or not math.isclose(sidestep_duration,gate['sidestep_duration_s'],abs_tol=1e-9):
        raise ValueError('Recording duration must match the confirmed native gait')
    return gate


def output_path(path,root=ROOT):
    target=Path(path).resolve()
    if target.suffix.lower()!='.mp4' or target.is_relative_to((Path(root)/'checkpoints').resolve()):
        raise ValueError('Recording output must be an MP4 outside the checkpoint directory')
    return target


def direction_summary(observations):
    """Recompute actual unloaded role metrics from the recorded native frames."""
    events={side:{'peak':[0.,0.],'hits':[0,0],'samples':[0,0]} for side in ('left','right')}
    for row in observations:
        mode=row.get('mode')
        if not row.get('valid_first_episode') or mode not in ('sidestep_left','sidestep_right'):continue
        event=events['left' if mode=='sidestep_left' else 'right']
        for foot in (0,1):
            height=row['clearance_m'][foot];loaded=row['loaded'][foot]
            if not loaded:event['peak'][foot]=max(event['peak'][foot],height)
            if not row['reference_contact'][foot] and row['reference_height_m'][foot]>.022:
                event['samples'][foot]+=1
                event['hits'][foot]+=int(not loaded and height>=.025)
    result={}
    for side,event in events.items():
        item={};lead=0 if side=='left' else 1
        for role,foot in (('lead',lead),('trailing',1-lead)):
            count=event['samples'][foot]
            item.update({role+'_peak_clearance_m':event['peak'][foot],
                role+'_high_swing_samples':count,
                role+'_lift_success_rate':event['hits'][foot]/count if count else None})
        result[side]=item
    return result


def validate_recording(report):
    """Single-film physical checks supplement, never substitute for, the gate."""
    if not (report.get('accepted') is True and report.get('gate_passed') is True
            and report.get('robot')=='q1_leg_upper_body_v6' and report.get('controller_version')=='q1_leg_v6'
            and report.get('backend')=='isaaclab' and report.get('proxy') is False
            and report.get('confirmation_episodes',0)>=128 and report.get('confirmation_stage')==2
            and report.get('gate_unchanged_after_recording') is True
            and report.get('completed_plan') is True and report.get('failure') is False
            and report.get('technical_error') is None and report.get('error') is None
            and report.get('reset_count_after_start')==0 and report.get('auto_resets_suppressed')==0
            and report.get('unexpected_episode_reset') is False
            and report.get('final_mode')=='stand' and report.get('final_sequence_active') is False
            and report.get('fps')==50 and report.get('resolution')==[1280,720]
            and type(report.get('frames')) is int and report['frames']>0
            and report['frames']==report.get('expected_frames')
            and len(report.get('foot_observations',[]))==report['frames']
            and report.get('plan')==recording_plan(report.get('sidestep_duration_s',1.6))):
        return False
    duration=report.get('actual_duration_s');expected=report.get('expected_duration_s')
    speed=report.get('final_stand_speed_m_s');forward=report.get('walk_forward_displacement_m')
    if not all(type(x) in (int,float) and math.isfinite(x) for x in (duration,expected,speed,forward)):
        return False
    planned_duration=report['plan'][-1]['end']
    if report['expected_frames']!=round(planned_duration*50.) or abs(expected-planned_duration)>1e-8:
        return False
    if abs(duration-report['frames']/50.)>1e-8 or abs(duration-expected)>.001 or speed>=.07 or forward<=.05:
        return False
    modes=[item['mode'] for item in report.get('mode_changes',[])]
    if modes.count('sidestep_left')!=1 or modes.count('sidestep_right')!=1:return False
    if report.get('foot_metrics',{}).get('valid_first_episode_frames')!=report['frames']:return False
    for index,item in enumerate(report['foot_observations']):
        if (item.get('valid_first_episode') is not True or item.get('reset_count_after_start')!=0
                or item.get('auto_resets_suppressed')!=0):return False
        if item.get('mode') not in ('stand','walk','recover','sidestep_left','sidestep_right'):return False
        phase=item.get('gesture_phase')
        if type(phase) not in (int,float) or not math.isfinite(phase) or not 0<=phase<=1.:return False
        stamp=item.get('time')
        if type(stamp) not in (int,float) or not math.isfinite(stamp) or abs(stamp-(index+1)/50.)>1e-6:return False
        for name in ('clearance_m','reference_height_m'):
            values=item.get(name)
            if not isinstance(values,list) or len(values)!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in values):return False
        for name in ('loaded','reference_contact'):
            values=item.get(name)
            if not isinstance(values,list) or len(values)!=2 or not all(type(x) is bool for x in values):return False
        positions=item.get('foot_link_positions_world_m')
        if not isinstance(positions,list) or len(positions)!=2:return False
        if any(not isinstance(point,list) or len(point)!=3 or not all(type(x) in (int,float) and math.isfinite(x) for x in point) for point in positions):return False
    if report.get('directions')!=direction_summary(report['foot_observations']):return False
    for side in ('left','right'):
        item=report.get('directions',{}).get(side,{})
        for role in ('lead','trailing'):
            peak=item.get(role+'_peak_clearance_m');rate=item.get(role+'_lift_success_rate')
            if not (type(peak) in (int,float) and math.isfinite(peak) and peak>=.025
                    and type(rate) in (int,float) and math.isfinite(rate) and .4<=rate<=1.
                    and item.get(role+'_high_swing_samples',0)>0):return False
    return validate_upper_recording(report)


def verify_artifact(video,root=ROOT,run_id=None,check_frames=True):
    """CPU verification rejects stale outputs even if Kit exits with code zero."""
    video=Path(video);root=Path(root)
    report=json.loads(video.with_suffix('.json').read_text())
    if not validate_recording(report):raise ValueError('Incomplete or failed v6 recording; inspect its candidate metadata')
    if run_id is not None and report.get('recording_run_id')!=run_id:
        raise ValueError('Accepted output is stale; this recording attempt did not complete')
    if not video.is_file() or video.stat().st_size<1024 or sha256(video)!=report.get('video_sha256'):
        raise ValueError('Accepted recording bytes do not match the metadata')
    gate=preflight_recording(root,report['sidestep_duration_s'])
    for key in ('checkpoint_sha256','contract','source_sha256','confirmation_episodes'):
        if report.get(key)!=gate.get(key):raise ValueError('Recording no longer matches the confirmed native policy/source')
    for relative in RECORDER_SOURCES:
        if report.get('recorder_sources',{}).get(relative)!=sha256(root/relative):
            raise ValueError('Recording sources changed after capture')
    if check_frames:
        import numpy as np
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames','-of','json',str(video)],check=True,capture_output=True,text=True)
        stream=json.loads(probe.stdout)['streams'][0]
        if (stream.get('width'),stream.get('height'),stream.get('avg_frame_rate'),int(stream.get('nb_read_frames',0)))!=(1280,720,'50/1',report['expected_frames']):
            raise ValueError('Actual video dimensions, FPS or decoded frame count do not match')
        def frame(t):
            raw=subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',str(video),'-frames:v','1',
                '-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True).stdout
            return np.frombuffer(raw,dtype=np.uint8).reshape(720,1280,3).astype(float)
        first=frame(.5)
        # Ignore the moving text strip: a changing timestamp alone is not proof
        # that the native viewport updated or that the robot actually moved.
        change=max(float(np.mean(np.max(abs(first[110:]-frame(t)[110:]),axis=2)>20)) for t in (2.,5.2,6.9))
        if change<.001:raise ValueError('Native viewport appears frozen beneath its metadata overlay')
    return report


def _json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [_json_safe(v) for v in value]
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=ROOT/'docs/results/q1_leg_upper_body_v6.mp4')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sidestep-duration',type=float,default=1.6)
    parser.add_argument('--validate-only',action='store_true',help='Run the full gate without importing Isaac or torch')
    known,_=parser.parse_known_args(argv)
    try:
        gate=preflight_recording(ROOT,known.sidestep_duration)
        target=output_path(known.video,ROOT)
    except (ValueError,OSError,KeyError) as error:
        if known.validate_only:
            print(json.dumps({'ready':False,'error':str(error)},ensure_ascii=False));return 1
        parser.exit(2,str(error)+'\n')
    if known.validate_only:
        print(json.dumps({'ready':True,'checkpoint':gate['checkpoint'],'confirmation_episodes':gate['confirmation_episodes'],
                          'plan':recording_plan(known.sidestep_duration)},indent=2));return 0

    # This import and all GPU construction occur strictly after the full gate.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args(argv);args.enable_cameras=True
    target=output_path(args.video,ROOT);target.parent.mkdir(parents=True,exist_ok=True)
    run_id=os.environ.get('Q1_UPPER_RECORDING_RUN_ID') or uuid.uuid4().hex
    candidate=target.with_name(target.stem+'.candidate-'+uuid.uuid4().hex[:12]+'.mp4')
    plan=recording_plan(args.sidestep_duration)
    report={'accepted':False,'gate_passed':True,'robot':'q1_leg_upper_body_v6','controller_version':'q1_leg_v6',
        'source':'native Isaac Sim PhysX PPO','backend':'isaaclab','proxy':False,
        'recording_run_id':run_id,'checkpoint':gate['checkpoint'],'checkpoint_sha256':gate['checkpoint_sha256'],
        'contract':gate['contract'],'source_sha256':gate['source_sha256'],
        'recorder_sources':{name:sha256(ROOT/name) for name in RECORDER_SOURCES},
        'confirmation_episodes':gate['confirmation_episodes'],'confirmation_stage':2,'seed':args.seed,
        'sidestep_duration_s':args.sidestep_duration,'plan':plan,'fps':50,'resolution':[1280,720],
        'frames':0,'expected_frames':round(plan[-1]['end']/.02),'actual_duration_s':0.,'expected_duration_s':plan[-1]['end'],
        'completed_plan':False,'failure':False,'technical_error':None,'error':None,
        'gate_unchanged_after_recording':False,'reset_count_after_start':0,'auto_resets_suppressed':0,
        'unexpected_episode_reset':False,'mode_changes':[],'root_positions':[],'foot_observations':[],
        'upper_observations':[],'poses':[],'upper_metrics':{},
        'root_forces_or_constraints':False,'rendered_pose_filtering':False,
        'final_mode':None,'final_sequence_active':False,'walk_forward_displacement_m':0.,'final_stand_speed_m_s':None,
        'scope':'Confirmed native CAD simulation with one recorded episode; real hardware unvalidated'}
    metrics=FootDiagnosticMetrics()
    app=env=writer=None
    try:
        launcher=AppLauncher(args);app=launcher.app
        import imageio.v2 as imageio
        import numpy as np
        import torch
        from PIL import Image,ImageDraw,ImageFont
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from leg_humanoid_lab.interactive_control import KeyboardController
        from runtime_compat import load_checkpoint,runner_cfg
        NativeEnv,cfg=policy_environment(ROOT,'v6')
        class RecordingEnv(NativeEnv):
            _manual_reset_allowed=True
            suppressed_resets=0
            def _reset_idx(self,env_ids):
                if self._manual_reset_allowed:super()._reset_idx(env_ids)
                else:self.suppressed_resets+=len(env_ids) if env_ids is not None else self.num_envs
        cfg.scene.num_envs=1;cfg.training_mode=False;cfg.curriculum_stage=2;cfg.episode_length_s=30.
        cfg.sim.device=args.device;cfg.seed=args.seed
        cfg.viewer.eye=(2.35,-2.55,1.5);cfg.viewer.lookat=(.05,0.,.7);cfg.viewer.resolution=(1280,720)
        env=RecordingEnv(cfg,render_mode='rgb_array');env.set_manual_control('stand',[0.,0.,0.],0.)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=1.);env._manual_reset_allowed=False
        runner=OnPolicyRunner(wrapped,runner_cfg(PPOCfg()),log_dir=None,device=args.device)
        load_checkpoint(runner,gate['checkpoint'],load_optimizer=False,load_iteration=False)
        policy=runner.get_inference_policy(device=env.device)
        if not math.isclose(env.step_dt,.02,abs_tol=1e-7):raise RuntimeError('Native capture requires 50-Hz control')
        if preflight_recording(ROOT,args.sidestep_duration)!=gate:raise RuntimeError('Confirmed policy/source changed during initialization')
        writer=imageio.get_writer(str(candidate),fps=50,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        except OSError:font=ImageFont.load_default()
        for _ in range(12):env.sim.render();env.render(recompute=True)
        controller=KeyboardController(sidestep_duration=args.sidestep_duration)
        command_clock=time.monotonic();last_segment=last_mode=control=None
        previous_length=int(env.episode_length_buf[0]);walk_start=walk_end=walk_forward=None;final_speeds=[]
        with torch.inference_mode():
            for i in range(report['expected_frames']):
                if not app.is_running():raise RuntimeError('Isaac Sim closed before the plan completed')
                t=i*env.step_dt;now=command_clock+t
                segment=next(item for item in plan if item['start']<=t<item['end'])
                if segment['name']!=last_segment:
                    if segment['name']=='stop':controller.stop(now)
                    elif segment['name']=='sidestep_sequence':
                        if controller.sequence(now) is not True:raise RuntimeError('Finite sequence command was rejected')
                    elif segment['name']=='W_forward':
                        d=env.robot.data;walk_start=(d.root_pos_w[0]-env.scene.env_origins[0]).cpu().tolist()
                        w,x,y,z=d.root_quat_w[0].cpu().tolist();yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                        walk_forward=(math.cos(yaw),math.sin(yaw))
                    last_segment=segment['name']
                if segment['name']=='W_forward' and controller.velocity(*segment['command'],now) is not True:
                    raise RuntimeError('W command was rejected')
                control=controller.tick(env.step_dt,now);env.set_manual_control(control['mode'],control['command'],control['phase'])
                _,_,done,_=wrapped.step(policy(wrapped.get_observations()))
                d=env.robot.data;root=(d.root_pos_w[0]-env.scene.env_origins[0]).cpu().tolist();quat=d.root_quat_w[0].cpu().tolist()
                velocity=d.root_lin_vel_b[0,:2].cpu().tolist();length=int(env.episode_length_buf[0])
                reset=previous_length>0 and length<=previous_length;previous_length=length
                report['unexpected_episode_reset']|=reset
                report['reset_count_after_start']+=int(reset)
                clearance,_,loaded,_=env.foot_step_metrics();clearance=clearance[0].cpu().tolist();loaded=loaded[0].cpu().tolist()
                ref_height=env.reference_height[0].cpu().tolist();contact=env.reference_contact[0].cpu().tolist()
                positions=d.body_link_pos_w[0,env.foot_body_ids].cpu().tolist()
                finite=all(math.isfinite(v) for v in root+quat+velocity+clearance+ref_height+[v for point in positions for v in point])
                failed=bool(done[0]) or reset or not finite or env.suppressed_resets>0
                report['failure']|=failed
                report['root_positions'].append(_json_safe({'time':(i+1)*env.step_dt,'root':root,'quaternion':quat}))
                report['foot_observations'].append(_json_safe({'time':(i+1)*env.step_dt,'clearance_m':clearance,'loaded':loaded,
                    'reference_height_m':ref_height,'reference_contact':contact,'foot_link_positions_world_m':positions,
                    'mode':control['mode'],'gesture_phase':control['phase'],
                    'valid_first_episode':not failed,'reset_count_after_start':report['reset_count_after_start'],
                    'auto_resets_suppressed':int(env.suppressed_resets)}))
                if not failed:
                    metrics.update(clearance,loaded,ref_height,contact,positions)
                upper=env.upper_body_measurements()
                observation={'time':(i+1)*env.step_dt,'mode':control['mode'],'gesture_phase':control['phase'],
                    'valid_first_episode':not failed,
                    'chest_tilt_deg':math.degrees(float(upper['chest_tilt_rad'][0])),
                    'chest_angular_xy_rad_s':float(upper['chest_angular_xy_rad_s'][0]),
                    'chest_angular_xyz_rad_s':float(upper['chest_angular_xyz_rad_s'][0]),
                    'arm_reference_mse_rad2':float(upper['arm_reference_mse_rad2'][0]),
                    'arm_joint_speed_mse_rad2_s2':float(upper['arm_joint_speed_mse_rad2_s2'][0]),
                    'hands_return_distance_m':upper['hands_return_distance_m'][0].cpu().tolist()}
                upper_finite=all(math.isfinite(v) for key,v in observation.items() if isinstance(v,float)) and all(
                    math.isfinite(v) for v in observation['hands_return_distance_m'])
                if not upper_finite:
                    failed=True;report['failure']=True;observation['valid_first_episode']=False
                    report['foot_observations'][-1]['valid_first_episode']=False
                report['upper_observations'].append(_json_safe(observation))
                report['poses'].append(_json_safe({'time':(i+1)*env.step_dt,
                    'q':d.joint_pos[0,env.jids].cpu().tolist(),'root':root,'quaternion':quat}))
                if control['mode']!=last_mode:
                    report['mode_changes'].append({'time':t,'mode':control['mode']});last_mode=control['mode']
                if segment['name']=='W_forward' and finite:walk_end=root
                if segment['name']=='final_stand' and t>=segment['end']-.5 and finite:final_speeds.append(math.hypot(*velocity))
                env.sim.render();pixels=np.asarray(env.render(recompute=True))
                if pixels.shape!=(720,1280,3):raise RuntimeError(f'Unexpected viewport dimensions {pixels.shape}')
                picture=Image.fromarray(pixels);draw=ImageDraw.Draw(picture);draw.rectangle((0,0,1280,96),fill=(18,18,18))
                draw.text((16,6),'V6 NATIVE PPO | full torso/arms/feet stage 2 confirmed on 128 episodes',fill=(160,245,180),font=font)
                draw.text((16,36),f"seed {args.seed} | {t+env.step_dt:.2f}s | {control['mode']}"+(' | FAILED - NO RESET' if failed else ''),fill=(245,245,245),font=font)
                measured=' | '.join(f"{side} sole {100*clearance[j]:.1f} cm ({'LOADED' if loaded[j] else 'AIR'})" for j,side in enumerate(('L','R')))
                draw.text((16,65),measured,fill=(255,195,40),font=font)
                writer.append_data(np.asarray(picture));report['frames']+=1
                if failed:report['error']='Native episode failed; stopped without resetting';break
        report['completed_plan']=report['frames']==report['expected_frames'] and not report['failure']
        report['final_mode']=control['mode'] if control else None;report['final_sequence_active']=bool(control and control['sequence_active'])
        report['final_stand_speed_m_s']=float(np.mean(final_speeds)) if final_speeds else None
        if walk_start is not None and walk_end is not None:
            report['walk_forward_displacement_m']=sum((walk_end[i]-walk_start[i])*walk_forward[i] for i in (0,1))
        report['gate_unchanged_after_recording']=preflight_recording(ROOT,args.sidestep_duration)==gate
    except BaseException as exception:
        report['technical_error']=str(exception)
    finally:
        if writer is not None:
            try:writer.close()
            except BaseException as exception:report['technical_error']=str(exception)
        report['actual_duration_s']=report['frames']/50.;report['foot_metrics']=metrics.summary()
        report['auto_resets_suppressed']=int(env.suppressed_resets) if env else 0
        report['directions']=direction_summary(report['foot_observations'])
        report['upper_metrics']=upper_summary(report['upper_observations'],report['plan'])
        report['accepted']=True
        report['accepted']=validate_recording(report)
        if not report['accepted'] and report['error'] is None and report['technical_error'] is None:
            report['error']='Single capture did not complete walking, both unloaded sidesteps and stable stopping'
        if candidate.exists():report['video_sha256']=sha256(candidate)
        metadata=candidate.with_suffix('.json');metadata.write_text(json.dumps(_json_safe(report),indent=2,allow_nan=False)+'\n')
        if report['accepted']:
            os.replace(candidate,target);os.replace(metadata,target.with_suffix('.json'))
        print('LIFT_VIDEO_RESULT '+json.dumps({key:report[key] for key in ('accepted','frames','actual_duration_s','failure','technical_error','error')})+' '+str(target if report['accepted'] else candidate),flush=True)
        if env is not None:env.close()
        if app is not None:app.close()
    return 0 if report['accepted'] else 1


if __name__=='__main__':raise SystemExit(main())
