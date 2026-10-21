#!/usr/bin/env python3
"""Record only the formally confirmed native stage-2 lift-v5 policy.

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
    'scripts/reinforcement_learning/rsl_rl/record_lift_v5.py',
    'scripts/reinforcement_learning/rsl_rl/record_interactive_diagnostic.py',
    'scripts/reinforcement_learning/rsl_rl/play_interactive_web.py',
)


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
    root=Path(root);folder=root/'checkpoints/isaac/lift_v5'
    version,selected,gate=selected_policy(root,folder,version='v5')
    if version!='v5' or not gate.get('ready'):
        raise ValueError(gate.get('error','Confirmed native lift-v5 policy is unavailable'))
    if selected.resolve()!=folder.resolve() or Path(gate['checkpoint']).resolve()!=(folder/'latest.pt').resolve():
        raise ValueError('Accepted recording requires the published lift-v5/latest.pt')
    if gate.get('confirmation_episodes',0)<128 or gate.get('contract',{}).get('robot')!='q1_leg_lift_v5':
        raise ValueError('Accepted recording requires the full native 128-episode v5 confirmation')
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
            and report.get('robot')=='q1_leg_lift_v5' and report.get('controller_version')=='q1_leg_v5'
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
    return True


def verify_artifact(video,root=ROOT,run_id=None,check_frames=True):
    """CPU verification rejects stale outputs even if Kit exits with code zero."""
    video=Path(video);root=Path(root)
    report=json.loads(video.with_suffix('.json').read_text())
    if not validate_recording(report):raise ValueError('Incomplete or failed v5 recording; inspect its candidate metadata')
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
    parser.add_argument('--video',type=Path,default=ROOT/'docs/results/q1_leg_lift_v5.mp4')
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
    run_id=os.environ.get('Q1_LIFT_RECORDING_RUN_ID') or uuid.uuid4().hex
    candidate=target.with_name(target.stem+'.candidate-'+uuid.uuid4().hex[:12]+'.mp4')
    plan=recording_plan(args.sidestep_duration)
    report={'accepted':False,'gate_passed':True,'robot':'q1_leg_lift_v5','controller_version':'q1_leg_v5',
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
        NativeEnv,cfg=policy_environment(ROOT,'v5')
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
                if control['mode']!=last_mode:
                    report['mode_changes'].append({'time':t,'mode':control['mode']});last_mode=control['mode']
                if segment['name']=='W_forward' and finite:walk_end=root
                if segment['name']=='final_stand' and t>=segment['end']-.5 and finite:final_speeds.append(math.hypot(*velocity))
                env.sim.render();pixels=np.asarray(env.render(recompute=True))
                if pixels.shape!=(720,1280,3):raise RuntimeError(f'Unexpected viewport dimensions {pixels.shape}')
                picture=Image.fromarray(pixels);draw=ImageDraw.Draw(picture);draw.rectangle((0,0,1280,96),fill=(18,18,18))
                draw.text((16,6),'NATIVE PPO | stage 2 independently confirmed on 128+ episodes',fill=(160,245,180),font=font)
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
