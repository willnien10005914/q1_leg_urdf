#!/usr/bin/env python3
"""Record an UNACCEPTED native PPO checkpoint, including its actual failure.

This diagnostic never promotes a policy, writes successful-media names, changes
the acceptance gate, teleports the robot, or restarts a failed episode.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import time

from play_interactive_web import ROOT, sha256


def diagnostic_plan(sidestep_duration=1.6,sequence_only=False,inplace_only=False):
    if sequence_only and inplace_only:
        raise ValueError('In-place and finite-sequence diagnostic modes are mutually exclusive')
    if inplace_only:
        return [
            {'name':'stand','start':0.,'end':1.,'command':None},
            {'name':'inplace_lift','start':1.,'end':5.5,'command':[0.,0.,0.]},
            {'name':'recover','start':5.5,'end':6.5,'command':[0.,0.,0.]},
            {'name':'final_stand','start':6.5,'end':8.,'command':None},
        ]
    if sequence_only:
        end=1.+.5+2*sidestep_duration+1.
        return [
            {"name":"stand","start":0.,"end":1.,"command":None},
            {"name":"sidestep_sequence","start":1.,"end":end,"command":None},
            {"name":"final_stand","start":end,"end":end+1.5,"command":None},
        ]
    sequence_end = 3.5 + .5 + 2 * sidestep_duration + 1.
    return [
        {"name": "stand", "start": 0., "end": 1., "command": None},
        {"name": "W_forward", "start": 1., "end": 2.5, "command": [.22, 0., 0.]},
        {"name": "stop", "start": 2.5, "end": 3.5, "command": [0., 0., 0.]},
        {"name": "sidestep_sequence", "start": 3.5, "end": sequence_end, "command": None},
        {"name": "final_stand", "start": sequence_end, "end": sequence_end + 1.5, "command": None},
    ]


def diagnostic_contract(saved,root=ROOT):
    """Check identity before Kit starts, without importing simulator modules."""
    root=Path(root)
    import sys
    sys.path.insert(0,str(root))
    from tools.run_upper_body_training import source_manifest
    state=json.loads((root/'outputs/upper_body_training_status.json').read_text())
    if state.get('completed_iterations')!=20000 or state.get('state') not in ('budget_exhausted','complete'):
        raise ValueError('This diagnostic requires the terminal 20000-update checkpoint')
    if source_manifest(Path(state['profile_path']))!=state['source_sha256']:
        raise ValueError('Frozen training source changed')
    expected=json.loads((Path(state['run'])/'contract.json').read_text())
    if saved!=expected or saved.get('robot')!='q1_leg_upper_body_v6':
        raise ValueError('Terminal V6 contract mismatch')
    return expected


class FootDiagnosticMetrics:
    """Accumulate actual native foot measurements; reference motion earns nothing."""
    def __init__(self):
        self.frames=0
        self.unloaded_peak=[0.,0.];self.unloaded_samples=[0,0]
        self.planned_samples=[0,0];self.planned_success=[0,0]
        self.first=None;self.last=None;self.peak_displacement=[0.,0.];self.path_length=[0.,0.]

    def update(self,clearance,loaded,reference_height,reference_contact,world_positions):
        if not all(math.isfinite(float(value)) for value in list(clearance)+list(reference_height)
                   +[v for position in world_positions for v in position]):
            raise ValueError('Native foot diagnostic contains nonfinite measurements')
        positions=[[float(v) for v in point] for point in world_positions]
        if self.first is None:self.first=[point[:] for point in positions]
        for foot in (0,1):
            if not loaded[foot]:
                self.unloaded_samples[foot]+=1
                self.unloaded_peak[foot]=max(self.unloaded_peak[foot],float(clearance[foot]))
            if not reference_contact[foot] and reference_height[foot]>.022:
                self.planned_samples[foot]+=1
                self.planned_success[foot]+=int(not loaded[foot] and clearance[foot]>=.025)
            displacement=math.dist(positions[foot],self.first[foot])
            self.peak_displacement[foot]=max(self.peak_displacement[foot],displacement)
            if self.last is not None:self.path_length[foot]+=math.dist(positions[foot],self.last[foot])
        self.last=positions;self.frames+=1

    def summary(self):
        displacement=[[self.last[foot][axis]-self.first[foot][axis] for axis in range(3)] for foot in (0,1)] if self.first is not None else None
        return {'foot_order':['left','right'],'valid_first_episode_frames':self.frames,
                'unloaded_peak_clearance_m_per_foot':self.unloaded_peak,
                'unloaded_samples_per_foot':self.unloaded_samples,
                'planned_high_swing_samples_per_foot':self.planned_samples,
                'planned_high_swing_success_rate_per_foot':[success/count if count else None for success,count in zip(self.planned_success,self.planned_samples)],
                'foot_world_peak_displacement_m_per_foot':self.peak_displacement,
                'foot_world_path_length_m_per_foot':self.path_length,
                'foot_world_displacement_xyz_m_per_foot':displacement,
                'measurement':'minimum native CAD sole height; physical loaded contact; native foot link origin world displacement',
                'planned_high_swing':'reference_contact=false and reference_height>0.022m; success requires clearance>=0.025m and unloaded contact',
                'scope':'Single diagnostic episode only; does not replace balanced acceptance gates'}


def validate_metadata(report):
    """A valid diagnostic may show a fall; it must never claim acceptance."""
    return bool(
        report.get("diagnostic") is True
        and report.get("gate_passed") is False
        and report.get("accepted") is False
        and report.get("proxy") is False
        and report.get("source") == "native Isaac Sim PhysX PPO"
        and report.get("overlay") == "EXPERIMENTAL / NOT ACCEPTED"
        and report.get("reset_count_after_start") == 0
        and isinstance(report.get("seed"), int)
        and isinstance(report.get("checkpoint_sha256"), str)
        and len(report["checkpoint_sha256"]) == 64
        and isinstance(report.get("frames"), int) and report["frames"] > 0
        and math.isfinite(report.get("actual_duration_s", float("nan")))
        and abs(report["actual_duration_s"] - report["frames"] / 50.) < 1e-7
        and isinstance(report.get("failure"), bool)
        and (not report["failure"] or bool(report.get("failure_reason")))
        and report.get("technical_error") is None
        and validate_task_metadata(report)
    )


def validate_task_metadata(report):
    """Retain compatibility with already-published v4 diagnostic metadata."""
    if 'robot' not in report:
        return True
    robot=report.get('robot');task=report.get('task')
    if robot not in ('q1_leg_upper_body_v6',) or task not in ('inplace_curriculum','finite_sidestep_sequence','interactive_diagnostic'):
        return False
    if not isinstance(report.get('control_profile'),str) or not report['control_profile']:
        return False
    if task=='inplace_curriculum' and (robot!='q1_leg_upper_body_v6' or report.get('task_label')!='In-place curriculum'):
        return False
    metrics=report.get('foot_metrics',{})
    frames=metrics.get('valid_first_episode_frames',-1)
    if type(frames) is not int or not 0<=frames<=report.get('frames',-1):
        return False
    for field in ('unloaded_peak_clearance_m_per_foot','foot_world_peak_displacement_m_per_foot','foot_world_path_length_m_per_foot'):
        values=metrics.get(field)
        if not isinstance(values,list) or len(values)!=2 or not all(isinstance(value,(int,float)) and math.isfinite(value) and value>=0 for value in values):
            return False
    samples=metrics.get('planned_high_swing_samples_per_foot')
    rates=metrics.get('planned_high_swing_success_rate_per_foot')
    if not isinstance(samples,list) or not isinstance(rates,list) or len(samples)!=2 or len(rates)!=2:
        return False
    for count,rate in zip(samples,rates):
        if type(count) is not int or count<0 or (count==0 and rate is not None):
            return False
        if count and (not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<=rate<=1):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/interactive_diagnostic.mp4")
    parser.add_argument("--seed", type=int, default=42)
    task_options=parser.add_mutually_exclusive_group()
    task_options.add_argument("--sequence-only",action="store_true",help="Diagnose the finite sidestep without an earlier walking segment")
    task_options.add_argument("--inplace-only",action="store_true",help="Diagnose native PPO in-place lifting; this is a curriculum task, not keyboard locomotion")
    parser.add_argument("--sidestep-duration", type=float, default=1.6)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.sidestep_duration) or args.sidestep_duration <= 0:
        parser.error("--sidestep-duration must be finite and positive")
    checkpoint, target = args.checkpoint.resolve(), args.video.resolve()
    if not checkpoint.is_file():
        parser.error("--checkpoint must identify an existing native PPO checkpoint")
    if (target.name in ("q1_leg_interactive_v4.mp4","q1_leg_lift_v5.mp4","q1_leg_interactive_v5.mp4")
            or target.is_relative_to((ROOT / "checkpoints").resolve())):
        parser.error("Diagnostic media must use a separate output name outside checkpoints")
    if target.exists() or target.with_suffix(".json").exists():
        parser.error("Diagnostic output already exists; choose another filename to preserve it")
    contract_path = args.contract or checkpoint.parent / "contract.json"
    contract = json.loads(contract_path.read_text())
    expected=diagnostic_contract(contract)
    is_lift=True
    if args.inplace_only and not is_lift:
        parser.error('--inplace-only requires the independent lift-v5 contract')
    checkpoint_sha = sha256(checkpoint)
    args.enable_cameras = True
    launcher = AppLauncher(args)
    app = launcher.app

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from leg_humanoid_lab.agent_cfg import PPOCfg
    from leg_humanoid_lab.interactive_control import KeyboardController
    if is_lift:
        from leg_humanoid_lab.upper_body_env import UpperBodyEnvCfg as EnvCfg,UpperBodyLegEnv as NativeEnv,upper_body_contract as lift_contract
        runtime_contract=lift_contract()
        if any(runtime_contract.get(key)!=value for key,value in expected.items()):
            app.close()
            raise ValueError('Runtime lift control identity does not match the saved candidate')
    else:
        from leg_humanoid_lab.interactive_env import InteractiveEnvCfg as EnvCfg,InteractiveLegEnv as NativeEnv
    from runtime_compat import load_checkpoint, runner_cfg

    class DiagnosticEnv(NativeEnv):
        _manual_reset_allowed = True
        suppressed_resets = 0

        def _reset_idx(self, env_ids):
            if self._manual_reset_allowed:
                super()._reset_idx(env_ids)
            else:
                self.suppressed_resets += len(env_ids) if env_ids is not None else self.num_envs

    plan = diagnostic_plan(args.sidestep_duration,args.sequence_only,args.inplace_only)
    task='inplace_curriculum' if args.inplace_only else 'finite_sidestep_sequence' if args.sequence_only else 'interactive_diagnostic'
    report = {"diagnostic": True, "gate_passed": False, "accepted": False,
        "source": "native Isaac Sim PhysX PPO", "proxy": False,
        "overlay": "EXPERIMENTAL / NOT ACCEPTED", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha, "contract": expected,
        "robot":expected['robot'],"control_profile":expected.get('control_profile','interactive_v4_native_residual'),
        "task":task,"task_label":"In-place curriculum" if args.inplace_only else "Finite sidestep sequence" if args.sequence_only else "Interactive diagnostic",
        "seed": args.seed, "fps": 50, "resolution": [1280, 720], "frames": 0,
        "actual_duration_s": 0., "expected_duration_s": plan[-1]["end"],
        "plan": plan, "mode_changes": [], "root_positions": [],"foot_observations":[],
        "failure": False, "failure_reason": None, "technical_error": None,
        "completed_plan": False, "reset_count_after_start": 0,
        "auto_resets_suppressed": 0,
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py",
            "source/leg_humanoid_lab/leg_humanoid_lab/interactive_control.py",
            "scripts/reinforcement_learning/rsl_rl/record_terminal_v6_diagnostic.py")},
        "scope": "Unaccepted training candidate; actual failure remains visible; not evidence of learned skill."}
    if is_lift:
        for name in ('source/leg_humanoid_lab/leg_humanoid_lab/lift_env.py',
                     'source/leg_humanoid_lab/leg_humanoid_lab/lift_objective.py','config/lift_control_v5.json'):
            report['source_sha256'][name]=sha256(ROOT/name)
    foot_metrics=FootDiagnosticMetrics()
    env, writer = None, None
    try:
        cfg = EnvCfg()
        cfg.scene.num_envs = 1
        cfg.training_mode = False
        if is_lift:cfg.curriculum_stage=0 if args.inplace_only else 1
        cfg.episode_length_s = 30.
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.viewer.eye, cfg.viewer.lookat = ((2.35, -2.55, 1.5), (.05, 0., .7)) if is_lift else ((3., -3., 1.8), (.1, 0., .8))
        cfg.viewer.resolution = (1280, 720)
        env = DiagnosticEnv(cfg, render_mode="rgb_array")
        env.set_manual_control("stand", [0., 0., 0.], 0.)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        env._manual_reset_allowed = False
        runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
        load_checkpoint(runner, checkpoint, load_optimizer=False, load_iteration=False)
        if sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("Checkpoint changed while loading; record a stable snapshot")
        policy = runner.get_inference_policy(device=env.device)
        if not math.isclose(env.step_dt, .02, rel_tol=0., abs_tol=1e-7):
            raise RuntimeError("Diagnostic recording requires 50 Hz native control")
        controller = KeyboardController(sidestep_duration=args.sidestep_duration)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(target), fps=50, codec="libx264", quality=8,
                                   macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 27)
            metric_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
            metric_font = font
        for _ in range(12):
            env.sim.render()
            env.render(recompute=True)
        last_segment, last_mode = None, None
        command_clock = time.monotonic()
        count = round(plan[-1]["end"] / env.step_dt)
        with torch.inference_mode():
            for i in range(count):
                if not app.is_running():
                    report.update(failure=True, failure_reason="Isaac Sim closed before the diagnostic finished")
                    break
                t, now = i * env.step_dt, command_clock + i * env.step_dt
                segment = next(item for item in plan if item["start"] <= t < item["end"])
                if segment["name"] != last_segment:
                    if segment["name"] == "stop":
                        controller.stop(now)
                    elif segment["name"] == "sidestep_sequence":
                        controller.sequence(now)
                    last_segment = segment["name"]
                if segment["name"] == "W_forward":
                    controller.velocity(*segment["command"], now)
                if args.inplace_only:
                    mode={'inplace_lift':'walk','recover':'recover'}.get(segment['name'],'stand')
                    control={'mode':mode,'command':[0.,0.,0.],'phase':0.}
                else:
                    control = controller.tick(env.step_dt, now)
                env.set_manual_control(control["mode"], control["command"], control["phase"])
                obs = wrapped.get_observations()
                _obs, _reward, done, _info = wrapped.step(policy(obs))
                data = env.robot.data
                root = (data.root_pos_w[0] - env.scene.env_origins[0]).cpu().tolist()
                quat = data.root_quat_w[0].cpu().tolist()
                finite = all(math.isfinite(value) for value in root + quat)
                if finite:
                    report["root_positions"].append({"time": (i + 1) * env.step_dt,
                                                       "root": root, "quaternion": quat})
                if control["mode"] != last_mode:
                    report["mode_changes"].append({"time": t, "mode": control["mode"]})
                    last_mode = control["mode"]
                failed = bool(done[0]) or not finite
                if failed:
                    report.update(failure=True, failure_reason="Native PhysX episode terminated; recording stops without resetting")
                if not failed:
                    clearance,_speed,loaded,_target=env.foot_step_metrics()
                    foot_world=data.body_link_pos_w[0,env.foot_body_ids].cpu().tolist()
                    clearance=clearance[0].cpu().tolist();loaded=loaded[0].cpu().tolist()
                    ref_height=env.reference_height[0].cpu().tolist()
                    ref_contact=env.reference_contact[0].cpu().tolist()
                    foot_metrics.update(clearance,loaded,ref_height,ref_contact,foot_world)
                    report['foot_observations'].append({'time':(i+1)*env.step_dt,'clearance_m':clearance,'loaded':loaded,
                        'reference_height_m':ref_height,'reference_contact':ref_contact,'foot_link_positions_world_m':foot_world})
                env.sim.render()
                pixels = np.asarray(env.render(recompute=True))
                if pixels.shape != (720, 1280, 3):
                    raise RuntimeError(f"Unexpected native viewport frame shape: {pixels.shape}")
                picture = Image.fromarray(pixels)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 1280, 86), fill=(18, 18, 18))
                draw.text((16, 7), report["overlay"], fill=(255, 195, 40), font=font)
                if args.inplace_only:
                    draw.text((745,7),'In-place curriculum',fill=(255,195,40),font=font)
                label = f"Native PhysX | seed {args.seed} | {t + env.step_dt:.2f}s | {control['mode']}"
                if failed:
                    label += " | FAILED - NO RESET"
                draw.text((16, 46), label, fill=(255, 95, 95) if failed else (245, 245, 245), font=font)
                if is_lift and not failed:
                    draw.rectangle((0, 660, 1280, 720), fill=(18, 18, 18))
                    measured = ' | '.join(f"{side} sole {max(0.,100*clearance[j]):.1f} cm ({'LOADED' if loaded[j] else 'AIR'})" for j, side in enumerate(('L', 'R')))
                    peaks = foot_metrics.unloaded_peak
                    draw.text((16, 666), measured, fill=(245, 245, 245), font=metric_font)
                    draw.text((16, 692), f"Measured unloaded peak L/R: {100*peaks[0]:.1f}/{100*peaks[1]:.1f} cm | native CAD + contact", fill=(255, 195, 40), font=metric_font)
                writer.append_data(np.asarray(picture))
                report["frames"] += 1
                if failed:
                    break
        report["completed_plan"] = report["frames"] == count and not report["failure"]
        report["final_mode"] = last_mode
    except BaseException as exception:
        report["technical_error"] = str(exception)
        raise
    finally:
        if writer is not None:
            writer.close()
        report["actual_duration_s"] = report["frames"] / 50.
        report["auto_resets_suppressed"] = int(env.suppressed_resets) if env else 0
        report['foot_metrics']=foot_metrics.summary()
        if target.exists():
            report["video_sha256"] = sha256(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("DIAGNOSTIC_RESULT " + json.dumps({key: report[key] for key in (
            "gate_passed", "seed", "frames", "actual_duration_s", "failure", "failure_reason", "technical_error")}), flush=True)
        if env is not None:
            env.close()
        app.close()
    if not validate_metadata(report):
        raise RuntimeError("Diagnostic capture metadata is incomplete; inspect its JSON before sharing")


if __name__ == "__main__":
    main()
