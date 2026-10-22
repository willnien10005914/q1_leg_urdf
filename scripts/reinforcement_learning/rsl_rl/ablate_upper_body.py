#!/usr/bin/env python3
"""Native upper-body ablation with a frozen PPO, never policy publication.

Compare actual chest attitude and joint/hand tracking; retain independent foot
measurements and every first termination. Rendering receives raw PhysX poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record-variant', choices=('baseline', 'upper_pd', 'upper_residual_guard', 'upper_pd_guard', 'waist_pd_guard'))
    parser.add_argument('--video', type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or args.replicates < 1:
        parser.error('Use a fresh output path and positive replicate count')
    if bool(args.video) != bool(args.record_variant):
        parser.error('--video and --record-variant must be specified together')
    if args.video and (args.video.exists() or args.video.suffix.lower() != '.mp4'):
        parser.error('Recording requires a fresh MP4 path')
    args.enable_cameras = bool(args.video)
    app = env = writer = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        app = AppLauncher(args).app
        import math
        import numpy as np
        import torch
        from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from leg_humanoid_lab.lift_env import LiftLegEnv, LiftEnvCfg, lift_contract
        from leg_humanoid_lab.robot import JOINTS, ROOT, fk
        from leg_humanoid_lab.upper_body_stability import (
            PROFILE, VARIANTS, guarded_targets, joint_drive_variants, tilt_rad)
        from runtime_compat import runner_cfg, load_checkpoint
        expected = lift_contract()
        saved = json.loads((args.checkpoint.parent / 'contract.json').read_text())
        for key in ('robot', 'urdf_sha256', 'gait_sha256', 'control_sha256',
                    'control_profile', 'motor', 'joint_names', 'action_size', 'observation_size'):
            if saved.get(key) != expected.get(key):
                raise ValueError('Frozen baseline checkpoint contract mismatch: ' + key)
        if saved.get('proxy') is not False:
            raise ValueError('Native CAD only')
        checkpoint_sha = digest(args.checkpoint)
        variants = (args.record_variant,) if args.video else VARIANTS
        cases = ('pair',) if args.video else ('stand', 'left', 'right', 'pair')
        replicates = 1 if args.video else args.replicates
        labels = [(name, case) for name in variants for _ in range(replicates) for case in cases]
        class ExperimentEnv(LiftLegEnv):
            capture_active = None
            recording_started = False
            suppressed_resets = 0
            def _reset_idx(self, env_ids):
                if args.video and self.recording_started:
                    self.suppressed_resets += len(env_ids) if env_ids is not None else self.num_envs
                    return
                super()._reset_idx(env_ids)

            def __init__(self, cfg, **kwargs):
                super().__init__(cfg, **kwargs)
                self.variant = torch.tensor([VARIANTS.index(name) for name, _ in labels], device=self.device)
                kp, kd = joint_drive_variants(self.robot.data.default_joint_stiffness,
                    self.robot.data.default_joint_damping, self.robot.joint_names, self.variant)
                # Restore the production leg values; default buffers predate
                # LiftLegEnv's native 800/32 drive override.
                for i, name in enumerate(self.robot.joint_names):
                    if any(part in name for part in ('hip_', 'knee_', 'ankle_')):
                        kp[:, i] = self.control_config['leg_stiffness']
                        kd[:, i] = self.control_config['leg_damping']
                self.robot.write_joint_stiffness_to_sim(kp)
                self.robot.write_joint_damping_to_sim(kd)
                self.robot.actuators['all'].stiffness.copy_(kp)
                self.robot.actuators['all'].damping.copy_(kd)
                self.upper_failures = []
                self.capture_step = 0

            def _pre_physics_step(self, actions):
                super()._pre_physics_step(actions)
                self.target = guarded_targets(self.target, self.refq, self.actions,
                                              self.variant, self.lower, self.upper)

            def _get_dones(self):
                terminated, truncated = super()._get_dones()
                if self.capture_active is not None:
                    self.capture_step += 1
                    for index in torch.where(self.capture_active & (terminated | truncated))[0].cpu().tolist():
                        forces = self.contacts.data.net_forces_w[index]
                        bad = [{'body': self.contacts.body_names[j], 'force_n': float(forces[j].norm())}
                               for j in self.other_bodies if float(forces[j].norm()) > 25.]
                        self.upper_failures.append({'index': index, 'time_s': self.capture_step * self.step_dt,
                            'terminated': bool(terminated[index]), 'truncated': bool(truncated[index]),
                            'nonfoot_contacts_over_25n': bad})
                return terminated, truncated

        cfg = LiftEnvCfg()
        cfg.scene.num_envs = len(labels)
        cfg.training_mode = False
        cfg.curriculum_stage = 1
        cfg.episode_length_s = 30.
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.viewer.eye = (2.35, -2.55, 1.5)
        cfg.viewer.lookat = (.05, 0., .7)
        cfg.viewer.resolution = (1280, 720)
        env = ExperimentEnv(cfg, render_mode='rgb_array' if args.video else None)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
        load_checkpoint(runner, args.checkpoint, load_optimizer=False, load_iteration=False)
        policy = runner.get_inference_policy(device=env.device)
        torso_ids, _ = env.robot.find_bodies(['center_torso_link'], preserve_order=True)
        hand_ids, _ = env.robot.find_bodies(['left_wrist_yaw_link', 'right_wrist_yaw_link'], preserve_order=True)
        torso = torso_ids[0]
        env.set_manual_control('stand', [0., 0., 0.], 0.)
        wrapped.reset()
        env.recording_started = bool(args.video)
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        env.capture_active = active.clone()
        env.capture_step = 0
        env.upper_failures.clear()
        history = {}
        duration = 7.2 if args.video else 20.
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=50, codec='libx264', quality=8,
                                       macro_block_size=16, ffmpeg_params=['-movflags', '+faststart'])
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
            for _ in range(12):
                env.sim.render(); env.render(recompute=True)
        with torch.inference_mode():
            for step in range(round(duration / env.step_dt)):
                if not app.is_running():
                    raise RuntimeError('Simulator closed before the diagnostic completed')
                t = step * env.step_dt
                modes = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
                phases = torch.zeros(env.num_envs, device=env.device)
                for i, (_, case) in enumerate(labels):
                    if case == 'stand': continue
                    start, maneuver = 1.48, 1.6
                    elapsed = t - start
                    if 1. <= t < start:
                        modes[i] = 4
                    elif 0. <= elapsed < maneuver:
                        modes[i] = 3 if case == 'right' else 2
                        phases[i] = elapsed / maneuver
                    elif case == 'pair' and maneuver <= elapsed < 2 * maneuver:
                        modes[i] = 3; phases[i] = (elapsed - maneuver) / maneuver
                    elif (2 * maneuver if case == 'pair' else maneuver) <= elapsed < (2 * maneuver if case == 'pair' else maneuver) + 1.:
                        modes[i] = 4
                env.set_manual_control(modes, torch.zeros((env.num_envs, 3), device=env.device), phases)
                _, _, done, _ = wrapped.step(policy(wrapped.get_observations()))
                active &= ~done.reshape(-1).bool()
                data = env.robot.data
                clearance, speed, loaded, _ = env.foot_step_metrics()
                active &= (torch.isfinite(data.root_state_w).all(-1) & torch.isfinite(clearance).all(-1)
                           & torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.joint_vel).all(-1))
                env.capture_active &= active
                root_q = data.root_quat_w
                hand = quat_apply_inverse(root_q[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
                    (data.body_link_pos_w[:, hand_ids] - data.root_pos_w[:, None, :]).reshape(-1, 3)).reshape(-1, 2, 3)
                relative = quat_mul(quat_conjugate(root_q), data.body_quat_w[:, torso])
                values = {'valid': active, 'mode': modes, 'phase': phases, 'q': data.joint_pos[:, env.jids],
                    'qvel': data.joint_vel[:, env.jids], 'reference': env.refq, 'target': env.target,
                    'torque': data.applied_torque[:, env.jids], 'hands_body': hand,
                    'pelvis_tilt': tilt_rad(root_q), 'chest_tilt': tilt_rad(data.body_quat_w[:, torso]),
                    'waist_tilt': tilt_rad(relative), 'chest_angular_speed': data.body_ang_vel_w[:, torso].norm(dim=-1),
                    'root_position': data.root_pos_w - env.scene.env_origins,
                    'root_quaternion': root_q,
                    'root_speed': data.root_lin_vel_b[:, :2].norm(dim=-1),
                    'clearance': clearance, 'loaded': loaded, 'reference_height': env.reference_height,
                    'reference_contact': env.reference_contact, 'foot_speed': speed}
                for key, value in values.items():
                    history.setdefault(key, []).append(value.cpu().numpy().copy())
                if writer:
                    env.sim.render()
                    image = Image.fromarray(np.asarray(env.render(recompute=True)))
                    draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 1280, 100), fill=(18, 18, 18))
                    draw.text((16, 5), f'EXPERIMENTAL / NOT ACCEPTED | frozen PPO + {args.record_variant}', fill=(255, 190, 40), font=font)
                    draw.text((16, 36), f'Native PhysX | {t + env.step_dt:.2f}s | chest tilt {math.degrees(float(values["chest_tilt"][0])):.1f} deg', fill='white', font=font)
                    draw.text((16, 67), 'No retraining | native 100 Nm limits | no rendered-pose smoothing', fill='white', font=font)
                    writer.append_data(np.asarray(image))
                    if not bool(active.all()):
                        break  # Show the physical failure, never an auto-reset recovery.
        if writer:
            writer.close(); writer = None
        arrays = {key: np.stack(value) for key, value in history.items()}
        np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
        times = (np.arange(len(arrays['valid'])) + 1) * env.step_dt
        arm_ids = [i for i, n in enumerate(JOINTS) if 'shoulder_' in n or 'elbow_' in n]
        waist_ids = [i for i, n in enumerate(JOINTS) if 'waist_' in n]
        hand_reference = np.empty_like(arrays['hands_body'])
        cache = {}
        for t in range(len(times)):
            for i in range(env.num_envs):
                key = tuple(np.round(arrays['reference'][t, i], 7))
                if key not in cache:
                    transforms = fk(key)
                    cache[key] = np.stack([transforms[n][:3, 3] for n in ('left_wrist_yaw_link', 'right_wrist_yaw_link')])
                hand_reference[t, i] = cache[key]
        rows = []
        def rms(value): return float(np.sqrt(np.mean(np.square(value)))) if len(value) else None
        for i, (variant, case) in enumerate(labels):
            valid = arrays['valid'][:, i]
            dynamic = valid & (times >= 1.) & (times < (6.7 if case == 'pair' else 5.1))
            rest = valid & (times >= duration - 1.)
            row = {'variant': variant, 'case': case, 'survived': bool(valid.all()), 'valid_duration_s': float(valid.sum() * env.step_dt),
                'chest_tilt_rms_deg': math.degrees(rms(arrays['chest_tilt'][dynamic, i])) if dynamic.any() else None,
                'chest_tilt_peak_deg': math.degrees(float(arrays['chest_tilt'][dynamic, i].max())) if dynamic.any() else None,
                'pelvis_tilt_rms_deg': math.degrees(rms(arrays['pelvis_tilt'][dynamic, i])) if dynamic.any() else None,
                'waist_tilt_rms_deg': math.degrees(rms(arrays['waist_tilt'][dynamic, i])) if dynamic.any() else None,
                'chest_angular_speed_rms_rad_s': rms(arrays['chest_angular_speed'][dynamic, i]),
                'arm_reference_rmse_rad': rms((arrays['q'][:, i] - arrays['reference'][:, i])[dynamic][:, arm_ids]),
                'arm_drive_tracking_rmse_rad': rms((arrays['q'][:, i] - arrays['target'][:, i])[dynamic][:, arm_ids]),
                'waist_reference_rmse_rad': rms((arrays['q'][:, i] - arrays['reference'][:, i])[dynamic][:, waist_ids]),
                'final_stand_speed_m_s': float(arrays['root_speed'][rest, i].mean()) if rest.any() else None,
                'hand_return_error_m_per_hand': np.sqrt(np.mean(np.square(arrays['hands_body'][rest, i] - hand_reference[rest, i]), axis=(0, 2))).tolist() if rest.any() else None,
                'hand_return_distance_rms_m_per_hand': np.sqrt(np.mean(np.sum(np.square(arrays['hands_body'][rest, i] - hand_reference[rest, i]), axis=-1), axis=0)).tolist() if rest.any() else None,
                'upper_near_100nm_fraction': float((np.abs(arrays['torque'][dynamic, i][:, arm_ids + waist_ids]) >= 99.).mean()) if dynamic.any() else None,
                'directions': {}}
            for name, mode, lead in (('left', 2, 0), ('right', 3, 1)):
                selected = valid & (arrays['mode'][:, i] == mode)
                if not selected.any(): continue
                side = {}
                for role, foot in (('lead', lead), ('trailing', 1 - lead)):
                    high = selected & ~arrays['reference_contact'][:, i, foot] & (arrays['reference_height'][:, i, foot] > .022)
                    air = selected & ~arrays['loaded'][:, i, foot]
                    side[role + '_unloaded_peak_m'] = float(arrays['clearance'][air, i, foot].max()) if air.any() else 0.
                    side[role + '_high_swing_samples'] = int(high.sum())
                    side[role + '_lift_rate'] = float((~arrays['loaded'][high, i, foot] & (arrays['clearance'][high, i, foot] >= .025)).mean()) if high.any() else None
                row['directions'][name] = side
            rows.append(row)
        report = {'schema': 1, 'accepted': False, 'experimental': True, 'retrained': False,
            'source': 'native_physx_frozen_ppo_upper_body_ablation', 'checkpoint_sha256': checkpoint_sha,
            'baseline_control_sha256': expected['control_sha256'], 'profile': PROFILE,
            'profile_module_sha256': digest(ROOT / 'source/leg_humanoid_lab/leg_humanoid_lab/upper_body_stability.py'),
            'script_sha256': digest(__file__), 'seed': args.seed,
            'duration_s': len(times) * env.step_dt, 'expected_duration_s': duration,
            'rows': rows, 'first_terminations': env.upper_failures,
            'motor_effort_limit_nm': 100., 'motor_velocity_limit_rad_s': 10.,
            'root_forces_or_constraints': False, 'rendered_pose_filtering': False,
            'torso_measurement': 'center_torso_link world inclination and angular velocity; root upright alone does not measure chest stability',
            'hand_measurement': 'hand_return_error is per-coordinate RMS; hand_return_distance_rms is Euclidean distance RMS, both versus native reference FK in pelvis coordinates',
            'torque_measurement': 'Implicit actuator applied-torque estimate, not an independent sensor',
            'raw_trace': str(args.output.with_suffix('.npz'))}
        if args.video:
            report.update(video_sha256=digest(args.video), frames=len(times), failure=not rows[0]['survived'],
                          video=str(args.video), reset_count_after_start=0,
                          auto_resets_suppressed=env.suppressed_resets)
        if digest(args.checkpoint) != checkpoint_sha:
            raise RuntimeError('Frozen checkpoint changed during diagnosis')
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
        print('UPPER_BODY_DIAGNOSTIC ' + json.dumps({'output': str(args.output), 'survived': sum(row['survived'] for row in rows), 'episodes': len(rows)}), flush=True)
    except BaseException as error:
        args.output.with_suffix('.error.json').write_text(json.dumps({'error': str(error), 'accepted': False}) + '\n')
        raise
    finally:
        if writer is not None: writer.close()
        if env is not None: env.close()
        if app is not None: app.close()


if __name__ == '__main__':
    main()
