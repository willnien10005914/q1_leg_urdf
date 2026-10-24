#!/usr/bin/env python3
"""Native frozen-V6 waist-damping comparison; no training or acceptance claim."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=241)
    parser.add_argument('--legacy-v5-actor', action='store_true', help='Explicit native-interface transfer probe, never a V6 resume')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or args.replicates < 1:
        parser.error('Fresh output and positive replicate count required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = env = None
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from leg_humanoid_lab.upper_body_env import UpperBodyLegEnv, UpperBodyEnvCfg, upper_body_contract
        from leg_humanoid_lab.agent_cfg import PPOCfg
        from runtime_compat import runner_cfg, load_checkpoint
        expected = upper_body_contract()
        saved = json.loads((args.checkpoint.parent / 'contract.json').read_text())
        keys = ('urdf_sha256', 'gait_sha256', 'joint_names', 'action_size', 'observation_size') if args.legacy_v5_actor else ('robot', 'schema', 'urdf_sha256', 'gait_sha256', 'control_sha256', 'motor', 'joint_names')
        for k in keys:
            if saved.get(k) != expected.get(k):
                raise ValueError('Frozen native V6 checkpoint contract mismatch: ' + k)
        checkpoint_sha = digest(args.checkpoint)
        if args.legacy_v5_actor and (saved.get('robot') != 'q1_leg_lift_v5' or saved.get('proxy') is not False
                or saved.get('control_sha256') != expected['base_control_sha256']
                or checkpoint_sha != 'b4077ac2a5d3fd98de67e43165cc753263c01d60e33240562303f912e11dbab9'):
            raise ValueError('Legacy probe is restricted to the reviewed native 13115 actor')
        variants = (('original_kd4', 4.), ('waist_kd8', 8.), ('waist_kd12', 12.))
        cases = ('stand', 'inplace', 'left', 'right', 'pair')
        labels = [(v, kd, c) for v, kd in variants for _ in range(args.replicates) for c in cases]

        class ProbeEnv(UpperBodyLegEnv):
            capturing = None
            def __init__(self, cfg):
                super().__init__(cfg)
                self.failures = []
                self.capture_step = 0
                damping = self.robot.actuators['all'].damping.clone()
                for j, name in enumerate(self.robot.joint_names):
                    if 'waist_' in name:
                        damping[:, j] = torch.tensor([kd for _, kd, _ in labels], device=self.device)
                self.robot.write_joint_damping_to_sim(damping)
                self.robot.actuators['all'].damping.copy_(damping)

            def _get_dones(self):
                done, timeout = super()._get_dones()
                if self.capturing is not None:
                    self.capture_step += 1
                    for i in torch.where(self.capturing & (done | timeout))[0].cpu().tolist():
                        forces = self.contacts.data.net_forces_w[i]
                        bad = [{'body': self.contacts.body_names[j], 'force_n': float(forces[j].norm())}
                               for j in self.other_bodies if float(forces[j].norm()) > 25.]
                        self.failures.append({'index': i, 'time_s': self.capture_step * self.step_dt,
                                              'nonfoot_contacts_over_25n': bad})
                return done, timeout

        cfg = UpperBodyEnvCfg()
        cfg.scene.num_envs = len(labels)
        cfg.training_mode = False
        cfg.curriculum_stage = 0
        cfg.reward_revision = 2
        cfg.episode_length_s = 30.
        cfg.seed = args.seed
        cfg.sim.device = args.device
        env = ProbeEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        runner = OnPolicyRunner(wrapped, runner_cfg(PPOCfg()), log_dir=None, device=args.device)
        load_checkpoint(runner, args.checkpoint, load_optimizer=False, load_iteration=False)
        policy = runner.get_inference_policy(device=env.device)
        wrapped.reset()
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env.capturing = active.clone()
        arrays = {}
        with torch.inference_mode():
            for step in range(round(20. / env.step_dt)):
                if not app.is_running():
                    raise RuntimeError('Simulator closed during probe')
                t = step * env.step_dt
                modes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
                phases = torch.zeros(env.num_envs, device=env.device)
                for i, (_, _, case) in enumerate(labels):
                    if case == 'inplace':
                        if 1. <= t < 8.: modes[i] = 1
                    elif case != 'stand':
                        elapsed = t - 1.48
                        if 1. <= t < 1.48: modes[i] = 4
                        elif 0. <= elapsed < 1.6:
                            modes[i] = 3 if case == 'right' else 2
                            phases[i] = elapsed / 1.6
                        elif case == 'pair' and 1.6 <= elapsed < 3.2:
                            modes[i] = 3; phases[i] = (elapsed - 1.6) / 1.6
                        elif (3.2 if case == 'pair' else 1.6) <= elapsed < (4.2 if case == 'pair' else 2.6):
                            modes[i] = 4
                env.set_manual_control(modes, torch.zeros((env.num_envs, 3), device=env.device), phases)
                _, reward, done, _ = wrapped.step(policy(wrapped.get_observations()))
                active &= ~done.bool().reshape(-1)
                m = env.upper_body_measurements()
                height, speed, loaded, target = env.foot_step_metrics()
                active &= torch.isfinite(reward) & torch.isfinite(m['chest_tilt_rad']) & torch.isfinite(height).all(-1)
                env.capturing &= active
                data = {'valid': active, 'mode': modes, 'height': height, 'foot_speed': speed,
                        'loaded': loaded, 'reference_height': target, 'reference_contact': env.reference_contact,
                        'root_speed': env.robot.data.root_lin_vel_b[:, :2].norm(dim=-1), **m}
                for key, value in data.items():
                    arrays.setdefault(key, []).append(value.cpu().numpy().copy())
        arrays = {k: np.stack(v) for k, v in arrays.items()}
        np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
        times = (np.arange(len(arrays['valid'])) + 1) * env.step_dt
        rows = []
        for i, (variant, kd, case) in enumerate(labels):
            valid = arrays['valid'][:, i]
            dynamic = valid & (times >= (1. if case == 'inplace' else 1.48)) & (times < (8. if case == 'inplace' else 5.68 if case == 'pair' else 4.08))
            if case == 'stand': dynamic = valid
            tail = valid & (times >= 18.)
            moving = valid & (arrays['mode'][:, i] >= 1) & (arrays['mode'][:, i] <= 3)
            high = moving[:, None] & ~arrays['reference_contact'][:, i] & (arrays['reference_height'][:, i] > .022)
            stance = valid[:, None] & arrays['loaded'][:, i] & arrays['reference_contact'][:, i]
            def rms(k, selected):
                x = arrays[k][selected, i]
                return float(np.sqrt(np.mean(x ** 2))) if len(x) else None
            row = {'variant': variant, 'waist_kp': 70., 'waist_kd': kd, 'case': case,
                   'survived': bool(valid.all()), 'valid_duration_s': float(valid.sum() * env.step_dt),
                   'chest_dynamic_xy_rms_rad_s': rms('chest_angular_xy_rad_s', dynamic),
                   'chest_dynamic_xyz_rms_rad_s': rms('chest_angular_xyz_rad_s', dynamic),
                   'chest_peak_deg': float(np.degrees(arrays['chest_tilt_rad'][valid, i].max())) if valid.any() else None,
                   'waist_dynamic_speed_rms_rad_s': float(np.sqrt(arrays['waist_joint_speed_mse_rad2_s2'][dynamic, i].mean())) if dynamic.any() else None,
                   'tail_arm_speed_rms_rad_s': float(np.sqrt(arrays['arm_joint_speed_mse_rad2_s2'][tail, i].mean())) if tail.any() else None,
                   'tail_hand_distance_rms_m_per_hand': np.sqrt((arrays['hands_return_distance_m'][tail, i] ** 2).mean(0)).tolist() if tail.any() else None,
                   'tail_root_speed_m_s': float(arrays['root_speed'][tail, i].mean()) if tail.any() else None,
                   'stance_slip_rms_m_s': float(np.sqrt((arrays['foot_speed'][:, i] ** 2 * stance).sum() / max(stance.sum(), 1))),
                   'unloaded_peak_clearance_m_per_foot': [float(arrays['height'][moving & ~arrays['loaded'][:, i, f], i, f].max()) if (moving & ~arrays['loaded'][:, i, f]).any() else 0. for f in (0, 1)],
                   'high_swing_samples_per_foot': high.sum(0).tolist(),
                   'unloaded_lift_rate_per_foot': ((high & ~arrays['loaded'][:, i] & (arrays['height'][:, i] >= .025)).sum(0) / high.sum(0).clip(min=1)).tolist()}
            rows.append(row)
        report = {'experimental': True, 'accepted': False, 'retrained': False,
                  'actor_contract': saved['robot'], 'legacy_transfer_probe': args.legacy_v5_actor,
                  'checkpoint_sha256': checkpoint_sha, 'control_sha256': expected['control_sha256'],
                  'script_sha256': digest(__file__), 'seed': args.seed, 'duration_s': 20.,
                  'variants': variants, 'rows': rows, 'first_terminations': env.failures,
                  'root_forces_or_constraints': False, 'rendered_pose_filtering': False,
                  'motor_effort_limit_nm': 100., 'scope': 'Frozen actor physical damping comparison, no policy promotion'}
        if digest(args.checkpoint) != checkpoint_sha: raise RuntimeError('Probe checkpoint changed')
        with args.output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
        print('V6_DAMPING_PROBE', sum(r['survived'] for r in rows), '/', len(rows), flush=True)
    except BaseException as error:
        args.output.with_suffix('.error.json').write_text(json.dumps({'error': str(error)}) + '\n')
        raise
    finally:
        if env is not None: env.close()
        if app is not None: app.close()


if __name__ == '__main__':
    main()
