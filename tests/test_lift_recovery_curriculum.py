"""CPU checks for stage-zero stop-phase coverage and long recovery exposure.

Extract only the scheduling method: no Isaac Sim, GPU or robot-state writes.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]


def scheduler(count=1,stage=0):
    source=ast.parse((ROOT/'source/leg_humanoid_lab/leg_humanoid_lab/lift_env.py').read_text())
    cls=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='LiftLegEnv')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='_training_schedule')
    class Parent:
        def _training_schedule(self):
            self.stage_two_delegated=True
    harness=ast.ClassDef(name='Harness',bases=[ast.Name(id='Parent',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    module=ast.Module(body=[harness],type_ignores=[])
    namespace={'Parent':Parent,'torch':torch}
    exec(compile(ast.fix_missing_locations(module),'<lift-schedule-only>','exec'),namespace)
    env=namespace['Harness']()
    env.cfg=SimpleNamespace(curriculum_stage=stage)
    env.step_dt=.02
    env.episode_length_buf=torch.zeros(count,dtype=torch.long)
    env.mode=torch.zeros(count,dtype=torch.long)
    env.command_targets=torch.zeros(count,3)
    env.gesture_phase=torch.zeros(count)
    env.schedule_start=torch.full((count,),.5)
    env.walk_duration=torch.full((count,),4.9) # maps to a seven-second bout
    env.scenario=torch.zeros(count,dtype=torch.long)
    env.library={'sidestep_duration_s':1.6}
    return env


def test_seven_second_training_bout_matches_evaluation_stop_phase_with_long_tail():
    env=scheduler()
    count=0
    for step in range(1000):
        env.episode_length_buf[:]=step
        env._training_schedule()
        count+=int(env.mode[0]==1)
    assert count==350
    stop_phase=(count*env.step_dt/.9)%1.
    assert abs(stop_phase-7/.9%1.)<1e-9
    assert 20.-(.5+count*env.step_dt)>=12.
    assert env.mode[0]==0
    assert env.command_targets.abs().sum()==0


def test_many_stop_phases_are_covered_and_stand_only_scenarios_never_walk():
    generator=torch.Generator().manual_seed(71)
    env=scheduler(4096)
    env.walk_duration=3.5+2*torch.rand(4096,generator=generator)
    env.schedule_start=.3+.4*torch.rand(4096,generator=generator)
    duration=3.5+(env.walk_duration-3.5)*2.5
    start=torch.ceil(env.schedule_start/env.step_dt)
    end=torch.ceil((env.schedule_start+duration)/env.step_dt)
    phase=((end-start)*env.step_dt/.9).remainder(1.)
    bins=torch.bincount((phase*16).long().clamp(max=15),minlength=16)
    assert bins.min()>100
    assert ((phase-(7/.9%1.)).abs()<.012).sum()>30
    assert duration.min()>=3.5 and duration.max()<=8.5
    assert (20.-env.schedule_start-duration).min()>10.8
    env.scenario[:]=3
    env.episode_length_buf[:]=100
    env._training_schedule()
    assert (env.mode==0).all()


def test_other_stages_keep_their_existing_finite_timing_and_delegation():
    env=scheduler(4,stage=1)
    env.scenario=torch.tensor([0,1,2,3])
    env.episode_length_buf[:]=50 # t=.5, first maneuver
    env._training_schedule()
    assert env.mode.tolist()==[2,2,3,0]
    env.episode_length_buf[:]=125 # t=2, second maneuver for pair
    env._training_schedule()
    assert env.mode.tolist()==[3,4,4,0]
    env.cfg.curriculum_stage=2
    env._training_schedule()
    assert env.stage_two_delegated


def test_profile_keeps_the_total_budget_and_only_extends_stage_zero_episode():
    original=json.loads((ROOT/'config/lift_v5.json').read_text())
    proposed=json.loads((ROOT/'config/lift_v5_recovery20.json').read_text())
    assert proposed['training_episode_lengths_s']==[20,8,20]
    for field in ('num_envs','max_iterations','gait_file','learning_rate','symmetry',
                  'validation_episodes','confirmation_episodes','max_reward_revisions'):
        assert proposed[field]==original[field]
    assert proposed['max_iterations']==15000 and proposed['num_envs']==10240
