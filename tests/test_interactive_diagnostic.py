"""Diagnostic media retains failures and can never masquerade as accepted media."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/reinforcement_learning/rsl_rl'))
from record_interactive_diagnostic import diagnostic_plan, validate_metadata,diagnostic_contract,FootDiagnosticMetrics


def report():
    return {'diagnostic': True, 'gate_passed': False, 'accepted': False, 'proxy': False,
            'source': 'native Isaac Sim PhysX PPO', 'overlay': 'EXPERIMENTAL / NOT ACCEPTED',
            'reset_count_after_start': 0, 'seed': 42, 'checkpoint_sha256': 'a' * 64,
            'frames': 10, 'actual_duration_s': .2, 'failure': True,
            'failure_reason': 'Native episode terminated', 'technical_error': None}


def test_actual_failure_is_a_valid_diagnostic_but_acceptance_or_reset_is_not():
    value = report()
    assert validate_metadata(value)
    for key, bad in [('gate_passed', True), ('accepted', True), ('proxy', True),
                     ('reset_count_after_start', 1), ('overlay', ''),
                     ('failure_reason', None), ('frames', 0), ('actual_duration_s', .3),
                     ('technical_error', 'camera unavailable')]:
        assert not validate_metadata({**value, key: bad}), key
    assert validate_metadata({**value, 'failure': False, 'failure_reason': None})


def test_plan_is_finite_and_contains_only_native_control_requests():
    plan = diagnostic_plan()
    assert abs(plan[-1]['end'] - 9.7) < 1e-7
    assert all(a['end'] == b['start'] for a, b in zip(plan, plan[1:]))
    assert [item['name'] for item in plan] == ['stand', 'W_forward', 'stop', 'sidestep_sequence', 'final_stand']


def test_sequence_diagnostic_does_not_hide_a_failed_walk_in_the_same_episode():
    plan=diagnostic_plan(sequence_only=True)
    assert [item['name'] for item in plan]==['stand','sidestep_sequence','final_stand']
    assert abs(plan[-1]['end']-7.2)<1e-7
    assert all(a['end']==b['start'] for a,b in zip(plan,plan[1:]))


def test_inplace_diagnostic_is_an_explicit_zero_command_walk_curriculum_then_recovers():
    plan=diagnostic_plan(inplace_only=True)
    assert [(item['name'],item['start'],item['end']) for item in plan]==[
        ('stand',0.,1.),('inplace_lift',1.,5.5),('recover',5.5,6.5),('final_stand',6.5,8.)]
    assert plan[1]['command']==[0.,0.,0.]
    with pytest.raises(ValueError,match='mutually exclusive'):
        diagnostic_plan(sequence_only=True,inplace_only=True)


def test_v4_and_v5_contracts_pin_their_own_gait_and_v5_native_control():
    from play_interactive_web import sha256
    base=json.loads((ROOT/'config/robot_contract.json').read_text())
    common={'observation_size':110,'joint_names':base['joint_names'],
            'urdf_sha256':sha256(ROOT/'urdf/q1_inertia/quanta_robot_legged.urdf'),'proxy':False}
    saved_v4={**common,'robot':'q1_leg_interactive_v4','gait_sha256':sha256(ROOT/'config/v4_gaits.json')}
    assert diagnostic_contract(saved_v4)['robot']=='q1_leg_interactive_v4'
    control_path=ROOT/'config/lift_control_v5.json'
    control=json.loads(control_path.read_text())
    # Read the literal profile through the same public preflight mechanism;
    # neither fixture nor production validation imports Isaac or starts Kit.
    import ast
    tree=ast.parse((ROOT/'source/leg_humanoid_lab/leg_humanoid_lab/lift_objective.py').read_text())
    profile=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id=='CONTROL_PROFILE' for target in node.targets))
    saved_v5={**common,'robot':'q1_leg_lift_v5','gait_sha256':sha256(ROOT/control['gait_file']),
              'control_sha256':sha256(control_path),'control_profile':profile}
    assert diagnostic_contract(saved_v5)['control_profile']==profile
    for key,value in [('robot','proxy'),('gait_sha256',saved_v4['gait_sha256']),('control_sha256','0'*64),('control_profile','old_unbounded_control'),('proxy',True)]:
        with pytest.raises(ValueError):diagnostic_contract({**saved_v5,key:value})


def test_foot_measurements_require_physical_unloaded_motion_and_count_both_feet():
    metrics=FootDiagnosticMetrics()
    # A high reference and lifted-but-loaded foot cannot earn unloaded lift.
    metrics.update([.04,0.],[True,True],[.06,.06],[False,False],[[0.,0.,0.],[0.,1.,0.]])
    # Only actual left clearance earns one success; the right foot remains down.
    metrics.update([.035,0.],[False,False],[.06,.06],[False,False],[[.03,0.,.04],[0.,1.,0.]])
    # Stance observations do not dilute the planned swing denominator.
    metrics.update([0.,0.],[True,True],[0.,0.],[True,True],[[0.,0.,0.],[0.,1.,0.]])
    value=metrics.summary()
    assert value['unloaded_peak_clearance_m_per_foot']==[.035,0.]
    assert value['planned_high_swing_samples_per_foot']==[2,2]
    assert value['planned_high_swing_success_rate_per_foot']==[.5,0.]
    assert value['foot_world_peak_displacement_m_per_foot']==[.05,0.]
    assert value['foot_world_path_length_m_per_foot']==[.1,0.]
    assert value['foot_world_displacement_xyz_m_per_foot']==[[0.,0.,0.],[0.,0.,0.]]
    with pytest.raises(ValueError,match='nonfinite'):
        metrics.update([float('nan'),0.],[False,False],[.06,.06],[False,False],[[0.,0.,0.],[0.,1.,0.]])


def test_v5_metadata_identifies_curriculum_and_preserves_no_sample_evidence():
    metrics=FootDiagnosticMetrics()
    metrics.update([0.,0.],[True,True],[0.,0.],[True,True],[[0.,0.,0.],[0.,1.,0.]])
    value={**report(),'robot':'q1_leg_lift_v5','task':'inplace_curriculum','task_label':'In-place curriculum',
           'control_profile':'native_bounded','foot_metrics':metrics.summary()}
    assert value['foot_metrics']['planned_high_swing_success_rate_per_foot']==[None,None]
    assert validate_metadata(value)
    assert not validate_metadata({**value,'robot':'q1_leg_interactive_v4'})
    assert not validate_metadata({**value,'task_label':'Keyboard passed'})
    value['foot_metrics']['planned_high_swing_success_rate_per_foot']=[1.,1.]
    assert not validate_metadata(value)
