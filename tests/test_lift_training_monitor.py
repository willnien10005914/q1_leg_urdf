"""Lift curriculum safety rules run without importing Isaac or allocating a GPU."""
import importlib.util
import ast
import json
from pathlib import Path
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('lift_training_monitor',ROOT/'tools/run_lift_training.py')
monitor=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def profile():
    return monitor.validate_profile(json.loads((ROOT/'config/lift_v5.json').read_text()))


def report(stage=2,episodes=32,passed=True):
    return {'stage':stage,'episodes':episodes,'backend':'isaaclab','proxy':False,'score':1. if passed else .5,
            'passed':passed,'checks':{'native_swing_lift':passed},'failed_gates':[] if passed else ['native_swing_lift'],
            'cases':{name:{'episodes':episodes//len(monitor.STAGE_CASES[stage])} for name in monitor.STAGE_CASES[stage]}}


def fake_checkpoint(path):
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('resume/data.pkl',b'checkpoint-test-payload'*100)


def supervisor(tmp_path,monkeypatch,resume=False):
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    monkeypatch.setattr(monitor,'source_manifest',lambda config:{'fixture':'frozen'})
    return monitor.Supervisor(profile(),tmp_path/'profile.json',tmp_path/'run',resume=resume)


def test_plan_counts_only_new_v5_updates_and_has_no_expired_deadline():
    plan=profile()
    assert monitor.next_chunk_size(0,plan)==100
    assert monitor.next_chunk_size(1999,plan)==100
    assert monitor.next_chunk_size(2000,plan)==500
    assert monitor.next_chunk_size(2000,plan,stage=0)==100
    assert monitor.next_chunk_size(2000,plan,stage=1)==100
    assert monitor.next_chunk_size(14999,plan)==1
    assert monitor.next_chunk_size(15000,plan)==0
    assert plan['num_envs']*plan['num_steps_per_env']*plan['max_iterations']==3_686_400_000
    for key,value in [('num_envs',6144),('max_iterations',15001),('stage_passes_to_advance',1),('confirmation_episodes',32),('training_deadline_utc','2026-10-21T01:00:00+00:00')]:
        changed=dict(plan);changed[key]=value
        with pytest.raises(ValueError):monitor.validate_profile(changed)


def test_preparatory_pass_cannot_publish_or_substitute_for_interactive_confirmation():
    assert monitor.stage_gate_passed(report(0),0,32)
    assert monitor.stage_gate_passed(report(1),1,32)
    assert not monitor.confirmation_passed(report(0,128))
    assert not monitor.confirmation_passed(report(2,32))
    assert monitor.confirmation_passed(report(2,128))
    for key,value in [('proxy',True),('backend','preview'),('score',float('nan')),('checks',{}),('failed_gates',['lift'])]:
        changed=report(2,128);changed[key]=value
        assert not monitor.confirmation_passed(changed)
    incomplete=report(2,128);del incomplete['cases']['sequence']
    assert not monitor.confirmation_passed(incomplete)


def test_plateau_is_stage_and_revision_specific_never_a_success():
    plan=profile()
    values=[{'kind':'validation','stage':0,'reward_revision':0,'completed_iterations':100*(index+1),
             'score':score,'passed':False,'failed_gates':['inplace.right.lift']} for index,score in enumerate([.7,.701,.7,.702])]
    result=monitor.plateau_diagnostic(values,plan,0,0)
    assert result['failed_gate_counts']=={'inplace.right.lift':4}
    assert not result['successful'] and not result['acceptance_gates_changed']
    assert monitor.plateau_diagnostic(values[:3],plan,0,0) is None
    assert monitor.plateau_diagnostic(values,plan,1,0) is None
    assert monitor.plateau_diagnostic(values,plan,0,1) is None
    values[-1]['score']=float('nan')
    with pytest.raises(ValueError):monitor.plateau_diagnostic(values,plan,0,0)


def test_two_preparatory_passes_then_three_interactive_passes_require_separate_confirmation(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    confirmations=[];promoted=[]
    def evaluate(episodes,seed,kind):
        value=report(run.state['curriculum_stage'],episodes)
        summary={'kind':kind,'stage':run.state['curriculum_stage'],'reward_revision':0,'score':1.,'passed':True,
                 'completed_iterations':run.state['completed_iterations'],'failed_gates':[]}
        run.state['evaluations'].append(summary);run.state['last_evaluation']=summary
        if kind=='confirmation':confirmations.append((episodes,seed))
        return value
    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(run,'promote',lambda value:promoted.append(value))
    for count in range(1,8):
        run.state['completed_iterations']=count*100
        done=run.process_validation()
        assert done is (count==7)
        expected_stage=0 if count<2 else 1 if count<4 else 2
        assert run.state['curriculum_stage']==expected_stage
        assert len(confirmations)==(1 if count==7 else 0)
    assert confirmations==[(128,profile()['confirmation_seed'])]
    assert len(promoted)==1
    assert run.state['state']=='complete' and run.state['ready'] is True
    assert [(item['from'],item['to']) for item in run.state['stage_history']]==[(0,1),(1,2)]


def test_failed_confirmation_keeps_policy_unpublished_and_clears_streak(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    run.state.update(curriculum_stage=2,consecutive_passes=2,completed_iterations=700)
    def evaluate(episodes,seed,kind):
        value=report(2,episodes,kind!='confirmation')
        run.state['last_evaluation']={'kind':kind}
        return value
    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(run,'promote',lambda value:pytest.fail('Failed confirmation cannot publish'))
    assert not run.process_validation()
    assert run.state['consecutive_passes']==0 and not run.state['ready']


def test_saved_standalone_v5_chunk_preserves_new_update_count(tmp_path,monkeypatch):
    folder=tmp_path/'run';folder.mkdir()
    fake_checkpoint(folder/'resume.pt')
    (folder/'training_result.json').write_text(json.dumps({'completed_iterations':2,'num_envs':10240,
        'reward_revision':0,'curriculum_stage':0,'contract':{'robot':'q1_leg_lift_v5'}}))
    run=supervisor(tmp_path,monkeypatch,resume=True)
    assert run.state['completed_iterations']==2
    assert run.state['completed_transitions']==2*10240*24
    assert run.state['curriculum_stage']==0 and run.state['evaluations']==[]
    assert not run.state['ready']


def test_source_change_and_stale_result_cannot_mask_failed_worker(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    monkeypatch.setattr(monitor,'source_manifest',lambda config:{'fixture':'edited'})
    with pytest.raises(RuntimeError,match='sources changed'):run.assert_sources_unchanged()
    output=tmp_path/'result.json';output.write_text('{"completed_iterations":100}')
    before=monitor.timestamp(output)
    with pytest.raises(RuntimeError,match='not freshly written'):monitor.fresh_json(output,before,'worker result')
    output.write_text('invalid')
    with pytest.raises(RuntimeError,match='Invalid'):monitor.fresh_json(output,None,'worker result')


def test_chunk_retains_immutable_segment_and_resumable_checkpoint(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    run.profile['bootstrap']=None
    def child(command,phase):
        assert '--curriculum-stage' in command
        assert '--bootstrap' not in command
        fake_checkpoint(run.run/'resume.pt')
        contract={'robot':'q1_leg_lift_v5'}
        (run.run/'contract.json').write_text(json.dumps(contract))
        (run.run/'training_result.json').write_text(json.dumps({'completed_iterations':100,'num_envs':10240,
            'reward_revision':0,'curriculum_stage':0,'contract':contract,'checkpoint_sha256':monitor.sha256(run.run/'resume.pt')}))
    monkeypatch.setattr(run,'child',child)
    assert run.train_chunk()
    archived=run.run/'chunks/0000100_s0_r0'
    assert (archived/'resume.pt').read_bytes()==(run.run/'resume.pt').read_bytes()
    assert (archived/'contract.json').exists() and (archived/'training_result.json').exists()
    assert run.state['completed_iterations']==100


def test_bootstrap_discards_old_outputs_critic_optimizer_and_noise(tmp_path):
    """Exercise the real worker initializer without launching Isaac Kit."""
    import hashlib
    from types import SimpleNamespace
    import torch
    source=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/train_lift.py').read_text())
    selected=ast.Module(body=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in ('digest','initialize_actor_hidden')],type_ignores=[])
    namespace={'torch':torch,'hashlib':hashlib,'Path':Path,'policy_module':lambda runner:runner.actor,'checkpoint_for_runtime':lambda path:str(path)}
    exec(compile(selected,'train_lift_initializer','exec'),namespace)
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp=torch.nn.Sequential(torch.nn.Linear(110,8),torch.nn.ELU(),torch.nn.Linear(8,23))
            self.distribution=torch.nn.Module()
            self.distribution.std_param=torch.nn.Parameter(torch.full((23,),.12))
    old=Actor()
    with torch.no_grad():
        for parameter in old.parameters():parameter.fill_(3.)
    checkpoint=tmp_path/'old.pt'
    torch.save({'actor_state_dict':old.state_dict(),'critic_state_dict':{'poison':torch.ones(1)},'optimizer_state_dict':{'poison':1},'iter':641},checkpoint)
    actor=Actor();critic=torch.nn.Linear(110,1)
    critic_before={key:value.clone() for key,value in critic.state_dict().items()}
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
    runner=SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer,current_learning_iteration=641)
    result=namespace['initialize_actor_hidden'](runner,checkpoint)
    assert torch.all(actor.mlp[0].weight==3.)
    assert torch.all(actor.mlp[0].bias==3.)
    assert torch.all(actor.mlp(torch.randn(5,110))==0.)
    assert torch.all(actor.distribution.std_param==.12)
    assert all(torch.equal(value,critic_before[key]) for key,value in critic.state_dict().items())
    assert not optimizer.state and runner.current_learning_iteration==0
    assert result['actor_output_zeroed'] and result['critic']=='fresh' and result['optimizer']=='fresh'
