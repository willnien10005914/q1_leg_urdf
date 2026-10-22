"""CPU verification of V6 provenance, fresh budget, and independent full gates."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('upper_body_training_monitor', ROOT/'tools/run_upper_body_training.py')
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def profile():
    return monitor.validate_profile(json.loads((ROOT/'config/upper_body_v6.json').read_text()))


def report(stage=2, episodes=32, passed=True):
    return {'stage':stage,'episodes':episodes,'backend':'isaaclab','proxy':False,
            'score':1. if passed else .5,'passed':passed,
            'checks':{'native_swing_lift':passed,'upper_body.chest_peak':passed},
            'lower_body_checks':{'native_swing_lift':passed},'lower_body_passed':passed,
            'upper_body_checks':{'chest_peak':passed,'controlled_return':passed},
            'failed_gates':[] if passed else ['upper_body.chest_peak'],
            'cases':{name:{'episodes':episodes//len(monitor.STAGE_CASES[stage])}
                     for name in monitor.STAGE_CASES[stage]}}


def fake_checkpoint(path):
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('resume/data.pkl', b'checkpoint-test-payload'*100)


def supervisor(tmp_path, monkeypatch, resume=False):
    (tmp_path/'fixture.py').write_text('# fixed source\n')
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    monkeypatch.setattr(monitor,'source_manifest',lambda config:{'fixture.py':monitor.sha256(tmp_path/'fixture.py')})
    return monitor.Supervisor(profile(),tmp_path/'profile.json',tmp_path/'run',resume=resume)


def worker_functions():
    tree=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/train_upper_body.py').read_text())
    selected=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in
                             ('digest','validate_checkpoint_contract','validate_update_budget','initialize_complete_actor')],type_ignores=[])
    namespace={'torch':torch,'hashlib':hashlib,'Path':Path,'json':json,
               'policy_module':lambda runner:runner.actor,
               'checkpoint_for_runtime':lambda path:str(path),
               'reset_noise_std':lambda runner,std:runner.actor.distribution.std_param.data.fill_(std)}
    exec(compile(selected,'upper_body_worker_functions','exec'),namespace)
    return namespace


def test_fresh_twenty_thousand_budget_does_not_count_source_iterations():
    plan=profile()
    assert monitor.next_chunk_size(0,plan)==100
    assert monitor.next_chunk_size(1999,plan)==100
    assert monitor.next_chunk_size(2000,plan)==500
    assert monitor.next_chunk_size(2000,plan,stage=0)==100
    assert monitor.next_chunk_size(19999,plan)==1
    assert monitor.next_chunk_size(20000,plan)==0
    assert plan['num_envs']*plan['num_steps_per_env']*plan['max_iterations']==4_915_200_000
    budget=worker_functions()['validate_update_budget']
    assert budget(plan,19999,1)==20000
    for completed,iterations in [(19999,2),(20000,1),(-1,1),(0,0),(True,1)]:
        with pytest.raises(ValueError):budget(plan,completed,iterations)
    for key,value in [('num_envs',6144),('max_iterations',33115),('budget_mode','inherited_total'),
                      ('source_iteration',15000),('training_episode_lengths_s',[8,8,20]),('confirmation_episodes',32)]:
        changed=dict(plan);changed[key]=value
        with pytest.raises(ValueError):monitor.validate_profile(changed)


def test_lower_body_success_alone_never_accepts_v6():
    assert monitor.stage_gate_passed(report(0),0,32)
    assert monitor.stage_gate_passed(report(1),1,32)
    assert monitor.confirmation_passed(report(2,128))
    for key,value in [('upper_body_checks',{}),('upper_body_checks',{'chest':False}),('proxy',True),
                      ('backend','reference'),('score',float('nan')),('checks',{}),('failed_gates',['lift'])]:
        changed=report(2,128);changed[key]=value
        assert not monitor.confirmation_passed(changed)
    assert not monitor.confirmation_passed(report(2,32))
    assert not monitor.confirmation_passed(report(1,128))
    missing=report(2,128);del missing['cases']['sequence']
    assert not monitor.confirmation_passed(missing)


def test_curriculum_restarts_and_requires_two_two_three_plus_128(tmp_path, monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    assert run.state['completed_iterations']==0 and run.state['source_iteration']==13115
    assert run.state['curriculum_stage']==0 and not run.state['acceptance_inherited']
    assert (run.run/'source_snapshot/fixture.py').read_text()=='# fixed source\n'
    confirmations=[];promoted=[]
    def evaluate(episodes,seed,kind):
        value=report(run.state['curriculum_stage'],episodes)
        summary={'kind':kind,'stage':run.state['curriculum_stage'],'reward_revision':0,'score':1.,'passed':True,
                 'completed_iterations':run.state['completed_iterations'],'failed_gates':[]}
        run.state['evaluations'].append(summary);run.state['last_evaluation']=summary
        if kind=='confirmation':confirmations.append(episodes)
        return value
    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(run,'promote',lambda value:promoted.append(value))
    for count in range(1,8):
        run.state['completed_iterations']=count*100
        assert run.process_validation() is (count==7)
        assert run.state['curriculum_stage']==(0 if count<2 else 1 if count<4 else 2)
    assert confirmations==[128] and len(promoted)==1 and run.state['ready']


def test_failed_upper_confirmation_clears_streak_without_publishing(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    run.state.update(curriculum_stage=2,consecutive_passes=2,completed_iterations=700)
    def evaluate(episodes,seed,kind):
        value=report(2,episodes,kind!='confirmation')
        run.state['last_evaluation']={'kind':kind}
        return value
    monkeypatch.setattr(run,'evaluate',evaluate)
    monkeypatch.setattr(run,'promote',lambda value:pytest.fail('Failed upper-body confirmation cannot publish'))
    assert not run.process_validation()
    assert run.state['consecutive_passes']==0 and not run.state['ready']


def test_plateau_is_not_success_and_never_relaxes_upper_gates():
    values=[{'kind':'validation','stage':1,'reward_revision':0,'completed_iterations':100*(index+1),
             'score':score,'passed':False,'failed_gates':['upper_body.return']} for index,score in enumerate([.7,.701,.7,.702])]
    result=monitor.plateau_diagnostic(values,profile(),0,1)
    assert result['failed_gate_counts']=={'upper_body.return':4}
    assert not result['successful'] and not result['acceptance_gates_changed']
    assert monitor.plateau_diagnostic(values,profile(),1,1) is None
    assert monitor.plateau_diagnostic(values,profile(),0,2) is None


def test_source_changes_and_stale_json_stop_before_progress(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    (tmp_path/'fixture.py').write_text('# edited\n')
    with pytest.raises(RuntimeError,match='sources changed'):run.assert_sources_unchanged()
    output=tmp_path/'result.json';output.write_text('{"completed_iterations":100}')
    before=monitor.timestamp(output)
    with pytest.raises(RuntimeError,match='not freshly written'):monitor.fresh_json(output,before,'result')
    output.write_text('not json')
    with pytest.raises(RuntimeError,match='Invalid'):monitor.fresh_json(output,None,'result')


def test_training_chunk_retains_immutable_source_and_exact_new_count(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    run.profile['bootstrap']=None
    def child(command,phase):
        assert '--curriculum-stage' in command and '--initialize-policy' not in command
        fake_checkpoint(run.run/'resume.pt')
        contract={'robot':'q1_leg_upper_body_v6'}
        (run.run/'contract.json').write_text(json.dumps(contract))
        (run.run/'training_result.json').write_text(json.dumps({'completed_iterations':100,'num_envs':10240,
            'reward_revision':0,'curriculum_stage':0,'contract':contract,'checkpoint_sha256':monitor.sha256(run.run/'resume.pt'),
            'budget_mode':'new_updates','source_iteration':13115,
            'initialization':{'source_sha256':profile()['source_checkpoint_sha256']}}))
    monkeypatch.setattr(run,'child',child)
    assert run.train_chunk()
    archived=run.run/'chunks/0000100_s0_r0'
    assert (archived/'resume.pt').read_bytes()==(run.run/'resume.pt').read_bytes()
    assert run.state['completed_iterations']==100
    assert run.state['completed_transitions']==100*10240*24


def test_validation_only_retry_cannot_repeat_ppo_chunk(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    monkeypatch.setattr(monitor.time,'sleep',lambda seconds:None)
    calls=[]
    def child(command,phase):
        calls.append(phase)
        if len(calls)==1:raise RuntimeError('Native startup failed')
    monkeypatch.setattr(run,'child',child)
    run.validation_worker(['native','--evaluate'],'validating')
    assert calls==['validating','validating']
    assert run.state['completed_iterations']==0
    assert run.state['validation_worker_restart']['training_chunk_repeated'] is False
    with pytest.raises(ValueError):run.validation_worker(['native'],'training')


def test_complete_actor_preserves_output_but_resets_counter_critic_adam_and_std(tmp_path):
    functions=worker_functions()
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp=torch.nn.Sequential(torch.nn.Linear(110,8),torch.nn.ELU(),torch.nn.Linear(8,23))
            self.distribution=torch.nn.Module()
            self.distribution.std_param=torch.nn.Parameter(torch.full((23,),.12))
    old=Actor()
    with torch.no_grad():
        for parameter in old.parameters():parameter.fill_(.3)
    source=tmp_path/'source.pt'
    torch.save({'actor_state_dict':old.state_dict(),'critic_state_dict':{'poison':torch.ones(1)},
                'optimizer_state_dict':{'poison':1},'iter':13115,'infos':{'completed_iterations':13115}},source)
    contract={'robot':'q1_leg_lift_v5','schema':5,'urdf_sha256':'u'*64,'gait_sha256':'g'*64,
              'joint_names':[str(i) for i in range(23)],'action_size':23,'observation_size':110,
              'proxy':False,'control_sha256':'c'*64,'control_profile':'original'}
    (tmp_path/'contract.json').write_text(json.dumps(contract))
    expected={**contract,'robot':'q1_leg_upper_body_v6','schema':6,'control_sha256':'n'*64,'control_profile':'guarded'}
    plan={**profile(),'source_checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    actor=Actor();critic=torch.nn.Linear(110,1)
    critic_before={key:value.clone() for key,value in critic.state_dict().items()}
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
    runner=SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer,current_learning_iteration=13115)
    result=functions['initialize_complete_actor'](runner,source,expected,plan)
    for key,value in old.mlp.state_dict().items():assert torch.equal(value,actor.mlp.state_dict()[key])
    assert torch.all(actor.mlp[-1].weight==.3) and torch.all(actor.mlp[-1].bias==.3)
    assert torch.all(actor.distribution.std_param==.04)
    assert all(torch.equal(value,critic_before[key]) for key,value in critic.state_dict().items())
    assert not optimizer.state and runner.current_learning_iteration==0
    assert result['source_iteration']==13115 and result['completed_iterations']==0
    assert not result['actor_output_zeroed'] and not result['acceptance_inherited']
    assert result['critic']=='fresh' and result['optimizer']=='fresh'
    with pytest.raises(ValueError,match='SHA256'):
        functions['initialize_complete_actor'](runner,source,expected,{**plan,'source_checkpoint_sha256':'0'*64})
    with pytest.raises(ValueError,match='contract mismatch'):
        functions['validate_checkpoint_contract'](contract,expected)


def publication_history(run,monkeypatch,count=3):
    sources=run.state['source_sha256']
    sources.update({
        'source/leg_humanoid_lab/leg_humanoid_lab/upper_body_env.py':'e'*64,
        'source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py':'b'*64,
        'source/leg_humanoid_lab/leg_humanoid_lab/upper_body_evaluation.py':'v'*64})
    monkeypatch.setattr(monitor,'source_manifest',lambda config:dict(sources))
    run.state.update(curriculum_stage=2,consecutive_passes=3,completed_iterations=300)
    contract={'robot':'q1_leg_upper_body_v6','schema':6,'urdf_sha256':'u'*64,
              'gait_sha256':'g'*64,'control_sha256':'c'*64}
    for index in range(count):
        completed=(index+1)*100
        archive=run.run/'chunks'/f'{completed:07d}_s2_r0';archive.mkdir(parents=True)
        fake_checkpoint(archive/'resume.pt')
        (archive/'contract.json').write_text(json.dumps(contract))
        checkpoint_sha=monitor.sha256(archive/'resume.pt')
        value={**report(2,32),'completed_iterations':completed,'evaluation_seed':500000+completed,
               'checkpoint_sha256':checkpoint_sha,'robot':contract['robot'],'schema':6,
               'urdf_sha256':contract['urdf_sha256'],'gait_sha256':contract['gait_sha256'],
               'control_sha256':contract['control_sha256'],'upper_body_control_sha256':contract['control_sha256'],
               'environment_sha256':sources['source/leg_humanoid_lab/leg_humanoid_lab/upper_body_env.py'],
               'base_environment_sha256':sources['source/leg_humanoid_lab/leg_humanoid_lab/interactive_env.py'],
               'upper_body_evaluator_sha256':sources['source/leg_humanoid_lab/leg_humanoid_lab/upper_body_evaluation.py']}
        result={'completed_iterations':completed,'curriculum_stage':2,'reward_revision':0,'budget_mode':'new_updates',
                'checkpoint_sha256':checkpoint_sha,'initialization':{'source_sha256':profile()['source_checkpoint_sha256']}}
        (archive/'training_result.json').write_text(json.dumps(result))
        path=run.run/f'evaluation_{completed}_s2_{value["evaluation_seed"]}.json'
        path.write_text(json.dumps(value))
        run.state['evaluations'].append({'kind':'validation','stage':2,'reward_revision':0,'passed':True,'score':1.,
            'completed_iterations':completed,'seed':value['evaluation_seed'],'checkpoint_sha256':checkpoint_sha,
            'report':str(path),'report_sha256':monitor.sha256(path),'source_sha256':dict(sources)})
    return contract


def test_publication_rejects_three_success_summaries_with_only_two_full_reports(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    publication_history(run,monkeypatch,2)
    with pytest.raises(RuntimeError,match='three preceding complete'):
        run.preceding_stage2_validations()


def test_publication_embeds_three_full_reports_and_rejects_tampered_evidence(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    publication_history(run,monkeypatch)
    proofs=run.preceding_stage2_validations()
    assert len(proofs)==3
    assert [item['completed_iterations'] for item in proofs]==[100,200,300]
    assert all(item['upper_body_checks']['controlled_return'] for item in proofs)
    assert all(item['lower_body_checks']['native_swing_lift'] for item in proofs)
    assert all(item['validation_provenance']['report_sha256'] for item in proofs)
    path=Path(run.state['evaluations'][0]['report'])
    value=json.loads(path.read_text());value['upper_body_checks']['controlled_return']=False
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='incomplete, changed'):
        run.preceding_stage2_validations()
    # Updating a summary SHA cannot hide native gate failure in the sealed body.
    run.state['evaluations'][0]['report_sha256']=monitor.sha256(path)
    with pytest.raises(RuntimeError,match='native gates'):
        run.preceding_stage2_validations()


def test_publication_rejects_same_seed_or_wrong_archived_weights(tmp_path,monkeypatch):
    run=supervisor(tmp_path,monkeypatch)
    publication_history(run,monkeypatch)
    first,second=run.state['evaluations'][:2]
    path=Path(second['report']);value=json.loads(path.read_text())
    value['evaluation_seed']=first['seed'];path.write_text(json.dumps(value))
    second.update(seed=first['seed'],report_sha256=monitor.sha256(path))
    with pytest.raises(RuntimeError,match='native gates'):
        run.preceding_stage2_validations()
    value['evaluation_seed']=500200;path.write_text(json.dumps(value))
    second.update(seed=500200,report_sha256=monitor.sha256(path))
    checkpoint=run.run/'chunks/0000200_s2_r0/resume.pt'
    with zipfile.ZipFile(checkpoint,'a') as archive:archive.writestr('changed',b'changed after evaluation')
    with pytest.raises(RuntimeError,match='native gates'):
        run.preceding_stage2_validations()
