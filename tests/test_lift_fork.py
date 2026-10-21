"""Exercise the real actor-only control fork without launching Isaac Kit."""
import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT=Path(__file__).resolve().parents[1]


def worker_functions():
    source=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/train_lift.py').read_text())
    names={'digest','validate_checkpoint_contract','validate_update_budget','initialize_control_fork'}
    functions=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in names]
    runtime=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/runtime_compat.py').read_text())
    functions.extend(node for node in runtime.body if isinstance(node,ast.FunctionDef) and node.name=='reset_noise_std')
    namespace={'Path':Path,'hashlib':hashlib,'json':json,'torch':torch,
               'checkpoint_for_runtime':lambda path:str(path),'policy_module':lambda runner:runner.actor}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'native_lift_control_fork','exec'),namespace)
    return namespace


def contract(control='old'):
    return {'robot':'q1_leg_lift_v5','schema':5,'urdf_sha256':'u'*64,'gait_sha256':'g'*64,
            'joint_names':[f'joint_{index}' for index in range(23)],'action_size':23,'observation_size':110,
            'control_sha256':('a' if control=='old' else 'b')*64,'control_profile':control,
            'motor':{'control':control},'proxy':False}


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=torch.nn.Sequential(torch.nn.Linear(110,8),torch.nn.ELU(),torch.nn.Linear(8,23))
        self.distribution=torch.nn.Module()
        self.distribution.std_param=torch.nn.Parameter(torch.full((23,),.25))


class LegacyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Sequential(torch.nn.Linear(110,8),torch.nn.ELU(),torch.nn.Linear(8,23))
        self.critic=torch.nn.Linear(110,1)
        self.std=torch.nn.Parameter(torch.full((23,),.25))


def checkpoint(tmp_path,format='modern',iteration=1805):
    model=Actor() if format=='modern' else LegacyActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():parameter.fill_(3.)
    payload={'iter':iteration,'infos':{'completed_iterations':iteration},
             'optimizer_state_dict':{'poison':'source Adam must never load'}}
    if format=='modern':
        payload.update(actor_state_dict=model.state_dict(),critic_state_dict={'poison':torch.ones(1)})
    else:
        payload['model_state_dict']=model.state_dict()
    path=tmp_path/'source.pt';torch.save(payload,path)
    (tmp_path/'contract.json').write_text(json.dumps(contract()))
    return path,payload


def fresh_runner(format='modern'):
    actor=Actor() if format=='modern' else LegacyActorCritic()
    critic=torch.nn.Linear(110,1) if format=='modern' else actor.critic
    parameters=list(actor.parameters())+(list(critic.parameters()) if format=='modern' else [])
    optimizer=torch.optim.Adam(parameters)
    return SimpleNamespace(actor=actor,critic=critic,alg=SimpleNamespace(optimizer=optimizer),current_learning_iteration=0)


@pytest.mark.parametrize('source_format',['modern','legacy'])
@pytest.mark.parametrize('target_format',['modern','legacy'])
def test_fork_keeps_full_actor_output_with_fresh_critic_adam_and_explicit_noise(tmp_path,source_format,target_format):
    functions=worker_functions();path,_=checkpoint(tmp_path,source_format)
    runner=fresh_runner(target_format)
    critic_before={key:value.clone() for key,value in runner.critic.state_dict().items()}
    result=functions['initialize_control_fork'](runner,path,contract('new'),1805,.12)
    actor=runner.actor.mlp if target_format=='modern' else runner.actor.actor
    assert all(torch.all(parameter==3.) for parameter in actor.parameters())
    assert torch.all(actor[-1].weight==3.) and torch.all(actor[-1].bias==3.)
    assert all(torch.equal(value,critic_before[key]) for key,value in runner.critic.state_dict().items())
    assert not runner.alg.optimizer.state
    noise=runner.actor.distribution.std_param if target_format=='modern' else runner.actor.std
    assert torch.allclose(noise,torch.full((23,),.12))
    assert runner.current_learning_iteration==1805
    assert result['policy']=='control_fork' and result['inherited_completed_iterations']==1805
    assert result['source_payload_iteration']==1805 and not result['actor_output_zeroed']
    assert result['source_control_sha256']==contract()['control_sha256']
    assert result['new_control_sha256']==contract('new')['control_sha256']
    assert result['source_contract_sha256']==hashlib.sha256((tmp_path/'contract.json').read_bytes()).hexdigest()
    assert result['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not result['acceptance_inherited'] and result['curriculum_restarts_at']==0


def test_changed_controller_requires_explicit_fork_but_geometry_never_changes():
    validate=worker_functions()['validate_checkpoint_contract']
    validate(contract(),contract())
    with pytest.raises(ValueError,match='explicit --fork-policy'):
        validate(contract(),contract('new'))
    validate(contract(),contract('new'),fork_policy=True)
    for field,value in [('urdf_sha256','other'),('gait_sha256','other'),('action_size',22),
                        ('observation_size',109),('joint_names',['different']*23),('proxy',True)]:
        old=contract();old[field]=value
        with pytest.raises(ValueError):validate(old,contract('new'),fork_policy=True)


@pytest.mark.parametrize('iteration',[0,1804,1806,None,True])
def test_inherited_counter_must_match_actual_source_payload_before_any_actor_copy(tmp_path,iteration):
    functions=worker_functions();path,_=checkpoint(tmp_path,iteration=iteration)
    runner=fresh_runner();before={key:value.clone() for key,value in runner.actor.state_dict().items()}
    with pytest.raises(ValueError,match='source checkpoint iter'):
        functions['initialize_control_fork'](runner,path,contract('new'),1805,.12)
    assert all(torch.equal(value,before[key]) for key,value in runner.actor.state_dict().items())
    assert runner.current_learning_iteration==0 and not runner.alg.optimizer.state


def test_counter_conversion_and_infos_cannot_fabricate_lineage(tmp_path):
    functions=worker_functions();path,payload=checkpoint(tmp_path)
    payload['infos']['completed_iterations']=1804;torch.save(payload,path)
    with pytest.raises(ValueError,match='infos contradict'):
        functions['initialize_control_fork'](fresh_runner(),path,contract('new'),1805,.12)
    payload['infos']['completed_iterations']=1805;torch.save(payload,path)
    converted=tmp_path/'converted.pt';payload['iter']=0;torch.save(payload,converted)
    functions['checkpoint_for_runtime']=lambda unused:str(converted)
    with pytest.raises(ValueError,match='conversion changed'):
        functions['initialize_control_fork'](fresh_runner(),path,contract('new'),1805,.12)


def test_missing_output_or_nonfresh_optimizer_cannot_claim_a_complete_actor_fork(tmp_path):
    functions=worker_functions();path,payload=checkpoint(tmp_path)
    del payload['actor_state_dict']['mlp.2.weight'];torch.save(payload,path)
    with pytest.raises(ValueError,match='complete actor MLP'):
        functions['initialize_control_fork'](fresh_runner(),path,contract('new'),1805,.12)
    path,_=checkpoint(tmp_path)
    runner=fresh_runner();parameter=next(runner.actor.parameters())
    runner.alg.optimizer.state[parameter]={'step':torch.tensor(10.)}
    with pytest.raises(ValueError,match='fresh optimizer'):
        functions['initialize_control_fork'](runner,path,contract('new'),1805,.12)


def test_inherited_updates_remain_inside_the_total_15000_budget():
    budget=worker_functions()['validate_update_budget']
    assert budget({'max_iterations':15000},1805,100)==1905
    assert budget({'max_iterations':15000},1805,13195)==15000
    with pytest.raises(ValueError,match='total authorized'):
        budget({'max_iterations':15000},1805,13196)
    with pytest.raises(ValueError,match='total authorized'):
        budget({'max_iterations':20000},1805,13196)


def test_true_resume_preserves_learned_noise_and_uses_adam_loading(tmp_path):
    """Execute the real worker checkpoint branch, excluding the simulator."""
    path,payload=checkpoint(tmp_path)
    current=contract()
    functions=worker_functions();runner=fresh_runner()
    calls=[]
    def load(runner,path,*,load_optimizer,load_iteration):
        calls.append((load_optimizer,load_iteration))
        runner.actor.load_state_dict(payload['actor_state_dict'])
        runner.current_learning_iteration=payload['iter']
    functions.update(args=SimpleNamespace(checkpoint=path,bootstrap=False,fork_policy=False,evaluate=False,completed=1805),
                     expected=current,runner=runner,load_checkpoint=load)
    source=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/train_lift.py').read_text())
    checkpoint_branch=next(node for node in ast.walk(source) if isinstance(node,ast.If)
                           and isinstance(node.test,ast.Attribute) and ast.unparse(node.test)=='args.checkpoint')
    exec(compile(ast.Module(body=[copy.deepcopy(checkpoint_branch)],type_ignores=[]),'true_native_resume','exec'),functions)
    assert calls==[(True,True)]
    assert torch.all(runner.actor.distribution.std_param==3.),'ordinary resume silently reset learned noise'
    assert runner.current_learning_iteration==1805


def test_real_cli_requires_explicit_isolated_stage_zero_fork(tmp_path):
    source=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/train_lift.py').read_text())
    setup=[]
    guards=[]
    for node in source.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='AppLauncher.add_app_launcher_args':break
        if isinstance(node,(ast.Assign,ast.Expr)):setup.append(node)
    namespace={'argparse':argparse,'Path':Path,'__doc__':'lift fork CLI test'}
    exec(compile(ast.Module(body=setup,type_ignores=[]),'lift_fork_cli','exec'),namespace)
    for node in source.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='launcher' for target in node.targets):break
        if isinstance(node,ast.If):guards.append(node)
    parser=namespace['p']
    base=['--output-dir',str(tmp_path/'new'),'--checkpoint',str(tmp_path/'old/resume.pt'),'--completed','1805']
    for extra in (['--fork-policy','--bootstrap'],['--fork-policy','--evaluate']):
        with pytest.raises(SystemExit):parser.parse_args(base+extra)
    for extra in (['--completed','0'],['--curriculum-stage','1'],['--output-dir',str(tmp_path/'old')]):
        namespace['args']=parser.parse_args(base+['--fork-policy']+extra)
        with pytest.raises(SystemExit):
            exec(compile(ast.Module(body=guards,type_ignores=[]),'lift_fork_cli_guards','exec'),namespace)
    namespace['args']=parser.parse_args(base+['--fork-policy'])
    exec(compile(ast.Module(body=guards,type_ignores=[]),'lift_fork_cli_guards','exec'),namespace)
