"""Exercise stage scheduling without launching Isaac Sim."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('native_curriculum',Path(__file__).resolve().parents[1]/'tools/native_curriculum.py')
curriculum=importlib.util.module_from_spec(spec)
spec.loader.exec_module(curriculum)


def test_forced_budget_and_previous_stage_warm_start(tmp_path,monkeypatch):
    (tmp_path/'config').mkdir()
    (tmp_path/'outputs').mkdir()
    contract={'robot':'test-native','urdf_sha256':'same-source','proxy':False}
    (tmp_path/'config/robot_contract.json').write_text(json.dumps(contract))
    passed={'passed':True,'urdf_sha256':'same-source','proxy':False,'episodes':32}
    for skill in ('stand','sidestep'):
        folder=tmp_path/'checkpoints/isaac'/skill
        folder.mkdir(parents=True)
        (folder/'contract.json').write_text(json.dumps(contract))
        (folder/'evaluation.json').write_text(json.dumps(passed))
        (folder/'latest.pt').write_bytes(b'old-policy')
        (folder/'model_100.pt').write_bytes(b'old-snapshot')
    config=tmp_path/'config/budget.json'
    config.write_text(json.dumps({'contract':'test-native','num_envs':6144,'force_train':True,'confirmation_episodes':32,'select_current_run_only':True,'stages':[{'skill':'stand','iterations':8000},{'skill':'sidestep','iterations':8000,'warm_start':'previous_stage'}]}))
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        skill=command[command.index('--skill')+1]
        report=tmp_path/'checkpoints/isaac'/skill/'evaluation.json'
        previous=report.stat().st_mtime_ns
        report.write_text(json.dumps(passed))
        os.utime(report,ns=(previous+1,previous+1))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(curriculum,'ROOT',tmp_path)
    monkeypatch.setattr(curriculum.subprocess,'run',run)
    monkeypatch.setattr(curriculum.sys,'argv',['native_curriculum.py','--config',str(config),'--attempts','1'])
    assert curriculum.main()==0
    training=[c for c in calls if Path(c[1]).name=='train_native.sh']
    assert len(training)==2
    for command in training:
        assert command[command.index('--num_envs')+1]=='6144'
        assert command[command.index('--max_iterations')+1]=='8000'
    assert training[1][training[1].index('--checkpoint')+1]==str(tmp_path/'checkpoints/isaac/stand/latest.pt')
    selection=[c for c in calls if Path(c[1]).name=='select_policy.sh']
    assert len(selection)==2
    assert all('--current-run-only' in c for c in selection)
    assert len(list((tmp_path/'checkpoints/isaac/stand/runs').glob('*/model_100.pt')))==1
    final=json.loads((tmp_path/'outputs/training_status.json').read_text())
    assert final['state']=='passed' and final['envs']==6144 and final['iterations']==8000
