"""Lift acceptance rejects stationary, one-foot and post-reset fake success."""
from types import SimpleNamespace

import torch

from leg_humanoid_lab.lift_evaluation import STAGE_CASES, check_lift_gates, evaluate_lift


def complete_report(stage=0):
    cases={}
    for name in STAGE_CASES[stage]:
        item={'episodes':4,'survival_rate':1.,'upright':.995,'final_stand_speed_m_s':.01,'stop_observed_rate':1.,
              'stance_slip_m_s':.03,'foot_peak_clearance_m_per_foot':[.035,.035],
              'foot_lift_success_rate_per_foot':[.8,.8],'high_swing_samples_per_foot':[40.,40.]}
        if stage==1 and name!='stand':
            item['directions']={side:{'lead_peak_clearance_m':.035,'lead_lift_success_rate':.8,
                                     'lead_high_swing_samples':10.,'trailing_peak_clearance_m':.035,
                                     'trailing_lift_success_rate':.8,'trailing_high_swing_samples':10.}
                                for side in (('left','right') if name=='pair' else (name,))}
        cases[name]=item
    return {'stage':stage,'cases':cases}


def test_stage_specific_complete_reports_pass_and_missing_or_nonfinite_data_fail():
    for stage in (0,1):
        report=complete_report(stage)
        assert all(check_lift_gates(report).values())
        report['cases']['stand']['upright']=float('nan')
        assert not all(check_lift_gates(report).values())
        report=complete_report(stage)
        del report['cases'][STAGE_CASES[stage][-1]]
        assert not all(check_lift_gates(report).values())
    assert not all(check_lift_gates({'stage':2}).values())


def test_stationary_one_foot_and_missing_high_swing_cannot_pass():
    for field,value in (('foot_peak_clearance_m_per_foot',[0.,0.]),
                        ('foot_lift_success_rate_per_foot',[.95,0.]),
                        ('high_swing_samples_per_foot',[40.,0.])):
        report=complete_report()
        report['cases']['inplace'][field]=value
        assert not all(check_lift_gates(report).values())
    report=complete_report(1)
    # Global peaks from a good left maneuver cannot hide a dragging trailing
    # foot on the right maneuver of the same pair.
    report['cases']['pair']['directions']['right']['trailing_peak_clearance_m']=.002
    assert not all(check_lift_gates(report).values())


def transport_fixture(stage=0, lift='both', reset_index=None, report_done=True, count=4):
    """Synthetic tensor transport only; not a physics or learned-policy model."""
    gravity=torch.zeros(count,3);gravity[:,2]=-1.
    data=SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,.86]]*count),
                         root_lin_vel_b=torch.zeros(count,3),projected_gravity_b=gravity)
    clearance=torch.zeros(count,2);loaded=torch.ones(count,2,dtype=torch.bool)
    pending=torch.zeros(count,dtype=torch.long);pending_phase=torch.zeros(count)
    mode_history=[];command_history=[]

    def control(mode,command,phase):
        pending[:]=mode;pending_phase[:]=phase
        mode_history.append(mode.clone());command_history.append(command.clone())

    env=SimpleNamespace(num_envs=count,device='cpu',step_dt=.02,max_episode_length=1025,
        cfg=SimpleNamespace(allow_proxy=False,training_mode=True),
        library={'sidestep_duration_s':1.6,'walk_cycle_s':.9,'contract':'synthetic_test_transport'},
        robot=SimpleNamespace(data=data),reference_height=torch.zeros(count,2),
        reference_contact=torch.ones(count,2,dtype=torch.bool),
        episode_length_buf=torch.zeros(count,dtype=torch.long),
        reset_terminated=torch.zeros(count,dtype=torch.bool),set_manual_control=control,
        foot_step_metrics=lambda:(clearance,torch.zeros(count,2),loaded,None))

    class Wrapped:
        resets=0;steps=0
        def reset(self):
            self.resets+=1
            env.episode_length_buf.zero_()
            return torch.zeros(count,1),{}
        def get_observations(self): return torch.zeros(count,1)
        def step(self,action):
            self.steps+=1;env.episode_length_buf+=1
            env.reset_terminated.zero_()
            swing=torch.zeros(count,2,dtype=torch.bool)
            walking=pending==1
            swing[walking,0]=torch.sin(2*torch.pi*pending_phase[walking])>0
            swing[walking,1]=~swing[walking,0]
            for mode,lead in ((2,0),(3,1)):
                selected=pending==mode
                swing[selected,lead]=(pending_phase[selected]>.18)&(pending_phase[selected]<.44)
                swing[selected,1-lead]=(pending_phase[selected]>.66)&(pending_phase[selected]<.90)
            env.reference_contact[:]=~swing
            env.reference_height[:]=torch.where(swing,.04,.003)
            clearance.zero_();loaded.fill_(True)
            if lift!='none':
                actual=swing.clone()
                if lift=='left': actual[:,1]=False
                clearance[actual]=.04
                if lift!='loaded': loaded[actual]=False
            done=torch.zeros(count,dtype=torch.bool)
            if reset_index is not None and self.steps>=10:
                # The done frame and every later frame show an attractive
                # auto-reset pose. None may count toward the first episode.
                clearance[reset_index]=.25
                loaded[reset_index]=False
                env.reference_height[reset_index]=.04
                env.reference_contact[reset_index]=False
                if self.steps==10:
                    done[reset_index]=report_done
                    env.reset_terminated[reset_index]=True
                    env.episode_length_buf[reset_index]=0
            return self.get_observations(),torch.zeros(count),done,{}

    wrapped=Wrapped()
    runner=SimpleNamespace(get_inference_policy=lambda device:lambda obs:torch.zeros(count,23))
    return env,wrapped,runner,mode_history,command_history


def test_full_collector_rejects_stationary_and_one_foot_even_with_stable_root():
    for lift in ('none','left','loaded'):
        env,wrapped,runner,_,_=transport_fixture(lift=lift)
        report=evaluate_lift(runner,wrapped,env,episodes=4,seed=99,stage=0)
        assert not report['passed']
        assert report['cases']['inplace']['survival_rate']==1.
        if lift!='loaded':
            assert report['cases']['inplace']['foot_peak_clearance_m_per_foot'][1]==0.
        else:
            assert report['cases']['inplace']['foot_peak_clearance_m_per_foot'][1]>.035
        assert report['cases']['inplace']['foot_lift_success_rate_per_foot'][1]==0.
        assert wrapped.resets==1
        assert env.cfg.training_mode is True
        assert not hasattr(env,'manual_mode')


def test_first_episode_excludes_done_and_all_attractive_auto_reset_samples():
    env,wrapped,runner,_,_=transport_fixture(lift='none',reset_index=1)
    report=evaluate_lift(runner,wrapped,env,episodes=4,stage=0)
    first=report['per_episode'][1]
    assert first['case']=='inplace' and not first['survived']
    assert abs(first['duration_s']-.2)<1e-6
    assert first['foot_peak_clearance_m_per_foot']==[0.,0.]
    assert first['high_swing_samples_per_foot']==[0.,0.]
    assert report['cases']['inplace']['survival_rate']==.5
    assert not report['passed'] and wrapped.resets==1 and wrapped.steps==1000


def test_balanced_stage1_collector_keeps_left_right_pair_and_recovery_distinct():
    env,wrapped,runner,history,commands=transport_fixture(stage=1)
    report=evaluate_lift(runner,wrapped,env,episodes=4,stage=1)
    # A passing synthetic transport verifies collection and gates only.
    assert report['passed'] and report['score']==1.
    assert set(report['cases'])=={'stand','left','right','pair'}
    for index,expected in ((0,[0]),(1,[0,2,4,0]),(2,[0,3,4,0]),(3,[0,2,3,4,0])):
        changed=[]
        for row in history:
            value=int(row[index])
            if not changed or value!=changed[-1]: changed.append(value)
        assert changed==expected
    assert all(bool((command==0).all()) for command in commands)
    assert wrapped.resets==1 and report['reset_between_transitions'] is False
    assert 'stage 2' in report['scope']
    assert report['evaluation_seed']==70000 and report['episodes']==4


def test_unreported_episode_counter_reset_also_ends_first_episode():
    env,wrapped,runner,_,_=transport_fixture(lift='none',reset_index=1,report_done=False)
    report=evaluate_lift(runner,wrapped,env,episodes=4,stage=0)
    first=report['per_episode'][1]
    assert not first['survived'] and first['foot_peak_clearance_m_per_foot']==[0.,0.]
    assert abs(first['duration_s']-.2)<1e-6
    assert not report['passed']


def test_stop_observation_rate_has_an_explicit_gate_at_fourteen_of_sixteen():
    report=complete_report()
    item=report['cases']['stand']
    item.update(episodes=16,survival_rate=14/16,stop_observed_rate=14/16,
                final_stand_speed_m_s=.02)
    assert all(check_lift_gates(report).values())
    item['stop_observed_rate']=13/16
    checks=check_lift_gates(report)
    assert not checks['stand.stop_observed']
    assert checks['stand.stop'] and checks['stand.survival']
    item['stop_observed_rate']=0.
    assert not check_lift_gates(report)['stand.stop_observed']


def test_collector_averages_observed_stop_windows_without_fabricated_death_speed():
    for missing_per_case in (2,3):
        # Alternating tasks mean indices 0..3 remove exactly two observations
        # from each of the two 16-episode cases; 0..5 remove three each.
        env,wrapped,runner,_,_=transport_fixture(count=32,reset_index=list(range(2*missing_per_case)))
        report=evaluate_lift(runner,wrapped,env,episodes=32,stage=0)
        for name in ('stand','inplace'):
            item=report['cases'][name]
            assert item['stop_observed_episodes']==16-missing_per_case
            assert item['stop_observed_rate']==(16-missing_per_case)/16
            assert item['final_stand_speed_m_s']==0.
            assert report['checks'][f'{name}.stop']
        assert report['per_episode'][0]['final_stand_speed_m_s']==1e6
        assert not report['per_episode'][0]['stop_observed']
        if missing_per_case==2:
            assert report['passed']
        else:
            assert not report['passed']
            assert not report['checks']['stand.stop_observed']
            assert not report['checks']['inplace.stop_observed']
