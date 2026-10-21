import math
import pytest
from leg_humanoid_lab.interactive_control import KeyboardController

def test_velocity_limits_expiry_and_stop():
    c=KeyboardController()
    c.velocity(100,100,100,0)
    state=c.tick(.02,.02)
    assert 0<state['command'][0]<.22 and 0<state['command'][1]<.18
    for i in range(150):state=c.tick(.02,1+i*.02)
    assert max(map(abs,state['command']))<.001
    with pytest.raises(ValueError):c.velocity(math.nan,0,0,4)

def test_pair_has_finite_order_then_stands_without_teleport():
    c=KeyboardController(sidestep_duration=.2)
    assert c.sequence(0)
    assert not c.velocity(.2,0,0,.1)
    modes=[]
    for i in range(150):modes.append(c.tick(.02,i*.02)['mode'])
    assert 'sidestep_left' in modes and 'sidestep_right' in modes
    assert modes.index('sidestep_left')<modes.index('sidestep_right')
    assert modes[-1]=='stand' and not c.sequence_active

def test_stop_interrupts_pair_and_new_velocity_only_after_recovery():
    c=KeyboardController();c.sequence(0)
    for i in range(30):c.tick(.02,i*.02)
    c.stop(.6)
    assert c.tick(.02,.62)['mode']=='recover'
    assert not c.sequence_active

def test_pair_remains_locked_until_recovery_finishes():
    c=KeyboardController(sidestep_duration=.2);c.sequence(0)
    for i in range(60):state=c.tick(.02,i*.02)
    assert state['mode']=='recover' and state['sequence_active']
    assert not c.velocity(.2,0,0,1.2) and not c.sequence(1.2)
    for i in range(60,110):state=c.tick(.02,i*.02)
    assert state['mode']=='stand' and not state['sequence_active']
    assert c.velocity(.2,0,0,2.2)
