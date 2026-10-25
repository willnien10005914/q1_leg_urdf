import torch
from leg_humanoid_lab.balance_objective import support_capture_error,graded_cost

def sample(com_xy=(0.,0.),velocity=(0.,0.),loaded=(True,True)):
 com=torch.tensor([[*com_xy,.8]])
 vel=torch.tensor([[*velocity,0.]])
 feet=torch.tensor([[[0.,.1,0.],[0.,-.1,0.]]])
 return support_capture_error(com,vel,feet,torch.tensor([loaded]))

def test_supported_static_com_has_no_capture_error():
 assert sample().item()==0

def test_dynamic_velocity_can_leave_support_even_when_com_is_centered():
 assert sample(velocity=(1.,0.)).item()>.2

def test_only_actual_loaded_foot_provides_support():
 assert sample(com_xy=(0.,-.1)).item()==0
 assert sample(com_xy=(0.,-.1),loaded=(True,False)).item()>.1
 assert sample(loaded=(False,False)).item()>=.19

def test_capture_cost_retains_gradient_outside_support():
 value=torch.tensor([.08],requires_grad=True)
 cost=graded_cost(value,.04);cost.backward()
 assert value.grad.item()>0

def test_left_right_mirror_has_same_balance_error():
 assert torch.allclose(sample(com_xy=(0.,.14),loaded=(True,False)),sample(com_xy=(0.,-.14),loaded=(False,True)))
