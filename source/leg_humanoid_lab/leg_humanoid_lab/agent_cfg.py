"""Shared RSL-RL actor/critic architecture for training and live playback."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg,RslRlPpoActorCriticCfg,RslRlPpoAlgorithmCfg

@configclass
class PPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env=24
    max_iterations=2000
    save_interval=100
    experiment_name='q1_leg_native'
    empirical_normalization=False
    policy=RslRlPpoActorCriticCfg(init_noise_std=.25,actor_obs_normalization=False,critic_obs_normalization=False,actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],activation='elu')
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.003,num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.)
