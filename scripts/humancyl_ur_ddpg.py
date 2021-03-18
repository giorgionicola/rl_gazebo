import numpy as np
import gym
import rl_gazebo_env
import os
import datetime

import tensorflow as tf
from utils import Utils

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

env_name = 'HumanCylRobotEnv-v0'
env_ = gym.make(env_name)

ep_max_timesteps = env_.max_timesteps
env = DummyVecEnv([lambda: env_])

# the noise objects for DDPG
n_actions = 6
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                            sigma=0.15 * np.ones(n_actions),
                                            theta=0.3 * np.ones(n_actions))
layers = [400 for i in range(10)]
act_fun = 'relu'

gamma = 0.99
actor_lr = 1e-3
critic_lr = 1e-4
batch_size = 128
memory_limit = 200000
nb_rollout_steps = 60
nb_train_steps = 1

utils = Utils(env_name)
utils.create_log_model_dir(layers=layers,
                           act_fun=act_fun,
                           actor_lr=actor_lr,
                           critic_lr=critic_lr,
                           mem_size=memory_limit,
                           batch=batch_size,
                           n_rollout=nb_rollout_steps,
                           train_step=nb_train_steps)


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=layers,
                                           act_fun=tf.nn.relu,
                                           layer_norm=False,
                                           feature_extraction="mlp")


model = DDPG(policy=CustomPolicy,
             env=env,
             gamma=gamma,
             actor_lr=actor_lr,
             critic_lr=critic_lr,
             batch_size=batch_size,
             param_noise=None,
             action_noise=action_noise,
             memory_limit=memory_limit,
             nb_rollout_steps=nb_rollout_steps,
             nb_train_steps=nb_train_steps,
             verbose=1,
             tensorboard_log=utils.log_dir_path,
             normalize_observations=True
             )

total_training_steps = int(5e6)
step_checkpoints = 50000
n_checkpoints = int(total_training_steps / step_checkpoints)
last_model_ep = 0

# utils.save_params_simulation(algo_name='DDPG',
#                              model=model,
#                              env=env_,
#                              env_name=env_name,
#                              layers=layers,
#                              act_fun=act_fun,
#                              action_noise=action_noise,
#                              training_steps=total_training_steps)

for i in range(n_checkpoints):
    model.learn(total_timesteps=step_checkpoints,
                reset_num_timesteps=False)
    model.save(utils.save_dir_path + "/model_step" + str(last_model_ep + int(step_checkpoints/1000))+'k')
    last_model_ep = int(step_checkpoints/1000) * (i + 1)
