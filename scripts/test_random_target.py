import gym
import rl_gazebo_env
import os
from stable_baselines import DDPG
from utils_tests import Utils
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import json
import tensorflow as tf
import numpy as np
from termcolor import cprint
import csv
import datetime

env_name = 'Icra2020Env-v0'
model_name = 'nn_10x400_relu_ac_0.001_cr_0.0001_mem_200k_batch_128_roll_60_train_1'
model_date = '2019-09-07 23:10:31.660952'
model_step = '4000'

env_model_fold = env_name
model_path = env_model_fold + '/' + model_name + '/' + model_date

models_path = model_path + "/model_step" + str(model_step) + "k.pkl"
params_path = model_path + '/params_0.json'

env_ = gym.make('HumanCylRobotEnv-v0')
env_.training=False
env_.random_target = True
env_.max_timesteps = 5*env_.max_timesteps
env = DummyVecEnv([lambda: env_])

with open(params_path) as json_file:
    params = json.load(json_file)


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=params['model'][0]['policy']['layers'],
                                           act_fun=tf.nn.relu,
                                           layer_norm=False,
                                           feature_extraction="mlp")


model = DDPG.load(models_path, policy=CustomPolicy)

test_episodes = 5000

collision_counter = 0
success_counter = 0
timeover_counter = 0

success_target = []
collision_target = []
timeover_target = []

t_policy= []

for ep in range(test_episodes):

    if ep%100 ==0:
        print(ep)

    rnd_sample = np.random.rand(3)
    rand_target = [env_.target_range[0][0] + rnd_sample[0] * (env_.target_range[0][1] - env_.target_range[0][0]),
                   env_.target_range[1][ 0] + rnd_sample[1] * (env_.target_range[1][1] - env_.target_range[1][0]),
                   env_.target_range[2][0] + rnd_sample[2] * (env_.target_range[2][1] - env_.target_range[2][0])]

    env_.gazebo.set_model_state(model_name='target', position=rand_target)
    cube_targets = env_.neighbour_8_targets(rand_target)
    norm_cube_targets = [[None, None, None] for _ in range(8)]

    for i in range(8):
        norm_cube_targets[i] = env_.normalize_goal(cube_targets[i])

    p = [rand_target[i] - cube_targets[0][i] for i in range(3)]
    l = 0.2

    c0 = (1 - p[0] / l) * (1 - p[1] / l) * (1 - p[2] / l)
    c1 = (p[0] / l) * (1 - p[1] / l) * (1 - p[2] / l)
    c2 = (1 - p[0] / l) * (p[1] / l) * (1 - p[2] / l)
    c3 = (1 - p[0] / l) * (1 - p[1] / l) * (p[2] / l)
    c4 = (p[0] / l) * (1 - p[1] / l) * (p[2] / l)
    c5 = (1 - p[0] / l) * (p[1] / l) * (p[2] / l)
    c6 = (p[0] / l) * (p[1] / l) * (1 - p[2] / l)
    c7 = (p[0] / l) * (p[1] / l) * (p[2] / l)

    done = False
    begin = True
    collision = False
    state = env_.reset()
    env_.goal = rand_target
    
    
    
    while not done:
        
        t = datetime.datetime.now()
        
        state_with_target = np.concatenate((state, norm_cube_targets[0]))
        action0, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[1]))
        action1, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[2]))
        action2, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[3]))
        action3, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[4]))
        action4, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[5]))
        action5, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[6]))
        action6, _ = model.predict(state_with_target)

        state_with_target = np.concatenate((state, norm_cube_targets[7]))
        action7, _ = model.predict(state_with_target)

        action_merged = c0 * action0 + c1 * action1 + c2 * action2 + c3 * action3 + c4 * action4 + c5 * action5 + c6 * action6 + c7 * action7
        
        t_policy.append(datetime.datetime.now().microsecond - t.microsecond)
        if t_policy[-1] < 0:
            t_policy.pop(-1)
        if len(t_policy) == 1000:
            print(np.mean(t_policy), np.std(t_policy))
            t_policy = []
        
        state, rewards, done, info = env_.step(action_merged)

        if info == 'collision':
            done = True
            collision_counter += 1
            collision_target.append(rand_target)
        elif info == 'timeover':
            done = True
            timeover_counter += 1
            timeover_target.append(rand_target)
        elif info == True:
            done = True
            success_counter += 1
            success_target.append(rand_target)

success_rate = success_counter / test_episodes
collision_rate = collision_counter / test_episodes
timeover_rate = timeover_counter / test_episodes

# print('success rate:\t', success_rate)
# print('collision rate:\t', collision_rate)
# print('timeover rate:\t', timeover_rate)
#
# with open('/home/tartaglia/catkin_ws/src/rlgazebo/rl_gazebo/data/icra2020/results_merged_policies_'+model_step+'.csv',
#           'w+') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(['success_rate', 'collision_rate', 'timeover_rate'])
#     writer.writerow([success_rate, collision_rate, timeover_rate])
#     writer.writerow(['success_target'])
#     for i in range(len(success_target)):
#         writer.writerow([success_target[i][0], success_target[i][1], success_target[i][2]])
#     writer.writerow(['collision_target'])
#     for i in range(len(collision_target)):
#         writer.writerow([collision_target[i][0], collision_target[i][1], collision_target[i][2]])
#     writer.writerow(['timeover_target'])
#     for i in range(len(timeover_target)):
#         writer.writerow([timeover_target[i][0], timeover_target[i][1], timeover_target[i][2]])
