from rl_gazebo_env.base.RlGazeboEnv import RlGazeboEnv
from rl_gazebo_env.robots.UR import UR
from rl_gazebo_env.base.Gazebo import Gazebo
from rl_gazebo_env.robots.HumanCyl import HumanCylinder
from rl_gazebo_env.base.Utils import *

from gym.utils import seeding
from gym import spaces


class HumanCylUrEnv0(RlGazeboEnv):
    """
    Normalized between 0,+1
    rew:    dist
            coll=-5
            success = 20
    time=5.0
    """

    def __init__(self):
        """
        
        """
        super().__init__(command_period=0.02,
                         simulation_timestep=0.001,
                         max_time_ep=5.0)
        
        

        self.normalization_up = 1
        self.normalization_low = 0
        self.normalization = False

        self.ur = UR(model_name='ur',
                     joint_start_pos=[0.0, -2.5, 2.0, 0.0, 0.0, 0.0],
                     joint_start_vel=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     joint_pos_lim=[[-2 * np.pi, -2 * np.pi, -2 * np.pi, -np.pi, -np.pi, -np.pi],
                                    [2 * np.pi, 2 * np.pi, 2 * np.pi, np.pi, np.pi, np.pi]],
                     joint_vel_lim=[[-1.0 for _ in range(6)],
                                    [1.0 for _ in range(6)]],
                     range_rand_start_pos=[0.0 for _ in range(6)],
                     range_rand_start_vel=[0.0 for _ in range(6)],
                     normalization_low=0.0,
                     normalization_up=1.0,
                     learning_robot=True)


        self.human = HumanCylinder(model_name='human',
                                   walking_range=[[-2, 0.5],
                                                  [1, 2.0]],
                                   target_toll=0.1,
                                   max_speed=1.0,
                                   min_speed=0.3,
                                   max_freq_speed=1.0,
                                   fix_speed=False,
                                   fix_maxspeed=False,
                                   prob_speed_null=0.05,
                                   prob_stop_on_target=0.1,
                                   max_radius=0.5,
                                   min_radius=0.2,
                                   fix_radius=False,
                                   max_freq_radius=0.7,
                                   max_length=2.0,
                                   min_length=1.5,
                                   fix_length=True,
                                   max_freq_length=0.2,
                                   normalization=self.normalization,
                                   normalization_up=1.0,
                                   normalization_low=0.0)

        self.target_range = [[-0.5, 0.5],
                             [0.6, 1.0],
                             [0.8, 1.2]]

        step_target_x = 0.2
        step_target_y = 0.2
        step_target_z = 0.2
        self.target_x = [self.target_range[0][0] + step_target_x * i for i in
                         range(int(round((self.target_range[0][1] - self.target_range[0][0]) / step_target_x)) + 1)]
        self.target_y = [self.target_range[1][0] + step_target_y * i for i in
                         range(int(round((self.target_range[1][1] - self.target_range[1][0]) / step_target_y)) + 1)]
        self.target_z = [self.target_range[2][0] + step_target_z * i for i in
                         range(int(round((self.target_range[2][1] - self.target_range[2][0]) / step_target_z)) + 1)]

        self.n_target = len(self.target_x) * len(self.target_y) * len(self.target_z)
        self.target_list = []
        for _, x in enumerate(self.target_x):
            for _, y in enumerate(self.target_y):
                for _, z in enumerate(self.target_z):
                    self.target_list.append([x, y, z])

        self.target_list = np.array(self.target_list)
        self.count_target = 0

        self.goal = [None for _ in range(3)]
        self.goal_tollerance = 0.1

        self.max_time_ep = 5.0
        self.max_timesteps = int(self.max_time_ep / self.gazebo.command_period) + 1
        self.elapsed_timesteps = 0

        self.success_rew = 20
        self.collision_rew = -5
        self.action_rew = -0.01

        self.training = True
        self.tcp_frame = 'tcp'

        self.random_target = False

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float)
        self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float)

    def advance_simulation(self, actions):
        """
        
        :param actions:
        :return:
        """
        
        time = self._elapsed_timesteps * self.gazebo.timestep
        
        self.ur.update(actions)
        self.human.update_human(time)
        self.gazebo.advance_nsteps()

    def get_state_from_sim(self):
        """
        
        :return:
        """

        robot_state = self.ur.get_state()
        human_state = self.human.get_state()

        if not self.random_target:
            state = robot_state + human_state + self.goal.tolist()
            state = np.array(state)
            if self.normalization:
                normal_state = np.array(self.ur.normalize_state() + self.human.normalize_state() + self.goal_normal)
                return normal_state
            else:
                return state
        else:
            state = robot_state + human_state
            state = np.array(state)
            if self.normalization:
                normal_state = self.ur.normalize_state() + self.human.normalize_state()
                normal_state = np.array(normal_state)
                return normal_state
            else:
                return state,

    def get_reward(self, action, state=None):
        """
        
        :param action:
        :param state:
        :return:
        """
        
        tcp_position, _ = self.gazebo.get_tf(self.tcp_frame)
        dist = np.linalg.norm(tcp_position - np.array(self.goal))

        collisions, n_collisions = self.ur.get_collisions()

        done = False
        info = False

        if dist <= self.goal_tollerance and collisions is False:
            reward = self.success_rew
            done = True
            info = True
            # print('success')
        else:
            reward = - dist + self.action_rew * np.linalg.norm(action)

            if collisions:
                reward = reward + self.collision_rew
                info = 'collision'

        if self.training:
            info = {}
            
        return reward, done, info

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def env_reset(self):
        """
        
        :return:
        """
        self.gazebo.reset_world()

        if not self.random_target:
            self.count_target += 1
            if self.count_target == self.n_target:
                self.count_target = 0

            self.goal = self.target_list[self.count_target, :]
            self.goal_normal = self.normalize_goal(self.goal)
            self.gazebo.set_model_state(model_name='target', position=self.goal.tolist())

        ur_state = self.ur.reset()
        human_state = self.human.reset()

        #TODO togliere questa parte
        if not self.random_target:
            state = ur_state + human_state + self.goal_normal
        else:
            state = ur_state + human_state

        state = np.array(state)
        
        return state

    def normalize_goal(self, goal):
        """
        
        :param goal:
        :return:
        """

        goal_normal = [None for _ in range(len(goal))]
        for i in range(len(goal)):
            goal_normal[i] = normalize_to_01(goal[i], self.target_range[i][1], self.target_range[i][0])

            if self.normalization_up != 1 or self.normalization_low != 0:
                goal_normal[i] = normalize_from_01_to_any_range(goal_normal[i],
                                                                self.normalization_up,
                                                                self.normalization_low)

        return goal_normal

    def neighbour_8_targets(self, target):
        """
        
        :param target:
        :return:
        """
        # TODO Questa funzione non deve far parte dell'environment, deve essere una utils esterna
        for i, x in enumerate(self.target_x):
            if x > target[0]:
                edge_x = [self.target_x[i - 1], self.target_x[i]]
                break

        for i, y in enumerate(self.target_y):
            if y > target[1]:
                edge_y = [self.target_y[i - 1], self.target_y[i]]
                break

        for i, z in enumerate(self.target_z):
            if z > target[2]:
                edge_z = [self.target_z[i - 1], self.target_z[i]]
                break

        cube = [[edge_x[0], edge_y[0], edge_z[0]],
                [edge_x[1], edge_y[0], edge_z[0]],
                [edge_x[0], edge_y[1], edge_z[0]],
                [edge_x[0], edge_y[0], edge_z[1]],
                [edge_x[1], edge_y[0], edge_z[1]],
                [edge_x[0], edge_y[1], edge_z[1]],
                [edge_x[1], edge_y[1], edge_z[0]],
                [edge_x[1], edge_y[1], edge_z[1]], ]

        return cube

