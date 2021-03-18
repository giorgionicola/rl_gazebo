import gym
from gym.utils import seeding
from rl_gazebo_env.base.Gazebo import Gazebo
from copy import deepcopy
import numpy as np
import os
import json
from pprint import pprint

from typing import Union, Tuple, List, NoReturn, Dict


class RlGazeboEnv(gym.Env):
    """
    Base class for all environments. Each Environment must inherit form this class and override the empty methods
    
    """
    
    def __init__(self, max_time_ep: float, simulation_timestep: float, command_period: float):
        """
        
        :param max_time_ep: max length of an episode [s]
        :param simulation_timestep: Gazebo timestep length, it MUST be same of the value specified in the world file
        :param command_period: Period at which a new action from the policy is sent
        """
        assert type(max_time_ep) is float, 'max_time_ep must be a float'
        assert type(simulation_timestep) is float, 'simulation_time_setp must be a float'
        assert type(command_period) is float, 'command_period must be a float'
        assert command_period % simulation_timestep == 0.0, 'command period must be a multiple of simulation_timestep'
        
        self.gazebo = Gazebo(simulation_timestep=simulation_timestep,
                             command_period=command_period,
                             anonymous=False, )
        
        self.max_timesteps = int(max_time_ep / self.gazebo.command_period)
        self._elapsed_timesteps = 0
        self._old_state = None
        self._state = None
        self.name = type(self).__name__
    
    def step(self, actions: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply the actions given by the policy, advance the simulation, get the current state from the simulation,
        compute the reward and whether the episode is over
        
        :param actions: actions computed by the policy
        :return: state, reward, done, info
        """
        
        self.advance_simulation(actions)
        self._elapsed_timesteps += 1
        state = self.get_state_from_sim()
        reward, done, info = self.get_reward(actions)
        if not done:
            if self._elapsed_timesteps >= self.max_timesteps:
                done = True
        return state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode
        
        :return:
        """
        
        self._elapsed_timesteps = 0
        return self.env_reset()
    
    def env_reset(self) -> np.ndarray:
        """
        To be overridden for reset of the environment
        
        :return:
        """
        raise NotImplementedError
    
    def advance_simulation(self, actions) -> NoReturn:
        """
        To be overridden. Advance the simulation by sending the actions to the agents
        
        :param actions:
        :return:
        """
        
        raise NotImplementedError
    
    def get_state_from_sim(self) -> np.ndarray:
        """
        To be overridden. Return the state of the environment
        
        :return:
        """
        
        raise NotImplementedError
    
    def get_reward(self, actions) -> Tuple[float, bool, dict]:
        """
        To be overridden. Get the reward; done = True if the episode ends (not for time elapsed), info for additional
        info on the episode
        
        :param actions:
        :return: reward, done, info
        """
        
        raise NotImplementedError
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.np_random.seed(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
    
    def dump_params(self, folder):
        """
        Dumps the env params in a file named "env_name" in folder, it must be implemented in each environment
        
        :param folder: path to folder where save params
        :return:
        """
        
        assert type(folder) is str, 'folder must be a string'
        
        params = {'gazebo': self.gazebo.params,
                  'environment': self.__env_params}
        
        with open(folder + '/environment.json', 'w') as file:
            json.dump(params, file, indent=2)
    
    def print_params(self):
        pprint(self.gazebo.params)
        pprint(self.__env_params)
    
    def tensorboard(self):
        pass
    
    @property
    def __env_params(self) -> Dict:
        p = {'max_timestps': self.max_timesteps}
        p.update(self.env_params)
        return p
    
    @property
    def env_params(self):
        raise NotImplementedError
