from rl_gazebo_env.models.Model import Model
from rl_gazebo_env.base.Utils import *
from rl_gazebo_env.controllers.ControllerManager import ControllerManager
from rl_gazebo_env.trajectory.DatasetTrajectoryManager import DatasetTrajectoryManager
from rl_gazebo_env.end_effector.EndEffector import EndEffector

from copy import deepcopy
from typing import Union, List, NoReturn


class Robot(Model):
    """
    Robot extend Model to have one or more controllers managed by the ControllerManager, if the movements are based on a
    dataset it has also a DatasetManager. An EndEffector or a Tool can be attached to a Robot
    
    """
    _learning_robots = 0
    _not_learning_robots = 0
    _movement_type = ['action', 'trajectory', 'custom']
    
    def __init__(self, model_name: str, links: List[str], movement_type: str, reset_type: str, controllers: dict,
                 joints_groups: dict = None, joint_start_pos: List[float] = None, joint_start_vel: List[float] = None,
                 non_default_collision_links: List[str] = None, collision_links_mode: List[str] = None,
                 range_rand_start_pos: List[float] = None, range_rand_start_vel: List[float] = None,
                 path_traj_dataset: str = None, time_warp: dict = None, pause_at_target: dict = None,
                 rand_start_point: bool = None, revert_traj_prob: float = None, normalization_up: float = 1,
                 normalization_low: float = 0, learning_robot: bool = True, end_effector: 'EndEffector' = None,
                 sym_kinematic: bool= False, T_start: List[List]= None):
        
        """

        :param model_name: robot model name
        :param joints_groups: contains all joints infos, for dict structure see Joints() documentation
        :param links: list of robot links
        :param joint_start_pos: joints start position for reset
        :param joint_start_vel: joints start position for reset
        :param movement_type: (action, trajectory), action for instantaneous commands, (trajectory) for forwarding
            a trajectory
        :param reset_type: (fix, random, dataset), (fix) restart always at joint_start_pos and vel, (random) restart
            at joint_start_pos/vel + random(range_rand_start_pos/vel), (dataset) restart at a random point of a random
            trajectory from the dataset
        :param controllers: contains all controllers infos, for dict structure see ControllerManager() doc
        :param non_default_collision_links: links with non default collision mode, (default = all)
        :param collision_links_mode: collision mode for each non_default_collision_link(all, none, ghost, sensor, fixed)
        :param range_rand_start_pos: range start position for every joint, necessary if reset_type==random
        :param range_rand_start_vel: range start velocity for every joint, necessary if reset_type==random
        :param path_traj_dataset: absolute path to the trajectory dataset, if not specified it will be used the
            robot default
        :param rand_start_point: if True the trajectory will start from a random point, only at start
        :param time_warp: dict for making variable the trajetory time
        :param revert_traj_prob: the probability of reverting a trajectory
        :param normalization_up: upper value to normalize to
        :param normalization_low: lower value to normalize to
        :param learning_robot: if the robot is a agent training
        :param end_effector: end-effector to be attached to the robot
        :param T_start: Roto-translation matrix for symbolic kinematics for offset (default: eye(4))
        :param sym_kinematic: True to compute the symbolic kinematics
        """
        self.number = 0
        super().__init__(model_name=model_name, links=links, joints_groups=joints_groups, reset_type=reset_type,
                         joint_start_pos=joint_start_pos, joint_start_vel=joint_start_vel,
                         range_rand_start_pos=range_rand_start_pos, range_rand_start_vel=range_rand_start_vel,
                         non_default_collision_links=non_default_collision_links,
                         collision_links_mode=collision_links_mode, normalization_up=normalization_up,
                         normalization_low=normalization_low)
        
        
        
        assert type(learning_robot) is bool or learning_robot is None, 'learning robot must be a bool or None'
        self.learning_robot = learning_robot
        if learning_robot:
            self._learning_robots += 1
        else:
            self._not_learning_robots += 1
        
        assert movement_type in self._movement_type, 'The movement type must be one between %r' % self._movement_type
        if movement_type is 'dataset_trajectory':
            assert reset_type is 'dataset', 'If movement_type == dataset_trajectory -> reset_type must be dataset'
        
        if self.reset_type is 'dataset':
            assert path_traj_dataset is not None, 'Missing path to dataset of trajectories'
            self.dataset_trajectory_manager = DatasetTrajectoryManager(path_traj_dataset,
                                                                       time_warp=time_warp,
                                                                       random_start_point=rand_start_point,
                                                                       revert_traj_prob=revert_traj_prob,
                                                                       pause_at_target=pause_at_target)
            if joint_start_pos is not None:
                print_red_warn('reset_type is no_reset, joint_start_pos will be ignored')
            if joint_start_vel is not None:
                print_red_warn('reset_type is no_reset, joint_start_vel will be ignored')
            if range_rand_start_pos is not None:
                print_red_warn('reset_type is no_reset, range_rand_start_pos will be ignored')
            if joint_start_pos is not None:
                print_red_warn('reset_type is no_reset, range_rand_start_vel will be ignored')
            self.joint_start_pos = None
            self.joint_start_vel = None
            self.range_rand_start_pos = None
            self.range_rand_start_vel = None
        
        self.movement_type = movement_type
        
        self.controller_manager = ControllerManager(controllers=controllers,
                                                    robot_joints=self.joints,
                                                    namespace='/' + self.model_name)
        
        assert issubclass(type(end_effector), EndEffector) or (end_effector is None), \
            'end_effector must be a EndEffector or None'
        self.end_effector = end_effector
        
        assert type(sym_kinematic) is bool, 'symbolic_kinematic must be a bool'
        if sym_kinematic:
            # if T_start is not None:
                # assert type(T_start) is List and all([type(t) is List for t in T_start]), 'T_start must be a List[List]'
                # assert len(T_start) == 4 and all([len(t) ==4 for t in T_start]), 'T_start must be 4x4'
            self.T_start = T_start
            self.symbolic_kinematics()
        
        return
    
    def update(self, action: Union[list, np.ndarray] = None, controller_order: list = None,
               end_effector_action: Union[list, np.ndarray] = None, step_duration: float = None) -> NoReturn:
        """
        Apply the action to robot and end-effector
        
        :param action: robot actions
        :param controller_order: order of controllers
        :param end_effector_action: end_effector action
        :param step_duration: duration of a RL step, to check if the trajectory has finished
        :return:
        """
        
        if controller_order is None:
            controller_order = self.controller_manager.controller_names
        if self.movement_type is 'action':
            self.controller_manager.send_action_to_controllers(action, controller_order)
        elif self.movement_type is 'trajectory':
            if not self.dataset_trajectory_manager.hold_position:
                if self.dataset_trajectory_manager.current_trajectory.time_elapsed > \
                        self.dataset_trajectory_manager.current_trajectory.time[-1]:
                    if self.dataset_trajectory_manager.pause_at_target and \
                            self.dataset_trajectory_manager.pause_time < 0:
                        self.dataset_trajectory_manager.get_pause()
                    if not self.dataset_trajectory_manager.is_in_pause(step=step_duration):
                        self.dataset_trajectory_manager.get_next_trajectory()
                        if not self.dataset_trajectory_manager.hold_position:
                            self.controller_manager.send_trajectory_to_all_controllers(
                                self.dataset_trajectory_manager.current_trajectory)
                self.dataset_trajectory_manager.current_trajectory.time_elapsed += step_duration
        elif self.movement_type is 'custom':
            self.custom_movement()
        if end_effector_action is not None:
            self.end_effector.update(end_effector_action)
    
    def custom_movement(self):
        raise NotImplementedError
    
    def reset(self, traj_number: int = None, except_for= None) -> List[float]:
        """
        Reset the robot and the end-effector based on the reset_type specified
        
        :return total_state: robot + end effector state both not normalized
        """
        # These lines are copied from below
        
        
        if self.reset_type is 'custom':
            return self.custom_reset()
        elif self.reset_type is 'fix':
            start_position = deepcopy(self.joint_start_pos)
            start_velocity = deepcopy(self.joint_start_vel)
            self.set_joint_states(joints=self.joints.names, position=start_position, velocity=start_velocity)
        elif self.reset_type is 'random':
            start_position = deepcopy(self.joint_start_pos)
            start_velocity = deepcopy(self.joint_start_vel)
            for j in range(self.joints.dof):
                start_position[j] += np.random.uniform(self.range_rand_start_pos[j], -self.range_rand_start_pos[j])
                start_velocity[j] += np.random.uniform(self.range_rand_start_vel[j], -self.range_rand_start_vel[j])
                start_position[j] = clip_value(value=start_position[j],
                                               lower_range=self.joints.pos_limits[0][j],
                                               upper_range=self.joints.pos_limits[1][j])
                start_velocity[j] = clip_value(value=start_velocity[j],
                                               lower_range=self.joints.vel_limits[0][j],
                                               upper_range=self.joints.vel_limits[1][j])
            
            self.set_joint_states(joints=self.joints.names, position=start_position, velocity=start_velocity)
        elif self.reset_type is 'dataset':
            self.dataset_trajectory_manager.hold_position = False
            if traj_number is not None:
                start_position, start_velocity, _ = self.dataset_trajectory_manager.pick_trajectory(number=traj_number)
            else:
                start_position, start_velocity, _ = self.dataset_trajectory_manager.pick_random_trajectory(reset=True,
                                                                                                           except_for= except_for)
            self.controller_manager.cancel_all_controllers_goals()
        else:
            raise RuntimeError('reset type do not fall in any supported category')
        
        state_ef = []
        if self.end_effector is not None:
            state_ef = self.end_effector.reset()
        
        self.state = start_position + start_velocity
        total_state = self.state + state_ef
        
        return total_state
    
    def get_robot_state(self) -> List[float]:
        """
        Get the robot state including the end-effector
        
        :return state: robot + end effector state not normalized
        """
        
        state = self.get_state()
        if self.end_effector is not None:
            state_ef = self.end_effector.get_state()
            state += state_ef
        
        return state
    
    def brake_only_robot(self, brake: bool) -> NoReturn:
        """
        Set brakes only on the robot joints
        
        :param brake: True to set brakes, False to remove brakes
        :return:
        """
        assert type(brake) is bool, 'Brake command must be a bool'
        self.brake_joints(brake=brake, joints=self.joints)
    
    def brake_only_end_effector(self, brake: bool) -> NoReturn:
        """
        Set brakes only on the end-effector joints
        
        :param brake: True to set brakes, False to remove brakes
        :return:
        """
        assert type(brake) is bool, 'Brake command must be a bool'
        self.brake_joints(brake=brake, joints=self.end_effector.joints)
    
    def custom_reset(self) -> List[float]:
        """
        Method to be override for Custom resets
        
        :return state: robot + end effector state not normalized
        """
        raise NotImplementedError
    
    def symbolic_kinematics(self):
        """
        Compute the symbolic kinematics with output a set of Lambda functions
        :return:
        """
        
        raise NotImplementedError
    
    def get_cartesian_state(self):
        """
        compute the cartesian state based on the symbolic kinematics
        :return:
        """
        
        raise NotImplementedError
    
    @property
    def params(self):
        p = super().params
        p.update({'movement_type': self.movement_type,
                  'reset_type': self.reset_type,
                  'controllers': self.controller_manager.params,
                  'joint_start_pos': self.joint_start_pos,
                  'joint_start_vel': self.joint_start_vel,
                  'range_rand_start_pos': self.range_rand_start_pos,
                  'range_rand_start_vel': self.range_rand_start_vel,
                  'learning_robot': self.learning_robot})
        
        if self.reset_type is 'dataset':
            p['dataset_manager'] = self.dataset_trajectory_manager.params
        else:
            p['dataset_manager'] = None
        
        if self.end_effector is None:
            p['end_effector'] = None
        else:
            p['end_effector'] = self.end_effector.params
        
        return p
