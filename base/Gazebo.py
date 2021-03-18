import rospy
from rl_gazebo_msgs.srv import RunNsteps, AskTf, SetLinkCollisionMode, SetWorldCollision
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel, GetLinkState
from gazebo_msgs.msg import ModelState, LinkState
from std_srvs.srv import Empty

from rl_gazebo_env.base.Utils import print_red_warn
from rl_gazebo_env.robots import Robot
from rl_gazebo_env.models import Model
from rl_gazebo_env.controllers import JointBasedController, JointTrajectoryController

from numpy import ceil

from typing import NoReturn, List, Tuple


class Gazebo:
    """
    class to interface with Gazebo services, custom Gazebo world plugins and common nodes of the environment
    """
    
    def __init__(self, command_period: float, simulation_timestep: float, anonymous: bool = None):
        """
        Constructor
        
        :param command_period: frequency of the command sent to the agent, i.e. step duration
        :param simulation_timestep: length of a single simulation step in Gazebo, it must equal to the value in
            world file
        :param anonymous: set environment ROS node as anonymous
        """
        
        assert type(anonymous) is bool, 'anonymous must a bool'
        
        rospy.init_node('rl_gazebo_env', anonymous=anonymous)
        
        self.command_period = command_period
        self.timestep = simulation_timestep
        self.steps = int(command_period / self.timestep)
        
        self.__get_tf = rospy.ServiceProxy('/tf_service', AskTf)
        self.__advance_nsteps = rospy.ServiceProxy('/advance_n_steps', RunNsteps)
        self.__reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty, )
        self.__set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, )
        self.__delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        self.__spawn_model = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
        self.__get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.__set_links_collision = rospy.ServiceProxy('/set_links_collision', SetLinkCollisionMode)
        self.__set_world_collisions = rospy.ServiceProxy('/set_world_collision', SetWorldCollision)
        self.__pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.__unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    
    def advance_nsteps(self, steps: int = None) -> NoReturn:
        """
        Advance the simulation of steps, if not specified it advances of self.steps
        
        :param steps: steps to advance the simulation
        :return:
        """
        
        assert steps is None or type(steps) is int, 'steps if specified must be an int'
        rospy.wait_for_service('/advance_n_steps')
        try:
            if steps is None:
                self.__advance_nsteps(self.steps)
            else:
                assert steps > 0 and type(steps) is int, 'steps must be > 0 and int'
                self.__advance_nsteps(steps)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def reset_world(self) -> NoReturn:
        """
        Reset Gazebo World
        
        :return:
        """
        
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.__reset_world
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def set_model_state(self, model_name: str, position: List[float] = None, orientation: List[float] = None,
                        twist_linear: List[float] = None, twist_angular: List[float] = None) -> NoReturn:
        """
        Set a model state in Gazebo
        
        :param str model_name: model name
        :param position: model x-y-z position
        :param orientation: model x-y-z orientation
        :param twist_linear: model linear speed
        :param twist_angular: model rotation speed
        :return:
        """
        
        assert type(model_name) is str, 'model_name must be a string'
        assert type(position) is not None or type(orientation) is not None or type(twist_linear) is not None or \
               type(twist_angular) is not None, \
            'At least one among the fields (position, orientation, twist_linear, twist_angular) must be set'
        msg = ModelState()
        msg.model_name = model_name
        
        if position is not None:
            assert type(position) is list, 'position must be a list'
            assert len(position) is 3, 'position must be made of 3 elements'
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.position.z = position[2]
        
        if orientation is not None:
            assert type(orientation) is list, 'orientation must be a list'
            assert all([type(o) is float for o in orientation]), 'all orientation values must float'
            assert len(orientation) is 4, 'orientation must be made of 4 elements'
            msg.pose.orientation.x = orientation[0]
            msg.pose.orientation.y = orientation[1]
            msg.pose.orientation.z = orientation[2]
            msg.pose.orientation.w = orientation[3]
        
        if twist_linear is not None:
            assert type(twist_linear) is list, 'twist_linear must be a list'
            assert all([type(t_l) is float for t_l in twist_linear]), 'all twist_linear values must float'
            assert len(twist_linear) is 3, 'twist_linear must be made of 3 elements'
            msg.twist.linear.x = twist_linear[0]
            msg.twist.linear.y = twist_linear[1]
            msg.twist.linear.z = twist_linear[2]
        
        if twist_angular is not None:
            assert type(twist_angular) is list, 'twist_angular must be a list'
            assert all([type(t_a) is float for t_a in twist_angular]), 'all twist_angular values must float'
            assert len(twist_angular) is 3, 'position must be made of 3 elements'
            msg.twist.angular.x = twist_angular[0]
            msg.twist.angular.y = twist_angular[1]
            msg.twist.angular.z = twist_angular[2]
        
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.__set_model_state(msg)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def delete_model(self, model_name: str) -> NoReturn:
        """
        Delete model in Gazebo
        
        :param model_name: model name to be deleted
        :return:
        """
        assert type(model_name) is str, 'model_name must be a string'
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.__delete_model(model_name)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def get_tf(self, frame_name: str, ref_frame_name: str = None) -> Tuple[List[float], List[float]]:
        """
        Get the position and orientation from tf
        
        :param frame_name: frame requested
        :param ref_frame_name: reference frame, default is (world)
        :return:
        """
        assert type(frame_name) is str or type(frame_name) is None, 'Frame name must be a str'
        if ref_frame_name is None:
            ref_frame_name = 'world'
        
        rospy.wait_for_service('/tf_service', timeout=2.0)
        try:
            resp = self.__get_tf(ref_frame_name=ref_frame_name, frame_name=frame_name)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
            return [], []
        
        position = [resp.pose.position.x, resp.pose.position.y, resp.pose.position.z]
        orientation = [resp.pose.orientation.x,
                       resp.pose.orientation.y,
                       resp.pose.orientation.z,
                       resp.pose.orientation.w]
        
        return position, orientation
    
    def set_links_collision(self, model: str, links: list, mode: List[str]) -> bool:
        """
        Set collision link mode for links of model
        
        :param model: model name str
        :param links: list of link names
        :param mode: list od collision modality one for each link
        :return resp: True if successfull False if fail
        
        collision modes:
            - all : the link collide with each object not none
            - ghost : ghost links do not collide among them
            - none : no collision
            - sensor : it collides only with sensors like LIDAR
            - fixed : it collide only with fixed objects
        """
        
        collision_mode = ['all', 'ghost', 'none', 'sensor', 'fixed']
        
        assert type(model) is str, 'model must be a string'
        assert type(links) is list, 'links must be a list'
        assert all([type(l) is str for l in links]), 'each link value must be string'
        assert type(mode) is list, 'mode must be a list'
        assert all([type(m) is str for m in mode]), 'each mode value must be string'
        assert all([m in collision_mode for m in mode]), 'mode can only be [all, none, ghost, sensor, fixed]'
        assert len(links) == len(mode), 'links and mode must have the same elements'
        
        for value in mode:
            assert value in collision_mode, 'Only the following collision modelaities are supported %r' % mode
        
        rospy.wait_for_service('/set_links_collision', timeout=2.0)
        try:
            resp = self.__set_links_collision(model=model, links=links, mode=mode)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
        
        if resp.done is False:
            print_red_warn('Error in set_links_collision()')
        
        return resp.done
    
    def set_world_collisions(self, mode: bool) -> NoReturn:
        """
        Enable or disable all collisions in Gazebo
        
        :param mode: True collision on, False to switch off all the collision,
        :return:
        
        ATTENTION if one or more links have special collision modes they will have to set again manually
        """
        
        assert type(mode) is bool, 'mode must be a bool'
        
        rospy.wait_for_service('/set_world_collision', timeout=2.0)
        try:
            resp = self.__set_world_collisions(w_collisions=mode)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
        
        if resp.done is False:
            print_red_warn('Error in set_world_collision()')
    
    def pause_physics(self) -> NoReturn:
        """
        Pause gazebo physics -> Pause simulation
        
        :return:
        """
        rospy.wait_for_service('/gazebo/pause_physics', timeout=2.0)
        try:
            resp = self.__pause_physics()
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def unpause_physics(self) -> NoReturn:
        """
        Unpuase gazebo physics -> Unpause simulation
        
        :return:
        """
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=2.0)
        try:
            resp = self.__unpause_physics()
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def robots_go_to_start_position(self, robots_to_move: Robot, models_to_hold: List['Model'],
                                    controllers: List['JointTrajectoryController'], traj_duration: float) -> NoReturn:
        """
        move all the robots that cannot use set_joint_states() because of a Trajectory Position Controller by creating
            junction trajectories to the new start positions
        
        :param robots_to_move: Robot that requires to be moved
        :param models_to_hold: Models and Robot that should hold the current position
        :param controllers: Controllers that should receive the junction trajectory
        :param traj_duration: Duration of the trajectory
        :return:
        """
        assert type(controllers) is dict, 'controllers must be a dict of controllers divided per robot'
        
        for robot in robots_to_move:
            for c in robot.controller_manager.controller.values():
                if c.name not in controllers[robot.model_name]:
                    robot.brake_joints(brake=True, joints=c.joints_name)
        
        for model in models_to_hold:
            if model.joints.dof > 0:
                model.brake_all_joints(brake=True)
        self.set_world_collisions(mode=False)
        
        self.__create_and_send_junction_trajectories(robots_to_move=robots_to_move,
                                                     controllers=controllers,
                                                     duration=traj_duration)
        self.advance_nsteps(steps=int(traj_duration / self.timestep))
        self.set_world_collisions(mode=True)
        for model in models_to_hold:
            if model.joints.dof > 0:
                model.brake_all_joints(brake=False)
            if model.non_default_collision_link is not None:
                model.set_links_collision_mode_to_init()
        
        for robot in robots_to_move:
            robot.brake_all_joints(brake=False)
            if robot.non_default_collision_link is not None:
                robot.set_links_collision_mode_to_init()
            robot.controller_manager.send_trajectory_to_all_controllers(
                robot.dataset_trajectory_manager.current_trajectory)
    
    def __create_and_send_junction_trajectories(self, robots_to_move: List['Robot'],
                                                controllers: List['JointTrajectoryController'],
                                                duration: float) -> NoReturn:
        """
        Order the DatasetTrajectoryManager to create a junction trajectory and it is sent to the
        JointTrajectoryControllers

        :param robots_to_move: Robot that requires to be moved
        :param controllers: Controllers that should receive the junction trajectory
        :param duration: Duration of the junction trajectory
        :return:
        """
        for robot in robots_to_move:
            position = dict()
            velocity = dict()
            joint_position, joint_velocity, joint_order = robot.get_joint_states(joints=robot.joints.names)
            for c in controllers[robot.model_name]:
                position[c] = [None for _ in range(robot.controller_manager.controller[c].dof)]
                velocity[c] = [None for _ in range(robot.controller_manager.controller[c].dof)]
                for j, j_name in enumerate(joint_order):
                    if j_name in robot.controller_manager.controller[c].joints_name:
                        c_j_index = robot.controller_manager.controller[c].joints_name.index(j_name)
                        position[c][c_j_index] = joint_position[j]
                        velocity[c][c_j_index] = joint_velocity[j]
            
            trajectory = robot.dataset_trajectory_manager.connect_trajectory(position=position,
                                                                             velocity=velocity,
                                                                             controllers=controllers[robot.model_name],
                                                                             duration=duration)
            robot.controller_manager.send_trajectory_to_all_controllers(trajectory=trajectory)
    
    def get_link_state(self, link_name: str, ref_frame: str = 'world') -> Tuple[
        List[float], List[float], List[float], List[float]]:
        """
        Get the link state of a certain link: this is done by calling a Gazebo service

        :param link_name: The name of the link of which we want to get the state
        :param ref_frame: Name of the reference frame, default = world
        :return: state, sorted as position, orientation (quaternion) linear velocity and angular velocity
        """
        try:
            resp = self.__get_link_state(link_name, ref_frame)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
        
        position = [resp.link_state.pose.position.x, resp.link_state.pose.position.y, resp.link_state.pose.position.z]
        orientation = [resp.link_state.pose.orientation.x,
                       resp.link_state.pose.orientation.y,
                       resp.link_state.pose.orientation.z,
                       resp.link_state.pose.orientation.w]
        linear_velocities = [resp.link_state.twist.linear.x,
                             resp.link_state.twist.linear.y,
                             resp.link_state.twist.linear.z]
        angular_velocities = [resp.link_state.twist.angular.x,
                              resp.link_state.twist.angular.y,
                              resp.link_state.twist.angular.z]
        
        return position, orientation, linear_velocities, angular_velocities
    
    @property
    def params(self):
        return {'command_period': self.command_period,
                'timestep': self.timestep,
                'steps': self.steps}
