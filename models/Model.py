import rospy
from sensor_msgs.msg import JointState
from rl_gazebo_msgs.srv import GetJointStates, GetAllJointStates, SetJointStates, BrakeAll, BrakeJoints, \
    SetLinkCollisionMode, AskContacts
from rl_gazebo_env.base.Utils import *
from rl_gazebo_env.models.Joint import Joints

from typing import Union, Optional, List, NoReturn, Dict, Tuple

from copy import deepcopy

class Model:
    """
    A Model is any instance spawned in Gazebo, it can have joints but it is not controlled
    
    """
    _reset_type = ['fix', 'random', 'dataset', 'no_reset', 'custom']
    _collision_mode = ['all', 'none', 'ghost', 'sensor', 'fixed']
    
    def __init__(self, model_name: str, links: List[str], reset_type: Union[str, None], joints_groups: dict = None,
                 joint_start_pos: List[float] = None, joint_start_vel: List[float] = None,
                 range_rand_start_pos: List[float] = None, range_rand_start_vel: List[float] = None,
                 non_default_collision_links: List[str] = None, collision_links_mode: List[str] = None,
                 normalization_up: float = None, normalization_low: float = None):
        """
        
        :param model_name: model name
        :param links: links name
        :param reset_type: type of reset, no_reset only for joint-less models
        :param joints_groups: joint_groups dictionary
        :param joint_pos_limit: joints position limits
        :param joint_vel_limit: joints velocity limits
        :param wrapped_joints: joints to be wrapped, currently ony [0,2*pi)
        :param non_default_collision_links: links with non default collision mode (default=all)
        :param collision_links_mode: collision mode to be set
        :param normalization_up: upper value normalization
        :param normalization_low: lower value normalization
        """
        
        assert type(model_name) is str, 'model_name must be a string'
        self.model_name = model_name
        if links is not None:
            assert type(links) is list, 'links must be a list'
            assert all([type(l) == str for l in links]), 'Links element must be str'
        self.links = links
        self.joints = Joints(joint_groups=joints_groups)
        if joints_groups is not None:
            self.__joint_states = JointState()
            self.__joint_states.name = deepcopy(self.joints.names)
        
        assert reset_type in self._reset_type, 'The reset type must be one between %r' % self._reset_type
        self.reset_type = reset_type
        if reset_type is 'fix' or reset_type is 'random':
            assert type(joint_start_pos) is list, 'joint_start_pos must be a list'
            print(joint_start_pos, self.joints.dof)
            assert len(joint_start_pos) == self.joints.dof, 'joints_start_pos must be specified for every joint'
            assert all([type(j) == float for j in joint_start_pos]), 'joint_start_pos must be populated by floats'
            self.joint_start_pos = joint_start_pos
            assert type(joint_start_vel) is list, 'joint_start_vel must be a list'
            assert len(joint_start_vel) == self.joints.dof, 'joints_start_vel must be scefied for every joint'
            assert all([type(j) == float for j in joint_start_vel]), 'joint_start_vel must be populated by floats'
            self.joint_start_vel = joint_start_vel
            if reset_type is 'random':
                assert type(range_rand_start_pos) is list, \
                    'range_rand_start_pos must be specified if reset_type==random'
                assert len(range_rand_start_pos) == self.joints.dof, 'range_rand_start_pos must be set for every joint'
                assert all([type(j) == float for j in range_rand_start_pos]), \
                    'range_rand_start_pos must be populated by floats'
                self.range_rand_start_pos = range_rand_start_pos
                assert type(range_rand_start_vel) is list, \
                    'range_rand_start_vel must be specified if reset_type==random'
                assert len(range_rand_start_vel) == self.joints.dof, 'range_rand_start_vel must be set for every joint'
                assert all([type(j) == float for j in range_rand_start_pos]), \
                    'range_rand_start_pos must be populated by floats'
                self.range_rand_start_vel = range_rand_start_vel
                for j in range(self.joints.dof):
                    if range_rand_start_pos[j] + self.range_rand_start_pos[j] > self.joints.pos_limits[1][j] \
                            or range_rand_start_pos[j] + self.range_rand_start_pos[j] < self.joints.pos_limits[0][j]:
                        print_red_warn('robot_name: ' + str(model_name) + ' joint: ' + str(self.joints.names[j]) +
                                       ' random start exceed joint limits, it will be clipped')
        if reset_type is 'no_reset':
            assert joints_groups is None, 'If reset_type is no_reset the model must be joint-less'
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
        if reset_type is 'dataset':
            # reset_type == dataset is implemented in Robot since it is only for Robots
            pass
            
        if normalization_low is not None:
            assert type(normalization_low) is float or type(normalization_low) is int, \
                'normalization_low must be a number'
        self.normalization_low = normalization_low
        if normalization_up is not None:
            assert type(normalization_up) is float or type(normalization_up) is int, 'normalization_up must be a number'
        self.normalization_up = normalization_up
        
        self.__get_joint_states = rospy.ServiceProxy(self.model_name + '/get_joint_states', GetJointStates)
        self.__get_all_joint_states = rospy.ServiceProxy(self.model_name + '/get_all_joint_states', GetAllJointStates)
        self.__set_joint_states = rospy.ServiceProxy(self.model_name + '/set_joint_states', SetJointStates)
        self.__brake_all_joints = rospy.ServiceProxy(self.model_name + '/brake_all', BrakeAll)
        self.__brake_joints = rospy.ServiceProxy(self.model_name + '/brake_joints', BrakeJoints)
        self.__set_links_collision_mode = rospy.ServiceProxy('/set_links_collision', SetLinkCollisionMode)
        
        self._get_collision = dict()
        for link in self.links:
            self._get_collision[link] = rospy.ServiceProxy('/' + self.model_name + '/' + link + '/collision',
                                                           AskContacts)
        
        if non_default_collision_links is not None:
            assert type(non_default_collision_links) is list, 'non_default_collision_links must be a list'
            assert all(link in links for link in non_default_collision_links), \
                'All links in non_default_collision_links must be also in links'
        self.non_default_collision_link = non_default_collision_links
        self.collision_links_mode = collision_links_mode
        if collision_links_mode is not None and non_default_collision_links is not None:
            assert len(collision_links_mode) == len(non_default_collision_links), \
                'Collision_links_mode and non_default_collision_links mustc have the same length'
            assert all(mode in self._collision_mode for mode in collision_links_mode), \
                'collision mode can only be [all, none, ghost, sensor, fixed]'
            self.set_links_collision_mode_to_init()
        
        self.state = []
    
    def normalize(self, values: List[float], limits: List[List[float]]) -> List[float]:
        """
        Normalize the given value on the normalization range specified in init
        
        :param values: values to be normalized
        :param limits: limits over values are normalized
        :return:
        """
        normal_joints = [0.0 for _ in range(len(values))]
        for j in range(len(values)):
            normal_joints[j] = normalize_to_01(values[j],
                                               max_value=limits[1][j],
                                               min_value=limits[0][j])
        if self.normalization_up != 1 or self.normalization_low != 0:
            assert self.normalization_low is not None, 'normalization_low must be set'
            assert self.normalization_up is not None, 'normalization_up must be set'
            for j in range(len(values)):
                normal_joints[j] = normalize_from_01_to_any_range(normal_joints[j],
                                                                  upper_range=self.normalization_up,
                                                                  lower_range=self.normalization_low)
        return normal_joints
    
    def set_joint_states(self, joints: List[str], position: List[float] = None,
                         velocity: List[float] = None) -> NoReturn:
        """
        Set the joint state (poistion, velocity)
        
        :param joints: joints to be set
        :param position: position of the joints to be set, if not specified is 0 for all joints
        :param velocity: velocity of the joints to be set, if not specified is 0 for all joints
        :return:
        """
        assert all([j in self.joints.names for j in joints]), 'all values of joints must also be in self.joints'
        
        if position is not None:
            assert type(position) is list
            assert all([type(p) is float for p in position]), 'position values must be float'
            self.__joint_states.position = position
        else:
            self.__joint_states.position = [0 for _ in range(self.joints.dof)]
        
        if velocity is not None:
            assert type(position) is list
            assert all([type(p) is float for p in position]), 'position values must be float'
            self.__joint_states.velocity = velocity
        else:
            self.__joint_states.velocity = [0 for _ in range(self.joints.dof)]
        
        rospy.wait_for_service('/' + self.model_name + '/set_joint_states', timeout=2.0)
        try:
            self.__set_joint_states(self.__joint_states)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def get_state(self) -> List[float]:
        """
        Get the joint state,
        
        :return: state, sorted as position and velocity sorted also by order of the joints in self.joints
        """
        joint_positions, joint_velocities, joint_order = self.get_all_joint_states()
        
        order = []
        for j in self.joints.names:
            if j in joint_order:
                order.append(joint_order.index(j))

        joint_positions_sorted = [joint_positions[o] for o in order]
        joint_velocities_sorted = [joint_velocities[o] for o in order]
        
        joint_positions_sorted = self.joints.wrap_joints(joints_value=joint_positions_sorted)
        
        self.state = joint_positions_sorted + joint_velocities_sorted
        
        return self.state
    
    def normalize_state(self) -> List[float]:
        """
        
        :return: state normalized
        """
        
        state_normal = self.normalize(self.state[:self.joints.dof], self.joints.pos_limits)
        state_normal += self.normalize(self.state[self.joints.dof:], self.joints.vel_limits)
        
        return state_normal
    
    def get_all_joint_states(self) -> Tuple[List[float], List[float], List[str]]:
        """
        Get all the joints state
        
        :return:
        """
        
        rospy.wait_for_service('/' + self.model_name + '/get_all_joint_states', timeout=2.0)
        try:
            resp = self.__get_all_joint_states()
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
        
        joint_positions = resp.joint_states.position
        joint_velocities = resp.joint_states.velocity
        joint_order = resp.joint_states.name
        
        return joint_positions, joint_velocities, joint_order
    
    def get_joint_states(self, joints: List[str]) -> Tuple[List[float], List[float], List[str]]:
        """
        Get the state of the specified joints
        
        :param joints: joints name to get the joint state
        :return joint_positions, joint_velocities, joint_order:
        """
        assert all([j in self.joints.names for j in joints]), 'All joints requested must be in self.joints'
        
        rospy.wait_for_service('/' + self.model_name + '/get_joint_states', timeout=2.0)
        try:
            resp = self.__get_joint_states(joint_names=joints)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
        
        joint_positions = resp.joint_states.position
        joint_velocities = resp.joint_states.velocity
        joint_order = resp.joint_states.name
        
        return joint_positions, joint_velocities, joint_order
    
    def get_collisions(self) -> Tuple[bool, int]:
        """
        Get collision for every model link
        
        :return: collison, colliding_links
        """
        
        colliding_links = 0
        collision = False
        
        for link in self.links:
            rospy.wait_for_service('/' + self.model_name + '/' + link + '/collision', timeout=2.0)
            try:
                resp = self._get_collision[link]()
            except rospy.ServiceException as e:
                print('Service did not process request:' + str(e))
            
            if resp.ncontacts > 0:
                colliding_links += 1
        
        if colliding_links > 0:
            collision = True
        
        return collision, colliding_links
    
    def brake_all_joints(self, brake: bool) -> NoReturn:
        """
        Service to set brakes on all joints
        
        :param brake: True if set brakes, False if remove brakes
        :return:
        """
        assert type(brake) is bool, 'Brake command must be a bool'
        assert self.joints.dof > 0, 'self.dof must be > 0'
        rospy.wait_for_service('/' + self.model_name + '/brake_all', timeout=2.0)
        try:
            self.__brake_all_joints(brake)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def brake_joints(self, brake: bool, joints: List[str]) -> NoReturn:
        """
        Set brakes for a set of joints
        
        :param brake: True to enable brakes on joints, False to disable
        :param joints: Joints name to set Brake
        :return:
        """
        assert type(brake) is bool, 'Brake command must be a bool'
        assert type(joints) is list, 'joints must be a list of string'
        for j in joints:
            assert j in self.joints.names, '%r is not a joint of %r' % (j, self.model_name)
        
        rospy.wait_for_service('/' + self.model_name + '/brake_joints', timeout=2.0)
        try:
            resp = self.__brake_joints(brake=brake, joints=joints)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def set_links_collision_mode(self, links: List[str], mode: List[str]) -> NoReturn:
        """
        Set the collision mode for the link specified, it should be used only to set non standard collision mode i.e.
        all modes except from 'all'
        
        :param links: list of link to be modefied
        :param mode: list of collision mode, one for each link
        :return: resp
        """
        assert type(links) is list, 'links must be a list'
        assert type(mode) is list, 'mode must be a list'
        assert len(links) == len(mode), 'links and mode must have the same length'
        for l in links:
            assert l in self.links, '%r is not a link of %r' % l % self.model_name
        for m in mode:
            assert m in self._collision_mode, 'the collision mode must one of the following %r' % self._collision_mode
        
        rospy.wait_for_service('/set_links_collision', timeout=2.0)
        try:
            resp = self.__set_links_collision_mode(model=self.model_name, links=links, mode=mode)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def set_links_collision_mode_to_init(self) -> NoReturn:
        """
        Set to all non default collision link to mode specified in in init
        
        :return:
        """
        self.set_links_collision_mode(links=self.non_default_collision_link,
                                      mode=self.collision_links_mode)
    
    @property
    def params(self):
        return {'name': self.model_name,
                'links': self.links,
                'joints': self.joints.params,
                'non_default_collision_links': self.non_default_collision_link,
                'collision_links_mode': self.collision_links_mode,
                'normalization_up': self.normalization_up,
                'normalization_low': self.normalization_low}
