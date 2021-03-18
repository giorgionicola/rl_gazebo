import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_msgs.msg import Float64, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
import numpy as np
from copy import deepcopy

from typing import List, Union, NoReturn

from rl_gazebo_env.models.Joint import JointsGroup


class JointBasedController:
    """
    Parent class for all the joint based controllers
    
    """
    
    __compatible_hd_interfaces = ['position', 'velocity', 'effort']
    
    def __init__(self, name: str, controller: dict, joints_group: JointsGroup, namespace: str):
        assert type(name) is str, 'name must a string'
        assert type(controller) is dict, 'controller must be a dict'
        assert type(joints_group) is JointsGroup, 'joints must be a JointGroup'
        assert type(namespace) is str, 'namespace must a string'
        
        self.name = name
        self.namespace = namespace
        self.joints_name = deepcopy(joints_group.joints_name)
        self.dof = len(self.joints_name)
        
        assert controller['hd_interface'] in self.__compatible_hd_interfaces, 'The hardware interface must be one ' \
                                                                              'between [position velocity effort]'
        self.hd_interface = controller['hd_interface']
        
        if controller['limits'] is not None:
            assert len(controller['limits']) == 2, 'Limits must be made of 2 list'
            assert len(controller['limits'][0]) == joints_group.dof and \
                   len(controller['limits'][1]) == joints_group.dof, 'Limits number must be equal to joint number'
            assert all([controller['limits'][0][j] < controller['limits'][1][j] for j in range(joints_group.dof)]), \
                'Lower controller limits higher than upper limits'
            
            if self.hd_interface is 'velocity':
                assert all(
                    [joints_group.vel_limits[0][j] <= controller['limits'][0][j] <= joints_group.vel_limits[1][j] and
                     joints_group.vel_limits[0][j] <= controller['limits'][1][j] <= joints_group.vel_limits[1][j]
                     for j in range(joints_group.dof)]), 'controller limits must be within joints velocity limits'
                self.limits = deepcopy(controller['limits'])
            elif self.hd_interface is 'effort':
                assert all([joints_group.effort_limits[j] <= -controller['limits'][0][j] and
                            joints_group.effort_limits[j] <= controller['limits'][1][j]
                            for j in range(joints_group.dof)]), 'Controller limits must be with joint effort limits'
                self.limits = deepcopy(controller['limits'])
            elif self.hd_interface is 'position':
                self.limits = deepcopy(joints_group.pos_limits)
        else:
            if self.hd_interface is 'velocity':
                self.limits = deepcopy(joints_group.vel_limits)
            elif self.hd_interface is 'effort':
                self.limits = deepcopy(joints_group.effort_limits)
            elif self.hd_interface is 'position':
                self.limits = deepcopy(joints_group.pos_limits)
    
    @property
    def params(self):
        raise NotImplementedError


class JointController(JointBasedController):
    """
    JointController is a effort/position/velocity controller for only one joint
    
    """
    
    def __init__(self, name: str, controller: dict, joints_group: JointsGroup, namespace: str):
        """
        
        :param name: controller name
        :param jointsgroup: jointsgroup controlled
        :param namespace: namespace for the topic
        """
        super().__init__(name=name, controller=controller, joints_group=joints_group, namespace=namespace)
        
        self.__cmd_pub = rospy.Publisher(self.namespace + '/' + self.name + '/command', Float64,
                                         queue_size=1, )
        self.cmd = Float64()
    
    def command(self, command: float) -> NoReturn:
        """
        Publish command to controller
        
        :param command:
        :return:
        """
        
        assert all([self.limits[0][j] <= command <= self.limits[1][j] for j in range(self.dof)]), \
            'Command exceeding limits'
        
        self.cmd.data = command
        self.__cmd_pub.publish(self.cmd)
    
    @property
    def params(self):
        return {'type': 'JointController',
                'joints_name': self.joints_name,
                'hardware_interface': self.hd_interface,
                'limits': self.limits}


class JointGroupController(JointBasedController):
    """
    JointController is a effort/position/velocity controller for multiple joints
    
    """
    
    def __init__(self, name: str, controller: dict, joints_group: JointsGroup, namespace: str):
        """

        :param name: controller name
        :param joints: joints controlled
        :param namespace: namespace for the topic
        """
        
        super().__init__(name=name, controller=controller, joints_group=joints_group, namespace=namespace)
        
        self.__cmd_pub = rospy.Publisher(self.namespace + '/' + self.name + '/command', Float64MultiArray,
                                         queue_size=1, )
        self.cmd = Float64MultiArray()
    
    def command(self, command: Union[List[float], np.ndarray]) -> NoReturn:
        """
        Publish command to controller
        
        :param command: commands
        :return:
        """
        assert type(command) is list or type(
            command) is np.ndarray, 'command must be a list or a numpy array'
        # assert all([self.limits[0][j] <= command[j] <= self.limits[1][j] for j in range(self.dof)] )
        self.cmd.data = command
        self.__cmd_pub.publish(self.cmd)
    
    @property
    def params(self):
        return {'type': 'JointController',
                'joints_name': self.joints_name,
                'hardware_interface': self.hd_interface,
                'limits': self.limits}


class JointTrajectoryController(JointBasedController):
    """
    Controller for trajectories
    
    """
    
    def __init__(self, name: str, controller: dict, joints_group: JointsGroup, namespace: str):
        """
        
        :param name: controller name
        :param joints: joints controlled
        :param namespace: namespace for the action
        """
        
        super().__init__(name=name, controller=controller, joints_group=joints_group, namespace=namespace)
        
        self.__client = actionlib.SimpleActionClient(
            self.namespace + '/' + self.name + '/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.__client.wait_for_server()
        self.__goal = FollowJointTrajectoryGoal()
        
        self.__client.cancel_all_goals()
    
    def send_trajectory(self, trajectory_points: dict, trajectory_time_array: List[float]) -> NoReturn:
        """
        Send the trajectory to the controller, first position and velocity is discarded because it is the starting
        position i.e. the current robot position
        
        :param trajectory_points: trajectory position and velocity at every waypoint
        :param trajectory_time_array: time at every waypoint
        :return:
        """
        assert len(trajectory_points['position']) == len(trajectory_time_array), \
            'positions and time must have the same length: (' + str(len(trajectory_points['position'])) + ',' + \
            str(len(trajectory_time_array)) + ')'
        if trajectory_points['velocity']:
            assert len(trajectory_points['position']) == len(trajectory_points['velocity']), \
                'positions and velocity must have the same length: (' + str(len(trajectory_points['position'])) + \
                ',' + str(len(trajectory_points['velocity'])) + ')'
        
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joints_name
        
        self.__goal.trajectory = trajectory
        
        for p in range(len(trajectory_time_array)):
            if trajectory_time_array[p] == 0:
                pass
            else:
                point = JointTrajectoryPoint()
                
                point.positions = trajectory_points['position'][p].copy()
                if trajectory_points['velocity']:
                    point.velocities = trajectory_points['velocity'][p].copy()
                point.time_from_start = rospy.Duration(trajectory_time_array[p])
                
                self.__goal.trajectory.points.append(point)
        
        self.__client.send_goal(self.__goal)
    
    def check_status_traj(self) -> NoReturn:
        """
        Check the trajectory status, currently it checks only if is active or not
        
        :return:
        """
        status = self.__client.get_state()
        if status is actionlib.GoalStatus.SUCCEEDED:
            return True
        else:
            return False
    
    def cancel_all_goals(self) -> NoReturn:
        """
        Cancel all goals
        
        :return:
        """
        self.__client.cancel_all_goals()
    
    def wait_for_result(self) -> NoReturn:
        return self.__client.wait_for_result()
    
    @property
    def params(self):
        return {'type': 'JointTrajectoryController',
                'hd_interface': self.hd_interface,
                'joints': self.joints_name}


class TwistController:
    """
    Twist controller for a robot

    """
    
    def __init__(self, name: str, namespace: str):
        """

        :param name: controller name
        :param namespace: namespace for the topic
        """
        assert type(name) is str, 'name must a string'
        assert type(namespace) is str, 'namespace must a string'
        
        self.name = name
        self.namespace = namespace
        
        self.__cmd_pub = rospy.Publisher(self.namespace + '/cmd_vel', Twist, queue_size=1, )
        self.cmd = Twist()
    
    def command(self, command: Union[List[float], np.ndarray] = None) -> NoReturn:
        """
        Publish the Twist command
        
        :param command: speed command
        :return:
        """
        
        assert command is not None, 'command must be given'
        assert type(command) is list or type(command) is np.ndarray, \
            'command must be a list or a numpy array'
        assert len(command) is 6, 'All components must be given'
        
        self.cmd.linear.x = command[0]
        self.cmd.linear.y = command[1]
        self.cmd.linear.z = command[2]
        
        self.cmd.angular.x = command[3]
        self.cmd.angular.y = command[4]
        self.cmd.angular.z = command[5]
        
        self.__cmd_pub.publish(self.cmd)
    
    @property
    def params(self):
        return {'type': 'TwistController'}
