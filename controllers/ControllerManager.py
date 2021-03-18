from rl_gazebo_env.controllers.Controllers import *
from rl_gazebo_env.trajectory.Trajectory import Trajectory
from rl_gazebo_env.models.Joint import Joints
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, UnloadController, LoadController, \
    UnloadControllerRequest, LoadControllerRequest

from typing import List, NoReturn, Dict


class ControllerManager:
    """
    The controllerManager is a interface to easily handle robot that have multiple controllers, for example double arm
    robots
    
    """
    joint_controller_types = ['joint_controller',
                              'joint_group_controller',
                              'joint_trajectory_controller']
    body_controller_types = ['twist_controller']
    
    def __init__(self, controllers: dict, namespace: str, robot_joints: Joints = None):
        """
        
        :param controllers: dict
        :param namespace:
        """
        
        for _, c in controllers.items():
            assert type(c) is dict, 'single controllers must be dict'
            assert c['type'] in self.joint_controller_types or c['type'] in self.body_controller_types, \
                'Not compatible controller'
            if c['joints_group'] is not None:
                assert type(c['joints_group']) is str, 'joints_group must be a str'
        
        if robot_joints is not None:
            assert type(robot_joints) is Joints, 'robot_joints must be Joints'
        
        self.namespace = namespace
        self.controller_names = list(controllers.keys())
        self.n_controllers = len(controllers.keys())
        self.controller = dict()
        c_counter = 0
        
        self.__switch_controllers = rospy.ServiceProxy(namespace + '/controller_manager/switch_controller',
                                                       SwitchController)
        self.__load_controller = rospy.ServiceProxy(namespace + '/controller_manager/load_controller',
                                                    LoadController)
        self.__unload_controller = rospy.ServiceProxy(namespace + '/controller_manager/unload_controller',
                                                      UnloadController)
        
        for name in controllers.keys():
            c = controllers[name]
            if c['type'] in self.joint_controller_types:
                assert c['joints_group'] in robot_joints.jointgroups_name, \
                    'the joints group must be also in robot joints'
                
                if c['type'] is 'joint_controller':
                    self.controller[name] = JointController(name=name,
                                                            controller=c,
                                                            joints_group=robot_joints.groups[c['joints_group']],
                                                            namespace=self.namespace)
                
                if c['type'] is 'joint_group_controller':
                    self.controller[name] = JointGroupController(name=name,
                                                                 controller=c,
                                                                 joints_group=robot_joints.groups[c['joints_group']],
                                                                 namespace=self.namespace)
                
                if c['type'] is 'joint_trajectory_controller':
                    self.controller[name] = JointTrajectoryController(name=name,
                                                                      controller=c,
                                                                      joints_group=
                                                                      robot_joints.groups[c['joints_group']],
                                                                      namespace=self.namespace)
            else:
                if c['type'] is 'twist_controller':
                    self.controller[name] = TwistController(name=name,
                                                            namespace=self.namespace)
                
                c_counter += 1
    
    def send_action_to_controllers(self, action: List[float], controller_order: List[str]) -> NoReturn:
        
        """
        Send a set of actions each to a controller specified by controller_order
        
        :param action:
        :param controller_order:
        :return:
        """
        if self.n_controllers > 1:
            assert len(action) == self.n_controllers, \
                'if More than 1 controller action must be divided for each controller and the number must be equal'
            assert len(action) == len(controller_order), \
                'controller_order and action must have the same length'
            
            for action_subset, c_name in zip(action, controller_order):
                self.controller[c_name].command(action_subset)
        else:
            self.controller[self.controller_names[0]].command(action)
    
    def send_trajectory_to_all_controllers(self, trajectory: Trajectory) -> NoReturn:
        """
        Send a trajectory to all controllers involved
        
        :param trajectory: trajectory to be sent
        :return:
        """
        
        for c_name in trajectory.active_controllers:
            # TODO assert controller can accept trajectories
            self.controller[c_name].send_trajectory(trajectory_points=trajectory.controllers_traj[c_name],
                                                    trajectory_time_array=trajectory.time)
    
    def stop_all_controllers(self) -> NoReturn:
        """
        Stop all controllers
        
        :return:
        """
        msg = SwitchControllerRequest()
        msg.stop_controllers = self.controller_names
        msg.strictness = 2
        
        rospy.wait_for_service(self.namespace + '/controller_manager/switch_controller', timeout=2.0)
        try:
            self.__switch_controllers(msg)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def start_all_controllers(self) -> NoReturn:
        """
        Start all controllers
        
        :return:
        """
        msg = SwitchControllerRequest()
        msg.start_controllers = self.controller_names
        msg.strictness = 2
        
        rospy.wait_for_service(self.namespace + '/controller_manager/switch_controller', timeout=2.0)
        try:
            self.__switch_controllers(msg)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
        
        return
    
    def load_all_controllers(self) -> NoReturn:
        """
        Load all controllers
        
        :return:
        """
        for name in self.controller_names:
            self.load_controller(name)
        self.start_all_controllers()
        return
    
    def load_controller(self, name: str) -> NoReturn:
        """
        Load a controller
        
        :param name: controller name
        :return:
        """
        msg = LoadControllerRequest()
        msg.name = name
        rospy.wait_for_service(self.namespace + '/controller_manager/load_controller', timeout=2.0)
        try:
            self.__load_controller(msg)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def unload_all_controllers(self) -> NoReturn:
        """
        Unload all controllers
        
        :return:
        """
        self.stop_all_controllers()
        for name in self.controller_names:
            self.unload_controller(name)
        return
    
    def unload_controller(self, name: str):
        """
        Unload a controller
        
        :param name: controller name
        :return:
        """
        msg = UnloadControllerRequest()
        msg.name = name
        rospy.wait_for_service(self.namespace + '/controller_manager/unload_controller', timeout=2.0)
        try:
            self.__unload_controller(msg)
        except rospy.ServiceException as e:
            print('Service did not process request:' + str(e))
    
    def cancel_all_controllers_goals(self) -> NoReturn:
        """
        Cancel to all controllers the goal, only if they are TrajectoryController
        
        :return:
        """
        # TODO cancel_all_goals only for trajecty controllers
        for name in self.controller_names:
            self.controller[name].cancel_all_goals()
    
    @property
    def params(self):
        p = {'n_controllers': self.n_controllers}
        for name in self.controller_names:
            p.update({name: self.controller[name].params})
        return p
