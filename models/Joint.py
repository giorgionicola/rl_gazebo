from typing import List, NoReturn
from rl_gazebo_env.base.Utils import wrap_values


class Joint:
    """
    class for storing a joint
    """
    
    def __init__(self, joint: dict):
        """
        
        :param joint_dict: dictionary with all joints infos
        
        the dictionary must have the following structure:
            - pos_limits (Optional [ lower_limit, upper_limit ])
            - vel_limits (Optional [ lower_limit, upper_limit ])
            - effort_limits (Optional [ lower_limit, upper_limit ])
            - wrap (Optional bool)
            - wrap_limits (Optional [ lower_limit, upper_limit ])
        """
        
        if 'pos_limits' in joint and joint['pos_limits'] is not None:
            assert type(joint['pos_limits']) is list, 'limits must be a list'
            assert len(joint['pos_limits']) == 2, 'limits length must be 2'
            assert joint['pos_limits'][0] < joint['pos_limits'][1], 'limits must be [lower, upper]'
            self.pos_limits = joint['pos_limits']
        else:
            self.pos_limits = [-float("inf"), float("inf")]
        
        if 'vel_limits' in joint and joint['vel_limits'] is not None:
            assert type(joint['vel_limits']) is list, 'limits must be a list'
            assert len(joint['vel_limits']) == 2, 'limits length must be 2'
            assert joint['vel_limits'][0] < joint['vel_limits'][1], 'limits must be [lower, upper]'
            self.vel_limits = joint['vel_limits']
        else:
            self.vel_limits = [-float("inf"), float("inf")]
        
        if 'effort_limit' in joint and joint['effort_limit'] is not None:
            assert type(joint['effort_limit']) is float or type(joint['effort_limit']) is int, \
                'effort limits must be a numeber or None'
            self.effort_limit = joint['effort_limits']
        else:
            self.effort_limit = float("inf")
        
        if 'wrap' in joint:
            assert type(joint['wrap']) is bool
            self.wrap = joint['wrap']
        else:
            self.wrap = False
        
        if 'wrap_limits' in joint and joint['wrap_limits'] is not None:
            assert type(joint['wrap_limits']) is list, 'limits must be a list'
            assert len(joint['wrap_limits']) == 2, 'limits length must be 2'
            assert joint['wrap_limits'][0] <= joint['wrap_limits'][1], 'limits must be [lower, upper]'
            self.wrap_limits = joint['wrap_limits']
        else:
            self.wrap_limits = None
        
        if self.wrap:
            assert self.wrap_limits is not None, 'If wrap is True, than wrap_limits should be set '
    
    def wrap_joint(self, value: float) -> float:
        """
        wrap the joint value based on wrap_limits
        
        :param value: joint value
        :return:
        """
        if self.wrap:
            return wrap_values(value=value, low_limit=self.wrap_limits[0], up_limit=self.wrap_limits[1])
        else:
            return value
    
    @property
    def params(self):
        return {'pos_limits': self.pos_limits,
                'vel_limits': self.vel_limits,
                'effort_limit': self.effort_limit,
                'wrap': self.wrap,
                'wrap_limits': self.wrap_limits}


class JointsGroup:
    """
    Class storing a set of Joint(), for robots each controller is attached to a single JointGroup
    
    the dictionary should have the following structure for each joint:
            - name (str)
                - pos_limits (Optional [ lower_limit, upper_limit ])
                - vel_limits (Optional [ lower_limit, upper_limit ])
                - wrap (Optional bool)
                - wrap_limits (Optional [ lower_limit, upper_limit ])
    """
    
    def __init__(self, joint_group: dict):
        
        assert type(joint_group) is dict, 'Each joints_group must be a dict'
        assert len(joint_group.keys()) > 0, 'Empty joints group'
        
        self.joints = dict()
        self.joints_name = []
        
        for j in joint_group.keys():
            self.joints[j] = Joint(joint_group[j])
            self.joints_name.append(j)
        self.dof = len(self.joints_name)
    
    @property
    def vel_limits(self) -> List[List[float]]:
        """
        Get joints vel limit for the joints in the jointgroup

        :return:
        """
        joint_vel_limits = [[0 for _ in range(self.dof)], [0 for _ in range(self.dof)]]
        for counter, j_n in enumerate(self.joints_name):
            joint_vel_limits[0][counter] = self.joints[j_n].vel_limits[0]
            joint_vel_limits[1][counter] = self.joints[j_n].vel_limits[1]
        return joint_vel_limits
    
    @property
    def pos_limits(self) -> List[List[float]]:
        """
        Get joint pos limit for the joints in the jointgroup

        :return:
        """
        joint_pos_limits = [[0 for _ in range(self.dof)], [0 for _ in range(self.dof)]]
        for counter, j_n in enumerate(self.joints_name):
            joint_pos_limits[0][counter] = self.joints[j_n].pos_limits[0]
            joint_pos_limits[1][counter] = self.joints[j_n].pos_limits[1]
        return joint_pos_limits
    
    @property
    def effort_limits(self) -> List[float]:
        """
        Get joints effort limit for the joints in the jointgroup

        :return:
        """
        joints_effort_limit = [0 for _ in range(self.dof)]
        for counter, j_n in enumerate(self.joints_name):
            joints_effort_limit[counter] = self.joints[j_n].effort_limit
        return joints_effort_limit
    
    @property
    def params(self):
        p = dict()
        for j_n in self.joints_name:
            p.update({j_n: self.joints[j_n].params})
        return p


class Joints:
    """
    Class storing one or more JointGroup()
    """
    
    def __init__(self, joint_groups: dict):
        """

        :param joint_group: dictionary containing all joints infos
        """
        
        assert type(joint_groups) is dict or joint_groups is None, 'joints must be a dict or None'
        if joint_groups is None:
            self.dof = 0
            self.names = None
            self.groups = None
            self.jointgroups_name = None
        else:
            assert all([type(k) is str for k in joint_groups]), 'The keys must be str'
            self.groups = dict()
            self.names = []
            self.jointgroups_name = []
            for jg in joint_groups.keys():
                self.jointgroups_name.append(jg)
                self.groups[jg] = JointsGroup(joint_groups[jg])
                self.names += self.groups[jg].joints_name
            self.dof = len(self.names)
    
    def wrap_joints(self, joints_value: List[float], order: List[str] = None) -> List[float]:
        """
        wrap joints value if joint has wrap=True
        
        :param joints_value: all joints values
        :param order: order of joints value, default is self.names
        :return:
        """
        assert type(joints_value) is list, 'joints_value must be a list'
        assert len(joints_value) == self.dof, 'You must give all joints even the one not to be wrapped'
        assert all([type(j) is float for j in joints_value]), 'all joints_value must be float'
        
        if order is None:
            order = self.names
        
        wrapped_joints = [0.0 for _ in range(self.dof)]
        for counter, j_n in enumerate(order):
            for jg in self.groups.values():
                if j_n in jg.joints_name:
                    wrapped_joints[counter] = jg.joints[j_n].wrap_joint(joints_value[counter])
                    break
        
        return wrapped_joints
    
    @property
    def vel_limits(self) -> List[List[float]]:
        """
        Get joints vel limit for all joints in all jointgroups
        
        :return:
        """
        joints_vel_limits = [[], []]
        for _, j_g in self.groups.items():
            joints_vel_limits[0] += j_g.vel_limits[0]
            joints_vel_limits[1] += j_g.vel_limits[1]
        return joints_vel_limits
    
    @property
    def pos_limits(self) -> List[List[float]]:
        """
        Get joint pos limit for all joints in all jointgroups
        
        :return:
        """
        joints_pos_limits = [[],[]]
        for _, j_g in self.groups.items():
            joints_pos_limits[0] += j_g.pos_limits[0]
            joints_pos_limits[1] += j_g.pos_limits[1]
        return joints_pos_limits
    
    def effort_limits(self) -> List[float]:
        """
        Get the joints effort limit for all the joints groups
        
        :return:
        """

        joints_eff_limits = []
        for _, j_g in self.groups.items():
            joints_eff_limits += j_g.effort_limits
        return joints_eff_limits
    
    @property
    def params(self):
        p = {'dof': self.dof}
        if self.dof > 0:
            for j_g in self.jointgroups_name:
                p.update({j_g: self.groups[j_g].params})
        else:
            p.update({'joints': None})
        return p
