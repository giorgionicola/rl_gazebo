import rospy
from rl_gazebo_env.robots.Robot import Robot
from rl_gazebo_msgs.srv import SetCylSize
from rl_gazebo_env.base.Utils import *


# TODO da mettere a posto pesantemente

class HumanCylinder(Robot):
    """
    Class HumanCylinder. The human is depicted as a cyliinder walking that changes diomensions
    
    """
    
    def __init__(self,
                 model_name='human',
                 walking_range=None,
                 target_toll=0.1,
                 max_speed=0.5,
                 min_speed=0.1,
                 max_freq_speed=0.2,
                 prob_speed_null=0.05,
                 prob_stop_on_target=0.1,
                 prob_restart=0.01,
                 fix_speed=True,
                 fix_maxspeed=False,
                 fix_radius=False,
                 max_radius=0.5,
                 min_radius=0.1,
                 max_freq_radius=0.2,
                 fix_length=False,
                 max_length=2.0,
                 min_length=1.5,
                 max_freq_length=2,
                 normalization_up=1,
                 normalization_low=0,
                 normalization=False):
        
        self.__model_name = model_name
        
        self.normalization = normalization
        self.dof = 2
        
        link = []
        joint_start_pos = [0.0, 0.0]
        joint_start_vel = [0.0, 0.0]
        
        if walking_range is None:
            walking_range = [[-2, 0.5],
                              [1, 2.0]],
        else:
            assert np.array(walking_range).shape == (2, 2), 'must be a 2x2 '
            for i in range(2):
                assert walking_range[0][i] < walking_range[1][i], 'Axis ' + str(i) + ' has min value > max value'
        
        joint_vel_limit = [[-max_speed for _ in range(2)],
                           [max_speed for _ in range(2)]]
        joints_groups = {
            'jointgroup_vel': {
                'trasl_x': {
                    'pos_limits': [walking_range[0][0], walking_range[0][1]],
                    'vel_limits': [-max_speed, max_speed],
                    'effort_limits': 330.0,
                    'wrap': False,
                    'wrap_limits': None},
                'trasl_y': {
                    'pos_limits': [walking_range[1][0], walking_range[1][1]],
                    'vel_limits': [-max_speed, max_speed],
                    'effort_limits': 330.0,
                    'wrap': False,
                    'wrap_limits': None},
            }}
        
        controller = {'jointgroup_vel_controller': {'joints_group': 'jointgroup_vel',
                                         'type': 'joint_group_controller',
                                         'hd_interface': 'velocity',
                                         'limits': joint_vel_limit
                                         }
                      }
        
        super().__init__(model_name=model_name,
                         joints_groups=joints_groups,
                         normalization_up=normalization_up,
                         normalization_low=normalization_low,
                         controllers=controller,
                         reset_type='custom',
                         movement_type='custom',
                         learning_robot=False,
                         links=link,
                         joint_start_pos=joint_start_pos,
                         joint_start_vel=joint_start_vel)
        
        self.target_toll = target_toll
        self.fix_speed = fix_speed
        if not self.fix_speed:
            assert fix_maxspeed is False, 'fix_maxspeed cannot be True unless fix_speed == True'
        self.fix_maxspeed = fix_maxspeed
        assert max_speed >= min_speed, 'Max speed must greater or equal Min speed'
        self.max_speed = max_speed
        assert min_speed >= 0, 'Minimum speed must be a positive value'
        self.min_speed = min_speed
        
        self.fix_radius = fix_radius
        assert max_radius >= min_radius, 'Max radius must greater or equal Min radius'
        self.max_radius = max_radius
        assert min_radius >= 0, 'Min radius must be a positive value'
        self.min_radius = min_radius
        self.radius_max_freq = max_freq_radius
        self.radius = None
        
        self.fix_length = fix_length
        assert max_length >= min_length, 'Max length must greater or equal Min length'
        self.max_length = max_length
        assert min_length >= 0, 'Min length must be a positive value'
        self.min_length = min_length
        self.length_max_freq = max_freq_length
        self.length = None
        
        self.speed_max_freq = max_freq_speed
        self.prob_speed_null = prob_speed_null
        self.prob_stop_on_target = prob_stop_on_target
        self.prob_restart = prob_restart
        self.stop = False
        self.walk_speed_f = None
        
        if not self.fix_radius or not self.fix_length:
            self.__set_cyl_size = rospy.ServiceProxy('/' + model_name + '/set_cyl_size', SetCylSize)

    def get_state(self):
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
    
        self.state = joint_positions_sorted + joint_velocities_sorted + [self.radius]
    
        return self.state

    def normalize_state(self):
        """

        :return: state normalized
        """
            
        state_normal = self.normalize(self.state[:self.joints.dof], self.joints.pos_limits)
        state_normal += self.normalize(self.state[self.joints.dof:-1], self.joints.vel_limits)
        state_normal += self.normalize([self.radius], [[self.min_radius], [self.max_radius]])
    
        return state_normal
    
    def update_human(self, time):
        """
        
        :param time:
        :return:
        """
        
        self.update_speed(time)
        self.update_size(time)
    
    def custom_reset(self):
        """
        
        :return:
        """
        
        rand_position = np.random.uniform(low=self.joints.pos_limits[0], high=self.joints.pos_limits[1]).tolist()
        self.stop = False
        
        if self.fix_radius:
            self.radius_f = lambda t: self.max_radius
        else:
            self.radius_f = pseudo_random_function(max_ampl=self.max_radius,
                                                   min_ampl=self.min_radius,
                                                   max_freq=self.radius_max_freq)
        
        if self.fix_length:
            self.length_f = lambda t: self.max_length
        else:
            self.length_f = pseudo_random_function(max_ampl=self.max_length,
                                                   min_ampl=self.min_length,
                                                   max_freq=self.length_max_freq)
        
        if not self.fix_radius or not self.fix_length:
            self.update_size(0)
        
        self.walk_speed_f = self.new_walk_speed_f()
        start_speed = self.walk_speed_f(0)
        start_speed = clip_value(start_speed, self.max_speed, self.min_speed)
        dist_unit_vector = self.new_target(rand_position)
        speed_vector = [i * start_speed for i in dist_unit_vector]
        
        self.set_joint_states(joints=self.joints.names, position=rand_position, velocity=speed_vector)
        
        if self.normalization:
            if not self.fix_radius and not self.fix_length:
                state = self.normalize(rand_position, self.joints.pos_limits) +\
                        self.normalize(speed_vector, self.joints.vel_limits) +\
                         self.normalize(self.radius, [[self.min_radius], [self.max_radius]]) +\
                         self.normalize(self.length, [[self.min_length], [self.max_length]])
            elif not self.fix_radius and self.fix_length:
                
                state = self.normalize(rand_position, self.joints.pos_limits) + \
                        self.normalize(speed_vector, self.joints.vel_limits) + \
                        self.normalize([self.radius], [[self.min_radius], [self.max_radius]])
            elif self.fix_radius and not self.fix_length:
                state = self.normalize(rand_position, self.joints.pos_limits) + \
                        self.normalize(speed_vector, self.joints.vel_limits) + \
                        self.normalize(self.length, [[self.min_length], [self.max_length]])
            elif self.fix_radius and self.fix_length:
                state = self.normalize(rand_position, self.joints.pos_limits) + \
                        self.normalize(speed_vector, self.joints.vel_limits)
        else:
            if not self.fix_radius and not self.fix_length:
                state = rand_position + speed_vector + [self.radius] + [self.length]
            
            elif not self.fix_radius and self.fix_length:
                state = rand_position + speed_vector + [self.radius]
            
            elif self.fix_radius and not self.fix_length:
                state = rand_position + speed_vector + [self.length]
            
            elif self.fix_radius and self.fix_length:
                state = rand_position + speed_vector
        
        return state
    
    def update_speed(self, t):
        """
        
        :param t:
        :return:
        """
        
        joint_positions, _, _ = self.get_joint_states(joints=self.joints.names)
        dist_vector, norm_dist_vector = self.distance_from_target(joint_positions)
        dist_unit_vector = dist_vector/norm_dist_vector
        
        if norm_dist_vector < self.target_toll:
            dist_unit_vector = self.new_target(joint_positions)
            if np.random.rand() < self.prob_stop_on_target:
                self.stop = True
                
        if not self.stop:
            self.walk_speed_f = self.new_walk_speed_f()
            speed = self.walk_speed_f(t)
            if not self.fix_speed:
                if speed < self.min_speed:
                    speed = 0
                else:
                    speed = clip_value(speed, self.max_speed, self.min_speed, )
        else:
            if np.random.rand() < self.prob_restart:
                self.walk_speed_f = self.new_walk_speed_f()
                speed = self.walk_speed_f(t)
                if speed < self.min_speed:
                    speed = 0
                else:
                    speed = clip_value(speed, self.max_speed, self.min_speed)
                self.stop = False
            else:
                speed = 0
        speed_vector = [i * speed for i in dist_unit_vector]
        self.controller_manager.send_action_to_controllers(action=speed_vector,
                                                           controller_order=['jointgroup_vel_controller'])
        return
    
    def new_walk_speed_f(self):
        """
        
        :return:
        """
        if np.random.rand() < self.prob_speed_null:
            walk_speed_f = lambda t: 0
        else:
            if self.fix_speed:
                if self.fix_maxspeed:
                    walk_speed_f = lambda t: self.max_speed
                else:
                    if np.random.rand() > self.prob_speed_null:
                        walk_speed_f = lambda t: np.random.uniform(low=self.min_speed, high=self.max_speed)
            else:
                walk_speed_f = pseudo_random_function(max_ampl=self.max_speed,
                                                      min_ampl=self.min_speed,
                                                      max_freq=self.speed_max_freq)
        return walk_speed_f
    
    def update_size(self, t):
        """
        
        :param t:
        :return:
        """
        
        self.length = self.length_f(t)
        self.radius = self.radius_f(t)
        
        if not self.fix_length:
            self.length = clip_value(self.fix_length, self.max_length, self.min_length)
        
        if not self.fix_radius:
            if self.radius < self.min_radius:
                self.radius = self.min_radius
            
            if self.radius > self.max_radius:
                self.radius = self.max_radius
        
        self.set_cyl_size(radius=self.radius, length=self.length)
        
        return
    
    def distance_from_target(self, position):
        """
        
        :param position:
        :return:
        """
        
        dist_vector = np.array([self.target[0] - position[0],
                       self.target[1] - position[1]])
        norm_dist_vector = np.linalg.norm(dist_vector)
        
        return dist_vector, norm_dist_vector
    
    def new_target(self, position):
        """
        
        :param position:
        :return:
        """
        while True:
            self.target = np.random.uniform(low=[self.joints.pos_limits[0][j] + self.radius for j in range(self.dof)],
                                            high=[self.joints.pos_limits[1][j] - self.radius for j in range(self.dof)])
            self.target = self.target.tolist()
            dist_vector, norm_dist_vector = self.distance_from_target(position)
            
            if norm_dist_vector > 5 * self.target_toll:
                break
        dist_unit_vector = dist_vector / norm_dist_vector
        return dist_unit_vector
    
    def set_cyl_size(self, radius, length):
        """
        
        :param radius:
        :param length:
        :return:
        """
        rospy.wait_for_service('/' + self.model_name + '/set_cyl_size', timeout=2.0)
        try:
            self.__set_cyl_size(radius=radius, length=length)
        except rospy.ServiceException as e:
            print("/set_cylinder_size service call failed")
