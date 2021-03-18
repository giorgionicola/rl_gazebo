import numpy as np
from rl_gazebo_env.robots.Robot import Robot
from rl_gazebo_env.end_effector import EndEffector

import symengine
from symengine import Matrix, pi, cos, sin, Lambdify

from typing import List


class UR(Robot):
    """
    
    """
    
    def __init__(self, model_name: str = 'ur', links: list = None, active_joints: list = None,
                 joint_start_pos: List[float] = None, joint_start_vel: List[float] = None,
                 joint_pos_lim: List[List[float]] = None, joint_vel_lim: List[List[float]] = None,
                 reset_type: str = 'random', range_rand_start_pos: List[float] = None,
                 range_rand_start_vel: List[float] = None, normalization_up: float = 1.0,
                 normalization_low: float = 0.0, controller: dict = None, learning_robot: bool = True,
                 end_effector: 'EndEffector' = None, T_start: List[List] = None):
        print(joint_start_pos)
        
        if links is None:
            links = ['shoulder_link',
                     'upper_arm_link',
                     'forearm_link',
                     'wrist_1_link',
                     'wrist_2_link',
                     'wrist_3_link']
        
        if active_joints is None:
            
            joints_groups = {
                'Arm': {
                    'shoulder_pan_joint': {
                        'pos_limits': [-2 * np.pi, 2 * np.pi] if joint_pos_lim is None else [joint_pos_lim[0][0],
                                                                                             joint_pos_lim[1][0]],
                        'vel_limits': [-2 * np.pi, 2 * np.pi] if joint_vel_lim is None else [joint_vel_lim[0][0],
                                                                                             joint_vel_lim[1][0]],
                        'effort_limits': 330.0,
                        'wrap': False,
                        'wrap_limits': None},
                    'shoulder_lift_joint': {
                        'pos_limits': [-2 * np.pi, 2 * np.pi] if joint_pos_lim is None else [joint_pos_lim[0][1],
                                                                                             joint_pos_lim[1][1]],
                        'vel_limits': [-2 * np.pi, 2 * np.pi] if joint_vel_lim is None else [joint_vel_lim[0][1],
                                                                                             joint_vel_lim[1][1]],
                        'effort_limits': 330.0,
                        'wrap': False,
                        'wrap_limits': None},
                    'elbow_joint': {
                        'pos_limits': [-2 * np.pi, 2 * np.pi] if joint_pos_lim is None else [joint_pos_lim[0][2],
                                                                                             joint_pos_lim[1][2]],
                        'vel_limits': [-2 * np.pi, 2 * np.pi] if joint_vel_lim is None else [joint_vel_lim[0][2],
                                                                                             joint_vel_lim[1][2]],
                        'effort_limits': 150,
                        'wrap': False,
                        'wrap_limits': None},
                    'wrist_1_joint': {
                        'pos_limits': [-np.pi, np.pi] if joint_pos_lim is None else [joint_pos_lim[0][3],
                                                                                     joint_pos_lim[1][3]],
                        'vel_limits': [-np.pi, np.pi] if joint_vel_lim is None else [joint_vel_lim[0][2],
                                                                                     joint_vel_lim[1][2]],
                        'effort_limits': 54.0,
                        'wrap': False,
                        'wrap_limits': None},
                    'wrist_2_joint': {
                        'pos_limits': [-np.pi, np.pi] if joint_pos_lim is None else [joint_pos_lim[0][4],
                                                                                     joint_pos_lim[1][4]],
                        'vel_limits': [-np.pi, np.pi] if joint_vel_lim is None else [joint_vel_lim[0][4],
                                                                                     joint_vel_lim[1][4]],
                        'effort_limits': 54.0,
                        'wrap': False,
                        'wrap_limits': None},
                    'wrist_3_joint': {
                        'pos_limits': [-np.pi, np.pi] if joint_pos_lim is None else [joint_pos_lim[0][5],
                                                                                     joint_pos_lim[1][5]],
                        'vel_limits': [-np.pi, np.pi] if joint_vel_lim is None else [joint_vel_lim[0][5],
                                                                                     joint_vel_lim[1][5]],
                        'effort_limits': 54.0,
                        'wrap': False,
                        'wrap_limits': None}}
            }
        
        else:
            raise NotImplementedError()
        
        if joint_start_pos is None:
            joint_start_pos = [0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0]
        
        if joint_start_vel is None:
            joint_start_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        if controller is None:
            controller = {'jointgroup_vel_controller': {'joints_group': 'Arm',
                                                        'type': 'joint_group_controller',
                                                        'hd_interface': 'velocity',
                                                        'limits': [[-1.0 for _ in range(6)], [1.0 for _ in range(6)]]
                                                        }
                          }
        super().__init__(model_name=model_name,
                         joints_groups=joints_groups,
                         links=links,
                         joint_start_pos=joint_start_pos,
                         joint_start_vel=joint_start_vel,
                         movement_type='action',
                         reset_type=reset_type,
                         range_rand_start_pos=range_rand_start_pos,
                         range_rand_start_vel=range_rand_start_vel,
                         normalization_up=normalization_up,
                         normalization_low=normalization_low,
                         controllers=controller,
                         learning_robot=learning_robot,
                         end_effector=end_effector,
                         sym_kinematic=True,
                         T_start=T_start)
        
        self.T_elbow_l = lambda x: None
        self.T_tcp_l = lambda x: None
        self.J_elbow = lambda x: None
        self.J_tcp = lambda x: None
    
    def symbolic_kinematics(self):
        
        d1 = 0.1273
        a2 = -0.612
        a3 = -0.5723
        d4 = 0.163941
        d5 = 0.1157
        d6 = 0.0922
        
        shoulder_offset = 0.220941
        elbow_offset = -0.1719
        
        shoulder_height = d1
        upper_arm_length = -a2
        forearm_length = -a3
        wrist_1_length = d4 - elbow_offset - shoulder_offset
        wrist_2_length = d5
        wrist_3_length = d6
        
        q0, q1, q2, q3, q4, q5 = symengine.var('q0 q1 q2 q3 q4 q5')
        
        if self.T_start is None:
            self.T_start = Matrix([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        else:
            self.T_start = Matrix(self.T_start)
        
        shoulder_pan_j = Matrix([[cos(q0), -sin(q0), 0.0, 0.0],
                                 [sin(q0), cos(q0), 0.0, 0.0],
                                 [0.0, 0.0, 1.0, shoulder_height],
                                 [0.0, 0.0, 0.0, 1.0]])
        
        shoulder_lift_j = Matrix([[cos(q1 + pi / 2), 0.0, sin(q1 + pi / 2), 0.0],
                                  [0.0, 1.0, 0.0, shoulder_offset],
                                  [-sin(q1 + pi / 2), 0.0, cos(q1 + pi / 2), 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        
        elbow_j = Matrix([[cos(q2), 0.0, sin(q2), 0.0],
                          [0.0, 1.0, 0.0, elbow_offset],
                          [-sin(q2), 0.0, cos(q2), upper_arm_length],
                          [0.0, 0.0, 0.0, 1.0]])
        
        wrist1_j = Matrix([[cos(q3 + pi / 2), 0.0, sin(q3 + pi / 2), 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [-sin(q3 + pi / 2), 0.0, cos(q3 + pi / 2), forearm_length],
                           [0.0, 0.0, 0.0, 1.0]])
        
        wrist2_j = Matrix([[cos(q4), -sin(q4), 0.0, 0.0],
                           [sin(q4), cos(q4), 0.0, wrist_1_length],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        
        wrist3_j = Matrix([[cos(q5), 0.0, sin(q5), 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [-sin(q5), 0.0, cos(q5), wrist_2_length],
                           [0.0, 0.0, 0.0, 1.0]])
        
        tcp = Matrix([[cos(pi / 2), -sin(pi / 2), 0.0, 0.0],
                      [sin(pi / 2), cos(pi / 2), 0.0, wrist_3_length],
                      [0.0, 0.0, 1.0, 0],
                      [0.0, 0.0, 0.0, 1.0]])
        
        T_elbow = self.T_start * shoulder_pan_j * shoulder_lift_j * elbow_j
        T_tcp = self.T_start * shoulder_pan_j * shoulder_lift_j * elbow_j * wrist1_j * wrist2_j * wrist3_j * tcp
        
        P_elbow = T_elbow[0:3, 3]
        P_tcp = T_tcp[0:3, 3]
        
        x_elbow = Matrix([P_elbow[0], P_elbow[1], P_elbow[2]])
        x_tcp = Matrix([P_tcp[0], P_tcp[1], P_tcp[2]])
        
        J_elbow = Matrix([[x_elbow[0].diff(q0), x_elbow[0].diff(q1), x_elbow[0].diff(q2)],
                          [x_elbow[1].diff(q0), x_elbow[1].diff(q1), x_elbow[1].diff(q2)],
                          [x_elbow[2].diff(q0), x_elbow[2].diff(q1), x_elbow[2].diff(q2)]])
        
        J_tcp = Matrix([[x_tcp[0].diff(q0), x_tcp[0].diff(q1), x_tcp[0].diff(q2), x_tcp[0].diff(q3), x_tcp[0].diff(q4),
                         x_tcp[0].diff(q5)],
                        [x_tcp[1].diff(q0), x_tcp[1].diff(q1), x_tcp[1].diff(q2), x_tcp[1].diff(q3), x_tcp[1].diff(q4),
                         x_tcp[1].diff(q5)],
                        [x_tcp[2].diff(q0), x_tcp[2].diff(q1), x_tcp[2].diff(q2), x_tcp[2].diff(q3), x_tcp[2].diff(q4),
                         x_tcp[2].diff(q5)]])
        
        self.T_elbow_l = Lambdify([q0, q1, q2], T_elbow, backend='llvm')
        self.T_tcp_l = Lambdify([q0, q1, q2, q3, q4, q5], T_tcp, backend='llvm')
        self.P_elbow_l = Lambdify([q0, q1, q2], P_elbow, backend='llvm')
        self.P_tcp_l = Lambdify([q0, q1, q2, q3, q4, q5], P_tcp, backend='llvm')
        self.J_elbow_l = Lambdify([[q0, q1, q2]], J_elbow, backend='llvm')
        self.J_tcp_l = Lambdify([[q0, q1, q2, q3, q4, q5]], J_tcp, backend='llvm')
    
    def get_cartesian_state(self):
        joints_state = self.get_state()
        joints_position = joints_state[0:6]
        joints_speed = joints_state[6:]
        
        P_elbow = self.P_elbow_l(joints_position[0:3])
        P_tcp = self.P_tcp_l(joints_position)
        
        V_elbow = np.matmul(self.J_elbow_l(joints_position[0:3]), np.array(joints_speed[0:3]).reshape(3, 1))
        V_tcp = np.matmul(self.J_tcp_l(joints_position), np.array(joints_speed).reshape(6, 1))
        
        return P_elbow, P_tcp, V_elbow, V_tcp
