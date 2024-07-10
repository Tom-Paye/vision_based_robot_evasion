#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
# import messages_fr3
from messages_fr3.msg import Array2d

import vision_based_robot_evasion.kalman as kalman
import vision_based_robot_evasion.visuals as visuals

import numpy as np
import math
import time
import copy
import logging

"""
Right now, this function just reads values for robot and body positions whenever it cans,
and updates accordingly.

This makes speed estimation kinda crap, especially for the human.
Reading the timestamp of received info should help

TODO: Restructure the body geom class so it contains all info for both the arms and torso
TODO: Calculate body speed for the robot and human
    --> Robot : We read the jacobians, so it should be doable by reading joint velocities and multiplying
    --> Human : Naive approach: Divide motion between reads by timestamp diff btw/ reads
                    Problem: High noise
                Naive solution : smoothed model from past values with high weight, new
                measurement with low weight, weighted average of both?
TODO: Create clever law combining effects of distance and relative speed to apply force
    --> multiplicative force calculation : F = A x B
        A is a term depending on distance: Fmax * [!- max(D - Dmin, 0)]
        B is a term depending on speed : exp(V) => smaller force if we are moving away
"""

class geom_transformations:


    def Rz(self, a=0):
        c, s = np.cos(a), np.sin(a)
        T = np.array([ [c, -s, 0, 0],
                    [s, c , 0, 0],
                    [0 , 0, 1, 0],
                    [0 , 0, 0, 1]])
        return T

    def Ry(self, a=0):
        c, s = np.cos(a), np.sin(a)
        T = np.array([ [c , 0, s, 0],
                    [0 , 1, 0, 0],
                    [-s, 0, c, 0],
                    [0 , 0, 0, 1]])
        return T

    def Rx(self, a=0):
        c, s = np.cos(a), np.sin(a)
        T = np.array([ [1, 0, 0 , 0],
                    [0, c, -s, 0],
                    [0, s, c , 0],
                    [0, 0, 0 , 1]])
        return T

    def translation(self, a=[0, 0, 0]):
        x, y, z = a[0], a[1], a[2]
        T = np.array([ [1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])
        return T

def joint_to_cartesian(joint_states= [0., -0.8, 0., -2.36, 0., 1.57, 0.79]):
    """
    Transform joint states into joint poisitions in cartesian space
    TODO: More accurate volumetric description of the robot
    CURRENT GOAL:

    INPUT---------------------------

    joint_states: [7] vector of joint angles in radians TODO: figure out what position corresponds to 0 rads for each joint

    OUTPUT--------------------------

    joint_positions: [7, 3] array of cartesian positions of every joint

    joint_rotations: [7, 3, 3] array of cartesian rotations of every joint axis

    """

    """
    Geometry is taken from https://www.generationrobots.com/media/franka-emika-research-3-robot-datasheet.pdf
    or here https://frankaemika.github.io/docs/control_parameters.html 
    https://www.researchgate.net/figure/The-Elementary-Transform-Sequence-of-the-7-degree-offreedom-Franka-Emika-Panda_fig1_361785335 
    Because those specifications do not allow us to easily describe the robot as a set of cylinders,
    I made some modifications:
    - define the position of A3 as 316-82 = 234 mm away from A2
    - define A5 as 384-82 = 302mm away from A6
    - define A7 as 88 mm away from A6
    - add an 'A8' to represent the gripper, at distance 107mm from A7
    """

    link_lengths = [    [0, 0, 0.333],\
                        np.zeros(3),\
                        [0, 0, 0.316],\
                        [0.0825, 0, 0],\
                        [-0.0825, 0, 0.384],\
                        np.zeros(3),\
                        [0, 0, 0.107],\
                        [-0.088, 0, 0] ]
    
    # joint_states= np.zeros(8)

    Rx, Ry, Rz, translation = geom_transformations.Rx, geom_transformations.Ry, geom_transformations.RZ,\
                                geom_transformations.translation
    
    T00 = np.zeros([4,4])
    T00[3, 3] = 1
    
    T01 = Rz(joint_states[0]) @ translation(link_lengths[0])

    T02 = T01 @ Ry(joint_states[1])

    T03 = T02 @ Rz(joint_states[2]) @ translation(link_lengths[2])

    T04 = T03 @ Ry(-joint_states[3]) @ translation(link_lengths[3])
    
    T05 = T04 @ Rz(joint_states[4]) @ translation(link_lengths[4])

    T06 = T05 @ Ry(-joint_states[5])

    T07 = T06 @ Ry(joint_states[6]) @ translation(link_lengths[7]) @ Rx(np.pi) @ translation(link_lengths[6])

    Transforms = np.array([T00, T01, T02, T03, T04, T05, T06, T07])

    joint_positions = Transforms[:, 0:3, 3].reshape(8, 3)

    return joint_positions
    
    



    
def link_dists(pos_body, pos_robot):
    """
    Given the positions of two bodies, this function returns:
        - the minimum distance between them
        - the points on each body which are closest to the other
        - the direction connecting both bodies
        
    The bodies are considered as chains, not webs (each keypoint is connected to two other joints MAX)
    
    
    
    INPUT---------------------------
    Takes pos_body: [M, 3] array of keypoint positions IN PHYSICALLY CONSECUTIVE ORDER
    (i.e. don't feed it a hand keypoint followed by the head keypoint or it will assume you have antennae)

    Takes pos_robot: [N, 3] vector corresponding to robot joint positions (locations in cartesian space)

    OUTPUT--------------------------
    Returns dist: [P, Q] array of distances
        P = max(M, N) --> one row for each convolution of the longer effector positions on the shorter effector positions.
            Longer or shorter refers to the number of joints, not physical length
        Q = max(M, N)-1 --> because we only consider the distances from the P-1 segments of the shortest effector
        NOTE: in every row, there is one distance which corresponds to a segment connecting the longest effector end-to-end directly
            This distance should not be taken into account for control, and is set to 10m
        NOTE: dist[i, j] is the distance between the jth link of the smaller effector and the j+ith link of the longer one (mod P)

    Returns direc: [P, Q, 3] array of unit direction vectors along the shortest path between each body_segment-robot_segment pair
        if a segment on the robot is found to be intersecting a segment on the body (dist[i, j] == 0):
            the corresponding vector  direc[i, j, :] is [0, 0, 0]
        NOTE: The direct vectors point from the body to the robot

    Returnt t : [P, Q] array of position of 'intersection point' on each segment of the robot
        0 <= t[i, j] <= 1, as it represents a fraction of the segment length, starting from whichever point has a lower index in pos_robot
        Intersection point : on a segment A, the closest point to the segment B we're measuring a distance to

    Returnt u : [P, Q] array of position of 'intersection point' on each segment of the body
        0 <= u[i, j] <= 1, as it represents a fraction of the segment length, starting from whichever point has a lower index in pos_body

    NOTE: Temporarily modified so discm direc, t and u are only output for the minimum distances, not the whole arrays

    return closest_r : [k] array of indices corresponding to the segments of the robot where the closest points are located
    return closest_b : [k] array of indices corresponding to the segments of the body where the closest points are located

        
    https://www.sciencedirect.com/science/article/pii/0020019085900328 
    
    """


    # TODO: // implement marking of imaginary joints
    # TODO: // articulate body geometry into a format suitable for this function
    # TODO: Make the Kalman Filter 3D
    # TODO: Create estimation of body speeds
    # TODO: make kalman filter real-time
    # TODO: // fix the coordinate system
    
    logger = logging.getLogger(__name__)

    #####################
    # # FOR SIMULATION AND TESTING

    # links_r = np.array([[0, 0, 0], [1, 1, 0]])
    # links_b = np.array([[0, 2, 0], [1, 2, 0]])

    # links_r = np.array([[0, -3, 0], [3, 0, 0]])
    # links_b = np.array([[0, 0, 0], [3, -3, 0]])

    # links_r = np.array([[0, 0, 0], [1, 1, 0]])
    # links_b = np.array([[0, 1, 0], [1, 2, 0]])

    # links_r = np.array([[0, 0, 0], [2, 0, 0]])
    # links_b = np.array([[1, 2, 0], [1, 1, 0]])

    # links_r = np.array([[0, 0, 0], [1, 1, 0], [0, -3, 0], [3, 0, 0],
    #                     [0, 0, 0], [1, 1, 0], [0, 0, 0], [2, 0, 0],
    #                     [0, 0, 0], [1, 1, 0], [0, -3, 0], [3, 0, 0],
    #                     [0, 0, 0], [1, 1, 0], [0, 0, 0], [2, 0, 0]])
    # links_b = np.array([[0, 2, 0], [1, 2, 0], [0, 0, 0], [3, -3, 0],
    #                     [0, 1, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0],
    #                     [0, 2, 0], [1, 2, 0], [0, 0, 0], [3, -3, 0],
    #                     [0, 1, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0]])
    
    # links_r = np.array([[0, 0, 0], [1, 1, 0], [1, 2, 0], [2, 1, 0],
    #                     [3, 1, 0]])
    # links_b = np.array([[1/2, 4, 0], [1, 3, 0], [2, 2, 0], [5/2, 3, 0],
    #                     [4, 3, 0]])
    # pos_b = copy.copy(links_b)
    # pos_r = copy.copy(links_r)
    
    #####################
    
    
    links_b, links_r = pos_body, pos_robot
    
    m = len(links_b)
    n = len(links_r)
    p, q = 0, 0

    #####
    
    if m > n:
        arr_to_roll = links_b
        static_array = links_r
    else:
        arr_to_roll = links_r
        static_array = links_b
    n_rolls = max(m, n)
    n_remove = min(m, n)
    mat_roll = np.array([static_array]*n_rolls)
    mat_static = np.array([static_array]*n_rolls)
    bad_segments = []

    for i in range(n_rolls):
        new_layer = np.roll(arr_to_roll, -i, axis=0)
        new_layer = new_layer[0:min(m, n),:]
        mat_roll[i,:] = new_layer
        bad_segments.append([i, n_rolls-i-1])
    bad_segments = np.array(bad_segments[n_rolls-n_remove+1:])

    if m > n:
        links_b = mat_roll
        links_r = mat_static
    else:
        links_r = mat_roll
        links_b = mat_static

    pseudo_links = np.array([links_r[:, :-1, :], links_b[:, :-1, :]])

    d_r = np.diff(links_r, axis=1)
    d_b = np.diff(links_b, axis=1)
    d_rb = np.diff(pseudo_links, axis=0)[0]

    # calc length of both segments and check if parallel
    len_r = np.linalg.norm(d_r, axis=-1)**2
    len_b = np.linalg.norm(d_b, axis=-1)**2
    if 0 in len_r or 0 in len_b:
        logger.info('err: Link without length')
    R = np.einsum('ijk, ijk->ij', d_r, d_b)
    denom = len_r*len_b - R**2
    S1 = np.einsum('ijk, ijk->ij', d_r, d_rb)
    S2 = np.einsum('ijk, ijk->ij', d_b, d_rb)

    paral = (denom<0.001)
    t = (1-paral) * (S1*len_b-S2*R) / denom
    t = np.nan_to_num(t)
    t = np.clip(t, 0, 1)

    u = (t*R - S2) / (len_b)
    u = np.nan_to_num(u)
    u = np.clip(u, 0, 1)

    t = (u*R + S1) / len_r
    t = np.nan_to_num(t)
    t = np.clip(t, 0, 1)


    tp = np.transpose(np.array([t]*3), (1, 2, 0))
    up = np.transpose(np.array([u]*3), (1, 2, 0))
    diffs_3d = np.multiply(d_r, tp) - np.multiply(d_b, up) - d_rb
    dist = np.sqrt(np.sum(diffs_3d**2, axis=-1))

    distp = copy.copy(dist)
    dist = np.around(dist, decimals=6)
    distp[dist == 0] = 1000
    direc = np.multiply(diffs_3d, 1 / distp[:, :,  np.newaxis])
  
    
    [intersec_b_link, intersec_r_link] = np.nonzero(distp == 1000)
    direc[intersec_b_link, intersec_r_link, :] = [0, 0, 0]  # marks the places where the body and robot are believed to clip into another
    dist[bad_segments[:, 0], bad_segments[:, 1]] = 10 # marks the distances comparing imaginary axes (those that link both ends of each limb directly, for example)
    
    t = np.around(t, decimals=6)
    u = np.around(u, decimals=6)

    [i, j] = np.where(dist == np.min(dist))
    t = t[i, j]
    u = u[i, j]
    dist = dist[i, j]
    direc = direc[i, j,:]
    if m > n:
        closest_r = j
        closest_b = (j+i-1)%m
    else:
        closest_b = j
        closest_r = (j+i)%n

    # #######################
    # # Plots for testing purposes
    # class geom(): pass
    # geom.arm_pos = pos_b
    # geom.trunk_pos = pos_b
    # geom.robot_pos = pos_r
    # geom.arm_cp_idx = closest_b
    # geom.u = u
    # geom.trunk_cp_idx = closest_b
    # geom.v = u
    # geom.robot_cp_arm_idx = closest_r
    # geom.s = t
    # geom.robot_cp_trunk_idx = closest_r
    # geom.t = t

    # visuals.plot_skeletons(0, geom)
    # #######################

    # convert all results so only the 'base' end of a link is used (i.e. never have t or u == 1)
    closest_r = closest_r + np.floor(t)
    t = t * (1 - np.floor(t))
    
    closest_b = closest_b + np.floor(u)
    u = u * (1 - np.floor(u))

    if len(closest_r) > 1:
        # remove repeated values so we only have separate pairs
        full_info = np.hstack((dist[:, np.newaxis], direc,\
                               t[:, np.newaxis], u[:, np.newaxis], closest_r[:, np.newaxis], closest_b[:, np.newaxis]))
        unq = np.unique(full_info, axis=0)
        dist = unq[:,0]
        direc = unq[:,1:4]
        t = unq[:,4]
        u = unq[:,5]
        closest_r = unq[:,6].astype(int)
        closest_b = unq[:,7].astype(int)

    closest_r = closest_r.astype(int)
    closest_b = closest_b.astype(int)


    return dist, direc, t, u, closest_r, closest_b


class bbox_generator(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)

        self.initialize_variables()

        self.subscription_data = self.create_subscription(
            PoseArray,
            'kpt_data',
            self.data_callback,
            10)
        self.subscription_data  # prevent unused variable warning

        self.robot_joint_state = self.create_subscription(
            JointState,
            'robot_data',
            self.get_robot_joints,
            10)
        
        self.robot_joint_velocity = self.create_subscription(
            JointState,
            'robot_data',
            self.get_robot_joints,
            10)
        
        self.force_publisher_ = self.create_publisher(Array2d, 'repulsion_forces', 10)

        self.timer = self.create_timer(0.01, self.dist_callback)
        # self.timer = self.create_timer(0.01, self.kalman_callback)
        
    
    def initialize_variables(self):
        self.reset = 0.0
        self.x = []
        self.bodies = {}
        self.subject = '1'
        self.placeholder_Pe = np.array([[0., 0., 0.],
                                        [0., -.1, .3],
                                        [0., .1, .3],
                                        [0., 0., .6],
                                        [.1, .1, .7],
                                        [.1, -.1, .7],
                                        [.2, 0., .8],
                                        [.9, .2, .8],
                                        [.9, 0., .8],
                                        [1., 0., .8],
                                        [1., -.1, .6],
                                        [1., .1, .6],])    # placeholder end effector positions
        self.fig = 0
        self.max_dist = 3      # distance at which the robot feels a force exerted by proximity to a human
        self.min_dist = 0     # distance at which the robot is most strongly pushed bach by a human

    def kalman_callback(self):

        if np.any(self.bodies) and self.subject in self.bodies:
            if not np.any(self.x):
                self.x = [self.bodies[self.subject][0][0][0]]
            else:
                self.x.append(self.bodies[self.subject][0][0][0])
        if len(self.x)>1000:
            kalman.kaltest(self.x)
        

    def dist_callback(self):
        if self.subject in self.bodies:
            self.get_logger().debug('self.reset, np.any(self.bodies[self.subject][self.subject]):')
            self.get_logger().debug(str(self.reset and np.any(self.bodies[self.subject][0])))
            # Make the body outline fit the head
            if self.reset and np.any(self.bodies[self.subject][0]):
                # make one line from the left hand to the right
                arms = np.concatenate([np.flip(self.bodies[self.subject][0], axis=0), self.bodies[self.subject][1]])

                robot_pos = joint_to_cartesian(self.robot_joint_state)

                arms_dist, arms_direc, arms_t, arms_u, c_r_a, c_a_r = link_dists(arms, robot_pos) # self.placeholder_Pe, robot_pos
                trunk_dist, trunk_direc, trunk_t, trunk_u, c_r_t, c_t_r = link_dists(self.bodies[self.subject][0], robot_pos)
                # self.placeholder_Pe, robot_pos



                ###############
                # Plotting the distances
                class geom(): pass
                geom.arm_pos = arms
                geom.trunk_pos = self.bodies[self.subject][2]
                geom.robot_pos = robot_pos  # self.placeholder_Pe, robot_pos
                geom.arm_cp_idx = c_a_r
                geom.u = arms_u
                geom.trunk_cp_idx = c_t_r
                geom.v = trunk_u
                geom.robot_cp_arm_idx = c_r_a
                geom.s = arms_t
                geom.robot_cp_trunk_idx = c_r_t
                geom.t = trunk_t

                self.fig = visuals.plot_skeletons(self.fig, geom)

                ###############

                min_dist_arms = np.min(arms_dist)
                min_dist_trunk = np.min(trunk_dist)
                # min_dist = min(min_dist_arms, min_dist_trunk)
                # self.get_logger().info('Minimum distance:')
                # self.get_logger().info(str(min_dist))
                if min_dist_arms < min_dist_trunk:
                    body_geom = {'dist':arms_dist, 'direc':arms_direc,
                                't':arms_t      , 'u':arms_u,
                                'closest_r':c_r_a        , 'closest_b':c_a_r}
                else:
                    body_geom = {'dist':trunk_dist, 'direc':trunk_direc,
                                't':trunk_t      , 'u':trunk_u,
                                'closest_r':c_r_t        , 'closest_b':c_t_r}
                forces = self.force_estimator(body_geom, robot_pos)    # self.placeholder_Pe, robot_pos
                self.generate_repulsive_force_message(forces)                
                
    def force_estimator(self, body_geom, robot_pose):
        
        """
        Calculates the force imparted on the robot by the bounding box of another body
        This can be conceptualized as pushing on a part of the robot
        (as opposed to forces on the joint motors, which are determined in another function)

        For easier calculation, transform each force on a part of a link into:
        (a force at the end of that link closest to the base) + (a moment on that end)

        NOTE: This function assumes all joints are capable of the same torque.
        Another function should be implemented to give physical meanings to the output of this one,
        which merely gives forces/moments relative to the strongest force

        NOTE: The output of this function has one less row than the input, because we take 8 points
        (1 for each joint including the base) + 1 point for the EE, but we only output to the joints

        
        INPUT---------------------------
        body_geom: dict whose elements are the output of the "link_dists"
        
        robot_pose: [8, 3] vector corresponding to robot joint positions (locations in cartesian space)
        
        OUTPUT---------------------------
        forces: [7,3] vector of magnitudes of forces to be applied to the robot in 3D space
            force is given as a float between 0 and 1, 1 being the strongest
        
        // direc: [N, 3] array containing the direction along which the force is applied (with norm of 1)
        
        // application_segments: [N = 8] vector of robot segments on which to apply each force (segment 0 has an
                                                                                         extremity at the base)
        // application_dist: [N] vector of the distance along a segment at which a force is applied
        
        // force_vec: [N, 3] array containing the direction along which the force is applied scaled by the force (0 to 1)

        // moment:    [N, 3] array containing the direction along which the moment is applied scaled by the moment (0 to 1)

        TODO: Implement the rescaling, current forces / moments are left unscaled
        
        """
        

        direc = body_geom['direc']                      # unit vectors associated with each force
        application_segments = body_geom['closest_r']   # segment on which the force is applied
        application_dist = body_geom['t']               # fraction of each segment at which each force is applied

        levers = np.diff(robot_pose, axis = 0)          # vectors pointing along the robot links
        

        dists = copy.copy(body_geom['dist'])

        full_force_vec = np.zeros([8, 6])

        for i, force in enumerate(dists):
            vec = direc[i,:]
            seg = application_segments[i]
            dist = application_dist[i]

            force = 1 - (force-self.min_dist)/(self.max_dist - self.min_dist)
            force_vec = force * vec
            moment = np.zeros(3)

            if dist > 0:
                lever = levers[seg,:]
                moment = np.cross(lever*dist, force_vec)
            
            full_force_vec[seg,:] = full_force_vec[seg,:] + np.hstack((force_vec, moment))
            
        output = full_force_vec

        return output

        


    def generate_repulsive_force_message(self, forces):
        
        """
        Given an array of the repulsion forces to impose on the robot, it outputs a vector message to send them to 
        Curdin's code

        INPUT---------------------------
        forces : [7, 6] array of forces and moments applied to every joint of the robot
        
        OUTPUT---------------------------
        rep_msg: Custom ROS2 message of type Array2d containing:
            - array  : [n] vector (flattened array) encoding each component of each force on each joint
            - height : int number of rows of the unflattened array
            - width  : int number of cols of the unflattened array
        NOTE: [height, width] should be [6, 7] for 6DOF and 7 joints, but the C++ Eigen library is 
        COLUMN MAJOR order, which may need added translation
        """
        
        forces_flattened = forces.flatten(order='C')

        force_message = Array2d()
        force_message.array = list(forces_flattened.astype(float))
        [force_message.height, force_message.width]  = np.shape(forces)
        self.force_publisher_.publish(force_message)

        a = 2
        
    
    def force_translator(self, robot_pose, force_vec, application_segments, application_dist):
        
        """
        Translates body forces on the robot to moments on its joints
        --> assumes infinitely rigid links and 1 DOF joints
        --> for each joint, assumes all other joints are rigid
        NOTE: This is now done within Curdin's controller, this function is unnecessary
        
        INPUT---------------------------
        
        robot_pose: [M, 3] vector corresponding to robot joint positions (locations in cartesian space)
            NOTE: joints must be given in physically consecutive order (ie consecutive joints are physically connected
                                                                        by a single segment of the robot)
        
        force_vec: [N, 3] array containing the direction along which the force is applied scaled by the force (0 to 1)
            --> 1 line for each separate BODY force, NOT joint torque
        
        application_segments: [N] vector of robot segments on which to apply each force (segment 0 has an
                                                                                         extremity at the base)
        application_dist: [N] vector of the distance along a segment at which a force is applied

        jacobians: [N, 6, N] array containing the jacobians of each joint
            1st dim: Which joint is this jacobian for
            2nd dim: Which translation / rotation
            3rd dim: onto which joint does this map (joint torque = jacobian * Force on end effector)
            NOTE: the joints are the ones at the 'END' of the corresponding links
        
        OUTPUT---------------------------
        
        torques: [M] vector of torques on the robot's joints
            --> 1 row per joint
            TODO: define robot position by joint positions to disambiguate torque direction
            TODO: Rescale torques to account for the compliance of other joints and redistribute to prevent stall
            # TODO: add a function to calculate robot cartesian positions from angles
            
        
        """
        '''
                Substituting the joint torque with the end-effector force pre-multiplied with the Jaco-
                bian transposed
                τ = JTe Fe (3.85)
                yields the end-effector dynamics
                - Robot Dynamics

                https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html

                https://frankaemika.github.io/docs/franka_ros2.html#setup
                --> /home/tom/franka_ros2_ws/src/franka_ros2/franka_example_controllers/src
                    --> model_example_controller.cpp

                /home/tom/franka_ros2_ws/install/franka_semantic_components/include/franka_semantic_components
                franka_robot_model.hpp 
                --> describes the calculation of a jacobian
                    NOTE: Column major format?

                '''
        torques = np.zeros(len(robot_pose))
        
        for i, force in enumerate(force_vec):
            for j, joint in enumerate(robot_pose):
                
                # Determine the axis of the joint
                joint_axis = [1, 1, 1]
                
                # Determine moment of force on the joint
                force_pos = (robot_pose[application_segments[i],:]*(1-application_dist[i]))+\
                            (robot_pose[application_segments[i+1],:]*application_dist[i])
                force_relative_pos = force_pos - joint
                moment_raw = np.cross(force, force_relative_pos)
                
                # scale by the colinearity between moment and joint axis
                moment_scaled = np.dot(moment_raw, joint_axis)
                
                torques[j] += moment_scaled
        
        


    # def label_callback(self, msg):
    #     self.get_logger().info(msg.data)
    #     self.label = msg.data
    #     self.data = []
        
    def data_callback(self, msg):
        # self.get_logger().info(str(msg))
        # region = math.trunc(msg.w)
        limb_dict = {'left':0, 'right':1, 'trunk':2, '_stop':-1}
        region = msg.header.frame_id[2:]
        self.reset = ('stop' in msg.header.frame_id)


        if self.reset:
            if self.subject in self.bodies:
                for id in self.bodies.keys():
                    body = self.bodies[id]
                    # self.get_logger().info('Body' + id + ' Left side')
                    # self.get_logger().info(str(body[0]))
            else:
                pass
        else:
            body_id =msg.header.frame_id[0]
            if body_id == '-':
                body_id = int(msg.header.frame_id[0:2])
            else:
                body_id = int(msg.header.frame_id[0])

            poses_ros = msg.poses
            poses = []
            for i, pose in enumerate(poses_ros):
                poses.append([pose.position.x, pose.position.y, pose.position.z])
            poses = np.array(poses)

            threshold = 0.002
            if str(body_id) in self.bodies:
                self.bodies[str(body_id)][limb_dict[region]] = poses
            else:
                    self.bodies[str(body_id)] = [poses[0:3,:], poses[0:3,:], poses]

    def get_robot_joints(self, msg):
        """
        For now, it looks like the best we can do is get joint angles and transform that into a position manually
        so we just apply a matrix transformation to the list of joint angles to get the joint positions
        """

        







        


    



def main(args = None):
    
    rclpy.init(args=args)

    subscriber = bbox_generator()

    rclpy.spin(subscriber)




    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()
    
    print('done')
    
if __name__ == "__main__":
    main()