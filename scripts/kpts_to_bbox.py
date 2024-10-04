#!/usr/bin/env python3

import rclpy
import rclpy.callback_groups
from rclpy.node import Node
# from geometry_msgs.msg import PoseArray
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from geometry_msgs.msg import TransformStamped
# from geometry_msgs.msg import Quaternion
# from tf2_msgs.msg import TFMessage
# from std_msgs.msg import String
# from sensor_msgs.msg import JointState
# from urdf_parser_py.urdf import URDF
# from urdf_parser_py.urdf import Robot
# import messages_fr3
from messages_fr3.msg import Array2d
from messages_fr3.msg import IrregularDistArray

import scripts.run_stats as run_stats
import vision_based_robot_evasion.visuals as visuals

import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time
import copy
import logging
# from pathlib import Path
from skimage.util.shape import view_as_windows

# import subprocess
# import tempfile
# from urdfpy import URDF
# import trimesh
# import threading

# import tf2_ros
# import tf_transformations



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


def link_dists(pos_body, pos_robot, max_dist, false_link=-1):
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
    
    logger = logging.getLogger('link_dists')

    #####################
    # # FOR SIMULATION AND TESTING

    # joints_r = np.array([[0, 0, 0], [1, 1, 0]])
    # joints_b = np.array([[0, 2, 0], [1, 2, 0]])

    # joints_r = np.array([[0, -3, 0], [3, 0, 0]])
    # joints_b = np.array([[0, 0, 0], [3, -3, 0]])

    # joints_r = np.array([[0, 0, 0], [1, 1, 0]])
    # joints_b = np.array([[0, 1, 0], [1, 2, 0]])

    # joints_r = np.array([[0, 0, 0], [2, 0, 0]])
    # joints_b = np.array([[1, 2, 0], [1, 1, 0]])

    # joints_r = np.array([[0, 0, 0], [1, 1, 0], [0, -3, 0], [3, 0, 0],
    #                     [0, 0, 0], [1, 1, 0], [0, 0, 0], [2, 0, 0],
    #                     [0, 0, 0], [1, 1, 0], [0, -3, 0], [3, 0, 0],
    #                     [0, 0, 0], [1, 1, 0], [0, 0, 0], [2, 0, 0]])
    # joints_b = np.array([[0, 2, 0], [1, 2, 0], [0, 0, 0], [3, -3, 0],
    #                     [0, 1, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0],
    #                     [0, 2, 0], [1, 2, 0], [0, 0, 0], [3, -3, 0],
    #                     [0, 1, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0]])
    
    # joints_r = np.array([[0, 0, 0], [1, 1, 0], [1, 2, 0], [2, 1, 0],
    #                     [3, 1, 0]])
    # joints_b = np.array([[1/2, 4, 0], [1, 3, 0], [2, 2, 0], [5/2, 3, 0],
    #                     [4, 3, 0]])
    # pos_b = copy.copy(joints_b)
    # pos_r = copy.copy(joints_r)
    
    #####################
    
    # TODO:create dummy joint situated in the hand between both fingers, to prevent
    # a link being made between the fingers
    # /TODO: run distance finder on all gazilion links, but during the force transfer, only consider the important 7

    joints_b, joints_r = pos_body, pos_robot # [n_jts x 3 coordinates]
    
    n_joints_b = len(joints_b)
    n_joints_r = len(joints_r)
    p, q = 0, 0

    #####
    
    # Only roll over the larger array
    if n_joints_b > n_joints_r:
        arr_to_roll = joints_b
        array_not_rolled = joints_r
    else:
        arr_to_roll = joints_r
        array_not_rolled = joints_b

    # For every new rotation of the rolled array, one of the links actually doesn't exists, and should be disregarded
    n_rolls = max(n_joints_b, n_joints_r)
    n_compared_rows = min(n_joints_b, n_joints_r)
    # mat_roll = np.array([array_not_rolled]*n_rolls) 
    mat_static = np.array([array_not_rolled]*n_rolls)
    bad_segments = []


    ######## Speed up with fancy indexing: https://stackoverflow.com/questions/57272516/numpy-multiple-numpy-roll-of-1d-input-array
    window_shape = (n_rolls,3)
    mat_to_roll = np.concatenate((arr_to_roll,arr_to_roll[:-1]))
    mat_roll = view_as_windows(mat_to_roll, window_shape)[:,0,0:n_compared_rows,:]
    
    bad_segments = np.vstack((np.arange(n_rolls-n_compared_rows+1,n_rolls), np.arange(n_compared_rows-2,-1, -1))).T
    if false_link > -1:
        h_start = false_link
        h_end = h_start - n_rolls
        bad_segments_arm_trunk = np.vstack((np.arange(0,n_rolls), np.arange(h_start,h_end, -1))).T
        bad_segments_arm_trunk = bad_segments_arm_trunk[bad_segments_arm_trunk[:,1]%n_rolls < n_compared_rows ,:]
        bad_segments = np.vstack((bad_segments, bad_segments_arm_trunk)).astype(int)
    ############################
    # for i in range(n_rolls):
    #     new_layer = np.roll(arr_to_roll, -i, axis=0)
    #     new_layer = new_layer[0:n_compared_rows,:]
    #     mat_roll[i,:] = new_layer
    #     bad_segments.append([i, n_rolls-1 - i])
    # bad_segments = np.array(bad_segments[n_rolls-n_compared_rows+1:]) # +1 because there is 1 less links than segments

    if n_joints_b > n_joints_r:
        joints_b = mat_roll         # [n_jts x n_jnts_2 x 3 coordinates]
        joints_r = mat_static
    else:
        joints_r = mat_roll
        joints_b = mat_static

    link_origins = np.array([joints_r[:, :-1, :], joints_b[:, :-1, :]])

    links_r = np.diff(joints_r, axis=1)     # [n_jts x n_jnts_2-1 x 3 coordinates]
    links_b = np.diff(joints_b, axis=1)
    links_r_b = np.diff(link_origins, axis=0)[0]    # [n_jts x n_jnts_2-1 x 3 coordinates], the [0] is to remove 4th dim

    # Step 1
    # calc length of both segments and check if parallel
    D1 = np.linalg.norm(links_r, axis=-1)**2         # [n_jts x n_jnts_2-1]
    D2 = np.linalg.norm(links_b, axis=-1)**2

    # for all zero-length links, pretend length is very small, so the distance is 0. We need to set their m to 0 later on
    # This exists solely so numpy will stop throwing runtime errors about division by zero
    zeros_r = D1 == 0
    D1[zeros_r] = 0.0001
    zeros_b = D2 == 0
    D2[zeros_b] = 0.0001

    # if 0 in D1 or 0 in D2:
        # logger.info('err: Link without length')

    R = np.einsum('ijk, ijk->ij', links_r, links_b)     # [n_jts x n_jnts_2-1]
    S1 = np.einsum('ijk, ijk->ij', links_r, links_r_b)
    S2 = np.einsum('ijk, ijk->ij', links_b, links_r_b)

    denom = D1*D2 - R**2                                # [n_jts x n_jnts_2-1]
    paral = (np.abs(denom)<0.001)
    denom[paral] = 0.0001


    # Step 2
    t = (1-paral) * (S1*D2-S2*R) / denom            # t corresponds to the robot. it is the fraction of length along each link
    # t = np.nan_to_num(t)                              [n_jts x n_jnts_2-1]
    t = np.clip(t, 0, 1)
    t[zeros_r] = 0

    # Step 3
    u = (t*R - S2) / (D2)                           # u corresponds to the robot. it is the fraction of length along each link
    # u = np.nan_to_num(u)
    u = np.clip(u, 0, 1)
    u[zeros_b] = 0

    # Step 4
    t = (u*R + S1) / D1
    # t = np.nan_to_num(t)
    t = np.clip(t, 0, 1)
    t[zeros_r] = 0

    # Step 5
    link_scaling_r = np.transpose(np.array([t]*3), (1, 2, 0))       # [n_jts x n_jnts_2-1 x 3 dims]
    link_scaling_b = np.transpose(np.array([u]*3), (1, 2, 0))
    diffs_3d = links_r * link_scaling_r - links_b * link_scaling_b - links_r_b  # 3d vectors btw each link pair
    dist = np.sqrt(np.sum(diffs_3d**2, axis=-1))    # DD    [n_jts x n_jnts_2-1]

    # Unit direction betwen each link pair. Where both links intersect, set the direction close to 0 instead
    distp = copy.copy(dist)
    dist = np.around(dist, decimals=6)
    distp[dist == 0] = 1000
    direc = np.multiply(diffs_3d, 1 / distp[:, :,  np.newaxis])     # [n_jts x n_jnts_2-1 x 3 dims]
  
    
    [intersec_b_link, intersec_r_link] = np.nonzero(distp == 1000)
    direc[intersec_b_link, intersec_r_link, :] = [0, 0, 0]  # marks the places where the body and robot are believed to clip into another
    dist[bad_segments[:, 0], bad_segments[:, 1]] = 10 # marks the distances comparing imaginary axes (those that link both ends of each limb directly, for example)
    
    t = np.around(t, decimals=6)
    u = np.around(u, decimals=6)

    # Fetch only the information related to the closest links
    # [i, j] = np.where(dist == np.min(dist))
     # Fetch only the information related to links closer than self.max_dist
    [i, j] = np.where(dist < max_dist)
    # if len(i)>1:
    #     a = 2
    t = t[i, j]                         # [N]
    u = u[i, j]                         # [N]
    dist = dist[i, j]                   # [N]
    direc = direc[i, j,:]               # [N x 3 dims]
    if n_joints_b > n_joints_r:     
        closest_r = j                   # [N]
        closest_b = (j+i)%n_joints_b -1     # I think the -1 is superfluous
    else:
        closest_b = j
        closest_r = (j+i)%n_joints_r -1 # [N]

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

    # remove repeated values so we only have separate pairs
    if len(closest_r) > 1:
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

    # if len(closest_r) >1:
    #     a=2

    return dist, direc, t, u, closest_r, closest_b


class kpts_to_bbox(Node):

    def __init__(self):
        super().__init__('kpts_to_bbox')
        
        self.logger = logging.getLogger('kpts_to_bbox')
        logging.basicConfig(level=logging.DEBUG)

        self.initialize_variables()
        
        # self.subscription_arms_pos = self.create_subscription(
        #     Array2d,      # PoseArray, Array2d
        #     'arms_cartesian_pos',
        #     self.arms_pos_callback,
        #     10)
        
        # self.subscription_trunk_pos = self.create_subscription(
        #     Array2d,      # PoseArray, Array2d
        #     'trunk_cartesian_pos',
        #     self.trunk_pos_callback,
        #     10)
        
        self.subscription_body_pos = self.create_subscription(
            Array2d,      # PoseArray, Array2d
            'body_cartesian_pos',
            self.body_pos_callback,
            10)

        self.subscription_robot_pos = self.create_subscription(
            Array2d,      # PoseArray, Array2d
            'robot_cartesian_pos',
            self.robot_pos_callback,
            10)


        # self.robot_joint_state = self.create_subscription(
        #     JointState,
        #     'franka/joint_states',
        #     self.get_joint_velocities(),
        #     10)
        
        # self.robot_joint_state = self.create_subscription(
        #     TFMessage,
        #     'tf',
        #     self.transform_callback,
        #     10,)
        
        
        # link_poses = self.compute_robot_pose(robot, node)
        
        # self.force_publisher_ = self.create_publisher(Array2d, 'repulsion_forces', 10)
        self.force_publisher_ = self.create_publisher(IrregularDistArray, 'repulsion_forces', 10)
        self.Ipot_publisher_ = self.create_publisher(Array2d, 'repulsion_Ipot', 10)
        self.Damping_publisher_ = self.create_publisher(Array2d, 'repulsion_Damper', 10)
        

        # a = rclpy.callback_groups.ReentrantCallbackGroup()
        self.timer = self.create_timer(0.01, self.dist_callback)
        # self.timer = self.create_timer(0.01, self.kalman_callback)
        
    
    def initialize_variables(self):
        self.reset = 0.0
        self.x = []
        self.bodies = {}
        self.subject = '0'
        obj = time.gmtime(0) 
        epoch = time.asctime(obj) 
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
        self.max_dist = 0.4      # distance at which the robot feels a force exerted by proximity to a human
        self.min_dist = 0.00     # distance at which the robot is most strongly pushed back by a human
        self.joint_pos = [0., -0.8, 0., 2.36, 0., 1.57, 0.79]
        self.joint_vel = [0., -0.0, 0., -0., 0., 0., 0.]
        self.robot_cartesian_positions = np.zeros((7, 3))
        # self.arms_cartesian_positions = np.zeros((6, 3))
        # self.trunk_cartesian_positions = np.zeros((7, 3))
        self.body_cartesian_positions = np.zeros((12, 3))

        # Spring and damping forces
        Max_force_per_spring = 10.                          # [N]
        self.K = Max_force_per_spring / self.max_dist       # [N/m]
        self.D = 1.2 * np.sqrt(self.K)                      # [Ns/m]

        self.force_rescaling = np.array([1.4, 1.4, 1.2, 1.2, 1., 1., 1.])        # to increase the fores on joints near the base
        self.num_wanted_forces = 25.                         # to push the robot away enough, but not so much that it flips out

        # body info timeout
        self.body_timestamp = time.time_ns()

        # publishing stats
        self.pub_counter = 0
        self.Dt = 0
        self.t0 = time.time()

        # debugging
        # self.debug_t = time.time()
        # self.debug_dt = 0.
        # self.debug_dist_loops = 0
        

    def dist_callback(self):
        
        if np.any(self.body_cartesian_positions) and np.any(self.robot_cartesian_positions):

            # arms = self.arms_cartesian_positions
            # trunk = self.trunk_cartesian_positions

            

            body_pos = self.body_cartesian_positions
            robot_pos = self.robot_cartesian_positions

            # body = np.vstack((arms, trunk))
            # false_link = len(arms)
            dist, direc, t, u, c_r_b, c_b_r = link_dists(body_pos, robot_pos, self.max_dist, self.false_link)

            # arms_dist, arms_direc, arms_t, arms_u, c_r_a, c_a_r = link_dists(arms, robot_pos, self.max_dist) # self.placeholder_Pe, robot_pos
            # trunk_dist, trunk_direc, trunk_t, trunk_u, c_r_t, c_t_r = link_dists(trunk, robot_pos, self.max_dist)
            # self.placeholder_Pe, robot_pos

            ##############
            # # Plotting the distances
            # class geom(): pass
            # geom.arm_pos = body_pos
            # geom.trunk_pos = body_pos
            # geom.robot_pos = robot_pos  # self.placeholder_Pe, robot_pos
            # geom.arm_cp_idx = c_b_r
            # geom.u = u
            # geom.trunk_cp_idx = c_b_r
            # geom.v = u
            # geom.robot_cp_arm_idx = c_r_b
            # geom.s = t
            # geom.robot_cp_trunk_idx = c_r_b
            # geom.t = t

            # class geom(): pass
            # geom.body_pos = body_pos
            # geom.robot_pos = robot_pos  # self.placeholder_Pe, robot_pos
            # geom.body_cp_idx = c_b_r
            # geom.u = u
            # geom.robot_cp_body_idx = c_r_b
            # geom.s = t

            # self.fig = visuals.plot_skeletons(self.fig, geom)

            ##############

            # # only continue if the safety bubbles have been breached
            # if not np.any(arms_dist):
            #     body_geom = {'dist':trunk_dist, 'direc':trunk_direc,
            #                 't':trunk_t      , 'u':trunk_u,
            #                 'closest_r':c_r_t        , 'closest_b':c_t_r}
            # elif not np.any(trunk_dist):
            #     body_geom = {'dist':arms_dist, 'direc':arms_direc,
            #                 't':arms_t      , 'u':arms_u,
            #                 'closest_r':c_r_a        , 'closest_b':c_a_r}
            # else:
            #     # min_dist_arms = np.min(arms_dist)
            #     # min_dist_trunk = np.min(trunk_dist)
            #     # min_dist = min(min_dist_arms, min_dist_trunk)
            #     # self.get_logger().info('Minimum distance:')
            #     # self.get_logger().info(str(min_dist))

            #     # # if only using the minimum distance
            #     # if min_dist_arms < min_dist_trunk:
            #     #     body_geom = {'dist':arms_dist, 'direc':arms_direc,
            #     #                 't':arms_t      , 'u':arms_u,
            #     #                 'closest_r':c_r_a        , 'closest_b':c_a_r}
            #     # else:
            #     #     body_geom = {'dist':trunk_dist, 'direc':trunk_direc,
            #     #                 't':trunk_t      , 'u':trunk_u,
            #     #                 'closest_r':c_r_t        , 'closest_b':c_t_r}
                    
            #     # if using every distance under max_dist
            #     dist = np.hstack((arms_dist, trunk_dist))
            #     direc = np.concatenate((arms_direc, trunk_direc), axis = 0)
            #     t = np.hstack((arms_t, trunk_t))
            #     u = np.hstack((arms_u, trunk_u))
            #     closest_r = np.hstack((c_r_a, c_r_t))
            #     closest_b = np.hstack((c_a_r, c_t_r+len(c_a_r)))
            #     body_geom = {'dist':dist, 'direc':direc,
            #                     't':t      , 'u':u,
            #                     'closest_r':closest_r        , 'closest_b':closest_b}
            #     # forces = self.force_estimator(body_geom, robot_pos)    # self.placeholder_Pe, robot_pos

            #     # t3 = time.time()
            #     # dt = t3 - t2
            #     # if dt>0.05:
            #     #     self.logger.info("dist_callback 3 takes " + str(np.round(dt, 4)) + " seconds") 


  
            closest_r = c_r_b
            closest_b = c_b_r
            body_geom = {'dist':dist, 'direc':direc, 't':t , 'u':u, 'closest_r':closest_r, 'closest_b':closest_b}

            if time.time()>5:

            #     # forces = np.zeros((7, 6))
            #     # forces[3, 2] = 20

            #     self.generate_repulsive_force_message(forces)       
            # 

                # self.generate_distance_message(body_geom, robot_pos)
                self.generate_spring_forces_and_dampers(body_geom, robot_pos)
            # self.arms_cartesian_positions = np.zeros((6, 3))
            # self.trunk_cartesian_positions = np.zeros((6, 3))
            
            # reset the body object if the camera takes more than 1/00 Hz = 0.1s to send info
            current_time = time.time_ns()
            if current_time-self.body_timestamp > 1e8:
                self.body_cartesian_positions = np.zeros((12, 3))



    def generate_distance_message(self, body_geom, robot_pose):
        """
        -   Takes in the geometry of the robot as well as link distances
        -   Translates link distances to force precursors
        -   publishes the result out as a 1D vector
        """
        # Input : clip(max_dist-min_dist - abs(x-min_dist))     * application_dist if torque

        t0 = time.time()

        direc = body_geom['direc']                      # unit vectors associated with each force
        application_segments = body_geom['closest_r']   # segment on which the force is applied
        application_dist = body_geom['t']               # fraction of each segment at which each force is applied

        levers = np.diff(robot_pose, axis = 0)          # vectors pointing along the robot links
        link_lengths = np.linalg.norm(levers, axis=1)
        

        dists = copy.copy(body_geom['dist'])

        full_force_vec = np.zeros([len(robot_pose)-1, 6])


        #####################
        # spring_dists = np.clip(self.max_dist - self.min_dist - np.abs(dists-self.min_dist), 0, self.max_dist - self.min_dist)

        # low_mask = dists<self.min_dist
        # high_mask = 1-low_mask
        # spring_dists = -dists + self.max_dist*(np.ones(len(dists))+low_mask*(dists/self.min_dist - 1))  # this should return an actual distance in [m]
        spring_dists = np.clip(self.max_dist - dists, 0, self.max_dist)

        spring_vecs = spring_dists[:, np.newaxis] * direc

        # add and group spring vectors by joint
        joint_springs = []
        for i in range(len(robot_pose)):
            joint_springs.append( np.sum(spring_vecs[application_segments == i], axis=0) )

        # remove the 1st joint, which is the ground
        full_force_vec[:, 0:3] = np.array(joint_springs)[1:]

        
        ############################### treat springs individually
        dist_vecs = np.nan_to_num(dists[:, np.newaxis] * direc)
        dist_vecs = np.hstack((dist_vecs, dist_vecs*0))
        # self.logger.info('np.shape(dist_vecs)')
        # self.logger.info(str(np.shape(dist_vecs)))
        # dist_vecs = np.nan_to_num(spring_vecs)


        list_lengths = []

        dists_flattened = np.array([])
        for i in range(1, 8):
            vec = dist_vecs[application_segments == i]
            dists_flattened = np.append(dists_flattened, vec)
            list_lengths.append(len(vec))
        for i in range(8, len(robot_pose)):
            vec = dist_vecs[application_segments == i]
            dists_flattened = np.append(dists_flattened, vec)
            list_lengths[6] = list_lengths[6] + len(vec)

        dists_flattened = np.nan_to_num(dists_flattened).round(3)

        if np.any(dists_flattened):

            dist_message = IrregularDistArray()
            dist_message.array = list(dists_flattened.astype(float))
            dist_message.dimension2  = list_lengths
            self.force_publisher_.publish(dist_message)
            self.publishing_stats()
            # self.logger.info('list_lengths')
            # self.logger.info(str(list_lengths))

        

        # ###############################

        # ####################ACCOUNT FOR INPUT OF SIZE 13###################
        # # For every joint after the last movable joint(link_1-7), apply its torque/forces to the previous joint
        # for i in range(len(full_force_vec)-7):
        #     l = link_lengths[-i-1]
        #     if l == 0:
        #         length_multiplier = np.eye(6)
        #     else:
        #         # length_multiplier_force = np.concatenate((np.eye(3), -np.eye(3)/l), axis=0)
        #         length_multiplier_force = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=0)
        #         # length_multiplier_moment = np.concatenate((np.eye(3)*l, np.eye(3)), axis=0)
        #         # consider that the links towards the EE are short enough to be the same point,
        #         # else the fingers will cause moment trouble
        #         length_multiplier_moment = np.concatenate((np.zeros((3, 3)), np.zeros((3, 3))), axis=0)
        #         length_multiplier = np.concatenate((length_multiplier_force, length_multiplier_moment), axis=1)
        #     full_force_vec[-i-2] += length_multiplier @ full_force_vec[-i-1].T
        
        # # # Remove joint 0, apply only its torque to link 1
        # # l = link_lengths[0]
        # # if l == 0:
        # #     length_multiplier = np.eye(6)
        # # else:
        # #     length_multiplier_force = np.concatenate((np.zeros((3,3)), np.zeros((3,3))), axis=0)
        # #     length_multiplier_moment = np.concatenate((np.zeros((3,3)), np.eye(3)), axis=0)
        # #     length_multiplier = np.concatenate((length_multiplier_force, length_multiplier_moment), axis=1)
        # # full_force_vec[1] += length_multiplier @ full_force_vec[0]   

        # full_force_vec = full_force_vec[0:7]

        # full_force_vec = np.nan_to_num(full_force_vec)

        # # rescale spring forces to make them stronger if few kpts are exerting force
        # # the idea is that forces should be weak enough to not cause problems when the full body 
        # # exerts them, but get strong enough that a single hand can push anything 
        # num_interactions = len(application_segments)
        # if num_interactions>0:
        #     num_desired_interactions = 12           # Theoretical max is num_body_links * num_robot_links, so around 7*18 = 100. kwik mafs.
        #     denom = min(num_desired_interactions, num_interactions)
        #     multiplier = num_desired_interactions / denom
        #     # self.logger.info('number of interactions: {0}'.format(num_interactions))
        #     # self.logger.info(str(multiplier))
        #     # self.logger.info(str(self.logger.info('multiplier: {0}'.format(multiplier))))
        #     # self.logger.info(str(multiplier))
        #     full_force_vec = full_force_vec * multiplier
        #     # full_force_vec = np.tile(full_force_vec,(multiplier, 1))

        


        # # transform to message
        # forces = full_force_vec

        # # force_scaling = np.array([87., 87., 87., 87., 12., 12., 12.]) / (self.max_dist - self.min_dist)
        # # total_force = force_scaling[:, np.newaxis] * full_force_vec
        # # self.logger.info("Total force requested: " + str(total_force) + " N / Nm")

        # if np.any(forces):

        #     forces_flattened = forces.flatten(order='C')

        #     force_message = Array2d()
        #     force_message.array = list(forces_flattened.astype(float))
        #     [force_message.height, force_message.width]  = np.shape(forces.T)
        #     self.force_publisher_.publish(force_message)

        #     # self.logger.info(str(forces))

            # self.publishing_stats()

        # t1 = time.time()
        # dt = t1 - t0
        # if dt>0.05:
        #     self.logger.info("generate_distance_message takes " + str(np.round(dt, 4)) + " seconds") 


    def generate_spring_forces_and_dampers(self, body_geom, robot_pose):
        """
        -   Takes in the geometry of the robot as well as link distances
        -   Translates link distances to force precursors
        -   publishes the result out as a 1D vector
        """
        # Input : clip(max_dist-min_dist - abs(x-min_dist))     * application_dist if torque

        t0 = time.time()

        direc = body_geom['direc']                      # unit vectors associated with each force
        application_segments = body_geom['closest_r']   # segment on which the force is applied
        application_dist = body_geom['t']               # fraction of each segment at which each force is applied

        if not np.any(application_segments):
            return

        levers = np.diff(robot_pose, axis = 0)          # vectors pointing along the robot links
        link_lengths = np.linalg.norm(levers, axis=1) + 0.0000001
        

        dists = copy.copy(body_geom['dist'])

        Fpot_norm = self.K * np.clip(self.max_dist - dists, 0, self.max_dist)

        direc = np.hstack((direc, direc*0))

        # Add aditional moments on every joint for the forces applied partway along the link
        
        Mpot_application_segments = application_segments[application_dist>0]        # segment on which a moment is applied
        
        if np.any(Mpot_application_segments):
            Mpot_norm = Fpot_norm[application_dist>0] * link_lengths[Mpot_application_segments]\
                * application_dist[application_dist>0]
            
            Mpot_levers_unit = np.nan_to_num(levers[Mpot_application_segments, :] / link_lengths[Mpot_application_segments, None])

            Mpot_direc = np.cross(Mpot_levers_unit, direc[application_dist>0, 0:3][0])
            Mpot_direc = np.hstack((Mpot_direc*0, Mpot_direc))
        
        

            # Stack forces and moments into a single array
            Ipot_norm = np.hstack((Fpot_norm, Mpot_norm))

            Ipot_direc = np.vstack((direc, Mpot_direc))
            

            Ipot_application_segments = np.hstack((application_segments, Mpot_application_segments))
            # we add Mpot_application_segments+1 to apply the moment at the other end of the link, so the joint we care about reacts to it

        else:
            Ipot_norm, Ipot_direc, Ipot_application_segments = Fpot_norm, direc, application_segments

        # Add up all forces into a single vector
        Ipot_total = np.zeros((7, 6))
        for i in range(6):
            mask = Ipot_application_segments == i+1
            if np.any(mask):
                Ipot_part = Ipot_norm[mask, None] * Ipot_direc[mask,:]
                Ipot_total[i] = np.sum(Ipot_part, axis=0)

        for i in range(6, len(application_segments)):
            mask = Ipot_application_segments == i+1
            if np.any(mask):
                Ipot_part = Ipot_norm[mask, None] * Ipot_direc[mask,:]
                Ipot_total[6] = Ipot_total[6] + np.sum(Ipot_part, axis=0)

        # Ipot_total = Ipot_total * self.K


        # Calculate the system of dampers to operate on each joint
        Damping_moment_scaling = np.ones(len(Ipot_norm))
        if np.any(Mpot_application_segments):
            Damping_moment_scaling[-len(Mpot_norm):] = link_lengths[Mpot_application_segments]\
                * application_dist[application_dist>0]

        Damping_total = np.zeros((7, 6))
        for i in range(6):
            mask = Ipot_application_segments == i+1
            if np.any(mask):
                Damping_part = np.abs(Ipot_direc[mask,:]) * Damping_moment_scaling[mask,None]
                Damping_total[i] = np.sum(Damping_part, axis=0)
        for i in range(6, len(application_segments)):
            mask = Ipot_application_segments == i+1
            if np.any(mask):
                Damping_part = np.abs(Ipot_direc[mask,:]) * Damping_moment_scaling[mask,None]
                Damping_total[6] = Damping_total[6] + np.sum(Damping_part, axis=0)

        Damping_total = Damping_total * self.D

        ##### Rescale forces so those acting on the joints closer to base are stronger

        Ipot_total = Ipot_total * self.force_rescaling[:, None]
        Damping_total = Damping_total * self.force_rescaling[:, None]

        #####
        
        ##### rescale everything to act as if the same number of forces was always being applied
        
        num_actual_forces = len(Ipot_norm)

        if num_actual_forces>0:
            rescale_factor = self.num_wanted_forces / num_actual_forces

            Ipot_total = Ipot_total * rescale_factor
            Damping_total = Damping_total * rescale_factor

        #####

        
        
        
        # ##### Create fake forces and dampers for testing
        # Ipot_total = np.zeros((7, 6))
        # Ipot_total[0, 5] = 10

        # Damping_total = np.zeros((7, 6))
        # Damping_total[0, 5] = 10
        # #####


        if np.any(Ipot_total):

            Ipot_flattened = Ipot_total.flatten(order='C')

            Ipot_message = Array2d()
            Ipot_message.array = list(Ipot_flattened.astype(float))
            [Ipot_message.height, Ipot_message.width]  = np.shape(Ipot_total.T)
            self.Ipot_publisher_.publish(Ipot_message)

            # self.logger.info(str(Ipot_total))


            Damping_flattened = Damping_total.flatten(order='C')

            Damping_message = Array2d()
            Damping_message.array = list(Damping_flattened.astype(float))
            [Damping_message.height, Damping_message.width]  = np.shape(Damping_total.T)
            self.Damping_publisher_.publish(Damping_message)

            # self.logger.info('Damping_total')
            # self.logger.info(str(Damping_total))

            self.publishing_stats()



        
        
        
        






    def robot_pos_callback(self, msg):

        n_rows = msg.height
        n_cols = msg.width
        msg_array = np.array(msg.array)
        self.robot_cartesian_positions = np.reshape(msg_array,(n_rows,n_cols))

    # def arms_pos_callback(self, msg):

    #     n_rows = msg.height
    #     n_cols = msg.width
    #     msg_array = np.array(msg.array)
    #     self.arms_cartesian_positions = np.reshape(msg_array,(n_rows,n_cols))

    # def trunk_pos_callback(self, msg):

    #     n_rows = msg.height
    #     n_cols = msg.width
    #     msg_array = np.array(msg.array)
    #     self.trunk_cartesian_positions = np.reshape(msg_array,(n_rows,n_cols))

    def body_pos_callback(self, msg):

        n_rows = msg.height
        n_cols = msg.width
        msg_array = np.array(msg.array)
        self.false_link = msg_array[-1]
        self.body_cartesian_positions = np.reshape(msg_array[:-1],(n_rows,n_cols))
        self.body_timestamp = time.time_ns()

    
    def publishing_stats(self):

        t = time.time()
        dt = t - self.t0
        self.Dt = self.Dt + dt
        self.t0 = t
        self.pub_counter = self.pub_counter +1

        if dt>0.5:
            self.get_logger().info("Distances published after "+str(np.round(dt, 3))+" seconds")

        if self.Dt > 10:
            pub_freq = np.round(self.pub_counter / self.Dt, 3)
            self.get_logger().info("Distances published at "+str(pub_freq)+" Hz")
            self.Dt = 0
            self.pub_counter = 0

    

 

   

    







    



def main(args = None):
    
    rclpy.init(args=args)

    bbox_generator = kpts_to_bbox()
    # executor = rclpy.executors.MultiThreadedExecutor()
    
    # executor.add_node(bbox_generator)
    # executor.add_node(pose_publisher)

    # rclpy.spin(pose_publisher)
    rclpy.spin(bbox_generator)

    # executor_thread = threading.Thread(target=executor.spin, daemon=True)
    # executor_thread.start()

    # rate = pose_publisher.create_rate(2)
    # try:
    #     while rclpy.ok():
    #         # print('Help me body, you are my only hope')
    #         rate.sleep()
    # except KeyboardInterrupt:
    #     pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # bbox_generator.destroy_node()
    rclpy.shutdown()
    # executor_thread.join()
    
    print('done')
    
if __name__ == "__main__":
    main()
