#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import my_cpp_py_pkg.kalman as kalman


import numpy as np
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import math
import time
import copy

def rearrange_trunk(coords):
    """
    Current suggestion (uneducated)
    1) for each P: physically consecutive pair of keypoints in a group (eg wrist-elbow, elbow-shoulder, NOT wrist-shoulder)
        create the 3d axis joining them both
        1.1: Find the orthogonal vector OV to the axis which crosses the end effector EE
        1.2: Check if OV crosses between the KPts
            1.2.1 if yes, calculate Da(P), the min distance from the axis to the EE
            1.2.2 if no, find the Ca(P): the closest KPt on the axis to the EE.
    2) take the min of all Da, Ca
    This technique is not robust to hugging the robot, and may cause it to enter a blood rage.
    A simple, if computationally effective solution would be to exert pressure from each P to limit bang-bang interactions
    Complexity: O(num_nodes)*O(perp_axis)*constant

    Problem: We also need to save the direction along which the distance is the shortest, and take into account the robot
    links as well, which squares the whole complexity.
    For starters, it might be more practical to simply calculate the closest human-lint to robot-link distance and direction.

    Also, would a kalman filter on the human pos be a good idea? Is it already implemented by ZED? --> no,
    TODO:investigate kalman filtering
    """


    coords_l = coords[[0, 1, 2, 4], :]
    coords_r = coords[[0, 1, 3, 5], :]
    return coords_l, coords_r
    
def link_dists(pos_body, pos_robot):
    """
    INPUT---------------------------
    Takes pos_body: [M, 3] array of keypoint positions IN PHYSICALLY CONSECUTIVE ORDER
    (i.e. don't feed it a hand keypoint followed by the head keypoint or it will assume you have antennae)

    Takes pos_robot: [N, 3] vector corresponding to end effector position

    OUTPUT--------------------------
    Returns dist: [P, Q] array of distances
        P = min(M, N) --> one row for each convolution of the longer effector positions on the shorter effector positions.
            Longer or shorter refers to the number of joints, not physical length
        Q = P-1 --> because we only consider the distances from the P-1 segments of the shortest effector
        NOTE: in every row, there is one distance which corresponds to a segment connecting the longest effector end-to-end directly
            This distance should not be taken into account for control, and is set to 10m

    Returns direc: [P, Q, 3] array of unit direction vectors along the shortest path between each body_segment-robot_segment pair
        if a segment on the robot is found to be intersecting a segment on the body (dist[i, j] == 0):
            the corresponding vector  direc[i, j, :] is [0, 0, 0]
        NOTE: The direct vectors point from the body to the robot

    Returnt t : [P, Q] array of position of 'intersection point' on each segment of the robot
        0 <= t[i, j] <= 1, as it represents a fraction of the segment length, starting from whichever point has a lower index in pos_robot
        Intersection point : on a segment A, the closest point to the segment B we're measuring a distance to

    Returnt u : [P, Q] array of position of 'intersection point' on each segment of the body
        0 <= u[i, j] <= 1, as it represents a fraction of the segment length, starting from whichever point has a lower index in pos_body

        
    https://www.sciencedirect.com/science/article/pii/0020019085900328 
    """


    # # pos_body = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    # # pos_robot = [0, -3, 0]
    # pos_body = np.reshapos_robot(pos_body, [-1,3])
    # dP = np.diff(pos_body, axis=0) / np.linalg.norm(np.diff(pos_body, axis=0))
    # dpos_robot = pos_robot - pos_body[:-1, :]
    # c = np.diag(np.matmul(dP, np.transpose(dpos_robot)))     # distance of the plane along the axis from the first point

    # dist = np.zeros(len(dP))
    # for i in range(len(dist)):
    #     if c[i] <= 0:
    #         dist[i] = np.linalg.norm(pos_robot - pos_body[i])
    #     if c[i] >=1:
    #         dist[i] = np.linalg.norm(pos_robot - pos_body[i+1])
    #     if c[i] < 1 and c[i] > 0:
    #         dist[i] = np.linalg.norm(pos_robot - c[i] - pos_body[i])

    # TODO: implement marking of imaginary joints
    # TODO: articulate body geometry into a format suitable for this function
    # TODO: Make the Kalman Filter 3D
    # TODO: Create estimation of body speeds
    # TODO: make kalman filter real-time


    # links_r = np.array([[0, 0, 0], [1, 1, 0]])
    # links_b = np.array([[0, 2, 0], [1, 2, 0]])

    # links_r = np.array([[0, -3, 0], [3, 0, 0]])
    # links_b = np.array([[0, 0, 0], [3, -3, 0]])

    # links_r = np.array([[0, 0, 0], [1, 1, 0]])
    # links_b = np.array([[0, 1, 0], [1, 2, 0]])

    # links_r = np.array([[0, 0, 0], [2, 0, 0]])
    # links_b = np.array([[1, 2, 0], [1, 1, 0]])
    chkpt_1 = time.time()

    links_r = np.array([[0, 0, 0], [1, 1, 0], [0, -3, 0], [3, 0, 0],
                        [0, 0, 0], [1, 1, 0], [0, 0, 0], [2, 0, 0],
                        [0, 0, 0], [1, 1, 0], [0, -3, 0], [3, 0, 0],
                        [0, 0, 0], [1, 1, 0], [0, 0, 0], [2, 0, 0]])
    links_b = np.array([[0, 2, 0], [1, 2, 0], [0, 0, 0], [3, -3, 0],
                        [0, 1, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0],
                        [0, 2, 0], [1, 2, 0], [0, 0, 0], [3, -3, 0],
                        [0, 1, 0], [1, 2, 0], [1, 2, 0], [1, 1, 0]])
    
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
    mat_roll = np.array([static_array]*n_rolls)
    mat_static = np.array([static_array]*n_rolls)
    bad_segments = []

    for i in range(n_rolls):
        new_layer = np.roll(arr_to_roll, -i, axis=0)
        new_layer = new_layer[0:min(m, n),:]
        mat_roll[i,:] = new_layer
        bad_segments.append([i, n_rolls-i-1])
    bad_segments = np.array(bad_segments[1:])

    if m > n:
        links_b = mat_roll
        links_r = mat_static
    else:
        links_r = mat_roll
        links_b = mat_static
        
    # while n % m == 0:
    #     rep_array = np.ones(n).astype(int)
    #     rep_array[-1] = 2
    #     links_b = np.repeat(links_b, rep_array, axis=0)
    #     m = len(links_b)
    #     q += 1
    # links_b = np.tile(links_b, (n, 1))
    # links_r = np.tile(links_r, (m, 1))
    #####


    # links_r = np.repeat(links_r, m*np.ones(n).astype(int), axis=0)
    # links_b = np.tile(links_b, (n, 1))

    pseudo_links = np.array([links_r[:, :-1, :], links_b[:, :-1, :]])

    d_r = np.diff(links_r, axis=1)
    d_b = np.diff(links_b, axis=1)
    d_rb = np.diff(pseudo_links, axis=0)[0]

    # calc length of both segments and check if parallel
    len_r = np.linalg.norm(d_r, axis=-1)**2
    len_b = np.linalg.norm(d_b, axis=-1)**2
    if 0 in len_r or 0 in len_b:
        # self.get_logger().info('err: Link without length')
        print('err: Link without length')
    R = np.einsum('ijk, ijk->ij', d_r, d_b)
    # R = np.dot(d_r, d_b) # use np.einsum
    denom = len_r*len_b - R**2
    S1 = np.einsum('ijk, ijk->ij', d_r, d_rb)
    S2 = np.einsum('ijk, ijk->ij', d_b, d_rb)
    # S1 = np.dot(d_r, d_rb)
    # S2 = np.dot(d_b, d_rb)

    paral = (denom<0.001)
    # t = (1-paral) * (S1*len_b-S2*len_r) / denom
    t = (1-paral) * (S1*len_b-S2*R) / denom
    t = np.nan_to_num(t)
    t = np.clip(t, 0, 1)
    # if np.abs(denom) < 0.001:
    #     print('Parallel')
    #     t=0
    # else:
    #     t = (S1*len_b-S2*len_r) / denom
    #     if t>1:
    #         t=1
    #     if t<0:
    #         t=0

    u = (t*R - S2) / (len_b)
    u = np.nan_to_num(u)
    u = np.clip(u, 0, 1)
    # if u>1:
    #     u=1
    # if u<0:
    #     u=0

    t = (u*R + S1) / len_r
    t = np.clip(t, 0, 1)
    # if t>1:
    #     t=1
    # if t<0:
    #     t=0
    tp = np.transpose(np.array([t]*3), (1, 2, 0))
    up = np.transpose(np.array([u]*3), (1, 2, 0))
    diffs_3d = np.multiply(d_r, tp) - np.multiply(d_b, up) - d_rb
    dist = np.sqrt(np.sum(diffs_3d**2, axis=-1))
    # dist = np.sqrt(np.sum(( np.transpose(d_r, axes=(0,2,1))*t - d_b.T*u - d_rb.T )**2, axis=1))

    distp = copy.copy(dist)
    dist = np.around(dist, decimals=3)
    distp[dist == 0] = 1000
    direc = np.multiply(diffs_3d, 1 / distp[:, :,  np.newaxis])
  
    
    [intersec_b_link, intersec_r_link] = np.nonzero(distp == 1000)
    direc[intersec_b_link, intersec_r_link, :] = [0, 0, 0]  # marks the places where the body and robot are believed to clip into another
    dist[bad_segments[:, 0], bad_segments[:, 1]] = 10 # marks the distances comparing imaginary axes (those that link both ends of each limb directly, for example)
    
    t = np.around(t, decimals=3)
    u = np.around(u, decimals=3)

    chkpt_2 = time.time()
    elapsed_time = chkpt_2 - chkpt_1


    return dist, direc, t, u


class Subscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')

        self.initialize_variables()

        self.subscription_data = self.create_subscription(
            PoseArray,
            'kpt_data',
            self.data_callback,
            10)
        self.subscription_data  # prevent unused variable warning

        self.timer = self.create_timer(0.01, self.calc_callback)
        # self.timer = self.create_timer(0.01, self.kalman_callback)
        
    
    def initialize_variables(self):
        self.reset = 0.0
        self.x = []
        self.bodies = {}
        self.subject = '0'
        self.placeholder_Pe = np.array([[1., 0., 0.],
                                        [1., -.1, .3],
                                        [1., .1, .3],
                                        [1., 0., .6],
                                        [.9, .1, .7],
                                        [.9, -.1, .7],
                                        [.8, 0., .8],
                                        [.1, .2, .8],
                                        [.1, 0., .81],
                                        [0., 0., .8],
                                        [0., -.1, .6],
                                        [0., .1, .6],])    # placeholder end effector positions

    def kalman_callback(self):

        if np.any(self.bodies) and self.subject in self.bodies:
            if not np.any(self.x):
                self.x = [self.bodies[self.subject][0][0][0]]
            else:
                self.x.append(self.bodies[self.subject][0][0][0])
        if len(self.x)>1000:
            kalman.kaltest(self.x)
        

    def calc_callback(self):
        if self.subject in self.bodies:
            self.get_logger().info(str(self.reset and np.any(self.bodies[self.subject][0])))
            if self.reset and np.any(self.bodies[self.subject][0]):
                if len(self.bodies[self.subject][2]) >6:
                    [self.dt_l, self.dt_r] = rearrange_trunk(self.bodies[self.subject][2])
                left_dist, left_direc, left_t, left_u = link_dists(self.bodies[self.subject][0], self.placeholder_Pe)
                self.get_logger().info('Distances:')
                self.get_logger().info(str(left_dist))




    # def label_callback(self, msg):
    #     self.get_logger().info(msg.data)
    #     self.label = msg.data
    #     self.data = []
        
    def data_callback(self, msg):
        # self.get_logger().info(str(msg))
        # region = math.trunc(msg.w)
        region = msg.header.frame_id[1:]
        self.reset = (msg.header.frame_id[2:] == 'stop')
        # if self.reset:
        #     self.get_logger().info('left')
        #     self.get_logger().info(str(self.data_left))
        #     self.get_logger().info('right')
        #     self.get_logger().info(str(self.data_right))
        #     self.get_logger().info('trunk')
        #     self.get_logger().info(str(self.data_trunk))
        #     self.dl = self.data_left
        #     self.dr = self.data_right
        #     self.dt = self.data_trunk
        #     self.data_left = []
        #     self.data_right = []
        #     self.data_trunk = []
        # else:
        if self.reset and self.subject in self.bodies:
            self.get_logger().info('Body 0 Left side')
            self.get_logger().info(str(self.bodies[self.subject][0]))
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

            threshold = 0.2
            if str(body_id) in self.bodies:
                if region == 'left':
                    if np.shape(self.bodies[str(body_id)][0]) == np.shape(poses):
                        if np.any(np.linalg.norm(self.bodies[str(body_id)][0]-poses, axis=0) > threshold):
                            poses = self.bodies[str(body_id)][0]
                    self.bodies[str(body_id)][0] = poses
                if region == 'right':
                    if np.shape(self.bodies[str(body_id)][1]) == np.shape(poses):
                        if np.any(np.linalg.norm(self.bodies[str(body_id)][1]-poses, axis=0) > threshold):
                            poses = self.bodies[str(body_id)][1]
                    self.bodies[str(body_id)][1] = poses
                if region == 'trunk':
                    if np.shape(self.bodies[str(body_id)][2]) == np.shape(poses):
                        if np.any(np.linalg.norm(self.bodies[str(body_id)][2]-poses, axis=0) > threshold):
                            poses = self.bodies[str(body_id)][2]
                    self.bodies[str(body_id)][2] = poses
            else:
                    self.bodies[str(body_id)] = [poses[0:3,:], poses[0:3,:], poses]

        

            # data = []
            # data_ros = msg.poses
            # for pose in data_ros:
            #     data.append([pose.position.x, pose.position.y, pose.position.z])
            # data = np.array(data)
            # if region == 'left':
            #     self.data_left = data
            # if region == 'right':
            #     self.data_right = data
            # if region == 'trunk':
            #     self.data_trunk = data


            # if region == 0:
            #     if len(self.data_left) <1: #not np.any(self.data_left):
            #         self.data_left = np.array([[msg.x, msg.y, msg.z]])
            #     else:
            #         self.data_left = np.append(self.data_left, np.array([msg.x, msg.y, msg.z]))
            # if region == 1:
            #     if len(self.data_right) <1: #not np.any(self.data_right):
            #         self.data_right = np.array([[msg.x, msg.y, msg.z]])
            #     else:
            #         self.data_right = np.append(self.data_right, np.array([msg.x, msg.y, msg.z]))
            # if region == 2:
            #     if len(self.data_trunk) <1: #not np.any(self.data_trunk):
            #         self.data_trunk = np.array([[msg.x, msg.y, msg.z]])
            #     else:
            #         self.data_trunk = np.append(self.data_trunk, np.array([msg.x, msg.y, msg.z]))





        


    



def main(args = None):
    
    rclpy.init(args=args)

    subscriber = Subscriber()

    rclpy.spin(subscriber)




    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber.destroy_node()
    rclpy.shutdown()
    
    print('done')
    
if __name__ == "__main__":
    main()