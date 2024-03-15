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
    Takes pos_body: [N, 3] array of keypoint positions IN PHYSICALLY CONSECUTIVE ORDER
    (i.e. don't feed it a hand keypoint followed by the head keypoint or it will assume you have antennae)
    Takes pos_robot: [3] vector corresponding to end effector position
    returns the min distance to each body segment, i.e. each link between two joints
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

    # TODO: Fix the loop so it actually switches over limbs on the robot, and not joints
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
    links_r = np.repeat(links_r, m*np.ones(n).astype(int), axis=0)
    links_b = np.tile(links_b, (n, 1))

    pseudo_links = np.array([links_r[:-1, :], links_b[:-1, :]])

    d_r = np.diff(links_r, axis=0)
    d_b = np.diff(links_b, axis=0)
    d_rb = np.diff(pseudo_links, axis=0)[0]

    # calc length of both segments and check if parallel
    len_r = np.linalg.norm(d_r, axis=-1)**2
    len_b = np.linalg.norm(d_b, axis=-1)**2
    if 0 in len_r or 0 in len_b:
        print('err: Link without length')
    R = np.einsum('ij, ij->i', d_r, d_b)
    # R = np.dot(d_r, d_b) # use np.einsum
    denom = len_r*len_b - R**2
    S1 = np.einsum('ij, ij->i', d_r, d_rb)
    S2 = np.einsum('ij, ij->i', d_b, d_rb)
    # S1 = np.dot(d_r, d_rb)
    # S2 = np.dot(d_b, d_rb)

    paral = (denom<0.001)
    t = (1-paral) * (S1*len_b-S2*len_r) / denom
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

    dist = np.sqrt(np.sum(( d_r.T*t - d_b.T*u - d_rb.T )**2, axis=0))

    chkpt_2 = time.time()
    elapsed_time = chkpt_2 - chkpt_1


    return dist


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
                left_dist = link_dists(self.bodies[self.subject][0], self.placeholder_Pe)
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