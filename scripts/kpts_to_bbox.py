#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import cv2
import sys
import time
import numpy as np
import os
import copy
import _thread
from geometry_msgs.msg import Quaternion
import pickle
import math

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
    """

    coords = np.reshape(coords, [-1,3])

    coords_l = coords[[0, 1, 2, 4], :]
    coords_r = coords[[0, 1, 3, 5], :]
    return coords_l, coords_r
    
def link_dists(coords, Pe):
    """
    Takes coords: [N, 3] array of keypoint positions IN PHYSICALLY CONSECUTIVE ORDER
    (i.e. don't feed it a hand keypoint followed by the head keypoint or it will assume you have antennae)
    Takes Pe: [3] vector corresponding to end effector position
    returns the min distance to each body segment, i.e. each link between two joints
    """


    # coords = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    # Pe = [0, -3, 0]
    coords = np.reshape(coords, [-1,3])
    dP = np.diff(coords, axis=0) / np.linalg.norm(np.diff(coords, axis=0))
    dPe = Pe - coords[:-1, :]
    c = np.diag(np.matmul(dP, np.transpose(dPe)))     # distance of the plane along the axis from the first point

    dist = np.zeros(len(dP))
    for i in range(len(dist)):
        if c[i] <= 0:
            dist[i] = np.linalg.norm(Pe - coords[i])
        if c[i] >=1:
            dist[i] = np.linalg.norm(Pe - coords[i+1])
        if c[i] < 1 and c[i] > 0:
            dist[i] = np.linalg.norm(Pe - c[i] - coords[i])

    return dist


class Subscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        # self.subscription_label = self.create_subscription(
        #     String,
        #     'kpt_label',
        #     self.label_callback,
        #     10)
        # self.subscription_label  # prevent unused variable warning
        self.reset = 0.0
        self.data_left, self.data_right, self.data_trunk = [], [], []
        self.dl, self.dr, self.dt = [], [], []

        self.subscription_data = self.create_subscription(
            Quaternion,
            'kpt_data',
            self.data_callback,
            10)
        self.subscription_data  # prevent unused variable warning
        self.timer = self.create_timer(0.1, self.calc_callback)
        # self.calc_callback()
        
        

    def calc_callback(self):
        self.get_logger().info(str(self.reset and np.any(self.dt)))
        if self.reset and np.any(self.dt):
            [self.dt_l, self.dt_r] = rearrange_trunk(self.dt)
            placeholder_Pe = np.array([1, 1, 1])
            left_dist = link_dists(self.dl, placeholder_Pe)
            self.get_logger().info('Distances:')
            self.get_logger().info(str(left_dist))




    # def label_callback(self, msg):
    #     self.get_logger().info(msg.data)
    #     self.label = msg.data
    #     self.data = []
        
    def data_callback(self, msg):
        # self.get_logger().info(str(msg))
        region = math.trunc(msg.w)
        self.reset = (msg.w == -1)
        if self.reset:
            self.get_logger().info('left')
            self.get_logger().info(str(self.data_left))
            self.get_logger().info('right')
            self.get_logger().info(str(self.data_right))
            self.get_logger().info('trunk')
            self.get_logger().info(str(self.data_trunk))
            self.dl = self.data_left
            self.dr = self.data_right
            self.dt = self.data_trunk
            self.data_left = []
            self.data_right = []
            self.data_trunk = []
        else:    
            if region == 0:
                if len(self.data_left) <1: #not np.any(self.data_left):
                    self.data_left = np.array([[msg.x, msg.y, msg.z]])
                else:
                    self.data_left = np.append(self.data_left, np.array([msg.x, msg.y, msg.z]))
            if region == 1:
                if len(self.data_right) <1: #not np.any(self.data_right):
                    self.data_right = np.array([[msg.x, msg.y, msg.z]])
                else:
                    self.data_right = np.append(self.data_right, np.array([msg.x, msg.y, msg.z]))
            if region == 2:
                if len(self.data_trunk) <1: #not np.any(self.data_trunk):
                    self.data_trunk = np.array([[msg.x, msg.y, msg.z]])
                else:
                    self.data_trunk = np.append(self.data_trunk, np.array([msg.x, msg.y, msg.z]))





        


    



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