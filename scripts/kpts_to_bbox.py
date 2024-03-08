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

class Subscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        # self.subscription_label = self.create_subscription(
        #     String,
        #     'kpt_label',
        #     self.label_callback,
        #     10)
        # self.subscription_label  # prevent unused variable warning
        self.subscription_data = self.create_subscription(
            Quaternion,
            'kpt_data',
            self.data_callback,
            10)
        self.subscription_data  # prevent unused variable warning
        self.data_left = []
        self.data_right = []
        self.data_trunk = []

    # def label_callback(self, msg):
    #     self.get_logger().info(msg.data)
    #     self.label = msg.data
    #     self.data = []
        
    def data_callback(self, msg):
        # self.get_logger().info(str(msg))
        region = math.trunc(msg.w)
        reset = msg.w == -1
        if reset:
            self.get_logger().info('left')
            self.get_logger().info(str(self.data_left))
            self.get_logger().info('right')
            self.get_logger().info(str(self.data_right))
            self.get_logger().info('trunk')
            self.get_logger().info(str(self.data_trunk))
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