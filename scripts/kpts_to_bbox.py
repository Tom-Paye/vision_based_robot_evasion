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
from geometry_msgs.msg import Vector3
import pickle

class Subscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription_label = self.create_subscription(
            String,
            'kpt_label',
            self.label_callback,
            10)
        self.subscription_label  # prevent unused variable warning
        self.subscription_data = self.create_subscription(
            Vector3,
            'kpt_data',
            self.data_callback,
            10)
        self.subscription_data  # prevent unused variable warning

    def label_callback(self, msg):
        self.get_logger().info(msg.data)
        self.label = msg.data
        self.data = []
        
    def data_callback(self, msg):
        self.get_logger().info(str(msg))
        if self.data == []:
            self.data = [[msg.x, msg.y, msg.z]]
        else:
            self.data.append([msg.x, msg.y, msg.z])



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