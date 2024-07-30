#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String
import logging




class urdf_relay(Node):

    def __init__(self):
        super().__init__('robot_description')
        self.caught = 0
        self.listening_success = 0
        self.publishing_success = 0
        self.relayed_message = 0
        self.publisher = self.create_publisher(String, 'new_robot_description', 10)
        self.create_subscription(
            String,
            'robot_description',
            self.subscription_callback,
            10)
        self.create_timer(0.05, self.callback)
        
        # while self.publishing_success == 0:
        #     self.callback()
        #     time.sleep(0.001)

    def subscription_callback(self, input):
        
        self.relayed_message = input
        if type(self.relayed_message) == int:
            self.listening_success = 0
        else:
            self.listening_success = 1
        # logging.getLogger(__name__).info('Broadcasting robot description')
        # for i in range(500):
        #         self.publisher.publish(input)
        #         time.sleep(0.02)
        # self.destroy_node()

    def check_pub(self):
        pass

    def callback(self):
        if self.listening_success == 1:
            logging.getLogger(__name__).info('Broadcasting robot description')      
            print('Broadcasting robot description') 
            msg  = self.relayed_message
            for i in range(500):
                self.publisher.publish(self.relayed_message)
                time.sleep(0.02)
            self.publishing_success == 1
            self.destroy_node()

def main():
    rclpy.init()

    node = urdf_relay()

    rclpy.spin(node)

    # node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()