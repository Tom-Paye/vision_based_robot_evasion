#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from messages_fr3.srv import SetPose

import numpy as np
import time



class motion_client(Node):

    def __init__(self):
        super().__init__('motion_client')
        self.cli = self.create_client(SetPose, 'set_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetPose.Request()
        initial_pos = [0.5, 0.3, 0.4, 3.14, 0., 0.]
        future = self.send_request(initial_pos)
        time.sleep(2)
        self.time_0  = self.get_clock().now().nanoseconds/(10**9)
        # while not self.cli.(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')

        # self.timer = self.create_timer(0.01, self.draw_circle)
        self.custom_ee_pos()
        


    def send_request(self, pose):
        [x, y, z, roll, pitch, yaw] = pose
        self.req.x = x
        self.req.y = y
        self.req.z = z
        self.req.roll = roll
        self.req.pitch = pitch
        self.req.yaw = yaw
        return self.cli.call_async(self.req)


    
    def draw_circle(self):
        time = self.get_clock().now().nanoseconds/(10**9)
        time_from_start = (time-self.time_0) % 10
        travel = time_from_start * 2*np.pi / 10
        # print(travel)
        radius = 0.3

        x = 0.5
        y = radius * np.cos(travel)
        z = radius * np.sin(travel) + 0.4
        roll = 3.14
        pitch = 0.
        yaw = 0.
        self.send_request([x, y, z, roll, pitch, yaw])

    def custom_ee_pos(self):
        x = 0.5
        y = -0.3
        z = 0.4
        roll = 3.14
        pitch = 0.
        yaw = 0.
        self.send_request([x, y, z, roll, pitch, yaw])


    
def main():
    rclpy.init()

    client = motion_client()
    # basic_pos = [1, 1, 1, 1, 1, 1]
    # future = client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    # rclpy.spin_until_future_complete(client, future)
    # response = future.result()
    # client.get_logger().info(
    #     'Result of add_two_ints: for %d + %d = %d' %
    #     (int(sys.argv[1]), int(sys.argv[2]), response.sum))
    rclpy.spin(client)

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()