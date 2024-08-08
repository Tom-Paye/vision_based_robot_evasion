#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from messages_fr3.srv import SetPose

import numpy as np



class motion_client(Node):

    def __init__(self):
        super().__init__('motion_client')
        self.cli = self.create_client(SetPose, 'set_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        # self.req = SetPose.Request()
        self.timer = self.create_timer(0.01, self.draw_circle)


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
        time = self.get_clock().now()
        time_from_start = time % 10
        travel = time_from_start * 2

        x = 0.5
        y = 0.3 * np.cos(travel)
        z = 0.3 * np.sin(travel)
        roll = 0
        pitch = 0
        yaw = 0
        self.send_request(float([x, y, z, roll, pitch, yaw]))


    
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
    client.spin()

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()