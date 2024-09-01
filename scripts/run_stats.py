#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from messages_fr3.msg import Array2d

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path



class archivist(Node):

    def __init__(self):
        super().__init__('archivist')
        self.time_0 = time.time_ns()
        self.elapsed_time = 0
        self.time_kpt = []
        self.time_dist = []
        self.kpt_dist_delay = []

        self.subscription_kpt = self.create_subscription(
            Array2d,      # PoseArray, Array2d
            'body_cartesian_pos',
            self.kpt_listener,
            10)
        
        self.subscription_dist = self.create_subscription(
            Array2d,      # PoseArray, Array2d
            'repulsion_forces',
            self.dist_listener,
            10)
        
        self.timer = self.create_timer(120., self.compile_data)

        # self.compile_data()
        


    def kpt_listener(self, message):
        self.time_kpt.append(time.time_ns())
        # print('kpt message received!')
        # a = message

    def dist_listener(self, message):
        self.time_dist.append(time.time_ns())
        self.kpt_dist_delay.append(self.time_dist[-1]-self.time_kpt[-1])
        # print('dist message received!')
        # a = message

    
    def compile_data(self):
        
        dt=0
        ########################################
        # while(dt<5):
        #     self.time_kpt.append(time.time_ns())
        #     self.time_dist.append(time.time_ns())
        #     self.kpt_dist_delay.append(self.time_dist[-1]-self.time_kpt[-1])
        #     time.sleep(1)
        #     dt = dt+1

        ########################################
        
        time_kpt = np.array(self.time_kpt)
        time_dist = np.array(self.time_dist)
        kpt_dist_delay = np.array(self.kpt_dist_delay)     

        file_path = Path('/home/tom/Documents/Presentation/')
        delay_kpt = np.diff(time_kpt)
        np.savetxt(file_path.joinpath('kpt_delays'), delay_kpt)

        delay_dist = np.diff(time_dist)
        np.savetxt(file_path.joinpath('delay_dist'), delay_dist)

        np.savetxt(file_path.joinpath('kpt_dist_delay'), kpt_dist_delay)



        mean_kpt_delay = np.mean(delay_kpt)
        std_dev_kpt_delay = np.std(delay_kpt)
        print(f"kpt: mean: {mean_kpt_delay}, std_dev: {std_dev_kpt_delay}")

        mean_dist_delay = np.mean(delay_dist)
        std_dev_dist_delay = np.std(delay_dist)
        print(f"dist: mean: {mean_dist_delay}, std_dev: {std_dev_dist_delay}")

        mean_kpt_dist_delay = np.mean(kpt_dist_delay)
        std_dev_kpt_dist_delay = np.std(kpt_dist_delay)
        print(f"kpt_dist: mean: {mean_kpt_dist_delay}, std_dev: {std_dev_kpt_dist_delay}")


        kpt_freq = 1/mean_kpt_delay*1e9
        dist_freq = 1/mean_dist_delay*1e9

        print(f"Frequency [Hz]: kpt: {kpt_freq}, dist: {dist_freq}")

        fig, (ax1, ax2) = plt.subplots(2,1)

        ax1.boxplot(delay_kpt/1e6)
        ax1.set_ylabel('Body tracking delay [ms]')
        ax2.boxplot(kpt_dist_delay/1e6)
        ax2.set_ylabel('Planning delay [ms]')

        plt.show()





def main():
    rclpy.init()

    node = archivist()
    
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()