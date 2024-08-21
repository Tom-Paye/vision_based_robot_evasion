#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from urdf_parser_py.urdf import URDF
from urdf_parser_py.urdf import Robot

from messages_fr3.msg import Array2d
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String

import numpy as np
from scipy.spatial.transform import Rotation as R

import time
import copy
import logging

from skimage.util.shape import view_as_windows

from urdfpy import URDF


############ Standalone functions ###############


def calculate_transforms(robot, tf_world):
    link_poses = {}
    legacy_link = np.eye(4)
    for link in robot.links:
        link_poses[link.name] = legacy_link
        if link.name in tf_world:
            link_poses[link.name] = create_transform_matrix(tf_world[link.name][0], tf_world[link.name][1])
        legacy_link = link_poses[link.name]
    return link_poses 


def create_transform_matrix(translation, rotation):    
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = translation

    rot = R.from_quat(rotation)
    trans_matrix[:3, :3] = rot.as_matrix()

    return trans_matrix


############ /Standalone functions ##############


class robot_description(Node):

    def __init__(self):
        super().__init__('robot_description')
        self.caught = 0
        self.robot_des = self.create_subscription(
            String,
            'robot_description',
            self.description_callback,
            10)

    def description_callback(self, des_message):
        # only published to once, at the launch of the controller

        a = des_message
        robot = Robot.from_xml_string(a.data)
        # Check if robot model is valid
        robot.check_valid()
        # Print the robot model
        # print(robot)
        self.robot = robot
        self.caught = 1


class robot_pose(Node):

    def __init__(self):
        super().__init__('robot_pose')
        
        self.logger = logging.getLogger('robot_pose')
        logging.basicConfig(level=logging.DEBUG)

        # get the robot description
        caught = 0
        description_node = robot_description()
        self.logger.info('Waiting for robot_description message...')
        while not caught:
            rclpy.spin_once(description_node)
            caught = description_node.caught
            time.sleep(0.01)
        self.robot = description_node.robot
        self.logger.info('Robot model loaded!')  

        self.robot_joint_state = self.create_subscription(
            TFMessage,
            'tf',
            self.transform_callback,
            10,)     
        
        self.force_publisher_ = self.create_publisher(Array2d, 'robot_cartesian_pos', 10)


    def transform_callback(self, tf_message):
        
        t0 = time.time()

        robot_translation = np.zeros((10,3))
        robot_rotation = np.zeros((10,4))
        robot_rotation[:,-1] = np.ones(10)
        joint_name = [None] * 10

        total_trans = np.zeros(3)
        total_rot = np.array([0, 0, 0, 1])
        trm = R.from_quat(total_rot)
        idx = 0
        finger_rot_quat = np.zeros((2, 4))
        finger_rot_quat[:,-1] = np.ones(2)
        finger_trans = np.zeros((2, 3))

        for transform in tf_message.transforms:
            rot_ros = transform.transform.rotation
            trans_ros = transform.transform.translation
            id = transform.child_frame_id

            vec_t = np.array([trans_ros.x, trans_ros.y, trans_ros.z])
            vec_r = np.array([rot_ros.x, rot_ros.y, rot_ros.z, rot_ros.w])
            

            if id[-1].isnumeric():
                i = int(id[-1])-1

                quat_rot = trm.apply(vec_t)               # current translation, reoriented into world frame
                total_trans = total_trans + quat_rot        # total translation in WF
            
                srm = R.from_quat(vec_r)                    # rotation quat of current index in Current Frame
                trm = trm * srm                             # rotation quat of current index in WF               
                total_rot_quat = trm.as_quat()

                robot_rotation[idx+1] = total_rot_quat        # total rot and trans are applied to the next joint
                robot_translation[idx+1] = total_trans
                idx = idx + 1
            else:
                if id == 'panda_leftfinger':
                    i = -3

                    finger_rot_quat[0] = vec_r
                    finger_trans[0] = vec_t

                else:
                    i = -2

                finger_trans[1] = vec_t
                finger_rot_quat[1] = vec_r

            joint_name[i+1] = id

        # Deal with the fingers separately to not have to screw with the for loop
        robot_rotation[-2] = robot_rotation[-3]
        quat_rot = R.from_quat(robot_rotation[-3]).apply(finger_trans[0])
        robot_translation[-2] =  robot_translation[-3] + quat_rot

        robot_rotation[-1] = robot_rotation[-3]
        quat_rot = R.from_quat(robot_rotation[-3]).apply(finger_trans[1])
        robot_translation[-1] =  robot_translation[-3] + quat_rot

        tf_world = {}
        for i, joint in enumerate(joint_name[1:]):
            tf_world[joint] = [robot_translation[i+1], robot_rotation[i+1]]
        
        link_poses = calculate_transforms(self.robot, tf_world)
        pos_world = np.array(list(link_poses.values()))[:, 0:3, -1]

        self.robot_cartesian_positions = pos_world

        self.publish_important_info()

        # # apply robot rotations to each robot axis, assuming that axis is locally [0, 0, 1]
        # joint_axes = np.zeros((7, 3))
        # joint_axes[:,-1] = np.ones(7)
        # self.axis_rot = R.from_quat(robot_rotation[1:-2]).apply(joint_axes)

        # t1 = time.time()
        # dt = t1 - t0
        # if dt>0.05:
        #     self.logger.info("transform_callback takes " + str(np.round(dt, 4)) + " seconds")
        # return link_poses

    def publish_important_info(self):
        
        robot_pos = self.robot_cartesian_positions

        pos_flattened = robot_pos.flatten(order='C')

        force_message = Array2d()
        force_message.array = list(pos_flattened.astype(float))
        [force_message.height, force_message.width]  = np.shape(robot_pos)
        self.force_publisher_.publish(force_message)



def main(args = None):
    
    rclpy.init(args=args)

    pose_publisher = robot_pose()

    rclpy.spin(pose_publisher)

    rclpy.shutdown()
    
    print('done')
    
if __name__ == "__main__":
    main()