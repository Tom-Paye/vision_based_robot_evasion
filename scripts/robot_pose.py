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

        self.bodies = {}
        self.offset = np.array([-0.0, 0., 0.0]) # position offset to correct zed bullshit


        self.robot_joint_state = self.create_subscription(
            TFMessage,
            'tf',
            self.robot_callback,
            10,)     
        
        self.subscription_data = self.create_subscription(
            Array2d,      # PoseArray, Array2d
            'kpt_data',
            self.body_callback,
            10)
        self.subscription_data  # prevent unused variable warning
        

        self.robot_pos_publisher_ = self.create_publisher(Array2d, 'robot_cartesian_pos', 10)
        # self.arms_publisher_ = self.create_publisher(Array2d, 'arms_cartesian_pos', 10)
        # self.trunk_publisher_ = self.create_publisher(Array2d, 'trunk_cartesian_pos', 10)
        self.body_cartesian_pos = self.create_publisher(Array2d, 'body_cartesian_pos', 10)


    def robot_callback(self, tf_message):
        
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
        # increase their offset to account for the fingertips being further away than the finger joints 
        robot_rotation[-2] = robot_rotation[-3]
        quat_rot = R.from_quat(robot_rotation[-3]).apply(finger_trans[0])
        robot_translation[-2] =  robot_translation[-3] + quat_rot * 1.5

        robot_rotation[-1] = robot_rotation[-3]
        quat_rot = R.from_quat(robot_rotation[-3]).apply(finger_trans[1])
        robot_translation[-1] =  robot_translation[-3] + quat_rot * 1.5

        tf_world = {}
        for i, joint in enumerate(joint_name[1:]):
            tf_world[joint] = [robot_translation[i+1], robot_rotation[i+1]]
        
        link_poses = calculate_transforms(self.robot, tf_world)
        pos_world = np.array(list(link_poses.values()))[:, 0:3, -1]

        self.robot_cartesian_positions = pos_world

        self.publish_robot_pos()

        # # apply robot rotations to each robot axis, assuming that axis is locally [0, 0, 1]
        # joint_axes = np.zeros((7, 3))
        # joint_axes[:,-1] = np.ones(7)
        # self.axis_rot = R.from_quat(robot_rotation[1:-2]).apply(joint_axes)

        # t1 = time.time()
        # dt = t1 - t0
        # if dt>0.05:
        #     self.logger.info("robot_callback takes " + str(np.round(dt, 4)) + " seconds")
        # return link_poses

    def publish_robot_pos(self):
        
        robot_pos = self.robot_cartesian_positions

        pos_flattened = robot_pos.flatten(order='C')

        robot_message = Array2d()
        robot_message.array = list(pos_flattened.astype(float))
        [robot_message.height, robot_message.width]  = np.shape(robot_pos)
        self.robot_pos_publisher_.publish(robot_message)


    def body_callback(self, msg):

        ################
        # limb_dict = {'left':0, 'right':1, 'trunk':2, '_stop':-1}
        # Reshape message into array
        t0 = time.time()
        n_rows = msg.height
        n_cols = msg.width
        msg_array = np.array(msg.array)
        kpt_array = np.reshape(msg_array,(n_rows,n_cols))
        for body in np.unique(kpt_array[:,0]):
            body_kpts = kpt_array[kpt_array[:,0]==body,1:]
            if not str(int(body)) in self.bodies:
                self.bodies[str(int(body))] = [[], [], [], [time.time()]]
            for limb in np.unique(body_kpts[:,0]):
                limb_kpts = body_kpts[body_kpts[:,0]==limb,1:]
                # print(limb_kpts)
                offset_mat = np.tile(self.offset, (len(limb_kpts),1))
                limb_kpts = limb_kpts + offset_mat
                # remove rows with nan
                limb_kpts = limb_kpts[~np.isnan(limb_kpts).any(axis=1), :]
                # print(limb_kpts)
                self.bodies[str(int(body))][int(limb)] = limb_kpts
                self.bodies[str(int(body))][3][0] = time.time()


        if bool(self.bodies):
            # if you move out of frame too long, you will be assigned a new body ID
            # so we also need to increment it here

            # Scan to find currently active bodies:
            deceased = []
            for subject in self.bodies:
                ct = time.time()
                # expect bodies to be updated at 10 Hz, with a little leeway
                if ct - self.bodies[subject][3][0]>0.01 or not np.any(self.bodies[subject][0]):
                    deceased.append(subject)
            for body in deceased: del self.bodies[body] 

        if bool(self.bodies):
            # Switch to using oldest known body
            subjects = list(self.bodies.keys())
            subject = np.min(np.array(subjects).astype(int))
            self.subject = str(subject)

            self.publish_body_pos()

    
    def publish_body_pos(self):

        arms = np.concatenate([np.flip(self.bodies[self.subject][0], axis=0), self.bodies[self.subject][1]])
        # arms_flattened = arms.flatten(order='C')
        # arms_message = Array2d()
        # arms_message.array = list(arms_flattened.astype(float))
        # [arms_message.height, arms_message.width]  = np.shape(arms)
        # self.arms_publisher_.publish(arms_message)

        trunk = self.bodies[self.subject][2]
        # trunk_flattened = trunk.flatten(order='C')
        # trunk_message = Array2d()
        # trunk_message.array = list(trunk_flattened.astype(float))
        # [trunk_message.height, trunk_message.width]  = np.shape(trunk)
        # self.trunk_publisher_.publish(trunk_message)

        body = np.vstack((arms, trunk))
        body_flattened = body.flatten(order='C')
        false_link = len(arms)
        body_flattened = np.append(body_flattened, false_link)
        body_message = Array2d()
        body_message.array = list(body_flattened.astype(float))
        [body_message.height, body_message.width]  = np.shape(body)
        self.body_cartesian_pos.publish(body_message)

        # self.logger.info('published body pos')





        



def main(args = None):
    
    rclpy.init(args=args)

    pose_publisher = robot_pose()

    rclpy.spin(pose_publisher)

    rclpy.shutdown()
    
    print('done')
    
if __name__ == "__main__":
    main()