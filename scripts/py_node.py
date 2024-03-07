#!/usr/bin/env python3

# import my_cpp_py_pkg.module_to_import
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 run my_cpp_py_pkg py_node.py

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory

import cv2
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import numpy as np
import os
import copy
import _thread
from geometry_msgs.msg import Vector3
import pickle

# def input_thread(a_list):
#     input()             # use input() in Python3
#     a_list.append(True)

def inverse_transform(R, t):
    T_inv = np.eye(4, 4)
    T_inv[:3, :3] = np.transpose(R)
    T_inv[:3, 3] = -1 * np.matmul(np.transpose(R), (t))
    return T_inv


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_vec = self.create_publisher(Vector3, 'kpt_label', 10)
        self.publisher_text = self.create_publisher(String, 'kpt_data', 10)
        
        self.i = 0
        self.key = ''
        # self.future.done = False

    def timer_callback(self, label, data):
        """"
        label: [str]
        data: 2D array of dim [x, 3] of doubles
        """
        
        [i, j] = np.shape(data)
        if j!=3:
            self.get_logger().Warning('position vectors are not the right dimension! \n'
                                      + 'expected 3 dimensions, got' + str(j) )
            exit()
            
        str_msg = String()
        str_msg.data = label
        self.publisher_text.publish(str_msg)
        
        for k in range(i):
            msg = Vector3()
            pos = data[k]
            [msg.x, msg.y, msg.z] = pos.astype(float)
            self.publisher_vec.publish(msg)
 
        self.get_logger().info(label)
        self.get_logger().info(str(data[0]))
        self.key = cv2.pollKey()
        self.i += 1

    def timer_caller(self, data):
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback(data))
        rclpy.spin_once(self)
        

def publisher(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

class local_functions():
    
    def __init__(self):
        self.user_params = self.init_user_params()
        self.zed_params = self.init_zed_params()
        self.senders = {}
        self.network_senders = {}
        
        # for conf in self.zed_params.fusion:
            
        self.connect_cams()
        print("self.senders started, running the fusion...")
        self.fusion = self.init_fusion()
        [self.bodies, self.svo_image, self.camera_identifiers] = self.subscribe_to_cam_outputs()

        [self.rt, self.bodies, self.single_bodies] = self.init_body_tracking_and_viewer()
        
        # a_list = []
        # _thread.start_new_thread(input_thread, (a_list,))

        self.chk = [False, False]
        if self.user_params.display_video < 3:
            self.chk = [True, True]

    def init_user_params(self):
        # For now, all parameters are defined in this script
        class user_params(): pass
        user_params.config_pth = '/home/tom/Downloads/Rec_1/calib/info/extrinsics.txt'
        #'/home/tom/Downloads/Rec_1/calib/info/extrinsics.txt', '/usr/local/zed/tools/zed_calib.json'
        user_params.video_src = 'SVO'                                       # SVO, Live
        user_params.svo_pth = '/usr/local/zed/samples/recording/playback/multi camera/cpp/build/'
        user_params.svo_prefix = 'std_SN'  #clean_SN, std_SN
        user_params.svo_suffix = '_720p_30fps.svo'
        user_params.display_video = 0                                       # 0: none, 1: cam 1, 2: cam 2, 3: both cams
        user_params.display_skeleton = False
        user_params.return_hands = False
        user_params.time_loop = False
        return user_params
        
        # return self.user_params
        
    def init_zed_params(self):
        # For now, all parameters are defined in this script
        print("This sample display the fused body tracking of multiple cameras.")
        print("It needs a Localization file in input. Generate it with ZED 360.")
        print("The cameras can either be plugged to your devices, or already running on the local network.")
        
        class zed_params(): pass
        
        if self.user_params.config_pth[-4:] == 'json':
            zed_params.fusion = sl.read_fusion_configuration_file(self.user_params.config_pth, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
        else:
            zed_params.fusion =  [] # list of both fusionconfigs
            objects = []
            file = open(self.user_params.config_pth,'rb')
            # print(file.read())
            ids = [32689769, 34783283]
            objects = (pickle.load(file))
            keysList = list(objects.keys())
            for i, key in enumerate(keysList):
                fus = sl.FusionConfiguration()
                
                R = objects[key][0][0]
                t = objects[key][0][1]
                # M = np.append(R, np.transpose([t]), axis=1)
                # M = np.append(M, [[0, 0, 0, 1]], axis=0)
                M = inverse_transform(R, t)

                T = sl.Transform()
                for j in range(4):
                    for k in range(4):
                        T[j,k] = M[j,k]
                
                fus.pose = T
                fus.serial_number = ids[i]
                zed_params.fusion.append(fus)
            
        
        
        
        
        
        if len(zed_params.fusion) <= 0:
            print("Invalid config file.")
            exit(1)
    
        
        
        # common parameters
        zed_params.init = sl.InitParameters()
        zed_params.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        zed_params.init.coordinate_units = sl.UNIT.METER
        zed_params.init.depth_mode = sl.DEPTH_MODE.ULTRA
        zed_params.init.camera_resolution = sl.RESOLUTION.HD720
        # zed_params.init.camera_fps = 30
        if self.user_params.video_src == 'SVO':
            zed_params.init.svo_real_time_mode = True
    
        zed_params.communication = sl.CommunicationParameters()
        zed_params.communication.set_for_shared_memory()
    
        zed_params.positional_tracking = sl.PositionalTrackingParameters()
        zed_params.positional_tracking.set_as_static = True
    
        zed_params.body_tracking = sl.BodyTrackingParameters()
        zed_params.body_tracking.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        if self.user_params.return_hands == True:
            zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_38
        else:
            zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_18
        zed_params.body_tracking.enable_body_fitting = True
        zed_params.body_tracking.enable_tracking = True
        
        zed_params.fusion_init = sl.InitFusionParameters()
        zed_params.fusion_init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        zed_params.fusion_init.coordinate_units = sl.UNIT.METER
        zed_params.fusion_init.output_performance_metrics = False
        zed_params.fusion_init.verbose = True
        
        zed_params.body_tracking_fusion = sl.BodyTrackingFusionParameters()
        zed_params.body_tracking_fusion.enable_tracking = True
        zed_params.body_tracking_fusion.enable_body_fitting = True

        # only for body_18
        zed_params.left_arm_keypoints = [5, 6, 7]
        zed_params.right_arm_keypoints = [2, 3, 4]
        zed_params.trunk_keypoints = [0, 1, 14, 15, 16, 17]

        # only for body_38
        zed_params.left_hand_keypoints = [30, 32, 34, 36]
        zed_params.right_hand_keypoints = [31, 33, 35 , 37]

        
        return zed_params
    
    
    def connect_cams(self):
    
        for conf in self.zed_params.fusion:
            print("Try to open ZED", conf.serial_number)
            self.zed_params.init.input = sl.InputType()
            # network cameras are already running, or so they should
            if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
                self.network_senders[conf.serial_number] = conf.serial_number
        
            # local camera needs to be run form here, in the same process than the fusion
            else:
                self.zed_params.init.input = conf.input_type
                
                self.senders[conf.serial_number] = sl.Camera()
        
                # self.zed_params.init.set_from_serial_number(conf.serial_number)
                if self.user_params.video_src == 'SVO':
                    self.zed_params.init.set_from_svo_file(self.user_params.svo_pth \
                              + self.user_params.svo_prefix + str(conf.serial_number)\
                                  + self.user_params.svo_suffix)
                else:
                    self.zed_params.init.set_from_serial_number(conf.serial_number)
                    
                status = self.senders[conf.serial_number].open(self.zed_params.init)
                
                if status != sl.ERROR_CODE.SUCCESS:
                    print("Error opening the camera", conf.serial_number, status)
                    del self.senders[conf.serial_number]
                    continue
                
        
                status = self.senders[conf.serial_number].enable_positional_tracking(self.zed_params.positional_tracking)
                if status != sl.ERROR_CODE.SUCCESS:
                    print("Error enabling the positional tracking of camera", conf.serial_number)
                    del self.senders[conf.serial_number]
                    continue
        
                status = self.senders[conf.serial_number].enable_body_tracking(self.zed_params.body_tracking)
                if status != sl.ERROR_CODE.SUCCESS:
                    print("Error enabling the body tracking of camera", conf.serial_number)
                    del self.senders[conf.serial_number]
                    continue
        
                self.senders[conf.serial_number].start_publishing(self.zed_params.communication)
        
            print("Camera", conf.serial_number, "is open")
            
        if len(self.senders) + len(self.network_senders) < 1:
            print("Not enough cameras")
            exit(1)
        
        
    def init_fusion(self):
        
        self.zed_params.communication = sl.CommunicationParameters()
        fusion = sl.Fusion()
        
    
        fusion.init(self.zed_params.fusion_init)
            
        print("Cameras in this configuration : ", len(self.zed_params.fusion))
        return fusion
        
        
    def subscribe_to_cam_outputs(self):
                
        bodies = sl.Bodies()        
        for serial in self.senders:
            zed = self.senders[serial]
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies)
                
        camera_identifiers = []
        svo_image = [None] * len(self.zed_params.fusion)
        for i in range(0, len(self.zed_params.fusion)):
            conf = self.zed_params.fusion[i]
            uuid = sl.CameraIdentifier()
            uuid.serial_number = conf.serial_number
            print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)
    
            status = self.fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
            if status != sl.FUSION_ERROR_CODE.SUCCESS:
                print("Unable to subscribe to", uuid.serial_number, status)
            else:
                camera_identifiers.append(uuid)
                svo_image[i] = sl.Mat()
                print("Subscribed.")
    
        if len(camera_identifiers) <= 0:
            print("No camera connected.")
            exit(1)
        return bodies, svo_image, camera_identifiers
    
    
    def init_body_tracking_and_viewer(self):
        
        self.fusion.enable_body_tracking(self.zed_params.body_tracking_fusion)
    
        rt = sl.BodyTrackingFusionRuntimeParameters()
        rt.skeleton_minimum_allowed_keypoints = 7
    
        if self.user_params.display_skeleton == True:
            self.viewer = gl.GLViewer()
            self.viewer.init()
    
        # Create ZED objects filled in the main loop
        bodies = sl.Bodies()
        single_bodies = [sl.Bodies]
        
        return rt, bodies, single_bodies
        
    
    def zed_loop(self):
        self.left_pos_all = np.array([[0, 0, 0]])
        self.right_pos_all = np.array([[0, 0, 0]])
        self.trunk_pos_all = np.array([[0, 0, 0]])
        for idx, serial in enumerate(self.senders):
            zed = self.senders[serial]
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(self.bodies)
                if (idx+1 == self.user_params.display_video) or (self.user_params.display_video == 3):
                    self.chk[idx] = self.fusion.retrieve_image(self.svo_image[idx], self.camera_identifiers[idx]) == sl.FUSION_ERROR_CODE.SUCCESS
                    if self.chk == [True, True]:
                        if self.svo_image[idx] != 0:
                            cv2.imshow("View"+str(idx), self.svo_image[idx].get_data()) #dislay both images to cv2
                                # key = cv2.waitKey(1) 
    
        if self.fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            
            # Retrieve detected objects
            self.fusion.retrieve_bodies(self.bodies, self.rt)
            self.fetch_skeleton()
            
            # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
            # for cam in self.camera_identifiers:
            #     fusion.retrieveBodies(self.single_bodies, rt, cam); 
            if (self.user_params.display_skeleton == True) and (self.viewer.is_available()):
                self.viewer.update_bodies(self.bodies)
                
    def fetch_skeleton(self):
        
        if len(self.bodies.body_list) > 0:
            for body in self.bodies.body_list:         
                
                if self.user_params.return_hands == True:
                    left_kpt_idx = self.zed_params.left_hand_keypoints
                    right_kpt_idx = self.zed_params.right_hand_keypoints
                else:
                    left_kpt_idx = self.zed_params.left_arm_keypoints
                    right_kpt_idx = self.zed_params.right_arm_keypoints
                trunk_kpt_idx = self.zed_params.trunk_keypoints

                left_matrix = np.array(body.keypoint[left_kpt_idx[0]]).reshape([1, 3])
                right_matrix = np.array(body.keypoint[right_kpt_idx[0]]).reshape([1, 3])
                trunk_matrix = np.array(body.keypoint[trunk_kpt_idx[0]]).reshape([1, 3])

                for h in range(len(left_kpt_idx)-1):
                    i = left_kpt_idx[h+1]
                    j = right_kpt_idx[h+1]
                    keypoint_left = np.array(body.keypoint[i]).reshape([1, 3])
                    keypoint_right = np.array(body.keypoint[j]).reshape([1, 3])  # loops from 50 to 70 (69 is last)
                    np.vstack((left_matrix, keypoint_left))  # left hand
                    np.vstack((right_matrix, keypoint_right))  # left hand, but in a mirror
                
                if self.user_params.return_hands == False:
                    for h in range(len(trunk_kpt_idx)-1):
                        k = trunk_kpt_idx[h+1]
                        keypoint_trunk = np.array(body.keypoint[k]).reshape([1, 3])
                        np.vstack((trunk_matrix, keypoint_trunk))
            
                left_pos = np.mean(left_matrix, axis=0)
                right_pos = np.mean(right_matrix, axis=0)
                trunk_pos = np.mean(trunk_matrix, axis=0)

                if not np.any(np.isnan(left_pos)):
                    lhp = np.array([left_pos[:]])
                    if not np.any(self.left_pos_all):
                        self.left_pos_all = lhp
                    else:
                        self.left_pos_all = np.append(self.left_pos_all, lhp, axis=0)

                if not np.any(np.isnan(right_pos)):
                    rhp = np.array([right_pos[:]])
                    if not np.any(self.right_pos_all):
                        self.right_pos_all = rhp
                    else:
                        self.right_pos_all = np.append(self.right_pos_all, rhp, axis=0)

                if not np.any(np.isnan(trunk_pos)):
                    thp = np.array([trunk_pos[:]])
                    if not np.any(self.trunk_pos_all):
                        self.trunk_pos_all = thp
                    else:
                        self.trunk_pos_all = np.append(self.trunk_pos_all, thp, axis=0)
            
                # print("shape of hand positions is ", np.shape(left_pos))

                
                
    def close(self):

        if (self.user_params.display_skeleton == True) and (self.viewer.is_available()):
            self.viewer.exit()

        cv2.destroyAllWindows()
        for sender in self.senders:
            self.senders[sender].close()


def main(args=None):
    

    rclpy.init(args=args)
    publisher = MinimalPublisher()
    cam = local_functions()
    key = ''
    checkpoint = time.time()
    
    while (key != ord("q")):

        if cam.user_params.time_loop:
            checkpoint_2 = time.time()
            print(checkpoint_2-checkpoint)
            checkpoint = checkpoint_2

        cam.zed_loop()
        output_l, output_r, output_t = cam.left_pos_all, cam.right_pos_all, cam.trunk_pos_all
        publisher.timer_callback('left', output_l)
        publisher.timer_callback('right', output_r)
        if cam.user_params.return_hands:
            publisher.timer_callback('trunk', output_t)

        key = cv2.pollKey()
            
    cam.close()
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    
    main()

    


