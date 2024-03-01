#!/usr/bin/env python3

# import my_cpp_py_pkg.module_to_import

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

# def input_thread(a_list):
#     input()             # use input() in Python3
#     a_list.append(True)

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_txt = self.create_publisher(String, 'topic', 10)
        self.publisher_zed = self.create_publisher(String, 'topic', 10)
        self.i = 0
        self.key = ''
        # self.future.done = False

    def timer_callback(self, data):
        # if self.key == ord("q"):
        #     self.future.done = True
        msg = String()
        # msg.data = 'Hello World: %d' % self.i
        msg.data = data + '  ::  ' + str(self.i)
        self.publisher_txt.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
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

        [self.rt, self.viewer, self.bodies, self.single_bodies] = self.init_body_tracking_and_viewer()
        
        # a_list = []
        # _thread.start_new_thread(input_thread, (a_list,))

        self.chk = [False, False]
        if self.user_params.display_video < 3:
            self.chk = [True, True]

    def init_user_params(self):
        # For now, all parameters are defined in this script
        class user_params(): pass
        user_params.config_pth = '/usr/local/zed/tools/zed_calib.json'
        user_params.video_src = 'SVO'                                       # SVO, Live
        user_params.svo_pth = '/usr/local/zed/samples/recording/playback/multi camera/cpp/build/clean_SN'
        user_params.svo_suffix = '_720p_30fps.svo'
        user_params.display_video = 2                                       # 0: none, 1: cam 1, 2: cam 2, 3: both cams
        user_params.display_skeleton = True
        return user_params
        
        # return self.user_params
        
    def init_zed_params(self):
        # For now, all parameters are defined in this script
        print("This sample display the fused body tracking of multiple cameras.")
        print("It needs a Localization file in input. Generate it with ZED 360.")
        print("The cameras can either be plugged to your devices, or already running on the local network.")
        
        class zed_params(): pass
        zed_params.fusion = sl.read_fusion_configuration_file(self.user_params.config_pth, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
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
        zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_38
        zed_params.body_tracking.enable_body_fitting = True
        zed_params.body_tracking.enable_tracking = True
        
        zed_params.fusion_init = sl.InitFusionParameters()
        zed_params.fusion_init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        zed_params.fusion_init.coordinate_units = sl.UNIT.METER
        zed_params.fusion_init.output_performance_metrics = False
        zed_params.fusion_init.verbose = True
        
        zed_params.body_tracking_fusion = sl.BodyTrackingFusionParameters()
        zed_params.body_tracking_fusion.enable_tracking = True
        zed_params.body_tracking_fusion.enable_body_fitting = False
        
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
                              + str(conf.serial_number) + self.user_params.svo_suffix)
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
            viewer = gl.GLViewer()
            viewer.init()
    
        # Create ZED objects filled in the main loop
        bodies = sl.Bodies()
        single_bodies = [sl.Bodies]
        
        return rt, viewer, bodies, single_bodies
        
    
    def zed_loop(self):
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
            [self.left_hand_pos_all, self.right_hand_pos_all] = self.fetch_skeleton()
            
            # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
            # for cam in self.camera_identifiers:
            #     fusion.retrieveBodies(self.single_bodies, rt, cam); 
            if (self.user_params.display_skeleton == True) and (self.viewer.is_available()):
                self.viewer.update_bodies(self.bodies)
        return 'zed_loop successful'
                
    def fetch_skeleton(self):
        
        left_hand_pos_all = np.array([[0, 0, 0]])
        right_hand_pos_all = np.array([[0, 0, 0]])
        
        if len(self.bodies.body_list) > 0:
            for body in self.bodies.body_list:
                left_hand_matrix = np.array(body.keypoint[16]).reshape([1, 3])
                right_hand_matrix = np.array(body.keypoint[17]).reshape([1, 3])
            
                for i in range(30, 37, 2):
                    keypoint_left = np.array(body.keypoint[i]).reshape([1, 3])
                    keypoint_right = np.array(body.keypoint[i + 1]).reshape([1, 3])  # loops from 50 to 70 (69 is last)
                    np.vstack((left_hand_matrix, keypoint_left))  # left hand
                    np.vstack((right_hand_matrix, keypoint_right))  # left hand
            
                left_hand_pos = np.mean(left_hand_matrix, axis=0)
                right_hand_pos = np.mean(right_hand_matrix, axis=0)

                if not np.any(np.isnan(left_hand_pos)):
                    lhp = np.array([left_hand_pos[:]])
                    if not np.any(left_hand_pos_all):
                        left_hand_pos_all = lhp
                    else:
                        left_hand_pos_all = np.append(left_hand_pos_all, lhp, axis=0)

                if not np.any(np.isnan(right_hand_pos)):
                    rhp = np.array([right_hand_pos[:]])
                    if not np.any(right_hand_pos_all):
                        right_hand_pos_all = rhp
                    else:
                        right_hand_pos_all = np.append(right_hand_pos_all, rhp, axis=0)
            
                # print("shape of hand positions is ", np.shape(left_hand_pos))
            
        return left_hand_pos_all, right_hand_pos_all
                
                
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
    # while (viewer.is_available()):
    checkpoint = time.time()
    while (key != ord("p")):

        checkpoint_2 = time.time()
        print(checkpoint_2-checkpoint)
        checkpoint = checkpoint_2
        # if a_list: # key == ord("q"):
        #     break
        # if key == ord("q"):
        #     break

        # zed = self.senders[30635524]
        # zed.retrieve_image(self.svo_image[1], sl.VIEW.SIDE_BY_SIDE)
        # cv2.imshow("View"+str(1), self.svo_image[1].get_data())


        
        output = cam.zed_loop()
        publisher.timer_callback(output)


        key = cv2.pollKey()
            
    cam.close()



    # rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    
    main()

    


