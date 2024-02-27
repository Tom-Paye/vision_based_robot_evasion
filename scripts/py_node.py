#!/usr/bin/env python3

# import my_cpp_py_pkg.module_to_import

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String

import cv2
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import numpy as np
import os
import copy


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def publisher(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


def initialize_parameters():
    # For now, all parameters are defined in this script
    global user_params
    class user_params: pass
    user_params.config_pth = '/usr/local/zed/tools/zed_calib.json'
    user_params.video_src = 'SVO'                                       # SVO, Live
    user_params.svo_pth = '/usr/local/zed/samples/recording/playback/multi camera/cpp/build/clean_SN'
    user_params.svo_suffix = '_720p_30fps.svo'
    user_params.display_video = True
    user_params.display_skeleton = True
    # return user_params

def initialize_view(): #user_params):
    global user_params, viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt
    print("This sample display the fused body tracking of multiple cameras.")
    print("It needs a Localization file in input. Generate it with ZED 360.")
    print("The cameras can either be plugged to your devices, taken from an SVO file, or already running on the local network.")

    global fusion_configurations
    fusion_configurations = sl.read_fusion_configuration_file(user_params.config_pth, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
    if len(fusion_configurations) <= 0:
        print("Invalid config file.")
        exit(1)

    # global senders
    # senders = {}
    # global network_senders
    # network_senders = {}

    # # common parameters
    # global init_params
    # init_params = sl.InitParameters()
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # init_params.coordinate_units = sl.UNIT.METER
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.camera_resolution = sl.RESOLUTION.HD1080

    global communication_parameters
    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()

    global positional_tracking_parameters
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True

    global body_tracking_parameters
    body_tracking_parameters = sl.BodyTrackingParameters()
    body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
    body_tracking_parameters.enable_body_fitting = True
    body_tracking_parameters.enable_tracking = True

    global init_params_all
    init_params_all = [None] * len(fusion_configurations)
    for i, conf in enumerate(fusion_configurations):
        init_params_all[i] = sl.InitParameters()
        init_params_all[i].coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params_all[i].coordinate_units = sl.UNIT.METER
        init_params_all[i].depth_mode = sl.DEPTH_MODE.ULTRA
        init_params_all[i].camera_resolution = sl.RESOLUTION.HD720
        # print("Try to open ZED", conf.serial_number)
        init_params_all[i].input = sl.InputType()
        # network cameras are already running, or so they should
        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            network_senders[conf.serial_number] = conf.serial_number

        # local camera needs to be run form here, in the same process than the fusion
        else:
            init_params_all[i].input = conf.input_type
            
            senders[conf.serial_number] = sl.Camera()

            # init_params.set_from_serial_number(conf.serial_number)
            if user_params.video_src == 'SVO':
                init_params_all[i].set_from_svo_file(user_params.svo_pth+str(conf.serial_number)+user_params.svo_suffix)
            else:
                init_params_all[i].set_from_serial_number(conf.serial_number)
            
        #     status = senders[conf.serial_number].open(init_params)
        #     if status != sl.ERROR_CODE.SUCCESS:
        #         print("Error opening the camera", conf.serial_number, status)
        #         del senders[conf.serial_number]
        #         continue
            
        #     status = senders[conf.serial_number].enable_positional_tracking(positional_tracking_parameters)
        #     if status != sl.ERROR_CODE.SUCCESS:
        #         print("Error enabling the positional tracking of camera", conf.serial_number)
        #         del senders[conf.serial_number]
        #         continue

        #     status = senders[conf.serial_number].enable_body_tracking(body_tracking_parameters)
        #     if status != sl.ERROR_CODE.SUCCESS:
        #         print("Error enabling the body tracking of camera", conf.serial_number)
        #         del senders[conf.serial_number]
        #         continue

        #     senders[conf.serial_number].start_publishing(communication_parameters)

        # print("Camera", conf.serial_number, "is open")
        
    
    # if len(senders) + len(network_senders) < 1:
    #     print("No enough cameras")
    #     exit(1)

    # print("Senders started, running the fusion...")
        
    global init_fusion_parameters
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER
    init_fusion_parameters.output_performance_metrics = False
    init_fusion_parameters.verbose = True

    communication_parameters = sl.CommunicationParameters()
    # global fusion
    # fusion = sl.Fusion()
    # global camera_identifiers
    # camera_identifiers = []

    # fusion.init(init_fusion_parameters)
        
    # print("Cameras in this configuration : ", len(fusion_configurations))

    # # warmup
    # global bodies
    # bodies = sl.Bodies()        
    # for serial in senders:
    #     zed = senders[serial]
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #         zed.retrieve_bodies(bodies)

    # global svo_image
    # svo_image = [None] * len(fusion_configurations)
    # for i in range(0, len(fusion_configurations)):
    #     conf = fusion_configurations[i]
    #     uuid = sl.CameraIdentifier()
    #     uuid.serial_number = conf.serial_number
    #     print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)

    #     status = fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
    #     if status != sl.FUSION_ERROR_CODE.SUCCESS:
    #         print("Unable to subscribe to", uuid.serial_number, status)
    #     else:
    #         camera_identifiers.append(uuid)
    #         svo_image[i] = sl.Mat()
    #         print("Subscribed.")

    # if len(camera_identifiers) <= 0:
    #     print("No camera connected.")
    #     exit(1)

    global body_tracking_fusion_params
    body_tracking_fusion_params = sl.BodyTrackingFusionParameters()
    body_tracking_fusion_params.enable_tracking = True
    body_tracking_fusion_params.enable_body_fitting = True
    
    # fusion.enable_body_tracking(body_tracking_fusion_params)

    # global rt
    # rt = sl.BodyTrackingFusionRuntimeParameters()
    # rt.skeleton_minimum_allowed_keypoints = 7
    # global viewer
    # viewer = False
    # if user_params.display_skeleton:
    #     viewer = gl.GLViewer()
    #     viewer.init()

    # # Create ZED objects filled in the main loop
    # bodies = sl.Bodies()
    # global single_bodies
    # single_bodies = [sl.Bodies]

    # return viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt

def loop_video(): #user_params, viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt):
    global user_params, viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt
    
    chk = [False, False]
    for serial in senders:
        zed = senders[serial]
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)

            # Display images in separate windows
            if user_params.display_video:
                for idx, cam_id in enumerate(camera_identifiers):
                    chk[idx] = fusion.retrieve_image(svo_image[idx], camera_identifiers[idx]) == sl.FUSION_ERROR_CODE.SUCCESS
                    if chk == [True, True]:
                        if svo_image[idx] != 0:
                            cv2.imshow("View"+str(idx), svo_image[idx].get_data()) #dislay both images to cv2
                            # cv2.waitKey(2) 

    fus_err = fusion.process()
    if fus_err == sl.FUSION_ERROR_CODE.SUCCESS:
    
        # Retrieve detected objects
        fusion.retrieve_bodies(bodies, rt)
        
        # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
        # for cam in camera_identifiers:
        #     fusion.retrieveBodies(single_bodies, rt, cam); 
        if user_params.display_skeleton:
            upd_err = viewer.update_bodies(bodies)

    # return viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt

def close_script(): #viewer, senders):
    global viewer, senders
    cv2.destroyAllWindows()
    for sender in senders:
        senders[sender].close()
    if user_params.display_skeleton:
        viewer.exit()



######################################################################################################################################################

########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String

import cv2
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import numpy as np
import os

def main():
    # user_params = initialize_parameters()
    global user_params
    initialize_parameters()

    # [viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt]\
    # = initialize_view(user_params)
    
    global senders
    senders = {}
    global network_senders
    network_senders = {}
    
    initialize_view()

    ################################################################################

    # Start the cameras
    for i, conf in enumerate(fusion_configurations):
        print("Try to open ZED", conf.serial_number)

        if not conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            senders[conf.serial_number] = sl.Camera()

        status = senders[conf.serial_number].open(init_params_all[i])
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error opening the camera", conf.serial_number, status)
            del senders[conf.serial_number]
            continue
        
        status = senders[conf.serial_number].enable_positional_tracking(positional_tracking_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error enabling the positional tracking of camera", conf.serial_number)
            del senders[conf.serial_number]
            continue

        status = senders[conf.serial_number].enable_body_tracking(body_tracking_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error enabling the body tracking of camera", conf.serial_number)
            del senders[conf.serial_number]
            continue

        senders[conf.serial_number].start_publishing(communication_parameters)
        print("Camera", conf.serial_number, "is open")
    if len(senders) + len(network_senders) < 1:
        print("No enough cameras")
        exit(1)
    print("Senders started, running the fusion...")

    # Perform fusion
    fusion = sl.Fusion()
    camera_identifiers = []
    fusion.init(init_fusion_parameters)
    print("Cameras in this configuration : ", len(fusion_configurations))

    # warmup
    global bodies
    bodies = sl.Bodies()        
    for serial in senders:
        zed = senders[serial]
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)

    global svo_image
    svo_image = [None] * len(fusion_configurations)
    for i in range(0, len(fusion_configurations)):
        conf = fusion_configurations[i]
        uuid = sl.CameraIdentifier()
        uuid.serial_number = conf.serial_number
        print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)

        status = fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
        if status != sl.FUSION_ERROR_CODE.SUCCESS:
            print("Unable to subscribe to", uuid.serial_number, status)
        else:
            camera_identifiers.append(uuid)
            svo_image[i] = sl.Mat()
            print("Subscribed.")

    if len(camera_identifiers) <= 0:
        print("No camera connected.")
        exit(1)

    fusion.enable_body_tracking(body_tracking_fusion_params)

    rt = sl.BodyTrackingFusionRuntimeParameters()
    rt.skeleton_minimum_allowed_keypoints = 7
    global viewer
    viewer = False
    if user_params.display_skeleton:
        viewer = gl.GLViewer()
        viewer.init()

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    single_bodies = [sl.Bodies]


    key = ''
    while key != 113:
        # [viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt]\
        #     = loop_video(user_params, viewer, bodies, single_bodies, senders, camera_identifiers, fusion, svo_image, rt)
        # loop_video()
        #################################################################################################


        #################################################################################################
        chk = [False, False]
        for serial in senders:
            zed = senders[serial]
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies)

                # Display images in separate windows
                if user_params.display_video:
                    for idx, cam_id in enumerate(camera_identifiers):
                        chk[idx] = fusion.retrieve_image(svo_image[idx], camera_identifiers[idx]) == sl.FUSION_ERROR_CODE.SUCCESS
                        if chk == [True, True]:
                            if svo_image[idx] != 0:
                                cv2.imshow("View"+str(idx), svo_image[idx].get_data()) #dislay both images to cv2
                                # cv2.waitKey(2) 

        fus_err = fusion.process()
        if fus_err == sl.FUSION_ERROR_CODE.SUCCESS:
        
            # Retrieve detected objects
            fusion.retrieve_bodies(bodies, rt)
            
            # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
            # for cam in camera_identifiers:
            #     fusion.retrieveBodies(single_bodies, rt, cam); 
            if user_params.display_skeleton:
                viewer.update_bodies(bodies)
        key = cv2.waitKey(1)
        

    close_script() #viewer, senders)

    # publisher()

if __name__ == "__main__":

    main()

    # print("This sample display the fused body tracking of multiple cameras.")
    # print("It needs a Localization file in input. Generate it with ZED 360.")
    # print("The cameras can either be plugged to your devices, taken from an SVO file, or already running on the local network.")

    # fusion_configurations = sl.read_fusion_configuration_file(user_params.config_pth, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
    # if len(fusion_configurations) <= 0:
    #     print("Invalid config file.")
    #     exit(1)

    # senders = {}
    # network_senders = {}

    # # common parameters
    # init_params = sl.InitParameters()
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # init_params.coordinate_units = sl.UNIT.METER
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.camera_resolution = sl.RESOLUTION.HD1080

    # communication_parameters = sl.CommunicationParameters()
    # communication_parameters.set_for_shared_memory()

    # positional_tracking_parameters = sl.PositionalTrackingParameters()
    # positional_tracking_parameters.set_as_static = True

    # body_tracking_parameters = sl.BodyTrackingParameters()
    # body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    # body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
    # body_tracking_parameters.enable_body_fitting = False
    # body_tracking_parameters.enable_tracking = False

    # for conf in fusion_configurations:
    #     print("Try to open ZED", conf.serial_number)
    #     init_params.input = sl.InputType()
    #     # network cameras are already running, or so they should
    #     if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
    #         network_senders[conf.serial_number] = conf.serial_number

    #     # local camera needs to be run form here, in the same process than the fusion
    #     else:
    #         init_params.input = conf.input_type
            
    #         senders[conf.serial_number] = sl.Camera()

    #         # init_params.set_from_serial_number(conf.serial_number)
    #         if user_params.video_src == 'SVO':
    #             init_params.set_from_svo_file(user_params.svo_pth+str(conf.serial_number)+user_params.svo_suffix)
    #         else:
    #             init_params.set_from_serial_number(conf.serial_number)
            
    #         status = senders[conf.serial_number].open(init_params)
    #         if status != sl.ERROR_CODE.SUCCESS:
    #             print("Error opening the camera", conf.serial_number, status)
    #             del senders[conf.serial_number]
    #             continue
            
    #         status = senders[conf.serial_number].enable_positional_tracking(positional_tracking_parameters)
    #         if status != sl.ERROR_CODE.SUCCESS:
    #             print("Error enabling the positional tracking of camera", conf.serial_number)
    #             del senders[conf.serial_number]
    #             continue

    #         status = senders[conf.serial_number].enable_body_tracking(body_tracking_parameters)
    #         if status != sl.ERROR_CODE.SUCCESS:
    #             print("Error enabling the body tracking of camera", conf.serial_number)
    #             del senders[conf.serial_number]
    #             continue

    #         senders[conf.serial_number].start_publishing(communication_parameters)

    #     print("Camera", conf.serial_number, "is open")
        
    
    # if len(senders) + len(network_senders) < 1:
    #     print("No enough cameras")
    #     exit(1)

    # print("Senders started, running the fusion...")
        
    # init_fusion_parameters = sl.InitFusionParameters()
    # init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # init_fusion_parameters.coordinate_units = sl.UNIT.METER
    # init_fusion_parameters.output_performance_metrics = False
    # init_fusion_parameters.verbose = True
    # communication_parameters = sl.CommunicationParameters()
    # fusion = sl.Fusion()
    # camera_identifiers = []

    # fusion.init(init_fusion_parameters)
        
    # print("Cameras in this configuration : ", len(fusion_configurations))

    # # warmup
    # bodies = sl.Bodies()        
    # for serial in senders:
    #     zed = senders[serial]
    #     if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #         zed.retrieve_bodies(bodies)

    # svo_image = [None] * len(fusion_configurations)
    # for i in range(0, len(fusion_configurations)):
    #     conf = fusion_configurations[i]
    #     uuid = sl.CameraIdentifier()
    #     uuid.serial_number = conf.serial_number
    #     print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)

    #     status = fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
    #     if status != sl.FUSION_ERROR_CODE.SUCCESS:
    #         print("Unable to subscribe to", uuid.serial_number, status)
    #     else:
    #         camera_identifiers.append(uuid)
    #         svo_image[i] = sl.Mat()
    #         print("Subscribed.")

    # if len(camera_identifiers) <= 0:
    #     print("No camera connected.")
    #     exit(1)

    # body_tracking_fusion_params = sl.BodyTrackingFusionParameters()
    # body_tracking_fusion_params.enable_tracking = True
    # body_tracking_fusion_params.enable_body_fitting = False
    
    # fusion.enable_body_tracking(body_tracking_fusion_params)

    # rt = sl.BodyTrackingFusionRuntimeParameters()
    # rt.skeleton_minimum_allowed_keypoints = 7
    # if user_params.display_skeleton:
    #     viewer = gl.GLViewer()
    #     viewer.init()

    # # Create ZED objects filled in the main loop
    # bodies = sl.Bodies()
    # single_bodies = [sl.Bodies]

    
    # while True:  #(viewer.is_available()):
    #     chk = [False, False]
    #     for serial in senders:
    #         zed = senders[serial]
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #             zed.retrieve_bodies(bodies)

    #             # Display images in separate windows
    #             if user_params.display_video:
    #                 for idx, cam_id in enumerate(camera_identifiers):
    #                     chk[idx] = fusion.retrieve_image(svo_image[idx], camera_identifiers[idx]) == sl.FUSION_ERROR_CODE.SUCCESS
    #                     if chk == [True, True]:
    #                         if svo_image[idx] != 0:
    #                             cv2.imshow("View"+str(idx), svo_image[idx].get_data()) #dislay both images to cv2
    #                             cv2.waitKey(2) 

    #     if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            
    #         # Retrieve detected objects
    #         fusion.retrieve_bodies(bodies, rt)
            
    #         # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
    #         # for cam in camera_identifiers:
    #         #     fusion.retrieveBodies(single_bodies, rt, cam); 
    #         viewer.update_bodies(bodies)
            
            
    # cv2.destroyAllWindows()
    # for sender in senders:
    #     senders[sender].close()
        
    # viewer.exit()



