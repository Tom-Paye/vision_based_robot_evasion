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


def init_user_params():
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
    
def init_zed_params():
    # For now, all parameters are defined in this script
    global fusion_configurations, senders, network_senders, init_params, communication_parameters, \
        positional_tracking_parameters, body_tracking_parameters
    print("This sample display the fused body tracking of multiple cameras.")
    print("It needs a Localization file in input. Generate it with ZED 360.")
    print("The cameras can either be plugged to your devices, or already running on the local network.")
    fusion_configurations = sl.read_fusion_configuration_file(user_params.config_pth, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
    if len(fusion_configurations) <= 0:
        print("Invalid config file.")
        exit(1)

    senders = {}
    network_senders = {}
    
    # common parameters
    init_params = sl.InitParameters()
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True

    body_tracking_parameters = sl.BodyTrackingParameters()
    body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
    body_tracking_parameters.enable_body_fitting = False
    body_tracking_parameters.enable_tracking = False
    # return user_params




def main():
    
    init_user_params()
    init_zed_params()
    

    for conf in fusion_configurations:
        print("Try to open ZED", conf.serial_number)
        init_params.input = sl.InputType()
        # network cameras are already running, or so they should
        if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
            network_senders[conf.serial_number] = conf.serial_number

        # local camera needs to be run form here, in the same process than the fusion
        else:
            init_params.input = conf.input_type
            
            senders[conf.serial_number] = sl.Camera()

            # init_params.set_from_serial_number(conf.serial_number)
            init_params.set_from_svo_file('/usr/local/zed/samples/recording/playback/multi camera/cpp/build/clean_SN'+str(conf.serial_number)+'_720p_30fps.svo')
            status = senders[conf.serial_number].open(init_params)
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
        
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER
    init_fusion_parameters.output_performance_metrics = False
    init_fusion_parameters.verbose = True
    communication_parameters = sl.CommunicationParameters()
    fusion = sl.Fusion()
    camera_identifiers = []

    fusion.init(init_fusion_parameters)
        
    print("Cameras in this configuration : ", len(fusion_configurations))

    # warmup
    bodies = sl.Bodies()        
    for serial in senders:
        zed = senders[serial]
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)

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

    body_tracking_fusion_params = sl.BodyTrackingFusionParameters()
    body_tracking_fusion_params.enable_tracking = True
    body_tracking_fusion_params.enable_body_fitting = False
    
    fusion.enable_body_tracking(body_tracking_fusion_params)

    rt = sl.BodyTrackingFusionRuntimeParameters()
    rt.skeleton_minimum_allowed_keypoints = 7
    viewer = gl.GLViewer()
    viewer.init()

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    single_bodies = [sl.Bodies]

    chk = [False, False]
    while (viewer.is_available()):
        for serial in senders:
            zed = senders[serial]
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies)
                # zed.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE)
                for idx, cam_id in enumerate(camera_identifiers):
                    chk[idx] = fusion.retrieve_image(svo_image[idx], camera_identifiers[idx]) == sl.FUSION_ERROR_CODE.SUCCESS
                    if chk == [True, True]:
                        if svo_image[idx] != 0:
                            cv2.imshow("View"+str(idx), svo_image[idx].get_data()) #dislay both images to cv2
                            cv2.waitKey(2) 

        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            
            # Retrieve detected objects
            fusion.retrieve_bodies(bodies, rt)
            
            # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
            # for cam in camera_identifiers:
            #     fusion.retrieveBodies(single_bodies, rt, cam); 
            viewer.update_bodies(bodies)
            
            
    cv2.destroyAllWindows()
    for sender in senders:
        senders[sender].close()
        
    viewer.exit()


if __name__ == "__main__":
    
    main()

    


