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
import my_cpp_py_pkg.ogl_viewer.viewer as gl
import numpy as np
import pickle
import json
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import my_cpp_py_pkg.visuals as visuals

def inverse_transform(R, t):
    T_inv = np.eye(4, 4)
    T_inv[:3, :3] = np.transpose(R)
    T_inv[:3, 3] = -1 * np.matmul(np.transpose(R), (t)) + np.array([1, 1, 0]).T
    return T_inv


class MinimalPublisher(Node):


    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_vec = self.create_publisher(PoseArray, 'kpt_data', 10)
        # self.publisher_vec = self.create_publisher(Quaternion, 'kpt_data', 10)
        # self.publisher_text = self.create_publisher(String, 'kpt_label', 10)

        self.body_parts = {
        'left' : 0,
        'right' : 1,
        'trunk' : 2,
        'stop' : -1
        }
        
        self.i = 0
        self.key = ''
        

    def timer_callback(self, label, body_id, data):
        """"
        label: [str]
        data: 2D array of dim [x, 3] of doubles
        """
        
        [i, j] = np.shape(data)
        if j!=3:
            self.get_logger().Warning('position vectors are not the right dimension! \n'
                                      + 'expected 3 dimensions, got' + str(j) )
            exit()

        # p = float(self.body_parts[label])
            
        # str_msg = String()
        # str_msg.data = label
        # self.publisher_text.publish(str_msg)
        
        msg_header = Header()
        msg_header.frame_id = str(body_id) + label
        msg_vec = PoseArray()
        msg_vec.header = msg_header
        # poses = set(i)
        for k in range(i):
            pose = Pose()
            point = Point()
            [point.x, point.y, point.z] = data[k].astype(float)
            pose.position = point
            # poses.add(pose)
            msg_vec.poses.append(pose)
            # msg_vec.poses.
        # msg_vec.poses = poses

        # msg = Quaternion()
        # for k in range(i):
        #     pos = data[k]
        #     [msg.x, msg.y, msg.z] = pos.astype(float)
        #     msg.w = p + 0.01*k
        #     self.publisher_vec.publish(msg)
 
        # self.get_logger().info(label)
        # # self.get_logger().info(str(data[0]))
        # self.get_logger().info(str(data))
        self.publisher_vec.publish(msg_vec)
        self.key = cv2.pollKey()
        self.i += 1

    def timer_caller(self, data):
        timer_period = 0.0001  # seconds
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

        [self.bodies, self.single_bodies] = self.init_body_tracking_and_viewer()

        self.chk = [False, False]
        if self.user_params.display_video < 3:
            self.chk = [True, True]

    def init_user_params(self):
        # For now, all parameters are defined in this script
        class user_params(): pass
        user_params.from_sl_config_file = False
        user_params.config_pth = '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib.json'
        # '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib.json'
        # '/home/tom/Downloads/Rec_1/calib/info/extrinsics.txt'
        # '/usr/local/zed/tools/zed_calib.json'
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
            if self.user_params.from_sl_config_file:
                zed_params.fusion = sl.read_fusion_configuration_file(self.user_params.config_pth,\
                                                                    sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
            else:
                zed_params.fusion =  [] # list of both fusionconfigs
                file = open(self.user_params.config_pth)
                configs = json.load(file)
                for config in configs:
                    fus = sl.FusionConfiguration()
                    fus.serial_number = int(config)

                    M = np.array(configs[config]['transform'])
                    N = np.array([[-1, 0,  0, 0.358],
                                  [ 0, 1,  0, 0.03],
                                  [ 0, 0, -1, 0.006],
                                  [ 0, 0,  0, 1]])
                    M = np.matmul(N, M)
                    T = sl.Transform()
                    for j in range(4):
                        for k in range(4):
                            T[j,k] = M[j,k]
                    fus.pose = T
                    zed_params.fusion.append(fus)
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
                # M = np.append(M, [[0, 0, 0, 1]], axis=0) # + np.array([1, 1, 1]).T
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
        zed_params.init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
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

        zed_params.body_tracking_runtime = sl.BodyTrackingRuntimeParameters()
        # zed_params.body_tracking_runtime.detection_confidence_threshold = 0.85 #confidence threshold actually doesnt work
        
        zed_params.fusion_init = sl.InitFusionParameters()
        zed_params.fusion_init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        zed_params.fusion_init.coordinate_units = sl.UNIT.METER
        zed_params.fusion_init.output_performance_metrics = False
        zed_params.fusion_init.verbose = True
        
        zed_params.body_tracking_fusion = sl.BodyTrackingFusionParameters()
        zed_params.body_tracking_fusion.enable_tracking = True
        zed_params.body_tracking_fusion.enable_body_fitting = True

        # rt = sl.BodyTrackingFusionRuntimeParameters()
        # rt.skeleton_minimum_allowed_keypoints = 7

        zed_params.body_tracking_fusion_runtime = sl.BodyTrackingFusionRuntimeParameters()
        zed_params.body_tracking_fusion_runtime.skeleton_minimum_allowed_keypoints = 7
        # zed_params.body_tracking_fusion_runtime.skeleton_minimum_allowed_camera = 1
        # zed_params.body_tracking_fusion_runtime.skeleton_smoothing = 0.5


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
                zed.retrieve_bodies(bodies, self.zed_params.body_tracking_runtime)
                
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
    
        # rt = sl.BodyTrackingFusionRuntimeParameters()
        # rt.skeleton_minimum_allowed_keypoints = 7
    
        if self.user_params.display_skeleton == True:
            self.viewer = gl.GLViewer()
            self.viewer.init()
    
        # Create ZED objects filled in the main loop
        bodies = sl.Bodies()
        single_bodies = [sl.Bodies]
        self.known_bodies = {}
        
        return bodies, single_bodies
        
    
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
            
            # TODO: Likely error in fusion initialisation that screws with the camera fusion
            # Retrieve detected objects
            self.fusion.retrieve_bodies(self.bodies, self.zed_params.body_tracking_fusion_runtime)
            self.fetch_skeleton()
            
            # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
            # for cam in self.camera_identifiers:
            #     fusion.retrieveBodies(self.single_bodies, rt, cam); 
            if (self.user_params.display_skeleton == True):
                print(str(self.viewer.is_available()))
                if (self.viewer.is_available()):
                    self.viewer.update_bodies(self.bodies)
            

            
                
    def fetch_skeleton(self):
        
        if len(self.bodies.body_list) > 0:
            # TODO: Check that the keypoints collected are the right ones
            # TODO: Check that the keypoints are taken along the right axis
            # TODO: Check that the transforéation matrix is applied right
            if self.user_params.return_hands == True:
                left_kpt_idx = self.zed_params.left_hand_keypoints
                right_kpt_idx = self.zed_params.right_hand_keypoints
            else:
                left_kpt_idx = self.zed_params.left_arm_keypoints
                right_kpt_idx = self.zed_params.right_arm_keypoints
            trunk_kpt_idx = self.zed_params.trunk_keypoints
            
            for body in self.bodies.body_list:    
                
                # left_matrix = np.array(body.keypoint[left_kpt_idx[0]]).reshape([1, 3])
                # right_matrix = np.array(body.keypoint[right_kpt_idx[0]]).reshape([1, 3])
                # trunk_matrix = np.array(body.keypoint[trunk_kpt_idx[0]]).reshape([1, 3])

                # for h in range(len(left_kpt_idx)-1):    # fetch coords of each kpt
                #     i = left_kpt_idx[h+1]
                #     j = right_kpt_idx[h+1]
                #     keypoint_left = np.array(body.keypoint[i]).reshape([1, 3])
                #     keypoint_right = np.array(body.keypoint[j]).reshape([1, 3])  # loops from 50 to 70 (69 is last)
                #     left_matrix = np.vstack((left_matrix, keypoint_left))  # left hand
                #     right_matrix = np.vstack((right_matrix, keypoint_right))  # left hand, but in a mirror
                left_matrix = np.array(body.keypoint[left_kpt_idx])
                right_matrix = np.array(body.keypoint[right_kpt_idx])
                
                
                if self.user_params.return_hands == False:
                    trunk_matrix = np.array(body.keypoint[trunk_kpt_idx])
                    # for h in range(len(trunk_kpt_idx)-1):
                    #     k = trunk_kpt_idx[h+1]
                    #     keypoint_trunk = np.array(body.keypoint[k]).reshape([1, 3])
                    #     trunk_matrix = np.vstack((trunk_matrix, keypoint_trunk))
            
                if self.user_params.return_hands:   # if hands, only return the mean kpt position
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
                else:
                    self.left_pos_all = left_matrix
                    self.right_pos_all = right_matrix
                    self.trunk_pos_all = trunk_matrix

                body_id = body.id
                if body_id in self.known_bodies:
                    self.known_bodies[body_id][0] = left_matrix
                    self.known_bodies[body_id][1] = right_matrix
                    self.known_bodies[body_id][2] = trunk_matrix
                else:
                    self.known_bodies[body_id] = [left_matrix, right_matrix, trunk_matrix]

                # if not np.any(np.isnan(trunk_pos)):
                #     thp = np.array([trunk_pos[:]])
                #     if not np.any(self.trunk_pos_all):
                #         self.trunk_pos_all = thp
                #     else:
                #         self.trunk_pos_all = np.append(self.trunk_pos_all, thp, axis=0)
            
        if self.trunk_pos_all[0][0] < -3:
            print("this loop done")
              
                
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
    fig = 0
    
    while (key != ord("q")):

        if cam.user_params.time_loop:
            checkpoint_2 = time.time()
            print(checkpoint_2-checkpoint)
            checkpoint = checkpoint_2

        cam.zed_loop()
        # output_l, output_r, output_t = cam.left_pos_all, cam.right_pos_all, cam.trunk_pos_all
        for body_id in cam.known_bodies:

            [output_l, output_r, output_t] = cam.known_bodies[body_id]
            publisher.timer_callback('left', body_id, output_l)
            publisher.timer_callback('right', body_id, output_r)
            if not cam.user_params.return_hands:
                publisher.timer_callback('trunk', body_id, output_t)
            publisher.timer_callback('stop', '-1', np.array([[-1, -1, -1]]))

        if not (not cam.known_bodies):
            disp_body = list(cam.known_bodies)[0]
            ###############
            class geom(): pass
            geom.arm_pos = cam.known_bodies[disp_body][0]
            geom.trunk_pos = cam.known_bodies[disp_body][1]
            geom.robot_pos = cam.known_bodies[disp_body][2]
            geom.arm_cp_idx = [0, 1]
            geom.u = np.eye(2)
            geom.trunk_cp_idx = [0, 1]
            geom.v = np.eye(2)
            geom.robot_cp_arm_idx = [0, 1]
            geom.s = np.eye(2)
            geom.robot_cp_trunk_idx = [0, 1]
            geom.t = np.eye(2)

            fig = visuals.plot_skeletons(fig, geom)

            ###############

        key = cv2.pollKey()
            
    cam.close()
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    
    main()

    


