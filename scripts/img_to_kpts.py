#!/usr/bin/env python3

# import my_cpp_py_pkg.module_to_import
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 run my_cpp_py_pkg py_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from std_msgs.msg import String

import pyzed.sl as sl
import my_cpp_py_pkg.visuals as visuals

from ament_index_python.packages import get_package_share_directory

import cv2
import my_cpp_py_pkg.cv_viewer.tracking_viewer as cv_viewer
from cv_bridge import CvBridge, CvBridgeError

import my_cpp_py_pkg.ogl_viewer.viewer as gl

import logging
import time
import numpy as np
import json
import copy




#############################################INITIALIZE USER PARAMETERS#####################################
"""
For now, assume we want to output body_34 data, with a two keypoints for the hands
example of code : https://community.stereolabs.com/t/re-identifiaction-in-fused-camera/3071
"""

def init_user_params():
    # For now, all parameters are defined in this script
    class user_params(): pass

    user_params.from_sl_config_file = False
    user_params.config_pth = '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib_2.json'
    # '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib_2.json'
    # '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib.json'
    # '/home/tom/Downloads/Rec_1/calib/info/extrinsics.txt'
    # '/usr/local/zed/tools/zed_calib.json'

    user_params.video_src = 'SVO'                                       # SVO, Live
    user_params.svo_pth = '/usr/local/zed/samples/recording/playback/multi camera/cpp/build/'
    user_params.svo_prefix = 'std_SN'  #clean_SN, std_SN
    user_params.svo_suffix = '_720p_30fps.svo'

    user_params.display_video = 1                                    # 0: none, 1: cam 1, 2: cam 2, 3: both cams
    user_params.display_skeleton = False    # DO NOT USE, OPENGL IS A CANCER WHICH SHOULD NEVER BE GIVEN RAM
    user_params.time_loop = False

    user_params.body_type = 'BODY_34' # BODY_18, BODY_34, BODY_38, for some reason we are stuck with 34
    user_params.real_time = True

    logging.basicConfig(level=logging.DEBUG)
    
    return user_params


def init_zed_params(user_params):

    """
    Assume the configuration file is a .json file containing only the 4 dimensional transform matrices
    All ZED operations are performed from the POV of the first given camera, for visualisation purposes
    """

    logger = logging.getLogger(__name__)

    class zed_params(): pass
    
    zed_params.fusion =  [] # list of both fusionconfigs
    file = open(user_params.config_pth)
    configs = json.load(file)

    zed_params.base_transform = np.array(configs['base']['transform'])
    for config in configs:
        M = np.array(configs[config]['transform'])
        if config != 'base':
            fus = sl.FusionConfiguration()
            fus.serial_number = int(config)

            """
            Assume Zed fusion sets 1 camera as the origin, rather than the world origin
            This means you have to transform the coordinates of the second camera to what its
            position is in the first camera's frame of reference
            --> ZED PYTHON API : subscribe pyzed::sl::Fusion
            https://www.stereolabs.com/docs/fusion/overview

            If you want to apply a different cam position for the sake of getting openGL to look the right way:
            - only add translation to the first cam, before any other operation           
            """

            if not np.any(zed_params.fusion):   # Cam 1
                N = copy.copy(M)    # table coords --> Cam 1 coords (Cam 1 as seen from table)
                # [obj]world frame = N * [obj]cam frame
                P = inverse_transform(N[0:3, 0:3], N[0:3, 3])
                M = np.eye(4)
                zed_params.cam_transform = N
            else:
                Q = inverse_transform(M[0:3, 0:3], M[0:3, 3])
                M = N @ Q
                # M = P @ M  
            # if user_params.display_skeleton == True:    # To relocate display so it is visible in GLViewer
            #     visual_correction = np.array([[0, 0,  0, 0],
            #                                   [0, 0,  0, 1],
            #                                   [0, 0, 0, -3],
            #                                   [0, 0,  0, 0]])
            #     M = M + visual_correction
            
            # visualize: 
            # Np = np.eye(4) + visual_correction
            # Pp = inverse_transform(Np[0:3, 0:3], Np[0:3, 3])
            # Pp = P - visual_correction
            # Mp = Pp @ M
            # visuals.plot_axes(Pp, M)
            # M = zed_params.base_transform @ M
            # visual_correction = np.array([[0, 0,  0, 1],
            #                               [0, 0,  0, 0],
            #                               [0, 0, 0, -1],
            #                               [0, 0,  0, 0]])
            # M = M - visual_correction
            T = sl.Transform()
            for j in range(4):
                for k in range(4):
                    T[j,k] = M[j,k]
            fus.pose = T
            zed_params.fusion.append(fus)
        else:
            zed_params.base_transform = M
     
    if len(zed_params.fusion) < 2:
        logger.error("Invalid config file.")
        exit(1)

    
    
    "Initialization parameters"
    zed_params.init = sl.InitParameters()
    zed_params.init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # RIGHT_HANDED_Y_UP, IMAGE
    zed_params.init.coordinate_units = sl.UNIT.METER
    zed_params.init.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # PERFORMANCE, NEURAL, ULTRA
    zed_params.init.camera_resolution = sl.RESOLUTION.HD720
    # zed_params.init.camera_fps = 30
    if user_params.video_src == 'SVO' and user_params.real_time == True:
        zed_params.init.svo_real_time_mode = True

    "Communication parameters"
    zed_params.communication = sl.CommunicationParameters()
    zed_params.communication.set_for_shared_memory()        # no clue what this does

    "Positional tracking parameters (Camera)"
    zed_params.positional_tracking = sl.PositionalTrackingParameters()
    zed_params.positional_tracking.set_as_static = True

    "Body tracking parameters"
    zed_params.body_tracking = sl.BodyTrackingParameters()
    zed_params.body_tracking.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST # FAST, MEDIUM, ACCURATE
    
    if user_params.body_type == 'BODY_34':
        zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_34
    if user_params.body_type == 'BODY_18':
        zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_18
    if user_params.body_type == 'BODY_38':
        zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_38

    zed_params.body_tracking.enable_body_fitting = True
    zed_params.body_tracking.enable_tracking = True

    zed_params.body_tracking_runtime = sl.BodyTrackingRuntimeParameters()
    # zed_params.body_tracking_runtime.detection_confidence_threshold = 0.85 #confidence threshold actually doesnt work
    # zed_params.body_tracking_runtime.detection_confidence_threshold = 100 # default value = 50
    # zed_params.body_tracking_runtime.minimum_keypoints_threshold = 12 # default value = 0

    "Fusion parameters"
    zed_params.fusion_init = sl.InitFusionParameters()
    zed_params.fusion_init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # RIGHT_HANDED_Y_UP, IMAGE
    zed_params.fusion_init.coordinate_units = sl.UNIT.METER
    zed_params.fusion_init.output_performance_metrics = False
    zed_params.fusion_init.verbose = True
    
    zed_params.body_tracking_fusion = sl.BodyTrackingFusionParameters()
    zed_params.body_tracking_fusion.enable_tracking = True
    zed_params.body_tracking_fusion.enable_body_fitting = True

    zed_params.body_tracking_fusion_runtime = sl.BodyTrackingFusionRuntimeParameters()
    zed_params.body_tracking_fusion_runtime.skeleton_minimum_allowed_keypoints = 7
    # zed_params.body_tracking_fusion_runtime.skeleton_minimum_allowed_camera = 1
    # zed_params.body_tracking_fusion_runtime.skeleton_smoothing = 0.5

    if user_params.body_type == 'BODY_18':
        zed_params.left_arm_keypoints = [5, 6, 7]
        zed_params.right_arm_keypoints = [2, 3, 4]
        zed_params.trunk_keypoints = [1, 0, 14, 16, 17, 15]
    if user_params.body_type == 'BODY_34':
        zed_params.left_arm_keypoints = [4, 5, 6, 7]
        zed_params.right_arm_keypoints = [11, 12, 13, 14]
        zed_params.trunk_keypoints = [2, 3, 26, 27, 30, 31, 28, 29]
    if user_params.body_type == 'BODY_38':
        zed_params.left_hand_keypoints = [30, 32, 34, 36]
        zed_params.right_hand_keypoints = [31, 33, 35 , 37]
 
    return zed_params

#############################################INITIALIZE OBJECTS AND FUNCTIONS###############################
def inverse_transform(R, t):
    T_inv = np.eye(4, 4)
    T_inv[:3, :3] = np.transpose(R)
    T_inv[:3, 3] = -1 * np.matmul(np.transpose(R), (t))# + np.array([1, 1, 0]).T
    return T_inv


def fetch_skeleton(bodies, user_params, zed_params, known_bodies):
    
    """
    For each detected body, save arm and trunk keypoint locations in a separate numpy array,
    all packaged as a dictionary value
    """
    
    if len(bodies.body_list) > 0:
        # TODO: Check that the keypoints collected are the right ones
        # TODO: Check that the keypoints are taken along the right axis
        # TODO: Check that the transforéation matrix is applied right
        
        left_kpt_idx = zed_params.left_arm_keypoints
        right_kpt_idx = zed_params.right_arm_keypoints
        trunk_kpt_idx = zed_params.trunk_keypoints
        
        for body in bodies.body_list:

            "https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1BodyData.html#acd55fe117eb554b1eddae3d5c23481f5"
            body_id = body.id
            body_unique_id = body.unique_object_id
            body_tracking_state = body.tracking_state # if the current body is being tracked
            body_action_state = body.action_state # idle or moving
            body_kpt = body.keypoint
            body_kpt_confidence = body.keypoint_confidence
            position = body.position

            # Only keep bodies close to the origin
            dist = np.linalg.norm(position)
            if dist < 3:

                ##################### Visual Check
                
                lsl = body.keypoint[[29, 28, 27, 26, 3, 2, 1, 0, 18, 19, 20, 21],:]
                rsl = body.keypoint[[31, 30, 27, 26, 3, 2, 1, 0, 22, 23, 24, 25],:]
                lss = body.keypoint[[2, 4, 5, 6, 7, 8, 9],:]
                rss = body.keypoint[[2, 11, 12, 13, 14, 15, 16],:]


                # lsl = np.hstack((lsl, np.ones((len(lsl), 1))))
                # rsl = np.hstack((rsl, np.ones((len(rsl), 1))))
                # lss = np.hstack((lss, np.ones((len(lss), 1))))
                # rss = np.hstack((rss, np.ones((len(rss), 1))))

                # R = zed_params.base_transform
                # # R = np.matmul(self.zed_params.cam_transform, self.user_params.pos_transform)
                # # R = self.zed_params.cam_transform
                # # Rp = inverse_transform(R[0:3, 0:3], R[0:3, 3])
                # # R = Rp
                
                # lsl = np.matmul(R, lsl.T).T[:, 0:3]
                # rsl = np.matmul(R, rsl.T).T[:, 0:3]
                # lss = np.matmul(R, lss.T).T[:, 0:3]
                # rss = np.matmul(R, rss.T).T[:, 0:3]


                # # visuals.quick_plot(lsl, rsl, lss, rss)
                

                #####################
                
                "select the keypoints we are interested in"
                left_matrix = np.array(body.keypoint[left_kpt_idx])
                right_matrix = np.array(body.keypoint[right_kpt_idx])
                trunk_matrix = np.array(body.keypoint[trunk_kpt_idx])
                
                    
                # "The transform to get from camera POV to world view"
                # R = zed_params.base_transform @ zed_params.cam_transform      

                # "Convert to 4D poses"
                # left_matrix = np.hstack((left_matrix, np.ones((len(left_matrix), 1))))
                # right_matrix = np.hstack((right_matrix, np.ones((len(right_matrix), 1))))
                # trunk_matrix = np.hstack((trunk_matrix, np.ones((len(trunk_matrix), 1))))

                # left_matrix = np.matmul(R, left_matrix.T).T[:, 0:3]
                # right_matrix = np.matmul(R, right_matrix.T).T[:, 0:3]
                # trunk_matrix = np.matmul(R, trunk_matrix.T).T[:, 0:3]

                


                known_bodies[body_id] = [left_matrix, right_matrix, trunk_matrix]

        return known_bodies
    

#############################################ZED API PROCESSES##############################################
class vision():
    
    def __init__(self, user_params, zed_params):
        self.logger = logging.getLogger(__name__)
        self.user_params = user_params
        self.zed_params = zed_params
        self.senders = {}
        self.network_senders = {}
        self.error = 0

        self.chk = [False, False]
        if self.user_params.display_video < 3:
            self.chk = [True, True]

    def connect_cams(self):
    
        for conf in self.zed_params.fusion:
            self.logger.info("Try to open ZED " + str(conf.serial_number))
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
                    self.logger.error("Error opening the camera", conf.serial_number, status)
                    del self.senders[conf.serial_number]
                    continue
                
        
                status = self.senders[conf.serial_number].enable_positional_tracking(self.zed_params.positional_tracking)
                if status != sl.ERROR_CODE.SUCCESS:
                    self.logger.error("Error enabling the positional tracking of camera", conf.serial_number)
                    del self.senders[conf.serial_number]
                    continue
        
                status = self.senders[conf.serial_number].enable_body_tracking(self.zed_params.body_tracking)
                if status != sl.ERROR_CODE.SUCCESS:
                    self.logger.error("Error enabling the body tracking of camera", conf.serial_number)
                    del self.senders[conf.serial_number]
                    continue
        
                self.senders[conf.serial_number].start_publishing(self.zed_params.communication)
        
            print("Camera", conf.serial_number, "is open")
            
        if len(self.senders) + len(self.network_senders) < 1:
            self.logger.error("Not enough cameras")
            exit(1)

    def init_fusion(self):
        
        self.logger.info("self.senders started, running the fusion...")
        fusion = sl.Fusion()
        
        status = fusion.init(self.zed_params.fusion_init)

        if status != sl.ERROR_CODE.SUCCESS:
            self.logger.error(status)
        else:    
            self.logger.info("Cameras in this configuration : ", len(self.zed_params.fusion))
        self.fusion = fusion
        
    
    def subscribe_to_cam_outputs(self):
                
        bodies = sl.Bodies()        
        for serial in self.senders:
            zed = self.senders[serial]
            status = zed.grab()
            if status == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies, self.zed_params.body_tracking_runtime)
            else:
                self.logger.error('Could not grab zed output during initialization')
                self.logger.error(status)
                
        camera_identifiers = []
        svo_image = [None] * len(self.zed_params.fusion)
        for i in range(0, len(self.zed_params.fusion)):
            conf = self.zed_params.fusion[i]
            uuid = sl.CameraIdentifier()
            uuid.serial_number = conf.serial_number
            self.logger.info("Subscribing to" + str(conf.serial_number) + str(conf.communication_parameters.comm_type))
    
            status = self.fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
            if status != sl.FUSION_ERROR_CODE.SUCCESS:
                self.logger.error("Unable to subscribe to", str(uuid.serial_number), status)
            else:
                camera_identifiers.append(uuid)
                svo_image[i] = sl.Mat()
                self.logger.info("Subscribed.")
    
        if len(camera_identifiers) <= 0:
            self.logger.error("No camera connected.")
            exit(1)
        
        self.bodies = bodies
        self.svo_image = svo_image
        self.camera_identifiers = camera_identifiers


    def init_body_tracking_and_viewer(self):
        
        self.fusion.enable_body_tracking(self.zed_params.body_tracking_fusion)
    
        if self.user_params.display_skeleton == True:
            self.viewer = gl.GLViewer()
            self.viewer.init()

        if self.user_params.display_video:
            bridge = CvBridge()
            # Get ZED camera information
            zed = self.senders[list(self.senders.keys())[0]]
            camera_info = zed.get_camera_information()

            # 2D viewer utilities
            self.display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                            min(camera_info.camera_configuration.resolution.height, 720))
            self.image_scale = [self.display_resolution.width / camera_info.camera_configuration.resolution.width
                , self.display_resolution.height / camera_info.camera_configuration.resolution.height]
    
        # Create ZED objects filled in the main loop
        single_bodies = [sl.Bodies]
        self.single_bodies = single_bodies


    def zed_loop(self):

        for idx, serial in enumerate(self.senders):
            zed = self.senders[serial]
            status = zed.grab()
            if status != sl.ERROR_CODE.SUCCESS:
                self.logger.error(status)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    self.error = 1
            else:
                status = zed.retrieve_bodies(self.bodies)
                if status != sl.ERROR_CODE.SUCCESS:
                    self.logger.error(status)
                    self.error = 1
                else:                         

                    # # If we want to display a video
                    # if (idx+1 == self.user_params.display_video) or (self.user_params.display_video == 3):
                    #     self.chk[idx] = self.fusion.retrieve_image(\
                    #         self.svo_image[idx], self.camera_identifiers[idx])\
                    #               == sl.FUSION_ERROR_CODE.SUCCESS
                    #     if self.chk == [True, True]:
                            
                    #         if self.svo_image[idx] != 0:
                    #             cv_viewer.render_2D(self.svo_image[idx].get_data(), self.image_scale, self.bodies.body_list, \
                    #                                 self.zed_params.body_tracking.enable_tracking,
                    #                                 self.zed_params.body_tracking.body_format)
                    #             cv2.imshow("View"+str(idx), self.svo_image[idx].get_data())
                    #             #print("confidence is ", detected_body_list[0].confidence)
                    #             # print("lenght of detecetd bodies is ", len(detected_body_list))



                    status = self.fusion.process()
                    if status != sl.FUSION_ERROR_CODE.SUCCESS:
                        self.logger.error(status)
                        # self.error = 1
                    else:
                        
                        
                        
                        
                        
                        # TODO: Likely error in fusion initialisation that screws with the camera fusion
                        # Retrieve detected objects
                        status = self.fusion.retrieve_bodies(self.bodies, self.zed_params.body_tracking_fusion_runtime)
                        if status != sl.FUSION_ERROR_CODE.SUCCESS:
                            self.logger.error(status)
                            self.error = 1
                        
                        if (self.user_params.display_skeleton == True):
                            if (self.viewer.is_available()):
                                self.viewer.update_bodies(self.bodies)
                            else:
                                self.logger.error('viewer unavailable')
                                self.error = 1

                        # If we want to display a video
                        if (idx+1 == self.user_params.display_video) or (self.user_params.display_video == 3):
                            self.chk[idx] = self.fusion.retrieve_image(\
                                self.svo_image[idx], self.camera_identifiers[idx])\
                                    == sl.FUSION_ERROR_CODE.SUCCESS
                            if self.chk == [True, True]:
                                
                                if self.svo_image[idx] != 0:
                                    cv_viewer.render_2D(self.svo_image[idx].get_data(), self.image_scale, self.bodies.body_list, \
                                                        self.zed_params.body_tracking.enable_tracking,
                                                        self.zed_params.body_tracking.body_format)
                                    img = cv2.imshow("View"+str(idx), self.svo_image[idx].get_data())
                                    for body in self.bodies:
                                        for ctpt = np.mean(body.keypoint_2d, axis=0)
                                    cv2.drawMarker(img, ctpt, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30)
                                    #print("confidence is ", detected_body_list[0].confidence)
                                    # print("lenght of detecetd bodies is ", len(detected_body_list))
                        
                    
        
        return status

    
    def close(self):

        if (self.user_params.display_skeleton == True) and (self.viewer.is_available()):
            self.viewer.exit()

        cv2.destroyAllWindows()
        for sender in self.senders:
            self.senders[sender].close()

#############################################ROS PROCESSES##################################################

class geometry_publisher(Node):


    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_vec = self.create_publisher(PoseArray, 'kpt_data', 10)
        self.body_parts = {
        'left' : 0,
        'right' : 1,
        'trunk' : 2,
        'stop' : -1
        }
        self.i = 0
        self.key = ''
        

    def callback(self, label, body_id, data):
        """"
        label:      [str]
        data:       2D array of dim [x, 3] of doubles
        body_id :   int
        """
        
        [i, j] = np.shape(data)
        if j!=3:
            self.get_logger().Warning('position vectors are not the right dimension! \n'
                                      + 'expected 3 dimensions, got' + str(j) )
            exit()
        
        # initialize message
        msg_header = Header()
        msg_vec = PoseArray()


        msg_header.frame_id = str(body_id) + '_' + label
        msg_header.stamp = self.get_clock().now().to_msg()
        
        msg_vec.header = msg_header
        # poses = set(i)
        for k in range(i):
            pose = Pose()
            point = Point()
            [point.x, point.y, point.z] = data[k].astype(float)
            pose.position = point
            msg_vec.poses.append(pose)
 
        self.get_logger().info(label)
        self.get_logger().info(str(data))
        self.publisher_vec.publish(msg_vec)
        self.key = cv2.pollKey()
        self.i += 1


    def publish_all_bodies(self, bodies):
        for body_id in list(bodies.keys()):
            body = bodies[body_id]
            self.callback('eft', body_id, body[0])
            self.callback('right', body_id, body[1])
            self.callback('trunk', body_id, body[2])
            self.callback('stop', '-1', np.array([[-1, -1, -1]]))


#############################################MAIN###########################################################


def main(args=None):
    
    known_bodies = {}
    user_params = init_user_params()
    zed_params = init_zed_params(user_params)

    rclpy.init(args=args)
    publisher = geometry_publisher()

    cam = vision(user_params, zed_params)

    cam.connect_cams()
    cam.init_fusion()
    cam.subscribe_to_cam_outputs()
    cam.init_body_tracking_and_viewer()
    
    end_flag = 0
    key = ''
    while not end_flag:
        
        cam.zed_loop()
        # fetch_skeleton(cam.bodies, user_params, zed_params, known_bodies)
        # publisher.publish_all_bodies(known_bodies)
        
        key = cv2.pollKey()
        if key == ord("q") or cam.error == 1:
            end_flag = 1

    cam.close()











































# def inverse_transform(R, t):
#     T_inv = np.eye(4, 4)
#     T_inv[:3, :3] = np.transpose(R)
#     T_inv[:3, 3] = -1 * np.matmul(np.transpose(R), (t))# + np.array([1, 1, 0]).T
#     return T_inv


# class MinimalPublisher(Node):


#     def __init__(self):
#         super().__init__('minimal_publisher')
#         self.publisher_vec = self.create_publisher(PoseArray, 'kpt_data', 10)
#         # self.publisher_vec = self.create_publisher(Quaternion, 'kpt_data', 10)
#         # self.publisher_text = self.create_publisher(String, 'kpt_label', 10)

#         self.body_parts = {
#         'left' : 0,
#         'right' : 1,
#         'trunk' : 2,
#         'stop' : -1
#         }
        
#         self.i = 0
#         self.key = ''
        

#     def timer_callback(self, label, body_id, data):
#         """"
#         label: [str]
#         data: 2D array of dim [x, 3] of doubles
#         """
        
#         [i, j] = np.shape(data)
#         if j!=3:
#             self.get_logger().Warning('position vectors are not the right dimension! \n'
#                                       + 'expected 3 dimensions, got' + str(j) )
#             exit()

#         # p = float(self.body_parts[label])
            
#         # str_msg = String()
#         # str_msg.data = label
#         # self.publisher_text.publish(str_msg)
        
#         msg_header = Header()
#         msg_header.frame_id = str(body_id) + label
#         msg_vec = PoseArray()
#         msg_vec.header = msg_header
#         # poses = set(i)
#         for k in range(i):
#             pose = Pose()
#             point = Point()
#             [point.x, point.y, point.z] = data[k].astype(float)
#             pose.position = point
#             # poses.add(pose)
#             msg_vec.poses.append(pose)
#             # msg_vec.poses.
#         # msg_vec.poses = poses

#         # msg = Quaternion()
#         # for k in range(i):
#         #     pos = data[k]
#         #     [msg.x, msg.y, msg.z] = pos.astype(float)
#         #     msg.w = p + 0.01*k
#         #     self.publisher_vec.publish(msg)
 
#         self.get_logger().info(label)
#         # self.get_logger().info(str(data[0]))
#         self.get_logger().info(str(data))
#         self.publisher_vec.publish(msg_vec)
#         self.key = cv2.pollKey()
#         self.i += 1

#     def timer_caller(self, data):
#         timer_period = 0.0001  # seconds
#         self.timer = self.create_timer(timer_period, self.timer_callback(data))
#         rclpy.spin_once(self)
        

# def publisher(args=None):
#     rclpy.init(args=args)

#     minimal_publisher = MinimalPublisher()

#     rclpy.spin(minimal_publisher)

#     # Destroy the node explicitly
#     # (optional - otherwise it will be done automatically
#     # when the garbage collector destroys the node object)
#     minimal_publisher.destroy_node()
#     rclpy.shutdown()

# class local_functions():
    
#     def __init__(self):
#         self.user_params = self.init_user_params()
#         self.zed_params = self.init_zed_params()
#         self.senders = {}
#         self.network_senders = {}
#         self.error = 0
        
#         # for conf in self.zed_params.fusion:
            
#         self.connect_cams()
#         print("self.senders started, running the fusion...")
#         self.fusion = self.init_fusion()
#         [self.bodies, self.svo_image, self.camera_identifiers] = self.subscribe_to_cam_outputs()

#         [self.bodies, self.single_bodies] = self.init_body_tracking_and_viewer()

#         self.chk = [False, False]
#         if self.user_params.display_video < 3:
#             self.chk = [True, True]

    # def init_user_params(self):
    #     # For now, all parameters are defined in this script
    #     class user_params(): pass

    #     user_params.from_sl_config_file = False
    #     user_params.config_pth = '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib_2.json'
    #     # '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib_2.json'
    #     # '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/zed_calib.json'
    #     # '/home/tom/Downloads/Rec_1/calib/info/extrinsics.txt'
    #     # '/usr/local/zed/tools/zed_calib.json'

    #     user_params.video_src = 'SVO'                                       # SVO, Live
    #     user_params.svo_pth = '/usr/local/zed/samples/recording/playback/multi camera/cpp/build/'
    #     user_params.svo_prefix = 'std_SN'  #clean_SN, std_SN
    #     user_params.svo_suffix = '_720p_30fps.svo'

    #     user_params.display_video = 1                                     # 0: none, 1: cam 1, 2: cam 2, 3: both cams
    #     user_params.display_skeleton = True
    #     user_params.return_hands = False
    #     user_params.time_loop = False

    #     user_params.body_type = 'BODY_34' # BODY_18, BODY_34, BODY_38, for some reason we are stuck with 34
    #     user_params.pos_transform = np.array([[-1, 0,  0, 0.358],
    #                                           [ 0, 1,  0, 0.03],
    #                                           [ 0, 0, -1, 0.006],
    #                                           [ 0, 0,  0, 1]])
    #     return user_params
        
        # return self.user_params
        
    # def init_zed_params(self):
    #     # For now, all parameters are defined in this script
    #     print("This sample display the fused body tracking of multiple cameras.")
    #     print("It needs a Localization file in input. Generate it with ZED 360.")
    #     print("The cameras can either be plugged to your devices, or already running on the local network.")
        
    #     class zed_params(): pass
        
    #     if self.user_params.config_pth[-4:] == 'json':
    #         if self.user_params.from_sl_config_file:
    #             zed_params.fusion = sl.read_fusion_configuration_file(self.user_params.config_pth,\
    #                                                                 sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER)
    #         else:
    #             zed_params.fusion =  [] # list of both fusionconfigs
    #             file = open(self.user_params.config_pth)
    #             configs = json.load(file)
    #             for config in configs:
    #                 fus = sl.FusionConfiguration()
    #                 fus.serial_number = int(config)

    #                 M = np.array(configs[config]['transform'])
    #                 # M[0:3, 0:3] = M[0:3, 0:3].T
    #                 # M[0:3, 3] = -np.matmul(M[0:3, 0:3], M[0:3, 3])
    #                 # N = np.array([[-1, 0,  0, 0.358],
    #                 #               [ 0, 1,  0, 0.03],
    #                 #               [ 0, 0, -1, 0.006],
    #                 #               [ 0, 0,  0, 1]])
    #                 # M = np.matmul(N, M)
    #                 # M[0:3, 0:3] = M[0:3, 0:3].T
    #                 # M[0:3, 3] = -np.matmul(M[0:3, 0:3], M[0:3, 3])
    #                 # M = M + np.array([[0, 0,  0, 1],
    #                 #                   [0, 0,  0, 1],
    #                 #                   [0, 0, 0, -2],
    #                 #                   [0, 0,  0, 0]])
                    

    #                 ######################
    #                 """
    #                 Assume Zed fusion sets 1 camera as the origin, rather than the world origin
    #                 This means you have to transform the coordinates of the second camera to what its
    #                 position is in the first camera's frame of reference
    #                 --> ZED PYTHON API : subscribe pyzed::sl::Fusion
    #                 https://www.stereolabs.com/docs/fusion/overview

    #                 If you want to apply a different cam position for the sake of getting openGL to look the right way:
    #                 - only add translation to the first cam!
    #                 - comment out the conversion to real world coordinates R
    #                 - disable "zed_params.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP" 
                    
    #                 """

    #                 if not np.any(zed_params.fusion):
    #                     if self.user_params.display_skeleton == True:
    #                         M = M + np.array([[0, 0,  0, 0],
    #                                   [0, 0,  0, 1],
    #                                   [0, 0, 0, -3],
    #                                   [0, 0,  0, 0]])
    #                     N = copy.copy(M)    # table coords --> Cam1 coords (Cam1 as seen from table)
    #                     P = inverse_transform(N[0:3, 0:3], N[0:3, 3])
    #                     M = np.eye(4)
    #                     zed_params.cam_transform = N
                        
                        
    #                     # R = np.array([[-1, 0,  0, 0.358],
    #                     #               [ 0, 1,  0, 0.03],
    #                     #               [ 0, 0, -1, 0.006],
    #                     #               [ 0, 0,  0, 1]])
    #                     # Rt = inverse_transform(R[0:3, 0:3], R[0:3, 3])
    #                     # Rp = np.array([[1, 0,  0, -0.358],
    #                     #               [ 0, 1,  0, -0.03],
    #                     #               [ 0, 0, 1, -0.006],
    #                     #               [ 0, 0,  0, 1]])
    #                     # M = np.matmul(R, M)
    #                 else:
    #                     Q = inverse_transform(M[0:3, 0:3], M[0:3, 3])
    #                     # M = np.matmul(P, M)
    #                     # M = np.matmul(Q, N)
    #                     M = P @ M
                        
    #                 #########################

                    
    #                 # visuals.plot_axes(M)
    #                 T = sl.Transform()
    #                 for j in range(4):
    #                     for k in range(4):
    #                         T[j,k] = M[j,k]
    #                 fus.pose = T
    #                 zed_params.fusion.append(fus)
    #     else:
    #         zed_params.fusion =  [] # list of both fusionconfigs
    #         objects = []
    #         file = open(self.user_params.config_pth,'rb')
    #         # print(file.read())
    #         ids = [32689769, 34783283]
    #         objects = (pickle.load(file))
    #         keysList = list(objects.keys())
    #         for i, key in enumerate(keysList):
    #             fus = sl.FusionConfiguration()
                
    #             R = objects[key][0][0]
    #             t = objects[key][0][1]
    #             # M = np.append(R, np.transpose([t]), axis=1)
    #             # M = np.append(M, [[0, 0, 0, 1]], axis=0) # + np.array([1, 1, 1]).T
    #             M = inverse_transform(R, t)

    #             T = sl.Transform()
    #             for j in range(4):
    #                 for k in range(4):
    #                     T[j,k] = M[j,k]
                
    #             fus.pose = T
    #             fus.serial_number = ids[i]
    #             zed_params.fusion.append(fus)
            
        
        
        
        
        
    #     if len(zed_params.fusion) <= 0:
    #         print("Invalid config file.")
    #         exit(1)
    
        
        
    #     # common parameters
    #     zed_params.init = sl.InitParameters()
    #     # zed_params.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #     zed_params.init.coordinate_units = sl.UNIT.METER
    #     zed_params.init.depth_mode = sl.DEPTH_MODE.NEURAL  # sl.DEPTH_MODE.PERFORMANCE, sl.DEPTH_MODE.NEURAL
    #     zed_params.init.camera_resolution = sl.RESOLUTION.HD720
    #     # zed_params.init.camera_fps = 30
    #     if self.user_params.video_src == 'SVO':
    #         zed_params.init.svo_real_time_mode = True
    
    #     zed_params.communication = sl.CommunicationParameters()
    #     zed_params.communication.set_for_shared_memory()
    
    #     zed_params.positional_tracking = sl.PositionalTrackingParameters()
    #     zed_params.positional_tracking.set_as_static = True
    
    #     zed_params.body_tracking = sl.BodyTrackingParameters()
    #     zed_params.body_tracking.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    #     if self.user_params.return_hands == True:
    #         zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_38
    #     else:
    #         if self.user_params.body_type == 'BODY_34':
    #             zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_34
    #         else:
    #             zed_params.body_tracking.body_format = sl.BODY_FORMAT.BODY_18
    #     zed_params.body_tracking.enable_body_fitting = True
    #     zed_params.body_tracking.enable_tracking = True

    #     zed_params.body_tracking_runtime = sl.BodyTrackingRuntimeParameters()
    #     # zed_params.body_tracking_runtime.detection_confidence_threshold = 0.85 #confidence threshold actually doesnt work
        
    #     zed_params.fusion_init = sl.InitFusionParameters()
    #     zed_params.fusion_init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #     zed_params.fusion_init.coordinate_units = sl.UNIT.METER
    #     zed_params.fusion_init.output_performance_metrics = False
    #     zed_params.fusion_init.verbose = True
        
    #     zed_params.body_tracking_fusion = sl.BodyTrackingFusionParameters()
    #     zed_params.body_tracking_fusion.enable_tracking = True
    #     zed_params.body_tracking_fusion.enable_body_fitting = True

    #     # rt = sl.BodyTrackingFusionRuntimeParameters()
    #     # rt.skeleton_minimum_allowed_keypoints = 7

    #     zed_params.body_tracking_fusion_runtime = sl.BodyTrackingFusionRuntimeParameters()
    #     zed_params.body_tracking_fusion_runtime.skeleton_minimum_allowed_keypoints = 7
    #     # zed_params.body_tracking_fusion_runtime.skeleton_minimum_allowed_camera = 1
    #     # zed_params.body_tracking_fusion_runtime.skeleton_smoothing = 0.5


    #     # only for body_18
    #     if self.user_params.body_type == 'BODY_18':
    #         zed_params.left_arm_keypoints = [5, 6, 7]
    #         zed_params.right_arm_keypoints = [2, 3, 4]
    #         zed_params.trunk_keypoints = [1, 0, 14, 16, 17, 15]
    #     if self.user_params.body_type == 'BODY_34':
    #         zed_params.left_arm_keypoints = [4, 5, 6, 7]
    #         zed_params.right_arm_keypoints = [11, 12, 13, 14]
    #         zed_params.trunk_keypoints = [2, 3, 26, 27, 30, 31, 28, 29]

    #     # only for body_38
    #     zed_params.left_hand_keypoints = [30, 32, 34, 36]
    #     zed_params.right_hand_keypoints = [31, 33, 35 , 37]

        
    #     return zed_params
    
    
    # def connect_cams(self):
    
    #     for conf in self.zed_params.fusion:
    #         print("Try to open ZED", conf.serial_number)
    #         self.zed_params.init.input = sl.InputType()
    #         # network cameras are already running, or so they should
    #         if conf.communication_parameters.comm_type == sl.COMM_TYPE.LOCAL_NETWORK:
    #             self.network_senders[conf.serial_number] = conf.serial_number
        
    #         # local camera needs to be run form here, in the same process than the fusion
    #         else:
    #             self.zed_params.init.input = conf.input_type
                
    #             self.senders[conf.serial_number] = sl.Camera()
        
    #             # self.zed_params.init.set_from_serial_number(conf.serial_number)
    #             if self.user_params.video_src == 'SVO':
    #                 self.zed_params.init.set_from_svo_file(self.user_params.svo_pth \
    #                           + self.user_params.svo_prefix + str(conf.serial_number)\
    #                               + self.user_params.svo_suffix)
    #             else:
    #                 self.zed_params.init.set_from_serial_number(conf.serial_number)
                    
    #             status = self.senders[conf.serial_number].open(self.zed_params.init)
                
    #             if status != sl.ERROR_CODE.SUCCESS:
    #                 print("Error opening the camera", conf.serial_number, status)
    #                 del self.senders[conf.serial_number]
    #                 continue
                
        
    #             status = self.senders[conf.serial_number].enable_positional_tracking(self.zed_params.positional_tracking)
    #             if status != sl.ERROR_CODE.SUCCESS:
    #                 print("Error enabling the positional tracking of camera", conf.serial_number)
    #                 del self.senders[conf.serial_number]
    #                 continue
        
    #             status = self.senders[conf.serial_number].enable_body_tracking(self.zed_params.body_tracking)
    #             if status != sl.ERROR_CODE.SUCCESS:
    #                 print("Error enabling the body tracking of camera", conf.serial_number)
    #                 del self.senders[conf.serial_number]
    #                 continue
        
    #             self.senders[conf.serial_number].start_publishing(self.zed_params.communication)
        
    #         print("Camera", conf.serial_number, "is open")
            
    #     if len(self.senders) + len(self.network_senders) < 1:
    #         print("Not enough cameras")
    #         exit(1)
        
        
    # def init_fusion(self):
        
    #     self.zed_params.communication = sl.CommunicationParameters()
    #     fusion = sl.Fusion()
        
    
    #     fusion.init(self.zed_params.fusion_init)
            
    #     print("Cameras in this configuration : ", len(self.zed_params.fusion))
    #     return fusion
        
        
    # def subscribe_to_cam_outputs(self):
                
    #     bodies = sl.Bodies()        
    #     for serial in self.senders:
    #         zed = self.senders[serial]
    #         if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #             zed.retrieve_bodies(bodies, self.zed_params.body_tracking_runtime)
                
    #     camera_identifiers = []
    #     svo_image = [None] * len(self.zed_params.fusion)
    #     for i in range(0, len(self.zed_params.fusion)):
    #         conf = self.zed_params.fusion[i]
    #         uuid = sl.CameraIdentifier()
    #         uuid.serial_number = conf.serial_number
    #         print("Subscribing to", conf.serial_number, conf.communication_parameters.comm_type)
    
    #         status = self.fusion.subscribe(uuid, conf.communication_parameters, conf.pose)
    #         if status != sl.FUSION_ERROR_CODE.SUCCESS:
    #             print("Unable to subscribe to", uuid.serial_number, status)
    #         else:
    #             camera_identifiers.append(uuid)
    #             svo_image[i] = sl.Mat()
    #             print("Subscribed.")
    
    #     if len(camera_identifiers) <= 0:
    #         print("No camera connected.")
    #         exit(1)
    #     return bodies, svo_image, camera_identifiers
    
    
    # def init_body_tracking_and_viewer(self):
        
    #     self.fusion.enable_body_tracking(self.zed_params.body_tracking_fusion)
    
    #     # rt = sl.BodyTrackingFusionRuntimeParameters()
    #     # rt.skeleton_minimum_allowed_keypoints = 7
    
    #     if self.user_params.display_skeleton == True:
    #         self.viewer = gl.GLViewer()
    #         self.viewer.init()
    
    #     # Create ZED objects filled in the main loop
    #     bodies = sl.Bodies()
    #     single_bodies = [sl.Bodies]
    #     self.known_bodies = {}
        
    #     return bodies, single_bodies
        
    
    # def zed_loop(self):
    #     self.left_pos_all = np.array([[0, 0, 0]])
    #     self.right_pos_all = np.array([[0, 0, 0]])
    #     self.trunk_pos_all = np.array([[0, 0, 0]])

    #     print('chk 1')
        

    #     # a = sl.Pose()
    #     # self.senders[32689769].get_position(a)
    #     # b = sl.Transform()
    #     # pos_data = a.pose_data(b)
        
    #     for idx, serial in enumerate(self.senders):
    #         zed = self.senders[serial]
    #         print('chk 2')
    #         grab_success = zed.grab()
    #         if grab_success == sl.ERROR_CODE.SUCCESS:
    #             print('chk 3')
    #             zed.retrieve_bodies(self.bodies)
                
    #             if (idx+1 == self.user_params.display_video) or (self.user_params.display_video == 3):
    #                 self.chk[idx] = self.fusion.retrieve_image(self.svo_image[idx], self.camera_identifiers[idx]) == sl.FUSION_ERROR_CODE.SUCCESS
    #                 if self.chk == [True, True]:
                        
    #                     if self.svo_image[idx] != 0:
    #                         cv2.imshow("View"+str(idx), self.svo_image[idx].get_data()) #dislay both images to cv2
    #                             # key = cv2.waitKey(1) 
    #         else:
    #             print('grab failed')
    #             self.error +=1
                            
        
    
    #     if self.fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            
    #         # TODO: Likely error in fusion initialisation that screws with the camera fusion
    #         # Retrieve detected objects
    #         self.fusion.retrieve_bodies(self.bodies, self.zed_params.body_tracking_fusion_runtime)
            
    #         self.fetch_skeleton()

            
            
    #         # for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
    #         # for cam in self.camera_identifiers:
    #         #     fusion.retrieveBodies(self.single_bodies, rt, cam); 
    #         if (self.user_params.display_skeleton == True):
    #             # print(str(self.viewer.is_available()))
    #             if (self.viewer.is_available()):
    #                 # if np.any(self.bodies.body_list):
    #                 #     bodies_to_view = self.bodies.body_list
    #                 #     for body in bodies_to_view:
    #                 #         body.keypoint = body.keypoint + np.array([-1, -1, 2])
    #                 self.viewer.update_bodies(self.bodies)
    #                 # self.viewer.update_bodies(bodies_to_view)

            

            
                
    # def fetch_skeleton(self):
        
    #     if len(self.bodies.body_list) > 0:
    #         # TODO: Check that the keypoints collected are the right ones
    #         # TODO: Check that the keypoints are taken along the right axis
    #         # TODO: Check that the transforéation matrix is applied right
    #         if self.user_params.return_hands == True:
    #             left_kpt_idx = self.zed_params.left_hand_keypoints
    #             right_kpt_idx = self.zed_params.right_hand_keypoints
    #         else:
    #             left_kpt_idx = self.zed_params.left_arm_keypoints
    #             right_kpt_idx = self.zed_params.right_arm_keypoints
    #         trunk_kpt_idx = self.zed_params.trunk_keypoints
            
    #         for body in self.bodies.body_list:

    #             # Only keep bodies close to the origin
    #             dist = np.linalg.norm(np.mean(body.keypoint[:], axis=0))
    #             if dist < 5:

    #                 # print('body id : ' + str(body.id))  
    #                 # print('body sz : ' + str(len(body.keypoint)))  
                    
    #                 # left_matrix = np.array(body.keypoint[left_kpt_idx[0]]).reshape([1, 3])
    #                 # right_matrix = np.array(body.keypoint[right_kpt_idx[0]]).reshape([1, 3])
    #                 # trunk_matrix = np.array(body.keypoint[trunk_kpt_idx[0]]).reshape([1, 3])

    #                 # for h in range(len(left_kpt_idx)-1):    # fetch coords of each kpt
    #                 #     i = left_kpt_idx[h+1]
    #                 #     j = right_kpt_idx[h+1]
    #                 #     keypoint_left = np.array(body.keypoint[i]).reshape([1, 3])
    #                 #     keypoint_right = np.array(body.keypoint[j]).reshape([1, 3])  # loops from 50 to 70 (69 is last)
    #                 #     left_matrix = np.vstack((left_matrix, keypoint_left))  # left hand
    #                 #     right_matrix = np.vstack((right_matrix, keypoint_right))  # left hand, but in a mirror
    #                 left_matrix = np.array(body.keypoint[left_kpt_idx])
    #                 # left_matrix = np.array(body.keypoint[:])
    #                 right_matrix = np.array(body.keypoint[right_kpt_idx])
                    
    #                 #####################
                    
    #                 # lsl = body.keypoint[[29, 28, 27, 26, 3, 2, 1, 0, 18, 19, 20, 21],:]
    #                 # rsl = body.keypoint[[31, 30, 27, 26, 3, 2, 1, 0, 22, 23, 24, 25],:]
    #                 # lss = body.keypoint[[2, 4, 5, 6, 7, 8, 9],:]
    #                 # rss = body.keypoint[[2, 11, 12, 13, 14, 15, 16],:]


    #                 # lsl = np.hstack((lsl, np.ones((len(lsl), 1))))
    #                 # rsl = np.hstack((rsl, np.ones((len(rsl), 1))))
    #                 # lss = np.hstack((lss, np.ones((len(lss), 1))))
    #                 # rss = np.hstack((rss, np.ones((len(rss), 1))))

    #                 # R = np.matmul(self.user_params.pos_transform, self.zed_params.cam_transform)
    #                 # # R = np.matmul(self.zed_params.cam_transform, self.user_params.pos_transform)
    #                 # # R = self.zed_params.cam_transform
    #                 # # Rp = inverse_transform(R[0:3, 0:3], R[0:3, 3])
    #                 # # R = Rp
                    
    #                 # lsl = np.matmul(R, lsl.T).T[:, 0:3]
    #                 # rsl = np.matmul(R, rsl.T).T[:, 0:3]
    #                 # lss = np.matmul(R, lss.T).T[:, 0:3]
    #                 # rss = np.matmul(R, rss.T).T[:, 0:3]


    #                 # # visuals.quick_plot(lsl, rsl, lss, rss)
                    

    #                 #####################
                    
    #                 if self.user_params.return_hands == False:
    #                     trunk_matrix = np.array(body.keypoint[trunk_kpt_idx])
    #                     # for h in range(len(trunk_kpt_idx)-1):
    #                     #     k = trunk_kpt_idx[h+1]
    #                     #     keypoint_trunk = np.array(body.keypoint[k]).reshape([1, 3])
    #                     #     trunk_matrix = np.vstack((trunk_matrix, keypoint_trunk))
                
    #                 if self.user_params.return_hands:   # if hands, only return the mean kpt position
    #                     left_pos = np.mean(left_matrix, axis=0)
    #                     right_pos = np.mean(right_matrix, axis=0)
    #                     trunk_pos = np.mean(trunk_matrix, axis=0)

    #                     if not np.any(np.isnan(left_pos)):
    #                         lhp = np.array([left_pos[:]])
    #                         if not np.any(self.left_pos_all):
    #                             self.left_pos_all = lhp
    #                         else:
    #                             self.left_pos_all = np.append(self.left_pos_all, lhp, axis=0)

    #                     if not np.any(np.isnan(right_pos)):
    #                         rhp = np.array([right_pos[:]])
    #                         if not np.any(self.right_pos_all):
    #                             self.right_pos_all = rhp
    #                         else:
    #                             self.right_pos_all = np.append(self.right_pos_all, rhp, axis=0)
    #                 else:
    #                     self.left_pos_all = left_matrix
    #                     self.right_pos_all = right_matrix
    #                     self.trunk_pos_all = trunk_matrix

    #                 R = np.matmul(self.user_params.pos_transform, self.zed_params.cam_transform)

    #                 left_matrix = np.hstack((left_matrix, np.ones((len(left_matrix), 1))))
    #                 right_matrix = np.hstack((right_matrix, np.ones((len(right_matrix), 1))))
    #                 trunk_matrix = np.hstack((trunk_matrix, np.ones((len(trunk_matrix), 1))))

    #                 left_matrix = np.matmul(R, left_matrix.T).T[:, 0:3]
    #                 right_matrix = np.matmul(R, right_matrix.T).T[:, 0:3]
    #                 trunk_matrix = np.matmul(R, trunk_matrix.T).T[:, 0:3]

    #                 body_id = body.id
    #                 if body_id in self.known_bodies:
    #                     self.known_bodies[body_id][0] = left_matrix
    #                     self.known_bodies[body_id][1] = right_matrix
    #                     self.known_bodies[body_id][2] = trunk_matrix
    #                 else:
    #                     self.known_bodies[body_id] = [left_matrix, right_matrix, trunk_matrix]

    #                 # if not np.any(np.isnan(trunk_pos)):
    #                 #     thp = np.array([trunk_pos[:]])
    #                 #     if not np.any(self.trunk_pos_all):
    #                 #         self.trunk_pos_all = thp
    #                 #     else:
    #                 #         self.trunk_pos_all = np.append(self.trunk_pos_all, thp, axis=0)
                
    #         if self.trunk_pos_all[0][0] < -3:
    #             print("this loop done")
              
                
    # def close(self):

    #     if (self.user_params.display_skeleton == True) and (self.viewer.is_available()):
    #         self.viewer.exit()

    #     cv2.destroyAllWindows()
    #     for sender in self.senders:
    #         self.senders[sender].close()


# def main(args=None):
    

#     rclpy.init(args=args)
#     publisher = MinimalPublisher()
#     cam = local_functions()
#     # key = ''
#     checkpoint = time.time()
#     # fig = 0
    
#     while (key != ord("q"))and (cam.error==0):

#         if cam.user_params.time_loop:
#             checkpoint_2 = time.time()
#             print(checkpoint_2-checkpoint)
#             checkpoint = checkpoint_2

#         cam.zed_loop()
#         # output_l, output_r, output_t = cam.left_pos_all, cam.right_pos_all, cam.trunk_pos_all
#         for body_id in cam.known_bodies:

#             [output_l, output_r, output_t] = cam.known_bodies[body_id]
#             publisher.timer_callback('left', body_id, output_l)
#             publisher.timer_callback('right', body_id, output_r)
#             if not cam.user_params.return_hands:
#                 publisher.timer_callback('trunk', body_id, output_t)
#             publisher.timer_callback('stop', '-1', np.array([[-1, -1, -1]]))

#         # if not (not cam.known_bodies):
#         #     disp_body = list(cam.known_bodies)[0]
#         #     ###############
#         #     class geom(): pass
#         #     geom.arm_pos = cam.known_bodies[disp_body][0]
#         #     geom.trunk_pos = cam.known_bodies[disp_body][1]
#         #     geom.robot_pos = cam.known_bodies[disp_body][2]
#         #     geom.arm_cp_idx = [0, 1]
#         #     geom.u = np.eye(2)
#         #     geom.trunk_cp_idx = [0, 1]
#         #     geom.v = np.eye(2)
#         #     geom.robot_cp_arm_idx = [0, 1]
#         #     geom.s = np.eye(2)
#         #     geom.robot_cp_trunk_idx = [0, 1]
#         #     geom.t = np.eye(2)

#         #     fig = visuals.plot_skeletons(fig, geom)

#         #     ###############

#         # key = cv2.pollKey()
#         # if (cam.user_params.display_skeleton == True):
#         #         if not (cam.viewer.is_available()):
#         #             key = ord("q")
            
#     cam.close()
#     publisher.destroy_node()
#     rclpy.shutdown()


if __name__ == "__main__":
    
    main()

    


