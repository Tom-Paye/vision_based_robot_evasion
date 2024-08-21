#!/usr/bin/env python3

# import vision_based_robot_evasion.module_to_import
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 run vision_based_robot_evasion img_to_kpts.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from std_msgs.msg import String
from messages_fr3.msg import Array2d

import pyzed.sl as sl

from ament_index_python.packages import get_package_share_directory

import cv2
import vision_based_robot_evasion.cv_viewer.tracking_viewer as cv_viewer
from cv_bridge import CvBridge, CvBridgeError

import vision_based_robot_evasion.ogl_viewer.viewer as gl
import vision_based_robot_evasion.visuals as visuals

import logging
import time
import numpy as np
import json
import copy
import pathlib




#############################################INITIALIZE USER PARAMETERS#####################################
"""
For now, assume we want to output body_34 data, with a two keypoints for the hands
example of code : https://community.stereolabs.com/t/re-identifiaction-in-fused-camera/3071
"""

def init_user_params():
    # For now, all parameters are defined in this script
    class user_params(): pass

    user_params.from_sl_config_file = False
    user_params.config_name = 'zed_calib_4.json'

    # External params
    user_params.video_src = 'Live'                                       # SVO, Live
    user_params.svo_pth = '/home/tom/Downloads/'
    # '/usr/local/zed/samples/recording/playback/multi camera/cpp/build/'
    user_params.svo_prefix = 'std_SN'  #clean_SN, std_SN
    user_params.svo_suffix = '_720p_30fps.svo'

    # View params
    user_params.display_video = 0                                   # 0: none, 1: cam 1, 2: cam 2, 3: both cams
    user_params.display_skeleton = False    # DO NOT USE, OPENGL IS A CANCER WHICH SHOULD NEVER BE GIVEN RAM
    user_params.display_fused_limbs = 0
    user_params.time_loop = False

    # Script params
    user_params.body_type = 'BODY_18' # BODY_18, BODY_34, BODY_38, for some reason we are stuck with 34
    user_params.real_time = True
    user_params.fusion = False

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
    current_pth = pathlib.Path(__file__).parent.resolve()
    a = current_pth.name
    if a == 'scripts':
        a = current_pth.parent.name
        config_pth = current_pth.parent / pathlib.Path(a + '/' + user_params.config_name)
    else:
        b = current_pth.parents[3]
        config_pth = b / pathlib.Path('src/' + a + '/' + a + '/' + user_params.config_name)
    file = open(config_pth)
    configs = json.load(file)

    zed_params.base_transform = np.array(configs['base']['transform'])
    R = zed_params.base_transform
    zed_params.R = {}
    # zed_params.S = inverse_transform(zed_params.R[0:3, 0:3], zed_params.R[0:3, 3])
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
            
            
            "One camera, the first to be loaded, is defined as the world origin, its position will be (0, H ,0)
            with H being its height, which depends on the condition mentioned above."
            - https://www.stereolabs.com/docs/fusion/zed360
            ==> the world origin is actually at a distance of -H in z?
            
            "The file also contains the camera's world data. It is defined by its rotation (radians) and its
            translation (meters). The position should be defined in UNIT::METER and COORDINATE_SYSTEM::IMAGE,
            the API read/write functions will handle the metric and coordinate system changes."
            - https://www.stereolabs.com/docs/fusion/overview
            
            "The rotations are Euler Angles.
            In the file the vector is XYZ,
            For you matrix M you write
            M.setRotationVector(sl::float3(X,Y,Z));"
            - https://community.stereolabs.com/t/multi-camera-point-cloud-fusion-using-room-calibration-file/3640
            
            "# Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between
            the two frames"
            - https://www.stereolabs.com/docs/positional-tracking/coordinate-frames
            
            "pose	The World position of the camera, regarding the other camera of the setup"
            - https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Fusion.html#a91061e5f56803c445cdf9ea7c0cf51c6
            -- update_pose
            """

            # if not np.any(zed_params.fusion):   # Cam 1
            #     N = copy.copy(M)    # table coords --> Cam 1 coords (Cam 1 as seen from table)
            #     # [obj]world frame = N * [obj]cam frame
            #     P = inverse_transform(N[0:3, 0:3], N[0:3, 3])
            #     M = np.eye(4)
            #     zed_params.cam_transform = N
            # else:
            #     Q = inverse_transform(M[0:3, 0:3], M[0:3, 3])
            #     # M = P @ M @ N
            #     # M = N @ M @ P 
            #     # M = M @ P
            #     # M = P @ M
            #     M = N @ Q

            # flip fwd-right-down coord system to the right-down-dwd one used by zed
            # coord_flip = np.array([ [0, 0, 1, 0],\
            #                         [1, 0, 0, 0],\
            #                         [0, 1, 0, 0],\
            #                         [0, 0, 0, 1] ])
            # coord_flip = np.array([ [0, 1, 0, 0],\
            #                         [0, 0, 1, 0],\
            #                         [1, 0, 0, 0],\
            #                         [0, 0, 0, 1] ])
            # M = M @ coord_flip
            # M = inverse_transform(M[0:3, 0:3], M[0:3, 3])
            zed_params.R[fus.serial_number] = R @ M
            M = np.eye(4)

            T = sl.Transform()
            for j in range(4):
                for k in range(4):
                    T[j,k] = M[j,k]
            fus.pose = T
            zed_params.fusion.append(fus)
     
    if len(zed_params.fusion) < 2:
        logger.error("Invalid config file.")
        exit(1)

    # coord_flip = np.array([ [0, 1, 0, 0],\
    #                         [0, 0, -1, 0],\
    #                         [-1, 0, 0, 0],\
    #                         [0, 0, 0, 1] ])
    # zed_params.R = zed_params.R @ zed_params.cam_transform
    # zed_params.R = coord_flip @ zed_params.R
    # zed_params.R = np.eye(4)
    # zed_params.R = zed_params.cam_transform @ zed_params.S

    
    
    "Initialization parameters"
    zed_params.init = sl.InitParameters()
    zed_params.init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # RIGHT_HANDED_Y_UP, IMAGE
    zed_params.init.coordinate_units = sl.UNIT.METER
    zed_params.init.depth_mode = sl.DEPTH_MODE.NEURAL  # PERFORMANCE, NEURAL, ULTRA
    zed_params.init.camera_resolution = sl.RESOLUTION.HD720
    zed_params.init.camera_fps = 30

    if user_params.video_src == 'SVO' and user_params.real_time == True:
        zed_params.init.svo_real_time_mode = True

    "Communication parameters"
    zed_params.communication = sl.CommunicationParameters()
    # zed_params.communication.set_for_shared_memory()        # no clue what this does

    "Positional tracking parameters (Camera)"
    zed_params.positional_tracking = sl.PositionalTrackingParameters()
    zed_params.positional_tracking.set_as_static = True
    zed_params.positional_tracking.set_gravity_as_origin = False

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

    # zed_params.runtime = sl.RuntimeParameters()
    # zed_params.runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    zed_params.body_tracking_runtime = sl.BodyTrackingRuntimeParameters()
    zed_params.body_tracking_runtime.detection_confidence_threshold = 70 #confidence threshold actually works?
    # zed_params.body_tracking_runtime.detection_confidence_threshold = 100 # default value = 50
    # zed_params.body_tracking_runtime.minimum_keypoints_threshold = 12 # default value = 0
    


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


def fill_in(pos, conf):
    """
    replaces positions with confidence 0 by the closest position with nonzero confidence
    """

    cx = np.where(conf==0)[0]
    cx_ = np.where(conf!=0)[0]

    dcx = np.subtract.outer(cx_, cx)

    id0 = np.argmin(np.abs(dcx), axis=0)
    id1 = np.arange(len(cx))
    id_c = dcx[id0, id1]

    pos[cx,:] = pos[cx+id_c,:]
    
    return pos


def fetch_skeleton(bodies, user_params, zed_params, known_bodies, fig):
    
    """
    For each detected body, save arm and trunk keypoint locations in a separate numpy array,
    all packaged as a dictionary value

    This function assumes that if two bodies are sufficiently close, then they are detected by different cameras.
    This is not safe.
    """
    
    if len(bodies) > 0:
               
        left_kpt_idx = zed_params.left_arm_keypoints
        right_kpt_idx = zed_params.right_arm_keypoints
        trunk_kpt_idx = zed_params.trunk_keypoints
        
        positions = []
        cam_idx = []
        for idx, object in enumerate(bodies):
            cam_idx.append(object[0])
            body = object[1]

            "https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1BodyData.html#acd55fe117eb554b1eddae3d5c23481f5"
            body_id = body.id
            body_unique_id = body.unique_object_id
            body_tracking_state = body.tracking_state # if the current body is being tracked
            body_action_state = body.action_state # idle or moving
            body_kpt = body.keypoint
            body_kpt_confidence = body.keypoint_confidence
            position = body.position

            R = zed_params.R[object[0]]
            position = (R @ np.append(position, 1))[0:3]

            positions.append(position)

            ###### Testing by never using cam 24
            # if object[0] == 30635524:
            #     body.confidence = 0
            ####################################

        positions = np.array(positions)
        # positions = np.array([
        #     [0, 0, 0],
        #     [1, 1, 1],
        #     [5, 5, 5], 
        #     [5, 5, 6]
        # ])
        
        # # only keep bodies seen by both cameras with similar positions
        # if len(positions)>1:
        #     positions_cam0 = positions[cam_idx == 0]
        #     positions_cam1 = positions[cam_idx == 1]
        #     # positions_0_tiled = np.tile(positions_cam0, (len(positions_cam1,1)))
        #     # positions_1_tiled = np.tile(positions_cam1, (len(positions_cam0,1)))
        #     distances = np.zeros((len(positions_cam0), len(positions_cam1)))
        #     for i, elem_0 in enumerate(positions_cam0):
        #         for j, elem_1 in enumerate(positions_cam1):
        #             distances[i, j] = np.linalg.norm(elem_0 - elem_1)
        #     # # only keep points less than 10 cm apart
        #     # acceptable_positions_0, acceptable_positions_1 = np.argwhere(distances < .1)
        #     # only keep the positions which are closest together
        #     acceptable_positions_0, acceptable_positions_1 = np.argwhere( not (distances - np.min(distances)))
        #     for i in range(len(acceptable_positions_0)):
        #         pair.append([i , closest_ordered[i, 0]])

        # # distances_btw_cams = np.linalg.norm(positions_0_tiled-positions_1_tiled, axis=1)
        # # sure_positions = distances_btw_cams < 0.1   # keep bodies seen as less than 10 cm appart by both cams



        if len(positions)>1:
            distances = np.ones([len(positions), len(positions)-1])*1000
            closest_h = np.tile(np.arange(len(positions)), (len(positions)-1, 1)).T
            closest_v = np.tile(np.arange(len(positions)-1), (len(positions), 1))
            closest = (closest_h+closest_v+1)%len(positions)
            pair = []
            for rot in range(len(positions)-1):
                pos_rot = np.roll(positions, -rot-1, axis=0)
                distances[:,rot] = np.linalg.norm(positions - pos_rot, axis=1)
            idx = np.argsort(distances, axis=1)
            # idx = np.array([idx.flatten(), closest_h.flatten()]).T
            # closest_ordered = closest[idx] 
            closest_ordered = np.take_along_axis(closest, idx, axis=1)  
            for i in range(len(positions)):
                if closest_ordered[closest_ordered[i, 0], 0] == i:
                    if not np.any(np.array(pair).flatten()==i):
                        pair.append([i , closest_ordered[i, 0]])
            
            for objects in pair:
                body_0 = bodies[objects[0]][1]
                cam_0 = bodies[objects[0]][0]
                body_1 = bodies[objects[1]][1]
                cam_1 = bodies[objects[1]][0]

                

                ########################################
                # Move keypoints to the correct coordinate system

                bodies_list = []

                for idx, body in enumerate([body_0, body_1]):
                    left_matrix = np.array(body.keypoint[left_kpt_idx])
                    right_matrix = np.array(body.keypoint[right_kpt_idx])
                    trunk_matrix = np.array(body.keypoint[trunk_kpt_idx])

                    "The transform to get from camera POV to world view"
                    R = zed_params.R[bodies[objects[idx]][0]]  

                    "Convert to 4D poses"
                    left_matrix = np.hstack((left_matrix, np.ones((len(left_matrix), 1))))
                    right_matrix = np.hstack((right_matrix, np.ones((len(right_matrix), 1))))
                    trunk_matrix = np.hstack((trunk_matrix, np.ones((len(trunk_matrix), 1))))

                    left_matrix = np.nan_to_num((R @ left_matrix.T).T[:, 0:3])
                    right_matrix = np.nan_to_num((R @ right_matrix.T).T[:, 0:3])
                    trunk_matrix = np.nan_to_num((R @ trunk_matrix.T).T[:, 0:3])

                    confidence = np.nan_to_num(body.keypoint_confidence)

                    conf_l = np.tile(confidence[left_kpt_idx], (3, 1)).T
                    conf_r = np.tile(confidence[right_kpt_idx], (3, 1)).T
                    conf_t = np.tile(confidence[trunk_kpt_idx], (3, 1)).T

                    bodies_list.append([left_matrix, conf_l, right_matrix, conf_r, trunk_matrix, conf_t])

                l0, cl0, r0, cr0, t0, ct0 = bodies_list[0]
                l1, cl1, r1, cr1, t1, ct1 = bodies_list[1]
                ########################################


                left_matrix = np.nan_to_num((l0*cl0 + l1*cl1) / (cl0+cl1 + 0.000000001))
                right_matrix = np.nan_to_num((r0*cr0 + r1*cr1) / (cr0+cr1 + 0.000000001))
                trunk_matrix = np.nan_to_num((t0*ct0 + t1*ct1) / (ct0+ct1 + 0.000000001))

                cl = (cl0+cl1)[:,0]
                cr = (cr0+cr1)[:,0]
                ct = (ct0+ct1)[:,0]


                

                # "The transform to get from camera 0 POV to world view"
                # R = zed_params.R    

                # "Convert to 4D poses"
                # left_matrix = np.hstack((left_matrix, np.ones((len(left_matrix), 1))))
                # right_matrix = np.hstack((right_matrix, np.ones((len(right_matrix), 1))))
                # trunk_matrix = np.hstack((trunk_matrix, np.ones((len(trunk_matrix), 1))))

                # left_matrix = np.matmul(R, left_matrix.T).T[:, 0:3]
                # right_matrix = np.matmul(R, right_matrix.T).T[:, 0:3]
                # trunk_matrix = np.matmul(R, trunk_matrix.T).T[:, 0:3]
    
                if user_params.display_fused_limbs:
                        ##################### Visual Check

                        # if body_id == 1:
                        geom = {'lsl' : left_matrix,
                                'rsl' : right_matrix,
                                'lss' : trunk_matrix,
                                'rss' : trunk_matrix}
                        
                        fig[body_id] = visuals.plot_held(fig[body_id], geom)
                        

                        #####################

                # if confidence_0>confidence_1:
                #     cam_id = cam_0
                # else:
                #     cam_id = cam_1
                # logging.getLogger().info("Dominant cam is "+ str(cam_id))
                # logging.getLogger().info("2 cams used")
        else:
            body = bodies[0][1]

            left_matrix = np.nan_to_num(np.array(body.keypoint[left_kpt_idx]))
            right_matrix = np.nan_to_num(np.array(body.keypoint[right_kpt_idx]))
            trunk_matrix = np.nan_to_num(np.array(body.keypoint[trunk_kpt_idx]))

            "The transform to get from camera POV to world view"
            R = zed_params.R[bodies[0][0]]  

            "Convert to 4D poses"
            left_matrix = np.hstack((left_matrix, np.ones((len(left_matrix), 1))))
            right_matrix = np.hstack((right_matrix, np.ones((len(right_matrix), 1))))
            trunk_matrix = np.hstack((trunk_matrix, np.ones((len(trunk_matrix), 1))))

            left_matrix = (R @ left_matrix.T).T[:, 0:3]
            right_matrix = (R @ right_matrix.T).T[:, 0:3]
            trunk_matrix = (R @ trunk_matrix.T).T[:, 0:3]

            confidence = np.nan_to_num(body.keypoint_confidence)
            cl = confidence[left_kpt_idx]
            cr = confidence[right_kpt_idx]
            ct = confidence[trunk_kpt_idx]

        

        # if any limb on the body has no keypoints, do not add the body
        if not np.any(cl) or not np.any(cr) or not np.any(ct):
            return known_bodies, fig
        # if a joint is has zero confidence on both cams, place it on the nearest joint
        if np.any(0 == cl):
            left_matrix = fill_in(left_matrix, cl)
        if np.any(0 == cr):
            right_matrix = fill_in(right_matrix, cr)
        if np.any(0 == ct):
            trunk_matrix = fill_in(trunk_matrix, ct)

        known_bodies[body_id] = [left_matrix, right_matrix, trunk_matrix, time.time()]

            # cam_id = bodies[0][0]
            # logging.getLogger().info("Dominant cam is "+ str(cam_id))
            # logging.getLogger().info("1 cam used")


    if type(known_bodies) != dict:
        known_bodies = {}

    # for key in list(known_bodies.keys()):
    #     if np.any(np.isnan(known_bodies[key][0])):
    #         a=2
    return known_bodies, fig
    

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
        
            # local camera needs to be run form here, in the same process as the fusion
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

                
                # self.zed_params.positional_tracking.set_initial_world_transform(conf.pose)
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


    def startup_split(self):
        bodies = sl.Bodies()        
        for serial in self.senders:
            zed = self.senders[serial]
            status = zed.grab()
            if status == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_bodies(bodies, self.zed_params.body_tracking_runtime)
            else:
                self.logger.error('Could not grab zed output during initialization')
                self.logger.error(status)
  
        self.bodies = []

        if self.user_params.display_video:
            self.image = sl.Mat()
            bridge = CvBridge()
            # Get ZED camera information
            zed = self.senders[list(self.senders.keys())[0]]
            camera_info = zed.get_camera_information()

            # 2D viewer utilities
            self.display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                            min(camera_info.camera_configuration.resolution.height, 720))
            self.image_scale = [self.display_resolution.width / camera_info.camera_configuration.resolution.width
                , self.display_resolution.height / camera_info.camera_configuration.resolution.height]


    def zed_loop(self):

        all_bodies = []
        # cam_check = np.zeros(len(self.senders))     # a verification that the camera has returned output
        for idx, serial in enumerate(self.senders):
            bodies = sl.Bodies()
            zed = self.senders[serial]
            status = zed.grab()
            if status != sl.ERROR_CODE.SUCCESS:
                self.logger.error(status)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    self.error = 1
            else:
                status = zed.retrieve_bodies(bodies)
                if status != sl.ERROR_CODE.SUCCESS:
                    self.logger.error(status)
                    self.error = 1
                if self.user_params.display_video and idx == 0:
                    zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    if len(self.bodies)>0:
                        cv_viewer.render_2D(self.image.get_data(), self.image_scale, self.bodies.body_list, \
                                                        self.zed_params.body_tracking.enable_tracking,
                                                        self.zed_params.body_tracking.body_format)
                    # else:
                    #     cv_viewer.render_2D(self.image.get_data(), self.image_scale, \
                    #                                     self.zed_params.body_tracking.enable_tracking,
                    #                                     self.zed_params.body_tracking.body_format)
                    cv2.imshow("View"+str(idx), self.image.get_data())

            for body in bodies.body_list:
                position = body.position
                # only keep bodies with high confidence
                confidence = body.confidence



                # Only keep bodies close to the origin
                dist = np.linalg.norm(position)
                if dist < 2 and confidence > 70:
                    all_bodies.append([serial, body])
                    # cam_check[idx] = 1

        # Only keep bodies seen by both cams
        # if not np.any(cam_check == 0):
        self.bodies = all_bodies

                        
                    
        
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
        # self.publisher_vec = self.create_publisher(PoseArray, 'kpt_data', 10)
        self.publisher_vec = self.create_publisher(Array2d, 'kpt_data', 10)
        self.message_array = []
        self.body_parts = {
        'left' : 0.,
        'right' : 1.,
        'trunk' : 2.,
        'stop' : -1.
        }
        self.i = 0
        self.key = ''
        self.t0 = time.time()
        self.pub_counter = 0
        self.Dt = 0
        

    def assemble_message(self, label, body_id, data):
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

        body_tag = np.ones(i) * float(body_id)
        limb_tag = self.body_parts[label] * np.ones(i)
        tag = np.vstack((body_tag, limb_tag)).T
        partial_message = np.hstack((tag, data))


        # initialize message if it is empty
        if  not np.any(self.message_array):
            self.message_array = partial_message
        else:
            self.message_array = np.vstack((self.message_array, partial_message))


    def callback(self):
        """
        kpt_data:   [i, j+2 array] of doubles
        col 0 says which body the keypoint belongs to
        col 1 says which limb the keypoint belongs to
        col 2,3,4 are kpt coordinates
        We assume kpts are in logical order
        """

        kpts_flattened = self.message_array.flatten(order='C')
        kpt_message = Array2d()
        kpt_message.array = list(kpts_flattened.astype(float))
        [kpt_message.height, kpt_message.width]  = np.shape(self.message_array)
        self.publisher_vec.publish(kpt_message)


        
        
        # self.get_logger().info(str(self.message_array))

    def publishing_stats(self):

        t = time.time()
        dt = t - self.t0
        self.Dt = self.Dt + dt
        self.t0 = t
        self.pub_counter = self.pub_counter +1

        if dt>0.05:
            self.get_logger().info("kpts published after "+str(np.round(dt, 3))+" seconds")

        if self.Dt > 10:
            pub_freq = np.round(self.pub_counter / self.Dt, 3)
            self.get_logger().info("kpts published at "+str(pub_freq)+" Hz")
            self.Dt = 0
            self.pub_counter = 0


        
        
        
        



    def publish_all_bodies(self, bodies):
        for body_id in list(bodies.keys()):
            body = bodies[body_id]

            self.assemble_message('left', body_id, body[0])
            self.assemble_message('right', body_id, body[1])
            self.assemble_message('trunk', body_id, body[2])
            self.callback()
            self.message_array = []

            self.publishing_stats()

    # /TODO: fix no new data available
    # /TODO: Fix publisher


#############################################MAIN###########################################################

class img_to_kpts(Node):
    def __init__(self):
        super().__init__("img_to_kpts")
        self.get_logger().info("img_to_kpts started")

        known_bodies = {}
        fig = [0, 0, 0, 0, 0]
        user_params = init_user_params()
        zed_params = init_zed_params(user_params)

        publisher = geometry_publisher()

        cam = vision(user_params, zed_params)

        cam.connect_cams()
        
        cam.startup_split()
        
        end_flag = 0
        key = ''
        
        while not end_flag:
            # T0 = time.time()
            
            cam.zed_loop()
            fetch_skeleton(cam.bodies, user_params, zed_params, known_bodies, fig)
            # T1 = time.time() - T0
            # Remove old bodies which are no longer tracked
            deceased = []
            for body in known_bodies:
                if time.time() - known_bodies[body][3] >0.2:
                    deceased.append(body)
            for body in deceased:
                del known_bodies[body]

            if 0 != np.size(list(known_bodies.keys())):
                publisher.publish_all_bodies(known_bodies)
            # T2 = time.time() -T0 - T1
            key = cv2.pollKey()
            if key == ord("q") or cam.error == 1:
                end_flag = 1

            # self.get_logger().info("T1:")
            # self.get_logger().info(str(T1))
            # self.get_logger().info("T2:")
            # self.get_logger().info(str(T2))

        cam.close()

    


def main(args=None):

    rclpy.init(args=args)
    node = img_to_kpts()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    




if __name__ == "__main__":
    
    main()

    


