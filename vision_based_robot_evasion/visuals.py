import numpy as np
import matplotlib.pyplot as plt
import copy


def plot_skeletons(fig, geom):
    
    """
    This function shows the position of the body and robot, and draws the lines
    connecting the closest points on the body an robot
    
    Input:----------------
    
    Takes fig: int or matplotlib figure object
        During initialization, feed an int to this function, and it will create
        the figure object on it's own
        For a call to update an existing figure, 'fig' must be the figure to be
        updated
        
    Takes geom, a passive class containing every geometry object to draw:
        
        - arm_pos : [n,3] array of keypoint positions of the arms
            (in physical order)
        - trunk_pos : [n,3] array of keypoint positions of the trunk
            (in physical order)
        - robot_pos : [n,3] array of keypoint positions of the robot
            (in physical order)
            
        - arm_cp_idx : integer, the lower index of the two arm keypoints whose
            segment is closest to the robot
        - u : Float, 0 < u < 1, starting from the arm keypoint with the lower
            index, the distance to go along the segment to get to the closest
            point to the robot, expressed as a fraction of total segment length
            
        - trunk_cp_idx : integer, the lower index of the two trunk keypoints
            whose segment is closest to the robot
        - v : Float, 0 < v < 1, starting from the trunk keypoint with the lower
            index, the distance to go along the segment to get to the closest
            point to the robot, expressed as a fraction of total segment length
            
        - robot_cp_arm_idx : integer, the lower index of the two robot
            keypoints whose segment is closest to the arm
        - s : Float, 0 < s < 1, starting from the robot keypoint with the lower
            index, the distance to go along the segment to get to the closest
            point to the arm, expressed as a fraction of total segment length
            
        - robot_cp_trunk_idx : integer, the lower index of the two robot
            keypoints whose segment is closest to the trunk
        - t : Float, 0 < t < 1, starting from the robot keypoint with the lower
            index, the distance to go along the segment to get to the closest
            point to the trunk, expressed as a fraction of total segment length
    """
    
    # unpack variables for better readability
    arm_pos, trunk_pos = geom.arm_pos, geom.trunk_pos
    robot_pos = geom.robot_pos
    # dist = geom.dist
    arm_cp_idx, u = geom.arm_cp_idx, geom.u
    trunk_cp_idx, v = geom.trunk_cp_idx, geom.v
    robot_cp_arm_idx, s = geom.robot_cp_arm_idx, geom.s
    robot_cp_trunk_idx, t = geom.robot_cp_trunk_idx, geom.t

    # print('arms: ')
    # for idx in arm_cp_idx:
    #     print(str(idx))
    # print('trunk: ')
    # for idx in trunk_cp_idx:
    #     print(str(idx))
    

    ################################/
    
    if type(u) == float:
        arm_cp = arm_pos[arm_cp_idx,:]*(1-u) + arm_pos[arm_cp_idx+1,:] * u
        trunk_cp = trunk_pos[trunk_cp_idx,:]*(1-v) + trunk_pos[trunk_cp_idx+1,:] * v
        robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*(1-s) + robot_pos[robot_cp_arm_idx+1,:] * s
        robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*(1-t) + robot_pos[robot_cp_trunk_idx+1,:] * t
        
        ra_pos = np.array([arm_cp, robot_cp_arm])
        rt_pos = np.array([trunk_cp, robot_cp_trunk])
    else:
        arm_cp = arm_pos[arm_cp_idx,:]*(1-u)[:, np.newaxis] + arm_pos[arm_cp_idx-1,:] * u[:, np.newaxis]
        trunk_cp = trunk_pos[trunk_cp_idx,:]*(1-v)[:, np.newaxis] + trunk_pos[trunk_cp_idx-1,:] * v[:, np.newaxis]
        robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*(1-s)[:, np.newaxis] + robot_pos[robot_cp_arm_idx-1,:] * s[:, np.newaxis]
        robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*(1-t)[:, np.newaxis] + robot_pos[robot_cp_trunk_idx-1,:] * t[:, np.newaxis]
        
        ra_pos = np.array([arm_cp, robot_cp_arm])
        rt_pos = np.array([trunk_cp, robot_cp_trunk])
    
    ################################\



    # Create the geometry linking human and robot
    # if type(arm_cp_idx) == int or type(arm_cp_idx) == float:
    #     draw_mode = 'simple'
    #     if type(u) == float:
    #         arm_cp = arm_pos[arm_cp_idx,:]*(1-u) + arm_pos[arm_cp_idx+1,:] * u
    #         trunk_cp = trunk_pos[trunk_cp_idx,:]*(1-v) + trunk_pos[trunk_cp_idx+1,:] * v
    #         robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*(1-s) + robot_pos[robot_cp_arm_idx+1,:] * s
    #         robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*(1-t) + robot_pos[robot_cp_trunk_idx+1,:] * t
            
    #         ra_pos = np.array([arm_cp, robot_cp_arm])
    #         rt_pos = np.array([trunk_cp, robot_cp_trunk])
    #     else:
    #         arm_cp = arm_pos[arm_cp_idx,:]*(1-u)[:, np.newaxis] + arm_pos[arm_cp_idx+1,:] * u[:, np.newaxis]
    #         trunk_cp = trunk_pos[trunk_cp_idx,:]*(1-v)[:, np.newaxis] + trunk_pos[trunk_cp_idx+1,:] * v[:, np.newaxis]
    #         robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*(1-s)[:, np.newaxis] + robot_pos[robot_cp_arm_idx+1,:] * s[:, np.newaxis]
    #         robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*(1-t)[:, np.newaxis] + robot_pos[robot_cp_trunk_idx+1,:] * t[:, np.newaxis]
            
    #         ra_pos = np.array([arm_cp, robot_cp_arm])
    #         rt_pos = np.array([trunk_cp, robot_cp_trunk])
    # else:
    #     draw_mode = 'hard'
    #     arm_cp_ids = copy.copy(arm_cp_idx)
    #     trunk_cp_ids = copy.copy(trunk_cp_idx)
    #     robot_cp_arm_ids = copy.copy(robot_cp_arm_idx)
    #     robot_cp_trunk_ids = copy.copy(robot_cp_trunk_idx)
    #     ties_list = []



    #     for i, arm_cp_idx in enumerate(arm_cp_ids):
    #         arm_cp_idx = int(arm_cp_idx)-1
    #         trunk_cp_idx = int(trunk_cp_ids[i])-1
    #         robot_cp_arm_idx = int(robot_cp_arm_ids[i])-1
    #         robot_cp_trunk_idx = int(robot_cp_trunk_ids[i])-1

    #         if type(u) == float:
    #             arm_cp = arm_pos[arm_cp_idx,:]*(1-u) + arm_pos[arm_cp_idx+1,:] * u
    #             trunk_cp = trunk_pos[trunk_cp_idx,:]*(1-v) + trunk_pos[trunk_cp_idx+1,:] * v
    #             robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*(1-s) + robot_pos[robot_cp_arm_idx+1,:] * s
    #             robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*(1-t) + robot_pos[robot_cp_trunk_idx+1,:] * t
                
    #             ra_pos = np.array([arm_cp, robot_cp_arm])
    #             rt_pos = np.array([trunk_cp, robot_cp_trunk])
    #         else:
    #             arm_cp = arm_pos[arm_cp_idx,:]*(1-u)[:, np.newaxis] + arm_pos[arm_cp_idx+1,:] * u[:, np.newaxis]
    #             trunk_cp = trunk_pos[trunk_cp_idx,:]*(1-v)[:, np.newaxis] + trunk_pos[trunk_cp_idx+1,:] * v[:, np.newaxis]
    #             robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*(1-s)[:, np.newaxis] + robot_pos[robot_cp_arm_idx+1,:] * s[:, np.newaxis]
    #             robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*(1-t)[:, np.newaxis] + robot_pos[robot_cp_trunk_idx+1,:] * t[:, np.newaxis]
                
    #             ra_pos = np.array([arm_cp, robot_cp_arm])
    #             rt_pos = np.array([trunk_cp, robot_cp_trunk])
    #         ties_list.append([arm_cp, trunk_cp, robot_cp_arm, robot_cp_trunk, ra_pos, rt_pos])

    # initialize or iterate figure
    plt.ion()
    
    if type(fig) == int:
        # create figure object
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        # ax.set_xlim(0, 4)
        # ax.set_ylim(0, 4)
        # ax.set_zlim(0, 4)
    else:
        # inherit and reset a figure object
        ax = fig.axes[0]
        for art in list(ax.lines):
            art.remove()
    
    # dummy_line_1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    # dummy_line_2 = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]])
    # ax.plot(dummy_line_1[:, 0], dummy_line_1[:, 1], dummy_line_1[:, 2], 'o-b')
    # ax.plot(dummy_line_2[:, 0], dummy_line_2[:, 1], dummy_line_2[:, 2], 'o-r')

    # plot the human and robot
        ax.plot(arm_pos[:, 0], arm_pos[:, 1], arm_pos[:, 2], 'o-b')
        ax.plot(trunk_pos[:, 0], trunk_pos[:, 1], trunk_pos[:, 2], 'o-g')
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 2], 'o-r')


    ################################/
    
    if type(u) == float:
        ax.plot(ra_pos[:, 0], ra_pos[:, 1], ra_pos[:, 2], '-k')
        ax.plot(rt_pos[:, 0], rt_pos[:, 1], rt_pos[:, 2], '-k')
    else:
        for i in range(np.shape(ra_pos)[1]):
            ax.plot(ra_pos[:, i, 0], ra_pos[:, i, 1], ra_pos[:, i, 2], '-k')
        for i in range(np.shape(rt_pos)[1]):
            ax.plot(rt_pos[:, i, 0], rt_pos[:, i, 1], rt_pos[:, i, 2], '-k')
    
    ################################\
    
    # # draw the lines from human to robot
    # if draw_mode == 'simple':
    #     if type(u) == float:
    #         ax.plot(ra_pos[:, 0], ra_pos[:, 1], ra_pos[:, 2], '-k')
    #         ax.plot(rt_pos[:, 0], rt_pos[:, 1], rt_pos[:, 2], '-k')
    #     else:
    #         for i in range(np.shape(ra_pos)[1]):
    #             ax.plot(ra_pos[:, i, 0], ra_pos[:, i, 1], ra_pos[:, i, 2], '-k')
    #         for i in range(np.shape(rt_pos)[1]):
    #             ax.plot(rt_pos[:, i, 0], rt_pos[:, i, 1], rt_pos[:, i, 2], '-k')
    # else:
    #     for item in ties_list:
    #         [arm_cp, trunk_cp, robot_cp_arm, robot_cp_trunk, ra_pos, rt_pos] = item
    #         if type(u) == float:
    #             ax.plot(ra_pos[:, 0], ra_pos[:, 1], ra_pos[:, 2], '-k')
    #             ax.plot(rt_pos[:, 0], rt_pos[:, 1], rt_pos[:, 2], '-k')
    #         else:
    #             for i in range(np.shape(ra_pos)[1]):
    #                 ax.plot(ra_pos[:, i, 0], ra_pos[:, i, 1], ra_pos[:, i, 2], '-k')
    #             for i in range(np.shape(rt_pos)[1]):
    #                 ax.plot(rt_pos[:, i, 0], rt_pos[:, i, 1], rt_pos[:, i, 2], '-k')


    set_axes_equal(ax)
    plt.pause(0.001)
    return fig


def plot_held(fig, geom):
    
    """
    This function shows the position of the body over time
    
    Input:----------------
    
    Takes fig: int or matplotlib figure object
        During initialization, feed an int to this function, and it will create
        the figure object on it's own
        For a call to update an existing figure, 'fig' must be the figure to be
        updated
        
    Takes geom, a dict containing every geometry object to draw:
        
        - XXX_pos : [n,3] array of keypoint positions of the arms
            (in physical order)
    """

    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'k'] 
    limbs = list(geom.keys())

    # initialize or iterate figure
    plt.ion()
    
    if type(fig) == int:
        # create figure object
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        plt.xlabel("X")
        plt.ylabel("Y")
        # ax.set_xlim(-1, 2)
        # ax.set_ylim(0, 2)
        # ax.set_zlim(-1, 2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 0)
        ax.set_zlim(-.5, .5)
    else:
        # inherit and reset a figure object
        ax = fig.axes[0]
        for art in list(ax.lines):
            art.remove()
    
    # plot the human and robot
    for i, limb in enumerate(limbs):
        pos = geom[limb]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-', color=colors[i])
    set_axes_equal(ax)
    plt.pause(0.001)
    return fig



def quick_plot(lsh, rsh, lsl, rsl):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(lsh[:,0], lsh[:,1], lsh[:,2], 'o-r')
    ax.plot(rsh[:,0], rsh[:,1], rsh[:,2], 'o-b')
    ax.plot(lsl[:,0], lsl[:,1], lsl[:,2], 'o-g')
    ax.plot(rsl[:,0], rsl[:,1], rsl[:,2], 'o-m')
    set_axes_equal(ax)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_axes(S, T):
    x = np.array([[0, 0, 0], [1, 0, 0]])
    y = np.array([[0, 0, 0], [0, 1, 0]])
    z = np.array([[0, 0, 0], [0, 0, 1]])


    xp = np.matmul(T[0:3, 0:3], x.T).T + T[0:3, 3]
    yp = np.matmul(T[0:3, 0:3], y.T).T + T[0:3, 3]
    zp = np.matmul(T[0:3, 0:3], z.T).T + T[0:3, 3]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:,0], x[:,1], x[:,2], 'o-r')
    ax.plot(y[:,0], y[:,1], y[:,2], 'o-g')
    ax.plot(z[:,0], z[:,1], z[:,2], 'o-b')
    ax.plot(xp[:,0], xp[:,1], xp[:,2], 'o-m')
    ax.plot(yp[:,0], yp[:,1], yp[:,2], 'o-y')
    ax.plot(zp[:,0], zp[:,1], zp[:,2], 'o-c')

    if np.any(S):
        # xps = np.matmul(S[0:3, 0:3], xp.T).T + S[0:3, 3]
        # yps = np.matmul(S[0:3, 0:3], yp.T).T + S[0:3, 3]
        # zps = np.matmul(S[0:3, 0:3], zp.T).T + S[0:3, 3]
        xps = np.matmul(S[0:3, 0:3], x.T).T + S[0:3, 3]
        yps = np.matmul(S[0:3, 0:3], y.T).T + S[0:3, 3]
        zps = np.matmul(S[0:3, 0:3], z.T).T + S[0:3, 3]
        ax.plot(xps[:,0], xps[:,1], xps[:,2], 'o-',color='maroon')
        ax.plot(yps[:,0], yps[:,1], yps[:,2], 'o-',color='darkgreen')
        ax.plot(zps[:,0], zps[:,1], zps[:,2], 'o-',color='navy')
    set_axes_equal(ax)
    plt.show()




def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    

def main():
    
    """
    Placeholder function to illustrate the workings of the visualizer
    
    """
    a = [[[0, 2, 1], [0, 2, 2]], [[0, 2, 2], [0, 2, 1]]]
    b = [[[0, 1, 1], [0, 3, 3]], [[0, 1, 1], [0, 3, 3]]]
    R = np.einsum('ijk, ijk->ij', a, b)
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    y  = x[2:]
    g = 2
    # fig = 0
    # class geom(): pass
    # geom.arm_pos = np.array([[-1., 0, 1.5], [-.2, 0., 1.5], [.2, 0., 1.5], [1., 0., 1.5]])
    # geom.trunk_pos = np.array([[0., 0, 1.2], [0., 0., 1.6], [0., 0., 1.8]])
    # geom.robot_pos = np.array([[0., 1., 0.], [0., 1., 1.], [1., 1., 1],
    #                            [1.5, 0.5, 1.]])
    # geom.arm_cp_idx = 1
    # geom.u = 0.2
    # geom.trunk_cp_idx = 0
    # geom.v = 0.7
    # geom.robot_cp_arm_idx = 2
    # geom.s = 0.5
    # geom.robot_cp_trunk_idx = 1
    # geom.t = 0.9
    
    # for i in range(2000):
    #     fig = plot_skeletons(fig, geom)
    #     geom.robot_pos = geom.robot_pos + np.array([[0., 0., 0.], [0., -.001, 0.],
    #                                                 [0., -.001, 0.01], [-0.001, -.001, 0.001]])
 

if __name__ == '__main__':
    main()
    
    """
    ['/home/tom/ros2_ws/src/my_cpp_py_pkg/install/my_cpp_py_pkg/lib/my_cpp_py_pkg',
    '/home/tom/ros2_ws/install/zed_topic_benchmark_interfaces/local/lib/python3.10/dist-packages',
    '/home/tom/ros2_ws/install/zed_interfaces/local/lib/python3.10/dist-packages',
    '/home/tom/ros2_ws/install/my_robot_controller/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/my_cpp_py_pkg/local/lib/python3.10/dist-packages',
    '/home/tom/ros2_ws/install/launch_testing_examples/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_pointcloud_publisher/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_minimal_subscriber/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_minimal_service/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_minimal_publisher/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_minimal_client/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_minimal_action_server/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_minimal_action_client/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_guard_conditions/lib/python3.10/site-packages',
    '/home/tom/ros2_ws/install/examples_rclpy_executors/lib/python3.10/site-packages',
    '/opt/ros/humble/lib/python3.10/site-packages',
    '/opt/ros/humble/local/lib/python3.10/dist-packages',
    '/ros2_ws/src/my_cpp_py_pkg',
    '/home/tom/ros2_ws/src/my_cpp_py_pkg/my_cpp_py_pkg/visualisations.py',
    '/home/tom/ros2_ws',
    '/usr/lib/python310.zip',
    '/usr/lib/python3.10',
    '/usr/lib/python3.10/lib-dynload',
    '/home/tom/.local/lib/python3.10/site-packages',
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/lib/python3/dist-packages']
    /home/tom/ros2_ws/install/my_cpp_py_pkg/local/lib/python3.10/dist-packages/my_cpp_py_pkg/__pycache__
    """
        
    