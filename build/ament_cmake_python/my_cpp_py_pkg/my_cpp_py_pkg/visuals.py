import numpy as np
import matplotlib.pyplot as plt


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
    arm_pos, trunk_pos, robot_pos = geom.arm_pos, geom.trunk_pos, geom.robot_pos
    # dist = geom.dist
    arm_cp_idx, u = geom.arm_cp_idx, geom.u
    trunk_cp_idx, v = geom.trunk_cp_idx, geom.v
    robot_cp_arm_idx, s = geom.robot_cp_arm_idx, geom.s
    robot_cp_trunk_idx, t = geom.robot_cp_trunk_idx, geom.t
    
    # Create the geometry linking human and robot
    arm_cp = arm_pos[arm_cp_idx,:]*u + arm_pos[arm_cp_idx+1,:] * (1-u)
    trunk_cp = trunk_pos[trunk_cp_idx,:]*v + trunk_pos[trunk_cp_idx+1,:] * (1-v)
    robot_cp_arm = robot_pos[robot_cp_arm_idx,:]*s + robot_pos[robot_cp_arm_idx+1,:] * (1-s)
    robot_cp_trunk = robot_pos[robot_cp_trunk_idx,:]*t + robot_pos[robot_cp_trunk_idx+1,:] * (1-t)
    
    ra_pos = np.array([arm_cp, robot_cp_arm])
    rt_pos = np.array([trunk_cp, robot_cp_trunk])
    
    # initialize or iterate figure
    plt.ion()
    
    if type(fig) == int:
        # create figure object
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
    else:
        # inherit and reset a figure object
        ax = fig.axes[0]
        [arms_line, trunk_line, robot_line, ra_line, rt_line] = ax.lines
        arms_line.remove()
        trunk_line.remove()
        robot_line.remove()
        ra_line.remove()
        rt_line.remove()
        
    # plot the human and robot
    arms_line, = ax.plot(arm_pos[:, 0], arm_pos[:, 1], arm_pos[:, 2], 'o-b')
    trunk_line, = ax.plot(trunk_pos[:, 0], trunk_pos[:, 1], trunk_pos[:, 2], 'o-g')
    robot_line, = ax.plot(robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 2], 'o-r')
    
    # draw the lines from human to robot
    ra_line = ax.plot(ra_pos[:, 0], ra_pos[:, 1], ra_pos[:, 2], '-k')
    rt_line = ax.plot(rt_pos[:, 0], rt_pos[:, 1], rt_pos[:, 2], '-k')
    
    plt.pause(0.0005)
    return fig


def main():
    
    """
    Placeholder function to illustrate the workings of the visualizer
    
    """
    
    fig = 0
    class geom(): pass
    geom.arm_pos = np.array([[-1., 0, 1.5], [-.2, 0., 1.5], [.2, 0., 1.5], [1., 0., 1.5]])
    geom.trunk_pos = np.array([[0., 0, 1.2], [0., 0., 1.6], [0., 0., 1.8]])
    geom.robot_pos = np.array([[0., 1., 0.], [0., 1., 1.], [1., 1., 1],
                               [1.5, 0.5, 1.]])
    geom.arm_cp_idx = 1
    geom.u = 0.2
    geom.trunk_cp_idx = 0
    geom.v = 0.7
    geom.robot_cp_arm_idx = 2
    geom.s = 0.5
    geom.robot_cp_trunk_idx = 1
    geom.t = 0.9
    
    for i in range(2000):
        fig = plot_skeletons(fig, geom)
        geom.robot_pos = geom.robot_pos + np.array([[0., 0., 0.], [0., -.001, 0.],
                                                    [0., -.001, 0.01], [-0.001, -.001, 0.001]])
 

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
        