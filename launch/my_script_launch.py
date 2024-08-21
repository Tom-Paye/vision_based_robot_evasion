import launch
import launch_ros.actions
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from pathlib import Path
from launch.actions import TimerAction




def generate_launch_description():

    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)

    ld = LaunchDescription([
        
        Node(
            package='vision_based_robot_evasion',
            executable='robot_pose.py',
            name='robot_pose'
        ),
        Node(
            package='vision_based_robot_evasion',
            executable='img_to_kpts.py',
            name='img_to_kpts'
        ),
        Node(
            package='vision_based_robot_evasion',
            executable='kpts_to_bbox.py',
            name='kpts_to_bbox',
        ),

        ##################################
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value='192.168.1.200', #'192.168.1.200', dont-care
            # default_value='dont-care', #'192.168.1.200', dont-care
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            # default_value='true',   # false
            default_value='false',   # false
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='true',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector.'),
        TimerAction(period = 6., actions = [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([PathJoinSubstitution(
                    [FindPackageShare('cartesian_impedance_control'), 'launch', 'cartesian_impedance_controller.launch.py'])]),
                launch_arguments={robot_ip_parameter_name: robot_ip,
                                load_gripper_parameter_name: load_gripper,
                                use_fake_hardware_parameter_name: use_fake_hardware,
                                fake_sensor_commands_parameter_name: fake_sensor_commands,
                                use_rviz_parameter_name: use_rviz
                                }.items(),
            ),
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource([PathJoinSubstitution(
            #         [FindPackageShare('cartesian_impedance_control'), 'run', 'user_input_server'])]),
            # )
        ])
        ######################################
        
    ])

    return ld

if __name__ == '__main__':
    generate_launch_description()