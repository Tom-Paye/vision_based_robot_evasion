import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

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

    ld = launch.LaunchDescription([

        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value='192.168.1.200',
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector.'),

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

        launch_ros.actions.Node(
            package='my_cpp_py_pkg',
            executable='kpts_to_bbox',
            name='kpts_to_bbox'),


    ])
    
    # img_to_kpts = launch_ros.actions.Node(
    #     package='my_cpp_py_pkg',
    #     executable='img_to_kpts',
    #     name='img_to_kpts'),

    # robot_state = launch_ros.actions.Node(
    #     package='franka_robot_state_broadcaster',
    #     executable='robot_state',
    #     name='robot_state'),
    # kpts_to_bbox = launch_ros.actions.Node(
    #     package='my_cpp_py_pkg',
    #     executable='kpts_to_bbox',
    #     name='kpts_to_bbox'),
    # cpp_node = launch_ros.actions.Node(
    #     package='my_cpp_py_pkg',
    #     executable='kpts_to_bbox',
    #     name='cpp_node'),

    # IncludeLaunchDescription(
    #         PythonLaunchDescriptionSource([PathJoinSubstitution(
    #             [FindPackageShare('franka_bringup'), 'launch', 'franka.launch.py'])]),
    #         launch_arguments={robot_ip_parameter_name: robot_ip,
    #                           load_gripper_parameter_name: load_gripper,
    #                           use_fake_hardware_parameter_name: use_fake_hardware,
    #                           fake_sensor_commands_parameter_name: fake_sensor_commands,
    #                           use_rviz_parameter_name: use_rviz
    #                           }.items(),
    #     ),


    # ld.add_action(kpts_to_bbox)

    return ld
        
  