import launch
import launch_ros.actions
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from pathlib import Path



def generate_launch_description():

    ld = LaunchDescription([
        # Node(
        #     package='vision_based_robot_evasion',
        #     executable='img_to_kpts.py',
        #     name='img_to_kpts'
        # ),
        Node(
            package='vision_based_robot_evasion',
            executable='kpts_to_bbox.py',
            name='kpts_to_bbox',
        )
        
    ])

    return ld

if __name__ == '__main__':
    generate_launch_description()