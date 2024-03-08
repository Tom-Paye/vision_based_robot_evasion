import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='my_cpp_py_pkg',
            executable='img_to_kpts',
            name='img_to_kpts'),
        launch_ros.actions.Node(
            package='my_cpp_py_pkg',
            executable='kpts_to_bbox',
            name='kpts_to_bbox'),
  ])