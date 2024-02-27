import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='my_cpp_py_pkg',
            executable='py_node',
            name='py_node'),
  ])