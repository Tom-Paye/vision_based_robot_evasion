#include "rclcpp/rclcpp.hpp"
// #include "my_cpp_py_pkg/cpp_header.hpp"

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<rclcpp::Node>("my_node_name");
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
#include <franka/robot.h>
#include <franka/gripper.h>
#include <franka/robot_state.h>