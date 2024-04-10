This project aims to use impedance based control to enforce evasion of human targets by a Franka Research 3 robot arm.
Perception of the environment, people, and objects is done through the use of two Zed2i stereo cameras.

Currently works on an Ubuntu 22.04 RT kernel, uses Franka Robotics and Stereolabs software and APIs, ROS2 Humble and CUDA.

Currently implemented: body detection (camera fusion needs a rework) and link-wise distance estimation between the operator and robot.

