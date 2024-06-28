This project aims to use impedance based control to enforce evasion of human targets by a Franka Research 3 robot arm.
Perception of the environment, people, and objects is done through the use of two Zed2i stereo cameras.

Currently works on an Ubuntu 22.04 RT kernel, uses Franka Robotics and Stereolabs software and APIs, ROS2 Humble and CUDA.

Currently implemented: body detection (camera fusion needs a rework) and link-wise distance estimation between the operator and robot.

Relies on Software from Frankaemika, Stereolabs, and Curdin Deplazes : https://github.com/CurdinDeplazes/cartesian_impedance_control,
forked at https://github.com/lambdacitizen/cartesian_impedance_control.git

Contents:
- img_to_kpts.py  : communicates with cameras / svo files, outputs the keypoints of the bodies detected via ROS2
- kpts_to_bbox.py : listens via ROS2 to body position and robot state, determines distances and forces between both
- Components for CV and OGL viewers
- viewer.py       : Contains basic python functions for graphing and visualization

How to run:

- configure both python scripts to preferences
- activate img_to_kpts.py: __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 run my_cpp_py_pkg img_to_kpts.py
- activate Curdin's controller:
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 launch cartesian_impedance_control cartesian_impedance_controller.launch.py
- activate kpts_to_bbox.py: __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ros2 run my_cpp_py_pkg kpts_to_bbox.py

TODO:

/1st goal: Talk to 2 different cameras over ros

/2nd goal: combine body tracking
  int goals:
  - /  can I run this from ROS?
  - /  Output actual hand coordinates in ROS
  - (create config file to load ROS and local options)
  - /  (perform better calibration and data recordings)
  - (make code easier to port: put all files in the same folder as the code)
  - Perform 3rd party sensor fusion and qualified position detection

3rd goal: create bubble
  - /  calculate the distances and directions between he robot and human
  - /  vectorize these calculations to work on both entire bodies at once
  - /  determine where on the robot to exert pseudo-force
  - determine force application policy (repulsion, dodging)
  - Justify the safety of the strategy

4th goal: fetch robot positions
  - set up the docker environment for the RT kernel
  - use codin's controller to connect to the robot
  - read position values from the franka broadcasters
  - reconstruct accurate robot geometry and volume

5th goal: Calculate jacobians to translate to moments of force
  - check out frankaemika's jakobians
  - how to create new jacobian along a link
  - implement in python or C++?
  - hijack Curdin's code to generate the torques straight from the forces

6th goal: Perform robot control to apply the calculated force
  - Talk to Curdin's controller over ros2

7th goal : redo fusion so it actually runs instead of stuttering pathetically

8th goal: incorporate 3D models, or show that it is unnecessary
  - body fitting models which are accurate and SAFE
  - infer body proportions using learned estimators?

