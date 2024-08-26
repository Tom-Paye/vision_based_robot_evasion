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
  - Perform 3rd party sensor fusion and qualified position detection, that still runs even if only one cam sees me
  - / improve location accuracy
  - / get an adapter so that all devices can be connected at once
  - debug img_to_kpts to figure out why the basis suddenly seems to change from time to time
  - performance increase : limit imgs_to_kpts to 30 fps to free up cpu, since kpts_to_bbox is the bottleneck
  - separate kpts_to_bbox into multiple nodes in order to enforce concurrent callbacks: https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process
  

3rd goal: create bubble
  - /  calculate the distances and directions between he robot and human
  - /  vectorize these calculations to work on both entire bodies at once
  - /  determine where on the robot to exert pseudo-force
  - determine force application policy (repulsion, dodging)
  - Justify the safety of the strategy
  - /  debug kpts to bbox to find out why it's so inefficient
  - /  output forces for every distance smaller than the bubble radius, to prevent bang bang       oscillation
  - Add a multiplicative factor to the force in the case of few repulsion interactions in order to
    enable stronger repulsion of the hand without overreacting when the full body is near

4th goal: fetch robot positions
  - / use Curdin's controller to connect to the robot
  - / read position values from the franka broadcasters
  - / reconstruct accurate robot geometry and volume

5th goal: Calculate jacobians to translate to moments of force
  - / check out frankaemika's jakobians
  - / how to create new jacobian along a link
  - implement in python or C++?
  - / hijack Curdin's code to generate the torques straight from the forces
  - / match forces to joint torques more correctly: test individual directions by sending fake forces
  - / reduce impedance parameters to more easily see results
  - There is more to this task than simply throwing a force at the controller to untangle. The controller only applies the component of the force which actually aligns with the joint. So you need great forces in order to rotate a joint the right way for it to permit another joint to do the movement you want, but these great forces also instantly cause the enabled joint to violate velocity conditions
  - Recalculate joint forces intelligently to prioritize movement in the joints that enable the motion you want
  ==> create secondary task of orienting joints the right way
  - / add scaling so forces get smaller towards the EE, use the official franka scaling
  - / add damping term to the force generator
  -  /Presence seems to induce rotation of the EE, I think that is due to the accumulated moments created when truncating the forces array // fixed in a messy way, need to revisit
  Problem : forces calculated here only affect a single joint, typically the EE. But, inverse kinematics disproportionately affect joints closer to the EE, while those near the base barely move at all.
  This means some directions of movement will have much higher gain compared to others.
  Solution: Perform pseudo-inverse kinematics preprocessing to feed forces onto all joints of the robot
  - / disable broadcasting old forces
  - the base position the robot wants to be in is not defined in the cartesian way, but in joint space. this means that the robot tries to keep every angle the same indepentently. So if my controller happens to point in the right direction on the right joint, it can interfere constructively with the base command and cause a velocity violation despite the robot not really adopting a good pose
  - reduce distance at which the force starts being applied
  - /add damping term to the force generator
  - add better prioritization between goals and repulsion. Null space?
  - instead of cartesian repulsion forces, it may be more intelligent to apply rotational impedance forces on every joint instead
  - add a component in the null space of the goal (NOT REPULSION) to bring the robot away from singularities

6th goal: Perform robot control to apply the calculated force
  -  / Talk to Curdin's controller over ros2

7th goal : / redo fusion so it actually runs instead of stuttering pathetically

8th goal: incorporate 3D models, or show that it is unnecessary
  - body fitting models which are accurate and SAFE
  - infer body proportions using learned estimators?

9th goal: Make it run smoothly
  - / Look for bottlenecks in CPU, GPU, RAM and disable power saving policy
  - / Check for waiting behavior between the nodes, make sure they are completely independent

10th goal : Package for use by others
  - Centralize all runtime parameters for easier tuning
  - Write doc
  - homogenize spring forces with Curdin

11th goal : Smarter repulsion
  - vary repulsion distances according to position request and joint speed, to limit unnecessary movement and enable handoffs
  - add a force component returning the robot to its 'neutral' state, steering away from singular positions
  - kalman filter the input bodies to get accurate cartesian speeds and further refine the damper between body and robot

For optimal performance:
Cameras high up
Not a black t-shirt!
