This project aims to use artificial potential fields and impedance based control to enforce evasion of human targets by a Franka Research 3 robot arm.
Perception of the environment, people, and objects is done through the use of two Zed2i stereo cameras.

Prerequisites:

Currently works on an Ubuntu 22.04 RT kernel, uses Franka Robotics and Stereolabs software and APIs, ROS2 Humble and CUDA 12.4. The version of CUDA really just depends on the ZED API and your setup.

Relies on Software from Frankaemika, Stereolabs (including python packages pyzed), and Curdin Deplazes : https://github.com/CurdinDeplazes/cartesian_impedance_control,
forked at https://github.com/lambdacitizen/cartesian_impedance_control.git

This project must be built in the same ROS2 workspace as https://github.com/CurdinDeplazes/cartesian_impedance_control and https://github.com/CurdinDeplazes/messages_fr3

Additional python environment modules:
pathlib
cv2
cv_bridge
logging
time
numpy
json
copy
urdf_parser_py
scipy
urdfpy

____________________

Project contents:
- img_to_kpts.py  : communicates with cameras / svo files, outputs the keypoints of the bodies detected via ROS2
- robot_pose.py : calculates the world frame positions of the robot joints, filters body info to obtain only relevant data
- kpts_to_bbox.py : calculates repulsion direction, spring force, and oriented spring constant
- run_stats.py : must be launched separately, outputs stats on cycle speeds of different nodes
- motion_generator.py : Uses the user input client to request robot positions and trajectories in CIC
- Components for CV and OGL viewers
- viewer.py       : Contains basic python functions for graphing and visualization, must be run separately or added to the launch file

____________________

Installation:
- Clone this repo in the same ROS 2 workspace as franka_ros2, cartesian_impedance_control and messages_fr3
- Adjust settings in img_to_kpts.py/init_user_params():
  - in calib.json, replace the matrices with the camera to world frame transitions in camera frame c_T_cw
  - if working off an SVO file rather than live camera feed, adjust the path to the file
  - do not bother with the visualization settings, I am fairly certain that I broke it
  - toggle the 'fusion' parameter to use the ZED fusion object rather than my workaround
- Adjust the settings in kpts_to_bbox/kpts_to_bbox.initialize_variables(self)
  - change publishing info within the publisher node itself to print out stats while the script runs
  - uncomment the block of code titled "Plotting the distances" to display the body and robot
    - the script called for visualization, visuals.py, can also be un-commented to also display interactions
    WARNING: This visualization significantly slows down the pipeline
- Modify whatever you want in the trajectory in motion_generator.py
- launch/my_script_launch.py: toggle the comments for robot_ip and use_fake_hardware parameters to run the system virtually

How to run:
- ros2 launch vision_based_robot_evasion my my_script_launch.py






____________________________________

TODO:

/1st goal: Talk to 2 different cameras over ros

/2nd goal: combine body tracking
  int goals:
  - /  can I run this from ROS?
  - /  Output actual hand coordinates in ROS
  - (create config file to load ROS and local options)
  - /  (perform better calibration and data recordings)
  - / (make code easier to port: put all files in the same folder as the code)
  - / Perform 3rd party sensor fusion and qualified position detection, that still runs even if only one cam sees me
  - / improve location accuracy
  - / get an adapter so that all devices can be connected at once
  - / debug img_to_kpts to figure out why the basis suddenly seems to change from time to time
  - / performance increase : limit imgs_to_kpts to 30 fps to free up cpu, since kpts_to_bbox is the bottleneck
  - / separate kpts_to_bbox into multiple nodes in order to enforce concurrent callbacks: https://robotics.stackexchange.com/questions/106026/ros2-multi-nodes-each-on-a-thread-in-same-process
  

3rd goal: create bubble
  - /  calculate the distances and directions between he robot and human
  - /  vectorize these calculations to work on both entire bodies at once
  - /  determine where on the robot to exert pseudo-force
  - / determine force application policy (repulsion, dodging)
  - / Justify the safety of the strategy
  - /  debug kpts to bbox to find out why it's so inefficient
  - /  output forces for every distance smaller than the bubble radius, to prevent bang bang       oscillation
  - / Add a multiplicative factor to the force in the case of few repulsion interactions in order to
    enable stronger repulsion of the hand without overreacting when the full body is near

4th goal: fetch robot positions
  - / use Curdin's controller to connect to the robot
  - / read position values from the franka broadcasters
  - / reconstruct accurate robot geometry and volume

5th goal: Calculate jacobians to translate to moments of force
  - / check out frankaemika's jakobians
  - / how to create new jacobian along a link
  - / implement in python or C++?
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
  - shift the wait for new kpts from the controller to the planner --> plan with new robot position info even if no new body data, for better tracking. But put a timeout so we forget very old bodies

For optimal performance:
Cameras high up
Not a black t-shirt!

presentation : 
Video with evasion for all joints, example with use case
benchmark code  and show times
How does it scale in complexity with the keypoint numbers
How does it compare with path planning
drawback and intuitive problems with the behaviour
