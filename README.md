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
- robot_pose.py : processes information about the robot and the detected bodies to reduce overhead later on. Outputs the keypoint positions in cartesian space as lists
- kpts_to_bbox.py : calculates repulsion direction, spring force, and oriented damping constants
- run_stats.py : must be launched separately, outputs stats on cycle speeds of different nodes by checking ROS messages
- motion_generator.py : Uses the user input client to request robot positions and trajectories in CIC
- Components for CV and OGL viewers
- visuals.py       : Contains basic python helper functions for graphing and visualization, can be called from other nodes such as kpts-to-bbox
- zed_calib_4 is the file used to contain the camera transformations

Generally, all complex scripts posess an __init__, init_user, etc. where important variables/parameters are defined and can be modified


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
Python version this was tested with:
Python 3.10.12

Pip list:
Package                              Version
------------------------------------ ----------------
ackermann-msgs                       2.0.2
action-msgs                          1.2.1
action-tutorials-interfaces          0.20.4
action-tutorials-py                  0.20.4
actionlib-msgs                       4.2.4
actuator-msgs                        0.0.1
addict                               2.4.0
ament-clang-format                   0.12.11
ament-cmake-test                     1.3.9
ament-copyright                      0.12.11
ament-cppcheck                       0.12.11
ament-cpplint                        0.12.11
ament-flake8                         0.12.11
ament-index-python                   1.4.0
ament-lint                           0.12.11
ament-lint-cmake                     0.12.11
ament-package                        0.14.0
ament-pep257                         0.12.11
ament-uncrustify                     0.12.11
ament-xmllint                        0.12.11
angles                               1.15.0
appdirs                              1.4.4
argcomplete                          1.8.1
asttokens                            2.4.1
attrs                                23.2.0
Babel                                2.8.0
bcrypt                               3.2.0
beautifulsoup4                       4.10.0
beniget                              0.4.1
blinker                              1.7.0
Brlapi                               0.8.3
Brotli                               1.0.9
builtin-interfaces                   1.2.1
catkin-pkg-modules                   1.0.0
certifi                              2020.6.20
chardet                              4.0.0
clarabel                             0.9.0
click                                8.1.7
colcon-argcomplete                   0.3.3
colcon-bash                          0.5.0
colcon-cd                            0.2.1
colcon-cmake                         0.2.28
colcon-common-extensions             0.3.0
colcon-core                          0.17.0
colcon-defaults                      0.2.8
colcon-devtools                      0.3.0
colcon-installed-package-information 0.2.1
colcon-library-path                  0.2.1
colcon-metadata                      0.2.5
colcon-notification                  0.3.0
colcon-output                        0.2.13
colcon-override-check                0.0.1
colcon-package-information           0.4.0
colcon-package-selection             0.2.10
colcon-parallel-executor             0.3.0
colcon-pkg-config                    0.1.0
colcon-powershell                    0.4.0
colcon-python-setup-py               0.2.8
colcon-recursive-crawl               0.2.3
colcon-ros                           0.5.0
colcon-test-result                   0.3.8
colcon-zsh                           0.5.0
colorama                             0.4.4
comm                                 0.2.1
command-not-found                    0.3
composition-interfaces               1.2.1
ConfigArgParse                       1.7
contourpy                            1.2.1
control-msgs                         4.6.0
controller-manager                   2.42.0
controller-manager-msgs              2.41.0
cov-core                             1.15.0
coverage                             6.2
cryptography                         3.4.8
cupshelpers                          1.0
cv-bridge                            3.2.1
cvxpy                                1.5.2
cycler                               0.12.1
Cython                               3.0.10
dash                                 2.16.0
dash-core-components                 2.0.0
dash-html-components                 2.0.0
dash-table                           5.0.0
dbus-python                          1.2.18
debugpy                              1.8.1
decorator                            4.4.2
defer                                1.0.6
demo-nodes-py                        0.20.4
diagnostic-msgs                      4.2.4
distlib                              0.3.4
distro                               1.7.0
distro-info                          1.1+ubuntu0.2
docutils                             0.17.1
domain-coordinator                   0.10.0
duplicity                            0.8.21
ecos                                 2.0.14
empy                                 3.3.4
example-interfaces                   0.9.3
examples-rclpy-executors             0.15.1
examples-rclpy-minimal-action-client 0.15.1
examples-rclpy-minimal-action-server 0.15.1
examples-rclpy-minimal-client        0.15.1
examples-rclpy-minimal-publisher     0.15.1
examples-rclpy-minimal-service       0.15.1
examples-rclpy-minimal-subscriber    0.15.1
exceptiongroup                       1.2.0
executing                            2.0.1
fasteners                            0.14.1
fastjsonschema                       2.19.1
filterpy                             1.4.5
flake8                               4.0.1
Flask                                3.0.2
fonttools                            4.53.0
franka_gripper                       0.1.9
franka_msgs                          0.1.9
freetype-py                          2.4.0
fs                                   2.4.12
future                               0.18.2
gast                                 0.5.2
generate-parameter-library-py        0.3.8
geometry-msgs                        4.2.4
html5lib                             1.1
httplib2                             0.20.2
idna                                 3.3
image-geometry                       3.2.1
imageio                              2.34.2
importlib-metadata                   4.6.4
indicator-cpufreq                    0.2.2
iniconfig                            1.1.1
interactive-markers                  2.3.2
ipython                              8.22.2
ipywidgets                           8.1.2
itsdangerous                         2.1.2
jedi                                 0.19.1
jeepney                              0.7.1
Jinja2                               3.1.3
joblib                               1.3.2
joint-state-publisher                2.4.0
joint-state-publisher-gui            2.4.0
jsonschema                           4.21.1
jsonschema-specifications            2023.12.1
jupyter_core                         5.7.1
jupyterlab_widgets                   3.0.10
keyboard                             0.13.5
keyring                              23.5.0
kiwisolver                           1.4.5
language-selector                    0.1
lark                                 1.1.1
laser-geometry                       2.4.0
launch                               1.0.6
launch-param-builder                 0.1.1
launch-ros                           0.19.7
launch-testing                       1.0.6
launch-testing-ros                   0.19.7
launch-xml                           1.0.6
launch-yaml                          1.0.6
launchpadlib                         1.10.16
lazr.restfulclient                   0.14.4
lazr.uri                             1.0.6
lazy_loader                          0.4
lifecycle-msgs                       1.2.1
lockfile                             0.12.2
logging-demo                         0.20.4
louis                                3.20.0
lxml                                 4.8.0
lz4                                  3.1.3+dfsg
macaroonbakery                       1.3.1
Mako                                 1.1.3
map-msgs                             2.1.0
mapping                              0.1.6
MarkupSafe                           2.1.5
matplotlib                           3.5.1
matplotlib-inline                    0.1.6
mccabe                               0.6.1
message-filters                      4.3.4
messages_fr3                         0.0.0
monotonic                            1.6
more-itertools                       8.10.0
moveit-configs-utils                 2.5.5
moveit-msgs                          2.2.1
mpi4py                               3.1.3
mpmath                               0.0.0
nav-msgs                             4.2.4
nbformat                             5.9.2
nest-asyncio                         1.6.0
netifaces                            0.11.0
networkx                             2.8.4
nose2                                0.9.2
notify2                              0.3
numpy                                1.26.4
oauthlib                             3.2.0
object-recognition-msgs              2.0.0
octomap-msgs                         2.0.0
olefile                              0.46
open3d                               0.18.0
opencv-python                        4.9.0.80
osqp                                 0.6.7.post1
osrf-pycommon                        2.0.2
packaging                            24.1
pandas                               2.2.2
paramiko                             2.9.3
parso                                0.8.3
pcl-msgs                             1.0.0
pendulum-msgs                        0.20.4
pexpect                              4.8.0
pillow                               10.2.0
pip                                  24.2
platformdirs                         4.2.0
plotly                               5.19.0
pluggy                               0.13.0
ply                                  3.11
prompt-toolkit                       3.0.43
protobuf                             3.12.4
psutil                               5.9.0
ptyprocess                           0.7.0
pure-eval                            0.2.2
py                                   1.10.0
pycairo                              1.20.1
pycodestyle                          2.8.0
pycollada                            0.6
pycups                               2.0.1
pydocstyle                           6.1.1
pydot                                1.4.2
pyflakes                             2.4.0
pyglet                               2.0.16
Pygments                             2.11.2
PyGObject                            3.42.1
pygraphviz                           1.7
PyJWT                                2.3.0
pymacaroons                          0.13.0
PyNaCl                               1.5.0
PyOpenGL                             3.1.0
pyparsing                            2.4.7
PyQt5                                5.15.6
PyQt5-sip                            12.9.1
PyQt6                                6.7.0
PyQt6-Qt6                            6.7.1
PyQt6-sip                            13.6.0
pyquaternion                         0.9.9
pyrender                             0.1.45
pyRFC3339                            1.1
pytest                               6.2.5
pytest-cov                           3.0.0
python-apt                           2.4.0+ubuntu3
python-dateutil                      2.9.0.post0
python-debian                        0.1.43+ubuntu1.1
python-qt-binding                    1.1.2
pythran                              0.10.0
pytz                                 2022.1
pyxdg                                0.27
PyYAML                               5.4.1
pyzed                                4.0
pyzedhub                             0.73
qdldl                                0.1.7.post4
qt-dotgraph                          2.2.3
qt-gui                               2.2.3
qt-gui-cpp                           2.2.3
qt-gui-py-common                     2.2.3
quality-of-service-demo-py           0.20.4
rcl-interfaces                       1.2.1
rclpy                                3.3.13
rcutils                              5.1.6
referencing                          0.33.0
reportlab                            3.6.8
requests                             2.25.1
resource-retriever                   3.1.2
retrying                             1.3.4
rmw-dds-common                       1.6.0
roman                                3.3
ros-gz-interfaces                    0.244.14
ros2action                           0.18.10
ros2bag                              0.15.11
ros2cli                              0.18.10
ros2component                        0.18.10
ros2controlcli                       2.41.0
ros2doctor                           0.18.10
ros2interface                        0.18.10
ros2launch                           0.19.7
ros2lifecycle                        0.18.10
ros2multicast                        0.18.10
ros2node                             0.18.10
ros2param                            0.18.10
ros2pkg                              0.18.10
ros2run                              0.18.10
ros2service                          0.18.10
ros2topic                            0.18.10
rosbag2-interfaces                   0.15.11
rosbag2-py                           0.15.11
rosdistro-modules                    0.9.1
rosgraph-msgs                        1.2.1
rosidl-adapter                       3.1.5
rosidl-cli                           3.1.5
rosidl-cmake                         3.1.5
rosidl-generator-c                   3.1.5
rosidl-generator-cpp                 3.1.5
rosidl-generator-py                  0.14.4
rosidl-parser                        3.1.5
rosidl-runtime-py                    0.9.3
rosidl-typesupport-c                 2.0.1
rosidl-typesupport-cpp               2.0.1
rosidl-typesupport-fastrtps-c        2.2.2
rosidl-typesupport-fastrtps-cpp      2.2.2
rosidl-typesupport-introspection-c   3.1.5
rosidl-typesupport-introspection-cpp 3.1.5
rospkg-modules                       1.5.1
rpds-py                              0.18.0
rpyutils                             0.2.1
rqt-action                           2.0.1
rqt-bag                              1.1.5
rqt-bag-plugins                      1.1.5
rqt-console                          2.0.3
rqt-graph                            1.3.1
rqt-gui                              1.1.7
rqt-gui-py                           1.1.7
rqt-msg                              1.2.0
rqt-plot                             1.1.2
rqt-publisher                        1.5.0
rqt-py-common                        1.1.7
rqt-py-console                       1.0.2
rqt-reconfigure                      1.1.2
rqt-service-caller                   1.0.5
rqt-shell                            1.0.2
rqt-srv                              1.0.3
rqt-topic                            1.5.0
scikit-image                         0.24.0
scikit-learn                         1.5.1
scipy                                1.14.0
screen-resolution-extra              0.0.0
scs                                  3.2.6
SecretStorage                        3.3.1
sensor-msgs                          4.2.4
sensor-msgs-py                       4.2.4
setuptools                           72.1.0
shape-msgs                           4.2.4
six                                  1.16.0
snowballstemmer                      2.2.0
soupsieve                            2.3.1
srdfdom                              2.0.4
sros2                                0.10.5
stack-data                           0.6.3
statistics-msgs                      1.2.1
std-msgs                             4.2.4
std-srvs                             4.2.4
stereo-msgs                          4.2.4
sympy                                1.9
systemd-python                       234
teleop-twist-keyboard                2.4.0
tenacity                             8.2.3
tf2-geometry-msgs                    0.25.7
tf2-kdl                              0.25.7
tf2-msgs                             0.25.7
tf2-py                               0.25.7
tf2-ros-py                           0.25.7
tf2-tools                            0.25.7
theora-image-transport               2.5.1
threadpoolctl                        3.3.0
tifffile                             2024.8.10
toml                                 0.10.2
topic-monitor                        0.20.4
tqdm                                 4.66.2
traitlets                            5.14.1
trajectory-msgs                      4.2.4
trash-cli                            0.24.5.26
trimesh                              4.4.3
turtlesim                            1.4.2
typeguard                            2.2.2
typing_extensions                    4.10.0
tzdata                               2024.1
ubuntu-drivers-common                0.0.0
ubuntu-pro-client                    8001
ufoLib2                              0.13.1
ufw                                  0.36.1
unattended-upgrades                  0.1
unicodedata2                         14.0.0
unique-identifier-msgs               2.2.1
urdf_parser_py                       0.0.4
urdfdom-py                           1.2.1
urdfpy                               0.0.22
urllib3                              1.26.5
usb-creator                          0.3.7
vision_based_robot_evasion           0.0.0
vision-msgs                          4.1.1
visualization-msgs                   4.2.4
wadllib                              1.3.6
wcwidth                              0.2.13
webencodings                         0.5.1
Werkzeug                             3.0.1
wheel                                0.43.0
widgetsnbextension                   4.0.10
xacro                                2.0.8
xdg                                  5
xkit                                 0.0.0
zipp                                 1.0.0



____________________________________