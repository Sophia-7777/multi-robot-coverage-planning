如果你以后已经熟悉了，不需要每次执行一堆检查命令。
编译：
cd ~/RESEARCH/multi_robot/ros_ws

conda activate bishe

source /opt/ros/noetic/setup.bash

catkin_make
source ~/RESEARCH/multi_robot/ros_ws/devel/setup.bash

运行：
rosrun tf static_transform_publisher \
0 0 0 0 0 0 \
world global_map 100

Terminal 1
conda activate bishe
source /opt/ros/noetic/setup.bash
source ~/RESEARCH/multi_robot/ros_ws/devel/setup.bash
export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/home/youcan7777/RESEARCH/multi_robot/ros_ws/devel/lib:/opt/ros/noetic/lib:/home/youcan7777/.mujoco/mujoco200_linux/bin
export TURTLEBOT3_MODEL=burger

roslaunch turtlebot3_gazebo turtlebot3_world.launch

roslaunch multi_robot_sim dual_turtlebot3.launch

roslaunch multi_robot_sim dual_turtlebot3_slam.launch

roslaunch multi_robot_sim dual_turtlebot3_slam+rviz.launch

roslaunch multi_robot_sim map_fusion.launch

roslaunch multi_robot_sim frontier_exploration.launch

rosrun multi_robot_sim frontier_evaluation_node.py

rosrun multi_robot_sim task_allocation_node.py

rosrun multi_robot_sim astar_planner_node.py

rosrun multi_robot_sim dwa_local_planner_node.py

rosrun multi_robot_sim final_path_controller.py

rosrun multi_robot_sim multi_robot_keyboard.py



rosrun multi_robot_sim frontier_detector.py

rosrun multi_robot_sim frontier_cluster_node.py

rosrun multi_robot_sim robot_frontier_cost_node.py

rosrun multi_robot_sim frontier_evaluation_node.py

rosrun multi_robot_sim task_allocation_node.py

rosrun tf view_frames（查看tf）


Terminal 2
conda activate bishe
source /opt/ros/noetic/setup.bash
source ~/RESEARCH/multi_robot/ros_ws/devel/setup.bash
export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/home/youcan7777/RESEARCH/multi_robot/ros_ws/devel/lib:/opt/ros/noetic/lib:/home/youcan7777/.mujoco/mujoco200_linux/bin
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping


Terminal 3
conda activate bishe
source /opt/ros/noetic/setup.bash
source ~/RESEARCH/multi_robot/ros_ws/devel/setup.bash
export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/home/youcan7777/RESEARCH/multi_robot/ros_ws/devel/lib:/opt/ros/noetic/lib:/home/youcan7777/.mujoco/mujoco200_linux/bin
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch



Terminal 4

需要检查的时候再开：

conda activate bishe
source /opt/ros/noetic/setup.bash
source ~/RESEARCH/multi_robot/ros_ws/devel/setup.bash
export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/home/youcan7777/RESEARCH/multi_robot/ros_ws/devel/lib:/opt/ros/noetic/lib:/home/youcan7777/.mujoco/mujoco200_linux/bin
export TURTLEBOT3_MODEL=burger

然后根据需要：

rostopic list

或者：

rostopic hz /scan

或者：

rostopic hz /odom

或者：

rostopic hz /map

或者：

rosrun tf tf_echo map base_footprint
