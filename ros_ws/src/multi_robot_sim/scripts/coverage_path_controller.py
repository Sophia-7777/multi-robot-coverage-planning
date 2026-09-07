#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coverage_path_controller.py

ROS1 Dual Robot Coverage Path Controller
Global Coverage Path + DWA Local Planner

======================================================================
系统结构
======================================================================

Robot1:

    /robot1_coverage_path
            |
            v
    Global BCD Coverage Path
            |
            v
      Lookahead Target
            |
            v
           DWA
       ↙    ↓     ↘
    Laser  Path   Robot2
       ↘    ↓     ↙
            |
            v
    /robot1/cmd_vel


Robot2:

    /robot2_coverage_path
            |
            v
    Global BCD Coverage Path
            |
            v
      Lookahead Target
            |
            v
           DWA
       ↙    ↓     ↘
    Laser  Path   Robot1
       ↘    ↓     ↙
            |
            v
    /robot2/cmd_vel


======================================================================
输入
======================================================================

    /robot1_coverage_path
    /robot2_coverage_path

    /gazebo/model_states

    /robot1/scan
    /robot2/scan


======================================================================
输出
======================================================================

    /robot1/cmd_vel
    /robot2/cmd_vel


======================================================================
DWA 结构
======================================================================

Global Coverage Path
        |
        v
Lookahead Target
        |
        v
Dynamic Window
        |
        +----------------------+
        |                      |
        v                      v
    LaserScan             Other Robot
        |                      |
        +----------+-----------+
                   |
                   v
             Collision Check
                   |
                   v
             Trajectory Score
                   |
        +----------+----------+
        |          |          |
        v          v          v
      Path      Heading     Speed
        |
        +----------+----------+
                   |
                   v
             Best (v, omega)
                   |
                   v
                cmd_vel


======================================================================
重要说明
======================================================================

1. coverage_radius 和 robot_radius 是两个不同概念。

   coverage_radius:
       传感器/覆盖区域半径。

   robot_radius:
       机器人实际碰撞半径。

2. 如果 coverage_radius = 0.5 m：

       不代表 robot_radius = 0.5 m。

3. DWA 的碰撞判断使用：

       robot_radius + obstacle_margin

4. 本程序默认：

       /robotX/scan

   的 LaserScan 坐标系与机器人 base_link 坐标系一致。

   如果你的 laser 相对于 base_link 存在明显的 TF 偏移，
   应进一步使用 TF 进行坐标转换。

5. DWA 不负责重新规划 BCD Cell。

   coverage_path.py:
       BCD + A* + Zig-Zag
       负责全局覆盖路径。

   coverage_path_controller.py:
       DWA
       负责局部速度和避障。

======================================================================
"""

import math

import numpy as np
import rospy

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan


class RobotPathTracker:

    def __init__(
        self,
        robot_id,
        robot_name,
        other_robot_name,
        path_topic,
        cmd_topic,
        scan_topic,
        model_states_topic
    ):

        self.robot_id = robot_id
        self.robot_name = robot_name
        self.other_robot_name = other_robot_name

        # ============================================================
        # Basic Parameters
        # ============================================================

        self.control_frequency = rospy.get_param(
            "~control_frequency",
            20.0
        )

        # ============================================================
        # Robot Velocity
        # ============================================================

        self.max_linear_speed = rospy.get_param(
            "~max_linear_speed",
            0.20
        )

        self.min_linear_speed = rospy.get_param(
            "~min_linear_speed",
            0.00
        )

        self.max_angular_speed = rospy.get_param(
            "~max_angular_speed",
            1.2
        )

        # ============================================================
        # DWA Acceleration
        # ============================================================

        # ------------------------------------------------------------
        # 注意：
        #
        # Dynamic Window 的速度变化应该使用控制周期：
        #
        #     control_dt = 1 / control_frequency
        #
        # 而不是 dwa_dt。
        # ------------------------------------------------------------

        self.max_linear_accel = rospy.get_param(
            "~max_linear_accel",
            1.0
        )

        self.max_angular_accel = rospy.get_param(
            "~max_angular_accel",
            2.5
        )

        # ============================================================
        # DWA Prediction
        # ============================================================

        self.dwa_dt = rospy.get_param(
            "~dwa_dt",
            0.10
        )

        self.dwa_predict_time = rospy.get_param(
            "~dwa_predict_time",
            1.2
        )

        self.linear_samples = rospy.get_param(
            "~linear_samples",
            7
        )

        self.angular_samples = rospy.get_param(
            "~angular_samples",
            15
        )

        # ============================================================
        # Robot Geometry
        # ============================================================

        # ------------------------------------------------------------
        # 覆盖半径
        #
        # 例如：
        #
        #     coverage_radius = 0.5 m
        #
        # 这里只是任务参数。
        # DWA 不直接把它作为机器人碰撞半径。
        # ------------------------------------------------------------

        self.coverage_radius = rospy.get_param(
            "~coverage_radius",
            0.5
        )

        # ------------------------------------------------------------
        # 机器人实际碰撞半径
        # ------------------------------------------------------------

        self.robot_radius = rospy.get_param(
            "~robot_radius",
            0.25
        )

        # ------------------------------------------------------------
        # 静态障碍物安全裕量
        # ------------------------------------------------------------

        self.obstacle_margin = rospy.get_param(
            "~obstacle_margin",
            0.10
        )

        # ------------------------------------------------------------
        # 两机器人额外安全距离
        # ------------------------------------------------------------

        self.robot_collision_margin = rospy.get_param(
            "~robot_collision_margin",
            0.15
        )

        # ============================================================
        # LaserScan
        # ============================================================

        self.scan_timeout = rospy.get_param(
            "~scan_timeout",
            0.5
        )

        # ============================================================
        # Global Path
        # ============================================================

        self.lookahead_distance = rospy.get_param(
            "~lookahead_distance",
            0.45
        )

        self.waypoint_tolerance = rospy.get_param(
            "~waypoint_tolerance",
            0.12
        )

        self.goal_tolerance = rospy.get_param(
            "~goal_tolerance",
            0.15
        )

        self.path_timeout = rospy.get_param(
            "~path_timeout",
            0.0
        )

        # ============================================================
        # DWA Score Weights
        # ============================================================

        # 路径贴合
        self.path_weight = rospy.get_param(
            "~path_weight",
            1.5
        )

        # 朝向
        self.heading_weight = rospy.get_param(
            "~heading_weight",
            1.0
        )

        # 速度
        self.velocity_weight = rospy.get_param(
            "~velocity_weight",
            1.5
        )

        # 静态障碍物
        self.obstacle_weight = rospy.get_param(
            "~obstacle_weight",
            1.0
        )

        # 另一台机器人
        self.robot_weight = rospy.get_param(
            "~robot_weight",
            2.0
        )

        # 沿全局路径前进
        self.progress_weight = rospy.get_param(
            "~progress_weight",
            2.0
        )

        # 角速度惩罚
        self.angular_penalty_weight = rospy.get_param(
            "~angular_penalty_weight",
            0.2
        )

        # ============================================================
        # Obstacle Clearance
        # ============================================================

        # ------------------------------------------------------------
        # 超过这个距离以后，不再额外奖励“离障碍物更远”。
        #
        # 这样可以避免：
        #
        #     障碍物距离 2m
        #
        # 时 DWA 仍然为了获得 clearance score 而降低速度。
        # ------------------------------------------------------------

        self.preferred_obstacle_distance = rospy.get_param(
            "~preferred_obstacle_distance",
            0.80
        )

        # ============================================================
        # Data
        # ============================================================

        self.path = None

        self.path_points = []

        # ------------------------------------------------------------
        # Current robot
        # ------------------------------------------------------------

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.robot_v = 0.0
        self.robot_w = 0.0

        # ------------------------------------------------------------
        # Other robot
        # ------------------------------------------------------------

        self.other_robot_x = None
        self.other_robot_y = None
        self.other_robot_yaw = None

        self.other_robot_v = 0.0
        self.other_robot_w = 0.0

        # ------------------------------------------------------------
        # LaserScan
        # ------------------------------------------------------------

        self.scan_msg = None

        self.obstacle_points_world = np.empty(
            (0, 2),
            dtype=np.float64
        )

        self.last_scan_time = None

        # ============================================================
        # State
        # ============================================================

        self.path_received = False
        self.pose_received = False
        self.scan_received = False

        self.current_target_index = 0

        self.finished = False

        self.last_path_time = None

        # ============================================================
        # Publisher
        # ============================================================

        self.cmd_pub = rospy.Publisher(
            cmd_topic,
            Twist,
            queue_size=1
        )

        # ============================================================
        # Subscribers
        # ============================================================

        self.path_sub = rospy.Subscriber(
            path_topic,
            Path,
            self.path_callback,
            queue_size=1
        )

        self.model_states_sub = rospy.Subscriber(
            model_states_topic,
            ModelStates,
            self.model_states_callback,
            queue_size=1
        )

        self.scan_sub = rospy.Subscriber(
            scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "Robot%d DWA Controller initialized.",
            self.robot_id
        )

        rospy.loginfo(
            "Robot name: %s",
            self.robot_name
        )

        rospy.loginfo(
            "Other robot: %s",
            self.other_robot_name
        )

        rospy.loginfo(
            "Coverage radius: %.3f m",
            self.coverage_radius
        )

        rospy.loginfo(
            "Robot radius: %.3f m",
            self.robot_radius
        )

        rospy.loginfo(
            "Obstacle margin: %.3f m",
            self.obstacle_margin
        )

        rospy.loginfo(
            "Max linear speed: %.3f m/s",
            self.max_linear_speed
        )

        rospy.loginfo(
            "Max angular speed: %.3f rad/s",
            self.max_angular_speed
        )

        rospy.loginfo(
            "Max linear acceleration: %.3f m/s^2",
            self.max_linear_accel
        )

        rospy.loginfo(
            "DWA dt: %.3f s",
            self.dwa_dt
        )

        rospy.loginfo(
            "DWA prediction: %.2f s",
            self.dwa_predict_time
        )

        rospy.loginfo(
            "Velocity weight: %.2f",
            self.velocity_weight
        )

        rospy.loginfo(
            "Progress weight: %.2f",
            self.progress_weight
        )

        rospy.loginfo(
            "Preferred obstacle distance: %.2f m",
            self.preferred_obstacle_distance
        )

        rospy.loginfo(
            "=========================================="
        )

    # =================================================================
    # Path callback
    # =================================================================

    def path_callback(self, msg):

        if len(msg.poses) == 0:

            rospy.logwarn(
                "Robot%d received empty Path.",
                self.robot_id
            )

            self.path = None
            self.path_points = []
            self.path_received = False

            self.stop()

            return

        self.path = msg

        self.path_points = []

        for pose in msg.poses:

            x = pose.pose.position.x
            y = pose.pose.position.y

            self.path_points.append(
                (x, y)
            )

        self.path_received = True

        self.finished = False

        self.current_target_index = 0

        self.last_path_time = rospy.Time.now()

        rospy.loginfo(
            "Robot%d received new Path: %d points.",
            self.robot_id,
            len(self.path_points)
        )

    # =================================================================
    # Gazebo Model States
    # =================================================================

    def model_states_callback(self, msg):

        # ============================================================
        # Current Robot
        # ============================================================

        if self.robot_name in msg.name:

            index = msg.name.index(
                self.robot_name
            )

            pose = msg.pose[index]

            self.robot_x = pose.position.x
            self.robot_y = pose.position.y

            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w

            self.robot_yaw = self.quaternion_to_yaw(
                qx,
                qy,
                qz,
                qw
            )

            # --------------------------------------------------------
            # Gazebo velocity
            # --------------------------------------------------------

            if index < len(msg.twist):

                self.robot_v = msg.twist[
                    index
                ].linear.x

                self.robot_w = msg.twist[
                    index
                ].angular.z

            # --------------------------------------------------------
            # DWA 不允许负向前速度
            # --------------------------------------------------------

            self.robot_v = max(
                0.0,
                self.robot_v
            )

            self.pose_received = True

        # ============================================================
        # Other Robot
        # ============================================================

        if self.other_robot_name in msg.name:

            other_index = msg.name.index(
                self.other_robot_name
            )

            other_pose = msg.pose[
                other_index
            ]

            self.other_robot_x = (
                other_pose.position.x
            )

            self.other_robot_y = (
                other_pose.position.y
            )

            qx = other_pose.orientation.x
            qy = other_pose.orientation.y
            qz = other_pose.orientation.z
            qw = other_pose.orientation.w

            self.other_robot_yaw = (
                self.quaternion_to_yaw(
                    qx,
                    qy,
                    qz,
                    qw
                )
            )

            if other_index < len(msg.twist):

                self.other_robot_v = (
                    msg.twist[
                        other_index
                    ].linear.x
                )

                self.other_robot_w = (
                    msg.twist[
                        other_index
                    ].angular.z
                )

            self.other_robot_v = max(
                0.0,
                self.other_robot_v
            )

    # =================================================================
    # LaserScan callback
    # =================================================================

    def scan_callback(self, msg):

        self.scan_msg = msg

        if not self.pose_received:

            return

        if self.robot_yaw is None:

            return

        # ============================================================
        # LaserScan -> World
        #
        # 默认：
        #
        # LaserScan frame == robot base frame
        #
        # 如果 laser 有 TF 偏移，需要改成 TF2 转换。
        # ============================================================

        points = []

        angle = msg.angle_min

        cos_yaw = math.cos(
            self.robot_yaw
        )

        sin_yaw = math.sin(
            self.robot_yaw
        )

        for distance in msg.ranges:

            # --------------------------------------------------------
            # Invalid
            # --------------------------------------------------------

            if not math.isfinite(distance):

                angle += msg.angle_increment

                continue

            if distance < msg.range_min:

                angle += msg.angle_increment

                continue

            if distance > msg.range_max:

                angle += msg.angle_increment

                continue

            # --------------------------------------------------------
            # Laser frame
            # --------------------------------------------------------

            local_x = (
                distance
                *
                math.cos(angle)
            )

            local_y = (
                distance
                *
                math.sin(angle)
            )

            # --------------------------------------------------------
            # Robot frame -> World
            # --------------------------------------------------------

            world_x = (
                self.robot_x
                +
                local_x * cos_yaw
                -
                local_y * sin_yaw
            )

            world_y = (
                self.robot_y
                +
                local_x * sin_yaw
                +
                local_y * cos_yaw
            )

            points.append(
                (
                    world_x,
                    world_y
                )
            )

            angle += msg.angle_increment

        if points:

            self.obstacle_points_world = np.asarray(
                points,
                dtype=np.float64
            )

        else:

            self.obstacle_points_world = np.empty(
                (0, 2),
                dtype=np.float64
            )

        self.scan_received = True

        self.last_scan_time = rospy.Time.now()

    # =================================================================
    # Check Scan freshness
    # =================================================================

    def scan_is_fresh(self):

        if not self.scan_received:

            return False

        if self.last_scan_time is None:

            return False

        age = (
            rospy.Time.now()
            -
            self.last_scan_time
        ).to_sec()

        return age <= self.scan_timeout

    # =================================================================
    # Quaternion -> yaw
    # =================================================================

    @staticmethod
    def quaternion_to_yaw(
        qx,
        qy,
        qz,
        qw
    ):

        sin_yaw = (
            2.0
            *
            (
                qw * qz
                +
                qx * qy
            )
        )

        cos_yaw = (
            1.0
            -
            2.0
            *
            (
                qy * qy
                +
                qz * qz
            )
        )

        return math.atan2(
            sin_yaw,
            cos_yaw
        )

    # =================================================================
    # Normalize angle
    # =================================================================

    @staticmethod
    def normalize_angle(angle):

        while angle > math.pi:

            angle -= 2.0 * math.pi

        while angle < -math.pi:

            angle += 2.0 * math.pi

        return angle

    # =================================================================
    # Distance
    # =================================================================

    @staticmethod
    def distance(
        x1,
        y1,
        x2,
        y2
    ):

        return math.sqrt(
            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )

    # =================================================================
    # Find nearest path index
    # =================================================================

    def find_nearest_path_index(self):

        if not self.path_points:

            return None

        min_distance = float("inf")

        nearest_index = self.current_target_index

        start_index = max(
            0,
            self.current_target_index - 10
        )

        for index in range(
            start_index,
            len(self.path_points)
        ):

            px, py = self.path_points[index]

            distance = self.distance(
                self.robot_x,
                self.robot_y,
                px,
                py
            )

            if distance < min_distance:

                min_distance = distance

                nearest_index = index

        return nearest_index

    # =================================================================
    # Find lookahead index
    # =================================================================

    def find_lookahead_index(
        self,
        nearest_index
    ):

        if nearest_index is None:

            return None

        for index in range(
            nearest_index,
            len(self.path_points)
        ):

            px, py = self.path_points[index]

            distance = self.distance(
                self.robot_x,
                self.robot_y,
                px,
                py
            )

            if distance >= self.lookahead_distance:

                return index

        return len(
            self.path_points
        ) - 1

    # =================================================================
    # Update target index
    # =================================================================

    def update_target_index(self):

        nearest_index = (
            self.find_nearest_path_index()
        )

        if nearest_index is None:

            return None

        lookahead_index = (
            self.find_lookahead_index(
                nearest_index
            )
        )

        if lookahead_index is None:

            return None

        self.current_target_index = max(
            self.current_target_index,
            lookahead_index
        )

        return self.current_target_index

    # =================================================================
    # Check final goal
    # =================================================================

    def check_goal(self):

        if not self.path_points:

            return False

        goal_x, goal_y = (
            self.path_points[-1]
        )

        distance = self.distance(
            self.robot_x,
            self.robot_y,
            goal_x,
            goal_y
        )

        if distance <= self.goal_tolerance:

            self.finished = True

            self.stop()

            rospy.loginfo(
                "=========================================="
            )

            rospy.loginfo(
                "Robot%d coverage path finished.",
                self.robot_id
            )

            rospy.loginfo(
                "Final distance: %.3f m",
                distance
            )

            rospy.loginfo(
                "=========================================="
            )

            return True

        return False

    # =================================================================
    # Dynamic Window
    # =================================================================

    def calculate_dynamic_window(
        self,
        current_v,
        current_w
    ):

        # ------------------------------------------------------------
        # 非常重要：
        #
        # Dynamic Window 是一个“控制周期内”的速度变化范围。
        #
        # 20 Hz:
        #
        #     control_dt = 0.05 s
        #
        # 不是 dwa_dt = 0.10 s。
        # ------------------------------------------------------------

        control_dt = (
            1.0
            /
            max(
                self.control_frequency,
                1e-6
            )
        )

        # ============================================================
        # Linear
        # ============================================================

        v_min = max(
            self.min_linear_speed,
            current_v
            -
            self.max_linear_accel
            *
            control_dt
        )

        v_max = min(
            self.max_linear_speed,
            current_v
            +
            self.max_linear_accel
            *
            control_dt
        )

        # ============================================================
        # Angular
        # ============================================================

        w_min = max(
            -self.max_angular_speed,
            current_w
            -
            self.max_angular_accel
            *
            control_dt
        )

        w_max = min(
            self.max_angular_speed,
            current_w
            +
            self.max_angular_accel
            *
            control_dt
        )

        return (
            v_min,
            v_max,
            w_min,
            w_max
        )

    # =================================================================
    # Predict trajectory
    # =================================================================

    def predict_trajectory(
        self,
        v,
        w
    ):

        trajectory = []

        x = self.robot_x
        y = self.robot_y
        yaw = self.robot_yaw

        elapsed_time = 0.0

        while elapsed_time <= self.dwa_predict_time:

            trajectory.append(
                (
                    x,
                    y,
                    yaw,
                    elapsed_time
                )
            )

            x += (
                v
                *
                math.cos(yaw)
                *
                self.dwa_dt
            )

            y += (
                v
                *
                math.sin(yaw)
                *
                self.dwa_dt
            )

            yaw += (
                w
                *
                self.dwa_dt
            )

            yaw = self.normalize_angle(
                yaw
            )

            elapsed_time += self.dwa_dt

        return trajectory

    # =================================================================
    # Collision with LaserScan obstacles
    # =================================================================

    def trajectory_collision_obstacles(
        self,
        trajectory
    ):

        if (
            self.obstacle_points_world
            is None
            or
            len(self.obstacle_points_world) == 0
        ):

            return False

        safe_radius = (
            self.robot_radius
            +
            self.obstacle_margin
        )

        safe_radius_sq = (
            safe_radius
            *
            safe_radius
        )

        obstacle_points = (
            self.obstacle_points_world
        )

        for state in trajectory:

            x = state[0]
            y = state[1]

            dx = (
                obstacle_points[:, 0]
                -
                x
            )

            dy = (
                obstacle_points[:, 1]
                -
                y
            )

            distance_sq = (
                dx * dx
                +
                dy * dy
            )

            if np.any(
                distance_sq
                <=
                safe_radius_sq
            ):

                return True

        return False

    # =================================================================
    # Predict other robot
    # =================================================================

    def predict_other_robot_position(
        self,
        time
    ):

        if (
            self.other_robot_x is None
            or
            self.other_robot_y is None
        ):

            return None

        # ------------------------------------------------------------
        # 简单恒定速度模型
        # ------------------------------------------------------------

        if self.other_robot_yaw is None:

            return (
                self.other_robot_x,
                self.other_robot_y
            )

        x = (
            self.other_robot_x
            +
            self.other_robot_v
            *
            math.cos(
                self.other_robot_yaw
            )
            *
            time
        )

        y = (
            self.other_robot_y
            +
            self.other_robot_v
            *
            math.sin(
                self.other_robot_yaw
            )
            *
            time
        )

        return (
            x,
            y
        )

    # =================================================================
    # Collision with other robot
    # =================================================================

    def trajectory_collision_robot(
        self,
        trajectory
    ):

        if (
            self.other_robot_x is None
            or
            self.other_robot_y is None
        ):

            return False

        safe_distance = (
            self.robot_radius
            +
            self.robot_collision_margin
            +
            self.robot_radius
        )

        safe_distance_sq = (
            safe_distance
            *
            safe_distance
        )

        for state in trajectory:

            x = state[0]
            y = state[1]
            time = state[3]

            other_position = (
                self.predict_other_robot_position(
                    time
                )
            )

            if other_position is None:

                continue

            other_x, other_y = (
                other_position
            )

            dx = (
                x
                -
                other_x
            )

            dy = (
                y
                -
                other_y
            )

            distance_sq = (
                dx * dx
                +
                dy * dy
            )

            if distance_sq <= safe_distance_sq:

                return True

        return False

    # =================================================================
    # Minimum static obstacle distance
    # =================================================================

    def minimum_obstacle_distance(
        self,
        trajectory
    ):

        min_distance = float("inf")

        if (
            self.obstacle_points_world
            is not None
            and
            len(self.obstacle_points_world) > 0
        ):

            points = (
                self.obstacle_points_world
            )

            for state in trajectory:

                x = state[0]
                y = state[1]

                dx = (
                    points[:, 0]
                    -
                    x
                )

                dy = (
                    points[:, 1]
                    -
                    y
                )

                distance_sq = (
                    dx * dx
                    +
                    dy * dy
                )

                local_min = math.sqrt(
                    float(
                        np.min(
                            distance_sq
                        )
                    )
                )

                if local_min < min_distance:

                    min_distance = local_min

        return min_distance

    # =================================================================
    # Minimum other robot distance
    # =================================================================

    def minimum_robot_distance(
        self,
        trajectory
    ):

        min_distance = float("inf")

        if (
            self.other_robot_x is None
            or
            self.other_robot_y is None
        ):

            return min_distance

        for state in trajectory:

            x = state[0]
            y = state[1]
            time = state[3]

            other_position = (
                self.predict_other_robot_position(
                    time
                )
            )

            if other_position is None:

                continue

            other_x, other_y = (
                other_position
            )

            distance = self.distance(
                x,
                y,
                other_x,
                other_y
            )

            if distance < min_distance:

                min_distance = distance

        return min_distance

    # =================================================================
    # Distance to global path
    # =================================================================

    def distance_to_global_path(
        self,
        x,
        y
    ):

        if not self.path_points:

            return float("inf")

        min_distance = float("inf")

        start = max(
            0,
            self.current_target_index - 10
        )

        end = min(
            len(self.path_points),
            self.current_target_index + 80
        )

        for index in range(
            start,
            end
        ):

            px, py = self.path_points[index]

            distance = self.distance(
                x,
                y,
                px,
                py
            )

            if distance < min_distance:

                min_distance = distance

        return min_distance

    # =================================================================
    # Get path progress
    # =================================================================

    def get_path_progress(
        self,
        x,
        y
    ):

        if not self.path_points:

            return self.current_target_index

        start = max(
            0,
            self.current_target_index - 10
        )

        end = min(
            len(self.path_points),
            self.current_target_index + 80
        )

        nearest_index = (
            self.current_target_index
        )

        min_distance = float("inf")

        for index in range(
            start,
            end
        ):

            px, py = self.path_points[index]

            distance = self.distance(
                x,
                y,
                px,
                py
            )

            if distance < min_distance:

                min_distance = distance

                nearest_index = index

        return nearest_index

    # =================================================================
    # Heading score
    # =================================================================

    def calculate_heading_score(
        self,
        x,
        y,
        yaw
    ):

        if not self.path_points:

            return 0.0

        # ------------------------------------------------------------
        # 找预测终点附近的路径点
        # ------------------------------------------------------------

        min_distance = float("inf")

        nearest_index = (
            self.current_target_index
        )

        start = max(
            0,
            self.current_target_index - 5
        )

        end = min(
            len(self.path_points),
            self.current_target_index + 80
        )

        for index in range(
            start,
            end
        ):

            px, py = self.path_points[index]

            distance = self.distance(
                x,
                y,
                px,
                py
            )

            if distance < min_distance:

                min_distance = distance

                nearest_index = index

        # ------------------------------------------------------------
        # 向前看一些路径点
        # ------------------------------------------------------------

        target_index = min(
            len(self.path_points) - 1,
            nearest_index + 5
        )

        target_x, target_y = (
            self.path_points[
                target_index
            ]
        )

        dx = target_x - x
        dy = target_y - y

        if (
            abs(dx) < 1e-6
            and
            abs(dy) < 1e-6
        ):

            return 1.0

        target_yaw = math.atan2(
            dy,
            dx
        )

        yaw_error = self.normalize_angle(
            target_yaw - yaw
        )

        # ------------------------------------------------------------
        # cos：
        #
        # 同向       -> 1
        # 90度       -> 0
        # 180度      -> -1
        # ------------------------------------------------------------

        return math.cos(
            yaw_error
        )

    # =================================================================
    # Static obstacle clearance score
    # =================================================================

    def calculate_obstacle_score(
        self,
        trajectory
    ):

        min_distance = (
            self.minimum_obstacle_distance(
                trajectory
            )
        )

        # ------------------------------------------------------------
        # 没有 LaserScan 障碍点
        # ------------------------------------------------------------

        if not math.isfinite(
            min_distance
        ):

            return 1.0

        safe_distance = (
            self.robot_radius
            +
            self.obstacle_margin
        )

        # ------------------------------------------------------------
        # 已经进入安全范围
        # ------------------------------------------------------------

        if min_distance <= safe_distance:

            return -1.0

        # ------------------------------------------------------------
        # 超过 preferred distance：
        #
        # 不再继续奖励。
        #
        # 例如：
        #
        # 0.8m -> 1.0
        # 1.5m -> 1.0
        # 2.5m -> 1.0
        #
        # 避免“离障碍物越远越慢”。
        # ------------------------------------------------------------

        if (
            min_distance
            >=
            self.preferred_obstacle_distance
        ):

            return 1.0

        # ------------------------------------------------------------
        # safe_distance ~ preferred_distance
        # ------------------------------------------------------------

        score = (
            min_distance
            -
            safe_distance
        ) / (
            self.preferred_obstacle_distance
            -
            safe_distance
        )

        return max(
            0.0,
            min(
                1.0,
                score
            )
        )

    # =================================================================
    # Other robot clearance score
    # =================================================================

    def calculate_robot_score(
        self,
        trajectory
    ):

        min_distance = (
            self.minimum_robot_distance(
                trajectory
            )
        )

        if not math.isfinite(
            min_distance
        ):

            return 1.0

        safe_distance = (
            self.robot_radius
            +
            self.robot_collision_margin
            +
            self.robot_radius
        )

        if min_distance <= safe_distance:

            return -1.0

        preferred_distance = (
            safe_distance
            +
            0.8
        )

        if min_distance >= preferred_distance:

            return 1.0

        score = (
            min_distance
            -
            safe_distance
        ) / (
            preferred_distance
            -
            safe_distance
        )

        return max(
            0.0,
            min(
                1.0,
                score
            )
        )

    # =================================================================
    # Calculate DWA score
    # =================================================================

    def calculate_dwa_score(
        self,
        trajectory,
        v,
        w
    ):

        if not trajectory:

            return -float("inf")

        final_x = trajectory[-1][0]
        final_y = trajectory[-1][1]
        final_yaw = trajectory[-1][2]

        # ============================================================
        # 1. Path score
        # ============================================================

        path_distance = (
            self.distance_to_global_path(
                final_x,
                final_y
            )
        )

        if not math.isfinite(
            path_distance
        ):

            path_score = 0.0

        else:

            # --------------------------------------------------------
            # 距离路径越近越好
            # --------------------------------------------------------

            path_scale = 0.50

            path_score = math.exp(
                -path_distance
                /
                path_scale
            )

        # ============================================================
        # 2. Heading score
        # ============================================================

        heading_score = (
            self.calculate_heading_score(
                final_x,
                final_y,
                final_yaw
            )
        )

        # ------------------------------------------------------------
        # 将 [-1, 1] 映射到 [0, 1]
        # ------------------------------------------------------------

        heading_score = (
            0.5
            *
            (
                heading_score
                +
                1.0
            )
        )

        # ============================================================
        # 3. Velocity score
        # ============================================================

        if self.max_linear_speed > 1e-6:

            velocity_score = (
                v
                /
                self.max_linear_speed
            )

        else:

            velocity_score = 0.0

        velocity_score = max(
            0.0,
            min(
                1.0,
                velocity_score
            )
        )

        # ============================================================
        # 4. Progress score
        # ============================================================

        progress_index = (
            self.get_path_progress(
                final_x,
                final_y
            )
        )

        progress = (
            progress_index
            -
            self.current_target_index
        )

        # ------------------------------------------------------------
        # 如果预测终点没有向前：
        #
        # score = 0
        #
        # 如果向前 10 个 Path point：
        #
        # score ≈ 1
        # ------------------------------------------------------------

        progress_score = (
            progress
            /
            10.0
        )

        progress_score = max(
            0.0,
            min(
                1.0,
                progress_score
            )
        )

        # ============================================================
        # 5. Static obstacle score
        # ============================================================

        obstacle_score = (
            self.calculate_obstacle_score(
                trajectory
            )
        )

        # ============================================================
        # 6. Other robot score
        # ============================================================

        robot_score = (
            self.calculate_robot_score(
                trajectory
            )
        )

        # ============================================================
        # 7. Angular penalty
        # ============================================================

        angular_penalty = (
            abs(w)
            /
            max(
                self.max_angular_speed,
                1e-6
            )
        )

        # ============================================================
        # 8. Total score
        # ============================================================

        score = (

            # Path following
            self.path_weight
            *
            path_score

            +

            # Heading
            self.heading_weight
            *
            heading_score

            +

            # Speed
            self.velocity_weight
            *
            velocity_score

            +

            # Forward progress
            self.progress_weight
            *
            progress_score

            +

            # Static obstacle
            self.obstacle_weight
            *
            obstacle_score

            +

            # Other robot
            self.robot_weight
            *
            robot_score

            -

            # Excessive turning
            self.angular_penalty_weight
            *
            angular_penalty
        )

        return score

    # =================================================================
    # DWA main
    # =================================================================

    def dwa_control(self):

        (
            v_min,
            v_max,
            w_min,
            w_max
        ) = self.calculate_dynamic_window(
            self.robot_v,
            self.robot_w
        )

        # ============================================================
        # Ensure valid window
        # ============================================================

        if v_max < v_min:

            v_max = v_min

        if w_max < w_min:

            w_max = w_min

        # ============================================================
        # Velocity Samples
        # ============================================================

        if self.linear_samples <= 1:

            v_samples = [
                v_max
            ]

        else:

            v_samples = np.linspace(
                v_min,
                v_max,
                self.linear_samples
            )

        # ============================================================
        # Angular Samples
        # ============================================================

        if self.angular_samples <= 1:

            w_samples = [
                0.0
            ]

        else:

            w_samples = np.linspace(
                w_min,
                w_max,
                self.angular_samples
            )

        # ============================================================
        # Best trajectory
        # ============================================================

        best_score = -float("inf")

        best_v = 0.0
        best_w = 0.0

        feasible_count = 0

        # ============================================================
        # Evaluate all velocity pairs
        # ============================================================

        for v in v_samples:

            for w in w_samples:

                v = float(v)
                w = float(w)

                trajectory = (
                    self.predict_trajectory(
                        v,
                        w
                    )
                )

                # ----------------------------------------------------
                # Static obstacle collision
                # ----------------------------------------------------

                if self.trajectory_collision_obstacles(
                    trajectory
                ):

                    continue

                # ----------------------------------------------------
                # Other robot collision
                # ----------------------------------------------------

                if self.trajectory_collision_robot(
                    trajectory
                ):

                    continue

                feasible_count += 1

                score = (
                    self.calculate_dwa_score(
                        trajectory,
                        v,
                        w
                    )
                )

                if score > best_score:

                    best_score = score

                    best_v = v
                    best_w = w

        # ============================================================
        # No safe trajectory
        # ============================================================

        if feasible_count == 0:

            rospy.logwarn_throttle(
                1.0,
                "Robot%d DWA: no safe trajectory. STOP.",
                self.robot_id
            )

            return (
                0.0,
                0.0,
                -float("inf")
            )

        return (
            best_v,
            best_w,
            best_score
        )

    # =================================================================
    # Control
    # =================================================================

    def control(self):

        # ============================================================
        # No Path
        # ============================================================

        if not self.path_received:

            self.stop()

            return

        # ============================================================
        # No Pose
        # ============================================================

        if not self.pose_received:

            self.stop()

            return

        # ============================================================
        # No fresh LaserScan
        # ============================================================

        if not self.scan_is_fresh():

            rospy.logwarn_throttle(
                2.0,
                "Robot%d waiting for fresh LaserScan.",
                self.robot_id
            )

            self.stop()

            return

        # ============================================================
        # Finished
        # ============================================================

        if self.finished:

            self.stop()

            return

        # ============================================================
        # Path timeout
        # ============================================================

        if (
            self.path_timeout > 0.0
            and
            self.last_path_time is not None
        ):

            age = (
                rospy.Time.now()
                -
                self.last_path_time
            ).to_sec()

            if age > self.path_timeout:

                rospy.logwarn_throttle(
                    2.0,
                    "Robot%d Path timeout. Stop.",
                    self.robot_id
                )

                self.stop()

                return

        # ============================================================
        # Goal
        # ============================================================

        if self.check_goal():

            return

        # ============================================================
        # Update global target
        # ============================================================

        target_index = (
            self.update_target_index()
        )

        if target_index is None:

            self.stop()

            return

        target_x, target_y = (
            self.path_points[
                target_index
            ]
        )

        target_distance = self.distance(
            self.robot_x,
            self.robot_y,
            target_x,
            target_y
        )

        # ============================================================
        # Waypoint passed
        # ============================================================

        if target_distance <= self.waypoint_tolerance:

            if (
                self.current_target_index
                <
                len(self.path_points) - 1
            ):

                self.current_target_index += 1

        # ============================================================
        # DWA
        # ============================================================

        (
            linear_x,
            angular_z,
            score
        ) = self.dwa_control()

        # ============================================================
        # Final goal speed reduction
        # ============================================================

        remaining_points = (
            len(self.path_points)
            -
            self.current_target_index
        )

        if remaining_points <= 20:

            distance_factor = min(
                1.0,
                target_distance
                /
                0.5
            )

            linear_x *= (
                0.3
                +
                0.7
                *
                distance_factor
            )

        # ============================================================
        # Do not reverse
        # ============================================================

        linear_x = max(
            0.0,
            linear_x
        )

        # ============================================================
        # Publish
        # ============================================================

        cmd = Twist()

        cmd.linear.x = linear_x

        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0

        cmd.angular.z = angular_z

        self.cmd_pub.publish(
            cmd
        )

    # =================================================================
    # Stop
    # =================================================================

    def stop(self):

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0

        cmd.angular.z = 0.0

        self.cmd_pub.publish(
            cmd
        )


# =====================================================================
# Dual Robot Controller
# =====================================================================

class DualRobotPathController:

    def __init__(self):

        rospy.init_node(
            "dual_robot_path_controller"
        )

        # ============================================================
        # Model States
        # ============================================================

        model_states_topic = rospy.get_param(
            "~model_states_topic",
            "/gazebo/model_states"
        )

        # ============================================================
        # Robot 1
        # ============================================================

        robot1_name = rospy.get_param(
            "~robot1_name",
            "robot1"
        )

        robot2_name = rospy.get_param(
            "~robot2_name",
            "robot2"
        )

        self.robot1 = RobotPathTracker(
            robot_id=1,

            robot_name=robot1_name,

            other_robot_name=robot2_name,

            path_topic=rospy.get_param(
                "~robot1_path_topic",
                "/robot1_coverage_path"
            ),

            cmd_topic=rospy.get_param(
                "~robot1_cmd_topic",
                "/robot1/cmd_vel"
            ),

            scan_topic=rospy.get_param(
                "~robot1_scan_topic",
                "/robot1/scan"
            ),

            model_states_topic=model_states_topic
        )

        # ============================================================
        # Robot 2
        # ============================================================

        self.robot2 = RobotPathTracker(
            robot_id=2,

            robot_name=robot2_name,

            other_robot_name=robot1_name,

            path_topic=rospy.get_param(
                "~robot2_path_topic",
                "/robot2_coverage_path"
            ),

            cmd_topic=rospy.get_param(
                "~robot2_cmd_topic",
                "/robot2/cmd_vel"
            ),

            scan_topic=rospy.get_param(
                "~robot2_scan_topic",
                "/robot2/scan"
            ),

            model_states_topic=model_states_topic
        )

        # ============================================================
        # Timer
        # ============================================================

        frequency = rospy.get_param(
            "~control_frequency",
            20.0
        )

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0
                /
                max(
                    frequency,
                    1e-6
                )
            ),
            self.control_callback
        )

        rospy.on_shutdown(
            self.shutdown
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "Dual Robot Coverage DWA Controller"
        )

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "Control frequency: %.1f Hz",
            frequency
        )

        rospy.loginfo(
            "Robot1: %s",
            self.robot1.robot_name
        )

        rospy.loginfo(
            "Robot2: %s",
            self.robot2.robot_name
        )

        rospy.loginfo(
            "=========================================="
        )

    # =================================================================
    # Control callback
    # =================================================================

    def control_callback(self, event):

        if rospy.is_shutdown():

            return

        # ------------------------------------------------------------
        # Robot1
        # ------------------------------------------------------------

        self.robot1.control()

        # ------------------------------------------------------------
        # Robot2
        # ------------------------------------------------------------

        self.robot2.control()

    # =================================================================
    # Shutdown
    # =================================================================

    def shutdown(self):

        rospy.loginfo(
            "Stopping both robots..."
        )

        self.robot1.stop()

        self.robot2.stop()


# =====================================================================
# Main
# =====================================================================

def main():

    try:

        controller = (
            DualRobotPathController()
        )

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()
