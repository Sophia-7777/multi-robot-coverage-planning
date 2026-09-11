#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coverage_path_controller.py

ROS1 Dual Robot Coverage Path Controller
Global BCD Coverage Path + Improved DWA Local Planner

主要改进：

1. 更精确的 Global Path Tracking
2. 使用整个预测轨迹计算 Path Error
3. 使用 Path Heading
4. 使用 Path Progress
5. 增加原地旋转 Recovery
6. no safe trajectory 不再永久 STOP
7. DWA 支持 0 m/s + 非零角速度
8. 静态障碍物与其他机器人分别处理
9. 更合理的 obstacle clearance
10. 自动从障碍物状态恢复
"""

import math

import numpy as np
import rospy

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan


# =====================================================================
# Robot Path Tracker
# =====================================================================

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
        # Control
        # ============================================================

        self.control_frequency = rospy.get_param(
            "~control_frequency",
            20.0
        )

        # ============================================================
        # Velocity
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
        # Acceleration
        # ============================================================

        self.max_linear_accel = rospy.get_param(
            "~max_linear_accel",
            1.5
        )

        self.max_angular_accel = rospy.get_param(
            "~max_angular_accel",
            3.5
        )

        # ============================================================
        # DWA prediction
        # ============================================================

        self.dwa_dt = rospy.get_param(
            "~dwa_dt",
            0.08
        )

        self.dwa_predict_time = rospy.get_param(
            "~dwa_predict_time",
            1.5
        )

        self.linear_samples = rospy.get_param(
            "~linear_samples",
            9
        )

        self.angular_samples = rospy.get_param(
            "~angular_samples",
            21
        )

        # ============================================================
        # Robot geometry
        # ============================================================

        self.coverage_radius = rospy.get_param(
            "~coverage_radius",
            0.5
        )

        self.robot_radius = rospy.get_param(
            "~robot_radius",
            0.25
        )

        self.obstacle_margin = rospy.get_param(
            "~obstacle_margin",
            0.08
        )

        self.robot_collision_margin = rospy.get_param(
            "~robot_collision_margin",
            0.15
        )

        # ============================================================
        # Laser
        # ============================================================

        self.scan_timeout = rospy.get_param(
            "~scan_timeout",
            0.5
        )

        # ============================================================
        # Global path
        # ============================================================

        self.lookahead_distance = rospy.get_param(
            "~lookahead_distance",
            0.40
        )

        self.waypoint_tolerance = rospy.get_param(
            "~waypoint_tolerance",
            0.10
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
        # Path tracking
        # ============================================================

        self.path_search_forward = rospy.get_param(
            "~path_search_forward",
            100
        )

        self.path_error_scale = rospy.get_param(
            "~path_error_scale",
            0.35
        )

        # ============================================================
        # DWA weights
        # ============================================================

        self.path_weight = rospy.get_param(
            "~path_weight",
            3.0
        )

        self.heading_weight = rospy.get_param(
            "~heading_weight",
            2.5
        )

        self.progress_weight = rospy.get_param(
            "~progress_weight",
            3.0
        )

        self.velocity_weight = rospy.get_param(
            "~velocity_weight",
            1.0
        )

        self.obstacle_weight = rospy.get_param(
            "~obstacle_weight",
            1.5
        )

        self.robot_weight = rospy.get_param(
            "~robot_weight",
            3.0
        )

        self.angular_penalty_weight = rospy.get_param(
            "~angular_penalty_weight",
            0.15
        )

        # ============================================================
        # Obstacle clearance
        # ============================================================

        self.preferred_obstacle_distance = rospy.get_param(
            "~preferred_obstacle_distance",
            0.70
        )

        # ============================================================
        # Recovery
        # ============================================================

        self.recovery_rotate_speed = rospy.get_param(
            "~recovery_rotate_speed",
            0.60
        )

        self.recovery_timeout = rospy.get_param(
            "~recovery_timeout",
            3.0
        )

        self.recovery_min_rotation = rospy.get_param(
            "~recovery_min_rotation",
            0.15
        )

        # ============================================================
        # Data
        # ============================================================

        self.path = None
        self.path_points = []

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.robot_v = 0.0
        self.robot_w = 0.0

        self.other_robot_x = None
        self.other_robot_y = None
        self.other_robot_yaw = None

        self.other_robot_v = 0.0
        self.other_robot_w = 0.0

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
        # Recovery state
        # ============================================================

        self.recovery_mode = False
        self.recovery_start_time = None
        self.recovery_direction = 1.0
        self.recovery_start_yaw = None

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

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "Robot%d Improved DWA Controller",
            self.robot_id
        )

        rospy.loginfo(
            "Robot: %s",
            self.robot_name
        )

        rospy.loginfo(
            "Other: %s",
            self.other_robot_name
        )

        rospy.loginfo(
            "Robot radius: %.2f",
            self.robot_radius
        )

        rospy.loginfo(
            "Max speed: %.2f",
            self.max_linear_speed
        )

        rospy.loginfo(
            "Lookahead: %.2f",
            self.lookahead_distance
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
                "Robot%d received empty path.",
                self.robot_id
            )

            self.path_points = []
            self.path_received = False

            self.stop()

            return

        self.path = msg

        self.path_points = []

        for pose in msg.poses:

            self.path_points.append(
                (
                    pose.pose.position.x,
                    pose.pose.position.y
                )
            )

        self.path_received = True

        self.finished = False

        self.current_target_index = 0

        self.last_path_time = rospy.Time.now()

        rospy.loginfo(
            "Robot%d received path: %d points.",
            self.robot_id,
            len(self.path_points)
        )

    # =================================================================
    # Model states
    # =================================================================

    def model_states_callback(self, msg):

        if self.robot_name in msg.name:

            index = msg.name.index(
                self.robot_name
            )

            pose = msg.pose[index]

            self.robot_x = pose.position.x
            self.robot_y = pose.position.y

            self.robot_yaw = (
                self.quaternion_to_yaw(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                )
            )

            if index < len(msg.twist):

                self.robot_v = max(
                    0.0,
                    msg.twist[index].linear.x
                )

                self.robot_w = (
                    msg.twist[index].angular.z
                )

            self.pose_received = True

        if self.other_robot_name in msg.name:

            index = msg.name.index(
                self.other_robot_name
            )

            pose = msg.pose[index]

            self.other_robot_x = pose.position.x
            self.other_robot_y = pose.position.y

            self.other_robot_yaw = (
                self.quaternion_to_yaw(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                )
            )

            if index < len(msg.twist):

                self.other_robot_v = max(
                    0.0,
                    msg.twist[index].linear.x
                )

                self.other_robot_w = (
                    msg.twist[index].angular.z
                )

    # =================================================================
    # Laser callback
    # =================================================================

    def scan_callback(self, msg):

        self.scan_msg = msg

        if not self.pose_received:

            return

        if self.robot_yaw is None:

            return

        points = []

        angle = msg.angle_min

        cos_yaw = math.cos(
            self.robot_yaw
        )

        sin_yaw = math.sin(
            self.robot_yaw
        )

        for distance in msg.ranges:

            if not math.isfinite(distance):

                angle += msg.angle_increment
                continue

            if distance < msg.range_min:

                angle += msg.angle_increment
                continue

            if distance > msg.range_max:

                angle += msg.angle_increment
                continue

            local_x = (
                distance *
                math.cos(angle)
            )

            local_y = (
                distance *
                math.sin(angle)
            )

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
    # Quaternion
    # =================================================================

    @staticmethod
    def quaternion_to_yaw(
        qx,
        qy,
        qz,
        qw
    ):

        sin_yaw = (
            2.0 *
            (
                qw * qz
                +
                qx * qy
            )
        )

        cos_yaw = (
            1.0 -
            2.0 *
            (
                qy * qy +
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

        return math.hypot(
            x2 - x1,
            y2 - y1
        )

    # =================================================================
    # Find nearest path point
    # =================================================================

    def find_nearest_path_index(self):

        if not self.path_points:

            return None

        start = max(
            0,
            self.current_target_index - 20
        )

        end = min(
            len(self.path_points),
            self.current_target_index
            +
            self.path_search_forward
        )

        min_distance = float("inf")

        nearest_index = (
            self.current_target_index
        )

        for index in range(
            start,
            end
        ):

            px, py = (
                self.path_points[index]
            )

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
    # Lookahead
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

            px, py = (
                self.path_points[index]
            )

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
    # Update target
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
    # Path heading
    # =================================================================

    def get_path_heading(
        self,
        index
    ):

        if not self.path_points:

            return self.robot_yaw

        i1 = max(
            0,
            index - 2
        )

        i2 = min(
            len(self.path_points) - 1,
            index + 3
        )

        x1, y1 = (
            self.path_points[i1]
        )

        x2, y2 = (
            self.path_points[i2]
        )

        dx = x2 - x1
        dy = y2 - y1

        if math.hypot(dx, dy) < 1e-6:

            return self.robot_yaw

        return math.atan2(
            dy,
            dx
        )

    # =================================================================
    # Goal
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
                "Robot%d path finished.",
                self.robot_id
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

        dt = (
            1.0 /
            max(
                self.control_frequency,
                1e-6
            )
        )

        v_min = max(
            0.0,
            current_v
            -
            self.max_linear_accel * dt
        )

        v_max = min(
            self.max_linear_speed,
            current_v
            +
            self.max_linear_accel * dt
        )

        w_min = max(
            -self.max_angular_speed,
            current_w
            -
            self.max_angular_accel * dt
        )

        w_max = min(
            self.max_angular_speed,
            current_w
            +
            self.max_angular_accel * dt
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

        elapsed = 0.0

        while elapsed <= self.dwa_predict_time:

            trajectory.append(
                (
                    x,
                    y,
                    yaw,
                    elapsed
                )
            )

            x += (
                v *
                math.cos(yaw) *
                self.dwa_dt
            )

            y += (
                v *
                math.sin(yaw) *
                self.dwa_dt
            )

            yaw += (
                w *
                self.dwa_dt
            )

            yaw = self.normalize_angle(
                yaw
            )

            elapsed += self.dwa_dt

        return trajectory

    # =================================================================
    # Obstacle collision
    # =================================================================

    def trajectory_collision_obstacles(
        self,
        trajectory
    ):

        if len(
            self.obstacle_points_world
        ) == 0:

            return False

        collision_distance = (
            self.robot_radius
            +
            self.obstacle_margin
        )

        collision_distance_sq = (
            collision_distance ** 2
        )

        points = (
            self.obstacle_points_world
        )

        # ------------------------------------------------------------
        # 对原地旋转：
        #
        # 不能简单把旋转中心和 LaserScan 障碍物判碰撞，
        # 否则机器人靠近障碍物时永远不能转向。
        #
        # 这里只判断机器人中心是否已经进入真正碰撞范围。
        # ------------------------------------------------------------

        for state in trajectory:

            x = state[0]
            y = state[1]

            dx = (
                points[:, 0] - x
            )

            dy = (
                points[:, 1] - y
            )

            distance_sq = (
                dx * dx +
                dy * dy
            )

            if np.any(
                distance_sq
                <
                collision_distance_sq
            ):

                return True

        return False

    # =================================================================
    # Other robot prediction
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

        yaw = (
            self.other_robot_yaw
            if self.other_robot_yaw is not None
            else 0.0
        )

        x = (
            self.other_robot_x
            +
            self.other_robot_v
            *
            math.cos(yaw)
            *
            time
        )

        y = (
            self.other_robot_y
            +
            self.other_robot_v
            *
            math.sin(yaw)
            *
            time
        )

        return x, y

    # =================================================================
    # Robot collision
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
            2.0 *
            self.robot_radius
            +
            self.robot_collision_margin
        )

        safe_distance_sq = (
            safe_distance ** 2
        )

        for state in trajectory:

            x = state[0]
            y = state[1]
            t = state[3]

            other = (
                self.predict_other_robot_position(
                    t
                )
            )

            if other is None:

                continue

            ox, oy = other

            dx = x - ox
            dy = y - oy

            if (
                dx * dx +
                dy * dy
                <
                safe_distance_sq
            ):

                return True

        return False

    # =================================================================
    # Average path error
    # =================================================================

    def trajectory_path_error(
        self,
        trajectory
    ):

        if not self.path_points:

            return float("inf")

        total_error = 0.0

        count = 0

        points = np.asarray(
            self.path_points,
            dtype=np.float64
        )

        # ------------------------------------------------------------
        # 只搜索当前附近路径
        # ------------------------------------------------------------

        start = max(
            0,
            self.current_target_index - 15
        )

        end = min(
            len(points),
            self.current_target_index
            +
            self.path_search_forward
        )

        local_points = points[
            start:end
        ]

        if len(local_points) == 0:

            return float("inf")

        for state in trajectory:

            x = state[0]
            y = state[1]

            dx = (
                local_points[:, 0] - x
            )

            dy = (
                local_points[:, 1] - y
            )

            distance_sq = (
                dx * dx +
                dy * dy
            )

            min_distance = math.sqrt(
                float(
                    np.min(
                        distance_sq
                    )
                )
            )

            total_error += min_distance

            count += 1

        if count == 0:

            return float("inf")

        return (
            total_error / count
        )

    # =================================================================
    # Final path index
    # =================================================================

    def trajectory_path_progress(
        self,
        trajectory
    ):

        if not self.path_points:

            return self.current_target_index

        final_x = trajectory[-1][0]
        final_y = trajectory[-1][1]

        start = max(
            0,
            self.current_target_index - 10
        )

        end = min(
            len(self.path_points),
            self.current_target_index
            +
            self.path_search_forward
        )

        min_distance = float("inf")

        best_index = (
            self.current_target_index
        )

        for index in range(
            start,
            end
        ):

            px, py = (
                self.path_points[index]
            )

            distance = self.distance(
                final_x,
                final_y,
                px,
                py
            )

            if distance < min_distance:

                min_distance = distance
                best_index = index

        return best_index

    # =================================================================
    # Trajectory heading error
    # =================================================================

    def trajectory_heading_error(
        self,
        trajectory
    ):

        if not self.path_points:

            return math.pi

        final_x = trajectory[-1][0]
        final_y = trajectory[-1][1]
        final_yaw = trajectory[-1][2]

        # ------------------------------------------------------------
        # 找预测终点最近路径点
        # ------------------------------------------------------------

        start = max(
            0,
            self.current_target_index - 10
        )

        end = min(
            len(self.path_points),
            self.current_target_index
            +
            self.path_search_forward
        )

        min_distance = float("inf")

        nearest_index = (
            self.current_target_index
        )

        for index in range(
            start,
            end
        ):

            px, py = (
                self.path_points[index]
            )

            distance = self.distance(
                final_x,
                final_y,
                px,
                py
            )

            if distance < min_distance:

                min_distance = distance
                nearest_index = index

        path_yaw = (
            self.get_path_heading(
                nearest_index
            )
        )

        error = abs(
            self.normalize_angle(
                path_yaw - final_yaw
            )
        )

        return error

    # =================================================================
    # Minimum obstacle distance
    # =================================================================

    def minimum_obstacle_distance(
        self,
        trajectory
    ):

        if len(
            self.obstacle_points_world
        ) == 0:

            return float("inf")

        points = (
            self.obstacle_points_world
        )

        minimum = float("inf")

        for state in trajectory:

            x = state[0]
            y = state[1]

            dx = (
                points[:, 0] - x
            )

            dy = (
                points[:, 1] - y
            )

            distance_sq = (
                dx * dx +
                dy * dy
            )

            d = math.sqrt(
                float(
                    np.min(
                        distance_sq
                    )
                )
            )

            minimum = min(
                minimum,
                d
            )

        return minimum

    # =================================================================
    # Obstacle score
    # =================================================================

    def obstacle_score(
        self,
        trajectory
    ):

        d = (
            self.minimum_obstacle_distance(
                trajectory
            )
        )

        if not math.isfinite(d):

            return 1.0

        collision_distance = (
            self.robot_radius
            +
            self.obstacle_margin
        )

        if d <= collision_distance:

            return -1.0

        if (
            d >=
            self.preferred_obstacle_distance
        ):

            return 1.0

        denominator = (
            self.preferred_obstacle_distance
            -
            collision_distance
        )

        if denominator <= 1e-6:

            return 0.0

        score = (
            d -
            collision_distance
        ) / denominator

        return max(
            0.0,
            min(
                1.0,
                score
            )
        )

    # =================================================================
    # Robot score
    # =================================================================

    def minimum_robot_distance(
        self,
        trajectory
    ):

        if (
            self.other_robot_x is None
            or
            self.other_robot_y is None
        ):

            return float("inf")

        minimum = float("inf")

        for state in trajectory:

            x = state[0]
            y = state[1]
            t = state[3]

            other = (
                self.predict_other_robot_position(
                    t
                )
            )

            if other is None:

                continue

            ox, oy = other

            d = self.distance(
                x,
                y,
                ox,
                oy
            )

            minimum = min(
                minimum,
                d
            )

        return minimum

    # =================================================================
    # Robot clearance score
    # =================================================================

    def robot_score(
        self,
        trajectory
    ):

        d = (
            self.minimum_robot_distance(
                trajectory
            )
        )

        if not math.isfinite(d):

            return 1.0

        collision_distance = (
            2.0 *
            self.robot_radius
            +
            self.robot_collision_margin
        )

        if d <= collision_distance:

            return -1.0

        preferred = (
            collision_distance +
            0.8
        )

        if d >= preferred:

            return 1.0

        return (
            d -
            collision_distance
        ) / (
            preferred -
            collision_distance
        )

    # =================================================================
    # DWA score
    # =================================================================

    def calculate_dwa_score(
        self,
        trajectory,
        v,
        w
    ):

        # ============================================================
        # Path error
        # ============================================================

        path_error = (
            self.trajectory_path_error(
                trajectory
            )
        )

        if not math.isfinite(
            path_error
        ):

            path_score = 0.0

        else:

            path_score = math.exp(
                -path_error /
                max(
                    self.path_error_scale,
                    1e-6
                )
            )

        # ============================================================
        # Heading
        # ============================================================

        heading_error = (
            self.trajectory_heading_error(
                trajectory
            )
        )

        heading_score = (
            1.0 -
            heading_error / math.pi
        )

        heading_score = max(
            0.0,
            min(
                1.0,
                heading_score
            )
        )

        # ============================================================
        # Progress
        # ============================================================

        progress_index = (
            self.trajectory_path_progress(
                trajectory
            )
        )

        progress = (
            progress_index -
            self.current_target_index
        )

        progress_score = (
            progress /
            15.0
        )

        progress_score = max(
            0.0,
            min(
                1.0,
                progress_score
            )
        )

        # ============================================================
        # Velocity
        # ============================================================

        velocity_score = (
            v /
            max(
                self.max_linear_speed,
                1e-6
            )
        )

        velocity_score = max(
            0.0,
            min(
                1.0,
                velocity_score
            )
        )

        # ============================================================
        # Obstacle
        # ============================================================

        obstacle_score = (
            self.obstacle_score(
                trajectory
            )
        )

        # ============================================================
        # Other robot
        # ============================================================

        robot_score = (
            self.robot_score(
                trajectory
            )
        )

        # ============================================================
        # Angular penalty
        # ============================================================

        angular_penalty = (
            abs(w) /
            max(
                self.max_angular_speed,
                1e-6
            )
        )

        # ============================================================
        # Total
        # ============================================================

        score = (

            self.path_weight *
            path_score

            +

            self.heading_weight *
            heading_score

            +

            self.progress_weight *
            progress_score

            +

            self.velocity_weight *
            velocity_score

            +

            self.obstacle_weight *
            obstacle_score

            +

            self.robot_weight *
            robot_score

            -

            self.angular_penalty_weight *
            angular_penalty
        )

        return score

    # =================================================================
    # DWA
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

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # 即使当前 v=0，也强制加入：
        #
        #     v=0
        #
        # 这样机器人可以原地旋转。
        # ------------------------------------------------------------

        v_samples = np.linspace(
            v_min,
            v_max,
            max(
                2,
                self.linear_samples
            )
        )

        v_samples = np.unique(
            np.append(
                v_samples,
                0.0
            )
        )

        w_samples = np.linspace(
            w_min,
            w_max,
            max(
                3,
                self.angular_samples
            )
        )

        # ------------------------------------------------------------
        # 如果角速度窗口太小，加入较大的恢复角速度
        # ------------------------------------------------------------

        w_samples = np.unique(
            np.append(
                w_samples,
                [
                    -self.recovery_rotate_speed,
                    self.recovery_rotate_speed
                ]
            )
        )

        best_score = -float("inf")

        best_v = 0.0
        best_w = 0.0

        feasible_count = 0

        # ============================================================
        # Search
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
                # Static obstacle
                # ----------------------------------------------------

                if self.trajectory_collision_obstacles(
                    trajectory
                ):

                    continue

                # ----------------------------------------------------
                # Other robot
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

                # ----------------------------------------------------
                # Prefer moving trajectories when scores similar
                # ----------------------------------------------------

                if (
                    v > 0.0
                    and
                    score > best_score
                ):

                    best_score = score
                    best_v = v
                    best_w = w

                elif (
                    v == 0.0
                    and
                    best_score == -float("inf")
                ):

                    best_score = score
                    best_v = v
                    best_w = w

        # ============================================================
        # No trajectory
        # ============================================================

        if feasible_count == 0:

            rospy.logwarn_throttle(
                1.0,
                "Robot%d: no safe DWA trajectory.",
                self.robot_id
            )

            return (
                None,
                None,
                -float("inf")
            )

        return (
            best_v,
            best_w,
            best_score
        )

    # =================================================================
    # Start recovery
    # =================================================================

    def start_recovery(self):

        if self.recovery_mode:

            return

        self.recovery_mode = True

        self.recovery_start_time = (
            rospy.Time.now()
        )

        self.recovery_start_yaw = (
            self.robot_yaw
        )

        # ------------------------------------------------------------
        # 根据 path heading 决定旋转方向
        # ------------------------------------------------------------

        target_index = (
            self.update_target_index()
        )

        if target_index is not None:

            path_yaw = (
                self.get_path_heading(
                    target_index
                )
            )

            yaw_error = self.normalize_angle(
                path_yaw -
                self.robot_yaw
            )

            if yaw_error >= 0.0:

                self.recovery_direction = 1.0

            else:

                self.recovery_direction = -1.0

        else:

            self.recovery_direction = 1.0

        rospy.logwarn(
            "Robot%d entering rotation recovery.",
            self.robot_id
        )

    # =================================================================
    # Recovery control
    # =================================================================

    def recovery_control(self):

        if not self.recovery_mode:

            return False

        elapsed = (
            rospy.Time.now()
            -
            self.recovery_start_time
        ).to_sec()

        # ------------------------------------------------------------
        # Timeout
        # ------------------------------------------------------------

        if elapsed >= self.recovery_timeout:

            rospy.logwarn(
                "Robot%d recovery timeout.",
                self.robot_id
            )

            self.recovery_mode = False

            return False

        # ------------------------------------------------------------
        # 当前 path heading
        # ------------------------------------------------------------

        target_index = (
            self.update_target_index()
        )

        if target_index is None:

            self.stop()

            return True

        path_yaw = (
            self.get_path_heading(
                target_index
            )
        )

        yaw_error = self.normalize_angle(
            path_yaw -
            self.robot_yaw
        )

        # ------------------------------------------------------------
        # 如果已经基本朝向 path
        # ------------------------------------------------------------

        if abs(yaw_error) < 0.20:

            self.recovery_mode = False

            rospy.loginfo(
                "Robot%d recovery finished.",
                self.robot_id
            )

            return False

        # ------------------------------------------------------------
        # 旋转
        # ------------------------------------------------------------

        angular = (
            self.recovery_direction
            *
            self.recovery_rotate_speed
        )

        # ------------------------------------------------------------
        # 如果方向判断发生变化
        # ------------------------------------------------------------

        if yaw_error > 0.0:

            angular = (
                self.recovery_rotate_speed
            )

        else:

            angular = (
                -self.recovery_rotate_speed
            )

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.angular.z = angular

        self.cmd_pub.publish(
            cmd
        )

        return True

    # =================================================================
    # Main control
    # =================================================================

    def control(self):

        # ============================================================
        # Path
        # ============================================================

        if not self.path_received:

            self.stop()

            return

        # ============================================================
        # Pose
        # ============================================================

        if not self.pose_received:

            self.stop()

            return

        # ============================================================
        # Scan
        # ============================================================

        if not self.scan_is_fresh():

            rospy.logwarn_throttle(
                2.0,
                "Robot%d waiting for LaserScan.",
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
        # Timeout
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

                self.stop()

                return

        # ============================================================
        # Goal
        # ============================================================

        if self.check_goal():

            return

        # ============================================================
        # Update target
        # ============================================================

        target_index = (
            self.update_target_index()
        )

        if target_index is None:

            self.stop()

            return

        # ============================================================
        # Recovery
        # ============================================================

        if self.recovery_mode:

            if self.recovery_control():

                return

        # ============================================================
        # DWA
        # ============================================================

        (
            linear_x,
            angular_z,
            score
        ) = self.dwa_control()

        # ============================================================
        # No feasible trajectory
        # ============================================================

        if linear_x is None:

            self.start_recovery()

            self.recovery_control()

            return

        # ============================================================
        # Target distance
        # ============================================================

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
        # Reduce speed when close to waypoint
        # ============================================================

        if target_distance < 0.30:

            factor = (
                target_distance /
                0.30
            )

            factor = max(
                0.25,
                min(
                    1.0,
                    factor
                )
            )

            linear_x *= factor

        # ============================================================
        # Final goal slow down
        # ============================================================

        goal_x, goal_y = (
            self.path_points[-1]
        )

        goal_distance = self.distance(
            self.robot_x,
            self.robot_y,
            goal_x,
            goal_y
        )

        if goal_distance < 0.50:

            factor = (
                goal_distance /
                0.50
            )

            factor = max(
                0.20,
                min(
                    1.0,
                    factor
                )
            )

            linear_x *= factor

        # ============================================================
        # Publish
        # ============================================================

        cmd = Twist()

        cmd.linear.x = max(
            0.0,
            min(
                self.max_linear_speed,
                linear_x
            )
        )

        cmd.angular.z = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                angular_z
            )
        )

        self.cmd_pub.publish(
            cmd
        )

    # =================================================================
    # Scan fresh
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
    # Stop
    # =================================================================

    def stop(self):

        cmd = Twist()

        cmd.linear.x = 0.0
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

        model_states_topic = rospy.get_param(
            "~model_states_topic",
            "/gazebo/model_states"
        )

        robot1_name = rospy.get_param(
            "~robot1_name",
            "robot1"
        )

        robot2_name = rospy.get_param(
            "~robot2_name",
            "robot2"
        )

        # ============================================================
        # Robot1
        # ============================================================

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
        # Robot2
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
                1.0 /
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

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "Dual Robot Coverage DWA Controller"
        )

        rospy.loginfo(
            "Control frequency: %.1f Hz",
            frequency
        )

        rospy.loginfo(
            "=========================================="
        )

    # =================================================================
    # Timer
    # =================================================================

    def control_callback(self, event):

        if rospy.is_shutdown():

            return

        self.robot1.control()

        self.robot2.control()

    # =================================================================
    # Shutdown
    # =================================================================

    def shutdown(self):

        rospy.loginfo(
            "Stopping both robots."
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


# =====================================================================
# Entry
# =====================================================================

if __name__ == "__main__":

    main()