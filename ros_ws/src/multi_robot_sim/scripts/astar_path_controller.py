#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
import tf2_ros

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan


class AStarPathTracker:

    def __init__(self):

        rospy.init_node("astar_path_tracker")

        # ============================================================
        # Robot configuration
        # ============================================================

        self.robots = rospy.get_param(
            "~robots",
            [
                "robot1",
                "robot2"
            ]
        )

        self.global_frame = rospy.get_param(
            "~global_frame",
            "global_map"
        )

        self.base_frames = rospy.get_param(
            "~base_frames",
            [
                "robot1/base_footprint",
                "robot2/base_footprint"
            ]
        )

        # ============================================================
        # Control parameters
        # ============================================================

        self.control_rate = rospy.get_param(
            "~control_rate",
            20.0
        )

        self.max_linear_speed = rospy.get_param(
            "~max_linear_speed",
            0.25
        )

        self.max_angular_speed = rospy.get_param(
            "~max_angular_speed",
            1.0
        )

        self.min_linear_speed = rospy.get_param(
            "~min_linear_speed",
            0.02
        )

        # ============================================================
        # Path tracking parameters
        # ============================================================

        self.lookahead_distance = rospy.get_param(
            "~lookahead_distance",
            0.40
        )

        self.goal_tolerance = rospy.get_param(
            "~goal_tolerance",
            0.15
        )

        self.k_linear = rospy.get_param(
            "~k_linear",
            0.8
        )

        self.k_heading = rospy.get_param(
            "~k_heading",
            1.8
        )

        self.k_lateral = rospy.get_param(
            "~k_lateral",
            1.5
        )

        # ============================================================
        # Laser safety
        # ============================================================

        self.robot_radius = rospy.get_param(
            "~robot_radius",
            0.105
        )

        self.safety_margin = rospy.get_param(
            "~safety_margin",
            0.08
        )

        self.stop_distance = rospy.get_param(
            "~stop_distance",
            0.00
        )

        self.slow_distance = rospy.get_param(
            "~slow_distance",
            0.50
        )

        self.scan_timeout = rospy.get_param(
            "~scan_timeout",
            0.5
        )

        # ============================================================
        # Laser angle
        # ============================================================

        self.front_angle = rospy.get_param(
            "~front_angle",
            math.radians(30.0)
        )

        self.side_angle = rospy.get_param(
            "~side_angle",
            math.radians(70.0)
        )

        # ============================================================
        # State
        # ============================================================

        self.paths = {}
        self.scans = {}
        self.scan_times = {}

        for robot in self.robots:

            self.paths[robot] = None
            self.scans[robot] = None
            self.scan_times[robot] = rospy.Time(0)

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(20.0)
        )

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        # ============================================================
        # Publishers
        # ============================================================

        self.cmd_pubs = {}

        for robot in self.robots:

            self.cmd_pubs[robot] = rospy.Publisher(
                "/{}/cmd_vel".format(robot),
                Twist,
                queue_size=1
            )

        # ============================================================
        # Subscribers
        # ============================================================

        self.path_subs = []
        self.scan_subs = []

        for robot in self.robots:

            path_topic = "/{}/global_path".format(robot)

            path_sub = rospy.Subscriber(
                path_topic,
                Path,
                lambda msg, r=robot:
                self.path_callback(msg, r),
                queue_size=1
            )

            self.path_subs.append(path_sub)

            scan_topic = "/{}/scan".format(robot)

            scan_sub = rospy.Subscriber(
                scan_topic,
                LaserScan,
                lambda msg, r=robot:
                self.scan_callback(msg, r),
                queue_size=1
            )

            self.scan_subs.append(scan_sub)

        # ============================================================
        # Timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 /
                max(self.control_rate, 1.0)
            ),
            self.control_callback
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo(
            "=============================================="
        )

        rospy.loginfo(
            "A* Path Tracker started."
        )

        rospy.loginfo(
            "Robots: %s",
            str(self.robots)
        )

        rospy.loginfo(
            "Global frame: %s",
            self.global_frame
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
            "Lookahead distance: %.3f m",
            self.lookahead_distance
        )

        rospy.loginfo(
            "Goal tolerance: %.3f m",
            self.goal_tolerance
        )

        rospy.loginfo(
            "=============================================="
        )

    # ================================================================
    # Path callback
    # ================================================================

    def path_callback(
        self,
        msg,
        robot
    ):

        if len(msg.poses) == 0:

            self.paths[robot] = None

            self.publish_stop(robot)

            rospy.logwarn(
                "[%s] Received empty A* path. Robot stopped.",
                robot
            )

            return

        self.paths[robot] = msg

        rospy.loginfo_throttle(
            2.0,
            "[%s] A* path received: %d points.",
            robot,
            len(msg.poses)
        )

    # ================================================================
    # Laser callback
    # ================================================================

    def scan_callback(
        self,
        msg,
        robot
    ):

        self.scans[robot] = msg

        self.scan_times[robot] = rospy.Time.now()

    # ================================================================
    # Main control
    # ================================================================

    def control_callback(
        self,
        event
    ):

        for robot in self.robots:

            path = self.paths[robot]

            if path is None:

                self.publish_stop(robot)

                continue

            scan = self.scans[robot]

            if scan is None:

                rospy.logwarn_throttle(
                    2.0,
                    "[%s] No LaserScan. Robot stopped.",
                    robot
                )

                self.publish_stop(robot)

                continue

            scan_age = (
                rospy.Time.now() -
                self.scan_times[robot]
            ).to_sec()

            if scan_age > self.scan_timeout:

                rospy.logwarn_throttle(
                    2.0,
                    "[%s] LaserScan timeout: %.2f s.",
                    robot,
                    scan_age
                )

                self.publish_stop(robot)

                continue

            cmd = self.compute_control(
                robot,
                path,
                scan
            )

            if cmd is None:

                self.publish_stop(robot)

            else:

                self.cmd_pubs[robot].publish(
                    cmd
                )

    # ================================================================
    # Compute control
    # ================================================================

    def compute_control(
        self,
        robot,
        path,
        scan
    ):

        # ============================================================
        # Transform A* path to robot coordinate frame
        # ============================================================

        local_path = self.transform_path_to_base(
            path,
            robot
        )

        if len(local_path) == 0:

            rospy.logwarn_throttle(
                2.0,
                "[%s] Cannot transform A* path.",
                robot
            )

            return None

        # ============================================================
        # Remove points behind robot when possible
        # ============================================================

        local_path = self.filter_path(
            local_path
        )

        if len(local_path) == 0:

            return None

        # ============================================================
        # Goal
        # ============================================================

        goal_x, goal_y = local_path[-1]

        goal_distance = math.hypot(
            goal_x,
            goal_y
        )

        if goal_distance <= self.goal_tolerance:

            rospy.loginfo_throttle(
                2.0,
                "[%s] A* goal reached.",
                robot
            )

            return Twist()

        # ============================================================
        # Find nearest path point
        # ============================================================

        nearest_index = self.find_nearest_point(
            local_path
        )

        # ============================================================
        # Find lookahead point
        # ============================================================

        target_index = self.find_lookahead_point(
            local_path,
            nearest_index
        )

        target_x, target_y = local_path[
            target_index
        ]

        # ============================================================
        # Calculate actual path tangent
        # ============================================================

        target_yaw = self.get_path_yaw(
            local_path,
            target_index
        )

        # ============================================================
        # Errors
        # ============================================================

        lateral_error = target_y

        heading_error = self.normalize_angle(
            target_yaw
        )

        target_distance = math.hypot(
            target_x,
            target_y
        )

        # ============================================================
        # Angular velocity
        # ============================================================

        angular_speed = (
            self.k_heading *
            heading_error
            +
            self.k_lateral *
            lateral_error
        )

        angular_speed = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                angular_speed
            )
        )

        # ============================================================
        # Base linear velocity
        # ============================================================

        linear_speed = (
            self.k_linear *
            target_distance
        )

        linear_speed = min(
            self.max_linear_speed,
            linear_speed
        )

        # ============================================================
        # Heading based speed control
        #
        # Don't force zero too early.
        # ============================================================

        abs_heading = abs(
            heading_error
        )

        if abs_heading > 1.5:

            linear_speed = 0.0

        elif abs_heading > 1.1:

            linear_speed *= 0.10

        elif abs_heading > 0.8:

            linear_speed *= 0.30

        elif abs_heading > 0.5:

            linear_speed *= 0.55

        elif abs_heading > 0.3:

            linear_speed *= 0.75

        # ============================================================
        # If target is behind robot
        # ============================================================

        if target_x < 0.0:

            linear_speed *= 0.10

            if abs_heading > 0.7:

                linear_speed = 0.0

        # ============================================================
        # Laser obstacle detection
        # ============================================================

        front_distance = self.get_front_distance(
            scan
        )

        left_distance = self.get_sector_distance(
            scan,
            0.0,
            self.side_angle
        )

        right_distance = self.get_sector_distance(
            scan,
            -self.side_angle,
            0.0
        )

        # ============================================================
        # Hard stop
        # ============================================================

        if front_distance <= self.stop_distance:

            rospy.logwarn_throttle(
                1.0,
                "[%s] Obstacle too close: %.3f m",
                robot,
                front_distance
            )

            cmd = Twist()

            cmd.linear.x = 0.0

            # Turn toward side with more clearance
            if left_distance > right_distance:

                cmd.angular.z = 0.30

            else:

                cmd.angular.z = -0.30

            return cmd

        # ============================================================
        # Slow down near obstacle
        # ============================================================

        if front_distance < self.slow_distance:

            denominator = (
                self.slow_distance -
                self.stop_distance
            )

            ratio = (
                front_distance -
                self.stop_distance
            ) / denominator

            ratio = max(
                0.0,
                min(
                    1.0,
                    ratio
                )
            )

            linear_speed *= ratio

        # ============================================================
        # Strong turning
        # ============================================================

        if abs(angular_speed) > 0.8:

            linear_speed *= 0.35

        elif abs(angular_speed) > 0.6:

            linear_speed *= 0.55

        # ============================================================
        # Near goal
        # ============================================================

        if goal_distance < (
            2.0 *
            self.goal_tolerance
        ):

            linear_speed *= 0.5

        # ============================================================
        # Minimum speed
        # ============================================================

        if (
            linear_speed > 0.0
            and
            linear_speed < self.min_linear_speed
            and
            goal_distance > self.goal_tolerance
            and
            front_distance > self.stop_distance
            and
            abs_heading < 0.8
        ):

            linear_speed = (
                self.min_linear_speed
            )

        # ============================================================
        # Final command
        # ============================================================

        cmd = Twist()

        cmd.linear.x = linear_speed
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = angular_speed

        return cmd

    # ================================================================
    # Filter path
    # ================================================================

    def filter_path(
        self,
        path
    ):

        if len(path) == 0:

            return []

        # Keep all points.
        #
        # We don't remove points aggressively because
        # the A* path itself may contain important turns.
        #
        return path

    # ================================================================
    # Find nearest path point
    # ================================================================

    def find_nearest_point(
        self,
        path
    ):

        best_index = 0

        best_distance = float("inf")

        for i, (x, y) in enumerate(path):

            distance = math.hypot(
                x,
                y
            )

            if distance < best_distance:

                best_distance = distance

                best_index = i

        return best_index

    # ================================================================
    # Find lookahead point
    # ================================================================

    def find_lookahead_point(
        self,
        path,
        start_index
    ):

        if len(path) <= 1:

            return 0

        if start_index >= len(path) - 1:

            return len(path) - 1

        accumulated_distance = 0.0

        for i in range(
            start_index + 1,
            len(path)
        ):

            x1, y1 = path[i - 1]

            x2, y2 = path[i]

            segment_distance = math.hypot(
                x2 - x1,
                y2 - y1
            )

            accumulated_distance += (
                segment_distance
            )

            if (
                accumulated_distance
                >=
                self.lookahead_distance
            ):

                return i

        return len(path) - 1

    # ================================================================
    # Calculate path tangent
    # ================================================================

    def get_path_yaw(
        self,
        path,
        index
    ):

        if len(path) <= 1:

            return 0.0

        # ------------------------------------------------------------
        # Prefer forward direction
        # ------------------------------------------------------------

        if index < len(path) - 1:

            x1, y1 = path[index]

            x2, y2 = path[index + 1]

        else:

            x1, y1 = path[index - 1]

            x2, y2 = path[index]

        dx = x2 - x1

        dy = y2 - y1

        distance = math.hypot(
            dx,
            dy
        )

        if distance < 1e-6:

            return 0.0

        return math.atan2(
            dy,
            dx
        )

    # ================================================================
    # Transform global path to robot base frame
    # ================================================================

    def transform_path_to_base(
        self,
        path_msg,
        robot
    ):

        try:

            robot_index = self.robots.index(
                robot
            )

            base_frame = (
                self.base_frames[
                    robot_index
                ]
            )

        except ValueError:

            rospy.logerr(
                "[%s] Robot configuration error.",
                robot
            )

            return []

        source_frame = (
            path_msg.header.frame_id
        )

        if not source_frame:

            source_frame = self.global_frame

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    base_frame,
                    source_frame,
                    rospy.Time(0),
                    timeout=rospy.Duration(0.2)
                )
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ) as e:

            rospy.logwarn_throttle(
                2.0,
                "[%s] TF failed: %s",
                robot,
                str(e)
            )

            return []

        # ============================================================
        # Transform translation
        # ============================================================

        tx = (
            transform.transform
            .translation.x
        )

        ty = (
            transform.transform
            .translation.y
        )

        # ============================================================
        # Transform rotation
        # ============================================================

        q = (
            transform.transform
            .rotation
        )

        yaw = math.atan2(
            2.0 * (
                q.w * q.z
                +
                q.x * q.y
            ),
            1.0 -
            2.0 * (
                q.y * q.y
                +
                q.z * q.z
            )
        )

        cos_yaw = math.cos(
            yaw
        )

        sin_yaw = math.sin(
            yaw
        )

        # ============================================================
        # Transform path
        # ============================================================

        local_path = []

        for pose in path_msg.poses:

            gx = (
                pose.pose
                .position.x
            )

            gy = (
                pose.pose
                .position.y
            )

            dx = gx - tx

            dy = gy - ty

            lx = (
                cos_yaw * dx
                +
                sin_yaw * dy
            )

            ly = (
                -sin_yaw * dx
                +
                cos_yaw * dy
            )

            local_path.append(
                (
                    lx,
                    ly
                )
            )

        return local_path

    # ================================================================
    # Get front obstacle distance
    # ================================================================

    def get_front_distance(
        self,
        scan
    ):

        min_distance = float("inf")

        angle = scan.angle_min

        for r in scan.ranges:

            if angle < -self.front_angle:

                angle += scan.angle_increment

                continue

            if angle > self.front_angle:

                break

            if math.isfinite(r):

                if (
                    r >= scan.range_min
                    and
                    r <= scan.range_max
                ):

                    min_distance = min(
                        min_distance,
                        r
                    )

            angle += scan.angle_increment

        return min_distance

    # ================================================================
    # Get sector minimum distance
    # ================================================================

    def get_sector_distance(
        self,
        scan,
        angle_min,
        angle_max
    ):

        min_distance = float("inf")

        angle = scan.angle_min

        for r in scan.ranges:

            if angle > angle_max:

                break

            if angle >= angle_min:

                if math.isfinite(r):

                    if (
                        r >= scan.range_min
                        and
                        r <= scan.range_max
                    ):

                        min_distance = min(
                            min_distance,
                            r
                        )

            angle += scan.angle_increment

        return min_distance

    # ================================================================
    # Publish stop
    # ================================================================

    def publish_stop(
        self,
        robot
    ):

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0

        self.cmd_pubs[
            robot
        ].publish(cmd)

    # ================================================================
    # Normalize angle
    # ================================================================

    @staticmethod
    def normalize_angle(
        angle
    ):

        while angle > math.pi:

            angle -= 2.0 * math.pi

        while angle < -math.pi:

            angle += 2.0 * math.pi

        return angle


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    try:

        node = AStarPathTracker()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass