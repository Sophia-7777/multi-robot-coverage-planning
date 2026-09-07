#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates

from coverage_planner import CoveragePlanner
from astar import AStarPlanner
from dwa import DWAPlanner


class RobotPlannerState:

    def __init__(self):

        self.pose = None
        self.velocity = None

        self.coverage_points = []
        self.coverage_index = 0

        self.global_path = []

        self.initialized = False
        self.finished = False


class MultiRobotPlannerManager:

    """
    Manage two robots.

    Robot 1:
        /robot1_region
        /robot1/cmd_vel

    Robot 2:
        /robot2_region
        /robot2/cmd_vel

    Global map:
        /covered_map

    Robot states:
        /gazebo/model_states
    """

    def __init__(self):

        rospy.init_node(
            "multi_robot_planner_manager"
        )

        # ============================================================
        # Parameters
        # ============================================================

        self.robot1_name = rospy.get_param(
            "~robot1_name",
            "robot1"
        )

        self.robot2_name = rospy.get_param(
            "~robot2_name",
            "robot2"
        )

        self.map_topic = rospy.get_param(
            "~map_topic",
            "/covered_map"
        )

        self.robot1_region_topic = rospy.get_param(
            "~robot1_region_topic",
            "/robot1_region"
        )

        self.robot2_region_topic = rospy.get_param(
            "~robot2_region_topic",
            "/robot2_region"
        )

        self.robot1_cmd_topic = rospy.get_param(
            "~robot1_cmd_topic",
            "/robot1/cmd_vel"
        )

        self.robot2_cmd_topic = rospy.get_param(
            "~robot2_cmd_topic",
            "/robot2/cmd_vel"
        )

        self.control_rate = rospy.get_param(
            "~control_rate",
            10.0
        )

        self.waypoint_tolerance = rospy.get_param(
            "~waypoint_tolerance",
            0.25
        )

        # ============================================================
        # Data
        # ============================================================

        self.global_map = None

        self.robot1_region = None
        self.robot2_region = None

        self.robot1 = RobotPlannerState()
        self.robot2 = RobotPlannerState()

        self.lock = threading.Lock()

        # ============================================================
        # Planners
        # ============================================================

        self.robot1_coverage = CoveragePlanner(
            robot_id=1,
            robot_width=0.45,
            coverage_width=0.50,
            waypoint_spacing=0.30
        )

        self.robot2_coverage = CoveragePlanner(
            robot_id=2,
            robot_width=0.45,
            coverage_width=0.50,
            waypoint_spacing=0.30
        )

        self.astar1 = AStarPlanner(
            allow_diagonal=False
        )

        self.astar2 = AStarPlanner(
            allow_diagonal=False
        )

        self.dwa1 = DWAPlanner(
            max_speed=0.6,
            max_yaw_rate=1.2
        )

        self.dwa2 = DWAPlanner(
            max_speed=0.6,
            max_yaw_rate=1.2
        )

        # ============================================================
        # Subscribers
        # ============================================================

        self.map_sub = rospy.Subscriber(
            self.map_topic,
            OccupancyGrid,
            self.map_callback,
            queue_size=1
        )

        self.robot1_region_sub = rospy.Subscriber(
            self.robot1_region_topic,
            OccupancyGrid,
            self.robot1_region_callback,
            queue_size=1
        )

        self.robot2_region_sub = rospy.Subscriber(
            self.robot2_region_topic,
            OccupancyGrid,
            self.robot2_region_callback,
            queue_size=1
        )

        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback,
            queue_size=1
        )

        # ============================================================
        # Publishers
        # ============================================================

        self.robot1_cmd_pub = rospy.Publisher(
            self.robot1_cmd_topic,
            Twist,
            queue_size=1
        )

        self.robot2_cmd_pub = rospy.Publisher(
            self.robot2_cmd_topic,
            Twist,
            queue_size=1
        )

        # ============================================================
        # Timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 /
                self.control_rate
            ),
            self.timer_callback
        )

        rospy.loginfo(
            "=================================================="
        )

        rospy.loginfo(
            "Multi Robot Planner Started"
        )

        rospy.loginfo(
            "Robot 1: %s",
            self.robot1_name
        )

        rospy.loginfo(
            "Robot 2: %s",
            self.robot2_name
        )

        rospy.loginfo(
            "=================================================="
        )

    # ================================================================
    # Map callback
    # ================================================================

    def map_callback(
        self,
        msg
    ):

        with self.lock:

            self.global_map = msg

    # ================================================================
    # Region callbacks
    # ================================================================

    def robot1_region_callback(
        self,
        msg
    ):

        with self.lock:

            self.robot1_region = msg

        self.robot1_coverage.update_map(
            msg
        )

    def robot2_region_callback(
        self,
        msg
    ):

        with self.lock:

            self.robot2_region = msg

        self.robot2_coverage.update_map(
            msg
        )

    # ================================================================
    # Model states
    # ================================================================

    def model_states_callback(
        self,
        msg
    ):

        try:

            index1 = msg.name.index(
                self.robot1_name
            )

            index2 = msg.name.index(
                self.robot2_name
            )

        except ValueError:

            rospy.logwarn_throttle(
                5.0,
                "Cannot find robot models."
            )

            return

        with self.lock:

            pose1 = msg.pose[index1]
            pose2 = msg.pose[index2]

            twist1 = msg.twist[index1]
            twist2 = msg.twist[index2]

            self.robot1.pose = pose1
            self.robot2.pose = pose2

            self.robot1.velocity = twist1
            self.robot2.velocity = twist2

    # ================================================================
    # Quaternion -> yaw
    # ================================================================

    @staticmethod
    def quaternion_to_yaw(
        orientation
    ):

        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w

        sin_yaw = (
            2.0 *
            (
                w * z +
                x * y
            )
        )

        cos_yaw = (
            1.0 -
            2.0 *
            (
                y * y +
                z * z
            )
        )

        return math.atan2(
            sin_yaw,
            cos_yaw
        )

    # ================================================================
    # World -> Grid
    # ================================================================

    @staticmethod
    def world_to_grid(
        x,
        y,
        map_msg
    ):

        resolution = (
            map_msg.info.resolution
        )

        origin_x = (
            map_msg.info.origin.position.x
        )

        origin_y = (
            map_msg.info.origin.position.y
        )

        gx = int(
            math.floor(
                (
                    x -
                    origin_x
                ) /
                resolution
            )
        )

        gy = int(
            math.floor(
                (
                    y -
                    origin_y
                ) /
                resolution
            )
        )

        return gx, gy

    # ================================================================
    # Region mask
    # ================================================================

    @staticmethod
    def region_to_mask(
        region_msg
    ):

        height = region_msg.info.height
        width = region_msg.info.width

        array = np.asarray(
            region_msg.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        return array == 100

    # ================================================================
    # Find nearest allowed cell
    # ================================================================

    @staticmethod
    def nearest_allowed_cell(
        start,
        allowed_mask,
        max_radius=20
    ):

        sx, sy = start

        height, width = (
            allowed_mask.shape
        )

        if (
            0 <= sx < width and
            0 <= sy < height
        ):

            if allowed_mask[
                sy,
                sx
            ]:

                return (
                    sx,
                    sy
                )

        for radius in range(
            1,
            max_radius + 1
        ):

            for dy in range(
                -radius,
                radius + 1
            ):

                for dx in range(
                    -radius,
                    radius + 1
                ):

                    x = sx + dx
                    y = sy + dy

                    if (
                        x < 0 or
                        x >= width or
                        y < 0 or
                        y >= height
                    ):

                        continue

                    if allowed_mask[
                        y,
                        x
                    ]:

                        return (
                            x,
                            y
                        )

        return None

    # ================================================================
    # Initialize robot planner
    # ================================================================

    def initialize_robot(
        self,
        robot,
        coverage_planner,
        region_msg
    ):

        if robot.initialized:
            return True

        if region_msg is None:
            return False

        if robot.pose is None:
            return False

        # ------------------------------------------------------------
        # Generate coverage points
        # ------------------------------------------------------------

        coverage_points = (
            coverage_planner
            .generate_coverage_path()
        )

        if not coverage_points:

            rospy.logerr(
                "Robot %d: "
                "Cannot generate coverage path.",
                coverage_planner.robot_id
            )

            return False

        robot.coverage_points = (
            coverage_points
        )

        robot.coverage_index = 0

        robot.initialized = True

        rospy.loginfo(
            "Robot %d initialized: "
            "%d coverage points.",
            coverage_planner.robot_id,
            len(coverage_points)
        )

        return True

    # ================================================================
    # Build A* path
    # ================================================================

    def build_astar_path(
        self,
        robot,
        region_msg,
        astar
    ):

        if self.global_map is None:
            return False

        if region_msg is None:
            return False

        if robot.pose is None:
            return False

        if robot.finished:
            return False

        if not robot.coverage_points:
            return False

        # ------------------------------------------------------------
        # Current waypoint
        # ------------------------------------------------------------

        index = (
            robot.coverage_index
        )

        if index >= len(
            robot.coverage_points
        ):

            robot.finished = True

            return False

        goal = (
            robot.coverage_points[index]
        )

        # ------------------------------------------------------------
        # Robot current grid position
        # ------------------------------------------------------------

        start = (
            self.world_to_grid(
                robot.pose.position.x,
                robot.pose.position.y,
                self.global_map
            )
        )

        # ------------------------------------------------------------
        # Own region mask
        # ------------------------------------------------------------

        allowed_mask = (
            self.region_to_mask(
                region_msg
            )
        )

        # ------------------------------------------------------------
        # Correct start if robot is slightly
        # outside its region.
        # ------------------------------------------------------------

        start = (
            self.nearest_allowed_cell(
                start,
                allowed_mask
            )
        )

        if start is None:

            rospy.logwarn(
                "Robot cannot find "
                "allowed start cell."
            )

            return False

        # ------------------------------------------------------------
        # Make sure goal is inside region
        # ------------------------------------------------------------

        gx, gy = goal

        if not allowed_mask[
            gy,
            gx
        ]:

            goal = (
                self.nearest_allowed_cell(
                    goal,
                    allowed_mask
                )
            )

            if goal is None:

                rospy.logwarn(
                    "Goal is not in robot region."
                )

                return False

        # ------------------------------------------------------------
        # Occupancy grid
        # ------------------------------------------------------------

        height = (
            self.global_map.info.height
        )

        width = (
            self.global_map.info.width
        )

        occupancy = np.asarray(
            self.global_map.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ------------------------------------------------------------
        # A*
        # ------------------------------------------------------------

        path = astar.plan(
            occupancy,
            start,
            goal,
            allowed_mask
        )

        if not path:

            rospy.logwarn(
                "Robot %d: "
                "A* failed.",
                coverage_planner.robot_id
            )

            return False

        # ------------------------------------------------------------
        # Convert grid path to world coordinates
        # ------------------------------------------------------------

        world_path = []

        resolution = (
            self.global_map.info.resolution
        )

        origin_x = (
            self.global_map.info.origin.position.x
        )

        origin_y = (
            self.global_map.info.origin.position.y
        )

        for px, py in path:

            wx = (
                origin_x +
                (px + 0.5) *
                resolution
            )

            wy = (
                origin_y +
                (py + 0.5) *
                resolution
            )

            world_path.append(
                (
                    wx,
                    wy
                )
            )

        robot.global_path = (
            world_path
        )

        return True

    # ================================================================
    # Check waypoint reached
    # ================================================================

    def check_waypoint(
        self,
        robot,
        coverage_planner
    ):

        if robot.finished:
            return

        if robot.pose is None:
            return

        if not robot.coverage_points:
            return

        index = (
            robot.coverage_index
        )

        if index >= len(
            robot.coverage_points
        ):

            robot.finished = True

            return

        gx, gy = (
            robot.coverage_points[index]
        )

        target = (
            coverage_planner
            .grid_to_world(
                gx,
                gy
            )
        )

        if target is None:
            return

        tx, ty = target

        dx = (
            robot.pose.position.x -
            tx
        )

        dy = (
            robot.pose.position.y -
            ty
        )

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        if distance <= (
            self.waypoint_tolerance
        ):

            robot.coverage_index += 1

            robot.global_path = []

            rospy.loginfo(
                "Robot %d reached waypoint "
                "%d / %d",
                coverage_planner.robot_id,
                robot.coverage_index,
                len(robot.coverage_points)
            )

            if robot.coverage_index >= len(
                robot.coverage_points
            ):

                robot.finished = True

                rospy.loginfo(
                    "=================================================="
                )

                rospy.loginfo(
                    "Robot %d COVERAGE FINISHED",
                    coverage_planner.robot_id
                )

                rospy.loginfo(
                    "=================================================="
                )

    # ================================================================
    # DWA for one robot
    # ================================================================

    def control_robot(
        self,
        robot,
        other_robot,
        region_msg,
        coverage_planner,
        astar,
        dwa
    ):

        cmd = Twist()

        if robot.pose is None:
            return cmd

        if self.global_map is None:
            return cmd

        if region_msg is None:
            return cmd

        # ------------------------------------------------------------
        # Finished
        # ------------------------------------------------------------

        if robot.finished:
            return cmd

        # ------------------------------------------------------------
        # Check current waypoint
        # ------------------------------------------------------------

        self.check_waypoint(
            robot,
            coverage_planner
        )

        if robot.finished:
            return cmd

        # ------------------------------------------------------------
        # Build A* path if necessary
        # ------------------------------------------------------------

        if not robot.global_path:

            success = (
                self.build_astar_path(
                    robot,
                    region_msg,
                    astar
                )
            )

            if not success:

                return cmd

        # ------------------------------------------------------------
        # Robot state
        # ------------------------------------------------------------

        yaw = (
            self.quaternion_to_yaw(
                robot.pose.orientation
            )
        )

        if robot.velocity is not None:

            vx = (
                robot.velocity.linear.x
            )

            vy = (
                robot.velocity.linear.y
            )

            current_v = math.sqrt(
                vx * vx +
                vy * vy
            )

            current_omega = (
                robot.velocity.angular.z
            )

        else:

            current_v = 0.0
            current_omega = 0.0

        state = [
            robot.pose.position.x,
            robot.pose.position.y,
            yaw,
            current_v,
            current_omega
        ]

        # ------------------------------------------------------------
        # Occupancy grid
        # ------------------------------------------------------------

        height = (
            self.global_map.info.height
        )

        width = (
            self.global_map.info.width
        )

        occupancy = np.asarray(
            self.global_map.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ------------------------------------------------------------
        # Other robot position
        # ------------------------------------------------------------

        other_position = None

        if other_robot.pose is not None:

            other_position = (
                other_robot.pose.position.x,
                other_robot.pose.position.y
            )

        # ------------------------------------------------------------
        # Configure DWA map
        # ------------------------------------------------------------

        dwa.set_map_info(
            self.global_map.info.resolution,
            self.global_map.info.origin.position.x,
            self.global_map.info.origin.position.y
        )

        # ------------------------------------------------------------
        # DWA
        # ------------------------------------------------------------

        v, omega = dwa.plan(
            state,
            robot.global_path,
            occupancy,
            other_position
        )

        cmd.linear.x = v
        cmd.angular.z = omega

        return cmd

    # ================================================================
    # Timer
    # ================================================================

    def timer_callback(
        self,
        event
    ):

        with self.lock:

            # --------------------------------------------------------
            # Snapshot data
            # --------------------------------------------------------

            global_map = self.global_map

            region1 = self.robot1_region
            region2 = self.robot2_region

            robot1_pose = self.robot1.pose
            robot2_pose = self.robot2.pose

        # ------------------------------------------------------------
        # Wait for map
        # ------------------------------------------------------------

        if global_map is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for /covered_map ..."
            )

            return

        # ------------------------------------------------------------
        # Wait for regions
        # ------------------------------------------------------------

        if region1 is None or region2 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for robot regions ..."
            )

            return

        # ------------------------------------------------------------
        # Wait for robots
        # ------------------------------------------------------------

        if (
            robot1_pose is None or
            robot2_pose is None
        ):

            rospy.logwarn_throttle(
                5.0,
                "Waiting for robot poses ..."
            )

            return

        # ============================================================
        # Initialize
        # ============================================================

        self.initialize_robot(
            self.robot1,
            self.robot1_coverage,
            region1
        )

        self.initialize_robot(
            self.robot2,
            self.robot2_coverage,
            region2
        )

        # ============================================================
        # Robot 1
        # ============================================================

        cmd1 = self.control_robot(
            self.robot1,
            self.robot2,
            region1,
            self.robot1_coverage,
            self.astar1,
            self.dwa1
        )

        # ============================================================
        # Robot 2
        # ============================================================

        cmd2 = self.control_robot(
            self.robot2,
            self.robot1,
            region2,
            self.robot2_coverage,
            self.astar2,
            self.dwa2
        )

        # ============================================================
        # Publish
        # ============================================================

        self.robot1_cmd_pub.publish(
            cmd1
        )

        self.robot2_cmd_pub.publish(
            cmd2
        )

    # ================================================================
    # Shutdown
    # ================================================================

    def stop(self):

        cmd = Twist()

        self.robot1_cmd_pub.publish(
            cmd
        )

        self.robot2_cmd_pub.publish(
            cmd
        )


# ====================================================================
# Main
# ====================================================================

def main():

    manager = None

    try:

        manager = (
            MultiRobotPlannerManager()
        )

        rospy.on_shutdown(
            manager.stop
        )

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()