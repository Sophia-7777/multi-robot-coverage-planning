#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32


class CoverageTracker:

    def __init__(self):

        rospy.init_node(
            "coverage_tracker_node"
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

        self.global_map_topic = rospy.get_param(
            "~global_map_topic",
            "/global_map"
        )

        self.covered_map_topic = rospy.get_param(
            "~covered_map_topic",
            "/covered_map"
        )

        self.coverage_topic = rospy.get_param(
            "~coverage_topic",
            "/coverage_percent"
        )

        # ------------------------------------------------------------
        # Robot footprint / coverage radius
        # ------------------------------------------------------------

        self.robot_radius = rospy.get_param(
            "~robot_radius",
            0.25
        )

        self.coverage_margin = rospy.get_param(
            "~coverage_margin",
            0.0
        )

        self.coverage_radius = (
            self.robot_radius +
            self.coverage_margin
        )

        # ------------------------------------------------------------
        # Free-space threshold
        # ------------------------------------------------------------

        self.free_threshold = rospy.get_param(
            "~free_threshold",
            25
        )

        # ------------------------------------------------------------
        # Update rate
        # ------------------------------------------------------------

        self.publish_rate = rospy.get_param(
            "~publish_rate",
            5.0
        )

        # ------------------------------------------------------------
        # Minimum robot movement before updating coverage
        # ------------------------------------------------------------

        self.min_move_distance = rospy.get_param(
            "~min_move_distance",
            0.02
        )

        # ============================================================
        # Data
        # ============================================================

        self.global_map = None

        self.robot1_pose = None
        self.robot2_pose = None

        self.lock = threading.Lock()

        # ============================================================
        # Coverage state
        # ============================================================

        self.covered_cells = set()

        self.last_robot1_position = None
        self.last_robot2_position = None

        # ============================================================
        # Map information
        # ============================================================

        self.resolution = None

        # ============================================================
        # Subscribers
        # ============================================================

        self.global_map_sub = rospy.Subscriber(
            self.global_map_topic,
            OccupancyGrid,
            self.global_map_callback,
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

        self.covered_map_pub = rospy.Publisher(
            self.covered_map_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True
        )

        self.coverage_pub = rospy.Publisher(
            self.coverage_topic,
            Float32,
            queue_size=1
        )

        # ============================================================
        # Timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.publish_rate
            ),
            self.timer_callback
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo(
            "=================================================="
        )

        rospy.loginfo(
            "Coverage Tracker Started"
        )

        rospy.loginfo(
            "Robot 1             : %s",
            self.robot1_name
        )

        rospy.loginfo(
            "Robot 2             : %s",
            self.robot2_name
        )

        rospy.loginfo(
            "Global map          : %s",
            self.global_map_topic
        )

        rospy.loginfo(
            "Covered map         : %s",
            self.covered_map_topic
        )

        rospy.loginfo(
            "Robot radius        : %.3f m",
            self.robot_radius
        )

        rospy.loginfo(
            "Coverage margin     : %.3f m",
            self.coverage_margin
        )

        rospy.loginfo(
            "Coverage radius     : %.3f m",
            self.coverage_radius
        )

        rospy.loginfo(
            "Free threshold      : %d",
            self.free_threshold
        )

        rospy.loginfo(
            "Small noise filter  : 3x3 morphological closing"
        )

        rospy.loginfo(
            "=================================================="
        )

    # ================================================================
    # Global map callback
    # ================================================================

    def global_map_callback(self, msg):

        with self.lock:

            if self.resolution is None:

                self.resolution = msg.info.resolution

                rospy.loginfo(
                    "Global map resolution: %.4f m/cell",
                    self.resolution
                )

            else:

                if abs(
                    self.resolution -
                    msg.info.resolution
                ) > 1e-6:

                    rospy.logerr_throttle(
                        5.0,
                        "Global map resolution changed: "
                        "%.4f -> %.4f",
                        self.resolution,
                        msg.info.resolution
                    )

                    return

            self.global_map = msg

    # ================================================================
    # Gazebo model states
    # ================================================================

    def model_states_callback(self, msg):

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
                "Cannot find robots in /gazebo/model_states"
            )

            return

        pose1 = msg.pose[index1]
        pose2 = msg.pose[index2]

        with self.lock:

            self.robot1_pose = pose1
            self.robot2_pose = pose2

    # ================================================================
    # Remove small non-free noise
    # ================================================================

    @staticmethod
    def remove_small_free_space_noise(
        free_mask
    ):
        """
        Apply a fixed 3x3 morphological closing:

            dilation -> erosion

        Purpose:
            Fill very small holes / gaps inside free space.

        No ROS parameter is added or modified.

        Input:
            free_mask == True  -> free

        Output:
            processed free mask
        """

        height, width = free_mask.shape

        # ------------------------------------------------------------
        # Dilation
        # ------------------------------------------------------------

        padded = np.pad(
            free_mask,
            pad_width=1,
            mode="constant",
            constant_values=False
        )

        dilated = np.zeros_like(
            free_mask,
            dtype=bool
        )

        for dy in range(3):

            for dx in range(3):

                dilated |= padded[
                    dy:dy + height,
                    dx:dx + width
                ]

        # ------------------------------------------------------------
        # Erosion
        # ------------------------------------------------------------

        padded = np.pad(
            dilated,
            pad_width=1,
            mode="constant",
            constant_values=False
        )

        closed = np.ones_like(
            free_mask,
            dtype=bool
        )

        for dy in range(3):

            for dx in range(3):

                closed &= padded[
                    dy:dy + height,
                    dx:dx + width
                ]

        return closed

    # ================================================================
    # Generate processed free-space mask
    # ================================================================

    def get_free_mask(
        self,
        map_array
    ):
        """
        Generate the final free-space mask.

        First:
            use the original free_threshold.

        Then:
            fill very small non-free noise using 3x3 closing.

        This processed mask is used consistently by:

            1. Coverage marking
            2. Covered map generation
            3. Coverage percentage calculation
        """

        original_free_mask = (
            (map_array >= 0)
            &
            (map_array <= self.free_threshold)
        )

        processed_free_mask = (
            self.remove_small_free_space_noise(
                original_free_mask
            )
        )

        return processed_free_mask

    # ================================================================
    # World position -> world grid coordinate
    # ================================================================

    def world_to_coverage_cell(
        self,
        x,
        y
    ):

        if self.resolution is None:

            return None

        gx = int(
            math.floor(
                x / self.resolution
            )
        )

        gy = int(
            math.floor(
                y / self.resolution
            )
        )

        return gx, gy

    # ================================================================
    # Coverage cell -> world coordinate
    # ================================================================

    def coverage_cell_to_world(
        self,
        gx,
        gy
    ):

        x = (
            gx + 0.5
        ) * self.resolution

        y = (
            gy + 0.5
        ) * self.resolution

        return x, y

    # ================================================================
    # Check whether world point is free
    # ================================================================

    def is_world_point_free(
        self,
        x,
        y,
        free_mask,
        map_msg
    ):
        """
        Check whether a world coordinate corresponds to the
        PROCESSED free-space mask.
        """

        resolution = map_msg.info.resolution

        origin_x = (
            map_msg.info.origin.position.x
        )

        origin_y = (
            map_msg.info.origin.position.y
        )

        width = map_msg.info.width
        height = map_msg.info.height

        col = int(
            math.floor(
                (
                    x -
                    origin_x
                ) / resolution
            )
        )

        row = int(
            math.floor(
                (
                    y -
                    origin_y
                ) / resolution
            )
        )

        if col < 0 or col >= width:

            return False

        if row < 0 or row >= height:

            return False

        return bool(
            free_mask[row, col]
        )

    # ================================================================
    # Mark robot coverage
    # ================================================================

    def mark_robot_coverage(
        self,
        pose,
        free_mask,
        map_msg
    ):

        if pose is None:

            return

        robot_x = pose.position.x
        robot_y = pose.position.y

        radius = self.coverage_radius

        radius_cells = int(
            math.ceil(
                radius /
                self.resolution
            )
        )

        center_gx, center_gy = (
            self.world_to_coverage_cell(
                robot_x,
                robot_y
            )
        )

        for dy in range(
            -radius_cells,
            radius_cells + 1
        ):

            for dx in range(
                -radius_cells,
                radius_cells + 1
            ):

                # ----------------------------------------------------
                # Circular footprint
                # ----------------------------------------------------

                distance = math.sqrt(
                    dx * dx +
                    dy * dy
                ) * self.resolution

                if distance > radius:

                    continue

                # ----------------------------------------------------
                # Coverage cell
                # ----------------------------------------------------

                gx = center_gx + dx
                gy = center_gy + dy

                # ----------------------------------------------------
                # Cell center in world coordinates
                # ----------------------------------------------------

                wx, wy = (
                    self.coverage_cell_to_world(
                        gx,
                        gy
                    )
                )

                # ----------------------------------------------------
                # Only mark processed free space
                # ----------------------------------------------------

                if not self.is_world_point_free(
                    wx,
                    wy,
                    free_mask,
                    map_msg
                ):

                    continue

                # ----------------------------------------------------
                # Store persistent coverage
                # ----------------------------------------------------

                self.covered_cells.add(
                    (
                        gx,
                        gy
                    )
                )

    # ================================================================
    # Distance between positions
    # ================================================================

    @staticmethod
    def position_distance(
        pose,
        previous_position
    ):

        if pose is None:

            return float("inf")

        if previous_position is None:

            return float("inf")

        dx = (
            pose.position.x -
            previous_position[0]
        )

        dy = (
            pose.position.y -
            previous_position[1]
        )

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # ================================================================
    # Publish covered map
    # ================================================================

    def create_covered_map(
        self,
        global_map,
        free_mask
    ):
        """
        Create an OccupancyGrid visualization:

            100 = processed free space + covered
              0 = processed free space + not covered
             -1 = non-free / unknown

        """

        width = global_map.info.width
        height = global_map.info.height

        resolution = (
            global_map.info.resolution
        )

        origin_x = (
            global_map.info.origin.position.x
        )

        origin_y = (
            global_map.info.origin.position.y
        )

        # ------------------------------------------------------------
        # Output
        #
        #   -1 = non-free / unknown
        #    0 = free but not covered
        #  100 = covered free space
        # ------------------------------------------------------------

        covered_array = np.full(
            (
                height,
                width
            ),
            -1,
            dtype=np.int8
        )

        # ------------------------------------------------------------
        # Processed free area
        # ------------------------------------------------------------

        covered_array[
            free_mask
        ] = 0

        # ------------------------------------------------------------
        # Persistent covered cells
        # ------------------------------------------------------------

        for gx, gy in self.covered_cells:

            wx, wy = (
                self.coverage_cell_to_world(
                    gx,
                    gy
                )
            )

            col = int(
                math.floor(
                    (
                        wx -
                        origin_x
                    ) / resolution
                )
            )

            row = int(
                math.floor(
                    (
                        wy -
                        origin_y
                    ) / resolution
                )
            )

            if col < 0 or col >= width:

                continue

            if row < 0 or row >= height:

                continue

            # Only display coverage on processed free space.
            if free_mask[row, col]:

                covered_array[
                    row,
                    col
                ] = 100

        # ------------------------------------------------------------
        # Create OccupancyGrid
        # ------------------------------------------------------------

        msg = OccupancyGrid()

        msg.header.stamp = rospy.Time.now()

        msg.header.frame_id = (
            global_map.header.frame_id
        )

        msg.info = global_map.info

        msg.data = (
            covered_array
            .flatten()
            .astype(np.int8)
            .tolist()
        )

        return msg

    # ================================================================
    # Calculate coverage percentage
    # ================================================================

    def calculate_coverage_percentage(
        self,
        global_map,
        free_mask
    ):

        width = global_map.info.width
        height = global_map.info.height

        resolution = (
            global_map.info.resolution
        )

        # ------------------------------------------------------------
        # Target free space
        #
        # IMPORTANT:
        # Use processed free mask, not original map.
        # ------------------------------------------------------------

        target_cells = int(
            np.count_nonzero(
                free_mask
            )
        )

        if target_cells == 0:

            return 0.0

        # ------------------------------------------------------------
        # Count covered cells
        # ------------------------------------------------------------

        covered_count = 0

        origin_x = (
            global_map.info.origin.position.x
        )

        origin_y = (
            global_map.info.origin.position.y
        )

        for gx, gy in self.covered_cells:

            wx, wy = (
                self.coverage_cell_to_world(
                    gx,
                    gy
                )
            )

            col = int(
                math.floor(
                    (
                        wx -
                        origin_x
                    ) / resolution
                )
            )

            row = int(
                math.floor(
                    (
                        wy -
                        origin_y
                    ) / resolution
                )
            )

            if col < 0 or col >= width:

                continue

            if row < 0 or row >= height:

                continue

            if free_mask[row, col]:

                covered_count += 1

        # ------------------------------------------------------------
        # Coverage percentage
        # ------------------------------------------------------------

        coverage = (
            100.0 *
            covered_count /
            target_cells
        )

        return coverage

    # ================================================================
    # Timer
    # ================================================================

    def timer_callback(
        self,
        event
    ):

        # ------------------------------------------------------------
        # Copy data
        # ------------------------------------------------------------

        with self.lock:

            global_map = self.global_map

            pose1 = self.robot1_pose
            pose2 = self.robot2_pose

        # ------------------------------------------------------------
        # Check global map
        # ------------------------------------------------------------

        if global_map is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for /global_map ..."
            )

            return

        # ------------------------------------------------------------
        # Check robot poses
        # ------------------------------------------------------------

        if pose1 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for Robot1 pose ..."
            )

            return

        if pose2 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for Robot2 pose ..."
            )

            return

        # ------------------------------------------------------------
        # Convert global map
        # ------------------------------------------------------------

        width = global_map.info.width
        height = global_map.info.height

        map_array = np.asarray(
            global_map.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ============================================================
        # Generate processed free-space mask
        # ============================================================

        free_mask = self.get_free_mask(
            map_array
        )

        # ============================================================
        # Robot 1
        # ============================================================

        distance1 = (
            self.position_distance(
                pose1,
                self.last_robot1_position
            )
        )

        if (
            self.last_robot1_position is None
            or
            distance1 >= self.min_move_distance
        ):

            with self.lock:

                self.mark_robot_coverage(
                    pose1,
                    free_mask,
                    global_map
                )

            self.last_robot1_position = (
                pose1.position.x,
                pose1.position.y
            )

        # ============================================================
        # Robot 2
        # ============================================================

        distance2 = (
            self.position_distance(
                pose2,
                self.last_robot2_position
            )
        )

        if (
            self.last_robot2_position is None
            or
            distance2 >= self.min_move_distance
        ):

            with self.lock:

                self.mark_robot_coverage(
                    pose2,
                    free_mask,
                    global_map
                )

            self.last_robot2_position = (
                pose2.position.x,
                pose2.position.y
            )

        # ============================================================
        # Create covered map
        # ============================================================

        with self.lock:

            covered_map = (
                self.create_covered_map(
                    global_map,
                    free_mask
                )
            )

            coverage = (
                self.calculate_coverage_percentage(
                    global_map,
                    free_mask
                )
            )

            covered_cell_count = len(
                self.covered_cells
            )

        # ============================================================
        # Publish
        # ============================================================

        self.covered_map_pub.publish(
            covered_map
        )

        self.coverage_pub.publish(
            Float32(data=coverage)
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo_throttle(
            2.0,
            "Coverage: %.2f %% | "
            "Covered cells: %d | "
            "Robot1 move: %.3f m | "
            "Robot2 move: %.3f m",
            coverage,
            covered_cell_count,
            distance1,
            distance2
        )


# ====================================================================
# Main
# ====================================================================

def main():

    try:

        node = CoverageTracker()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()

