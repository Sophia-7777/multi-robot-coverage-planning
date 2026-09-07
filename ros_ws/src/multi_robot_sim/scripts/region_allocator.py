#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq
import threading

import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates


class RegionAllocator:

    def __init__(self):

        rospy.init_node(
            "region_allocator_node"
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

        self.covered_map_topic = rospy.get_param(
            "~covered_map_topic",
            "/covered_map"
        )

        self.region_map_topic = rospy.get_param(
            "~region_map_topic",
            "/region_map"
        )

        self.robot1_region_topic = rospy.get_param(
            "~robot1_region_topic",
            "/robot1_region"
        )

        self.robot2_region_topic = rospy.get_param(
            "~robot2_region_topic",
            "/robot2_region"
        )

        # ------------------------------------------------------------
        # Update rate
        # ------------------------------------------------------------

        self.update_rate = rospy.get_param(
            "~update_rate",
            1.0
        )

        # ------------------------------------------------------------
        # Reallocate when robot moves
        #
        # IMPORTANT:
        # default = False
        #
        # Initial region allocation is performed only once.
        # ------------------------------------------------------------

        self.reallocate_on_robot_move = rospy.get_param(
            "~reallocate_on_robot_move",
            False
        )

        self.min_robot_move = rospy.get_param(
            "~min_robot_move",
            1.0
        )

        # ============================================================
        # Data
        # ============================================================

        self.covered_map = None

        self.robot1_pose = None
        self.robot2_pose = None

        self.lock = threading.Lock()

        # ============================================================
        # Allocation state
        # ============================================================

        self.region_assigned = False

        self.region_labels = None

        self.last_robot1_position = None
        self.last_robot2_position = None

        # ============================================================
        # Subscribers
        # ============================================================

        self.covered_map_sub = rospy.Subscriber(
            self.covered_map_topic,
            OccupancyGrid,
            self.covered_map_callback,
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

        self.region_map_pub = rospy.Publisher(
            self.region_map_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True
        )

        self.robot1_region_pub = rospy.Publisher(
            self.robot1_region_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True
        )

        self.robot2_region_pub = rospy.Publisher(
            self.robot2_region_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True
        )

        # ============================================================
        # Timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.update_rate
            ),
            self.timer_callback
        )

        rospy.loginfo(
            "=================================================="
        )

        rospy.loginfo(
            "Fast Region Allocator Started"
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
            "Input map           : %s",
            self.covered_map_topic
        )

        rospy.loginfo(
            "Region map          : %s",
            self.region_map_topic
        )

        rospy.loginfo(
            "Reallocate on move  : %s",
            self.reallocate_on_robot_move
        )

        rospy.loginfo(
            "=================================================="
        )

    # ================================================================
    # Covered map
    # ================================================================

    def covered_map_callback(
        self,
        msg
    ):

        with self.lock:

            self.covered_map = msg

    # ================================================================
    # Robot poses
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
                "Cannot find robots in /gazebo/model_states"
            )

            return

        with self.lock:

            self.robot1_pose = msg.pose[index1]
            self.robot2_pose = msg.pose[index2]

    # ================================================================
    # World -> Grid
    # ================================================================

    @staticmethod
    def world_to_grid(
        x,
        y,
        map_msg
    ):

        resolution = map_msg.info.resolution

        origin_x = (
            map_msg.info.origin.position.x
        )

        origin_y = (
            map_msg.info.origin.position.y
        )

        gx = int(
            math.floor(
                (x - origin_x) /
                resolution
            )
        )

        gy = int(
            math.floor(
                (y - origin_y) /
                resolution
            )
        )

        return gx, gy

    # ================================================================
    # Grid validity
    # ================================================================

    @staticmethod
    def valid_cell(
        gx,
        gy,
        width,
        height
    ):

        return (
            0 <= gx < width
            and
            0 <= gy < height
        )

    # ================================================================
    # Robot cell
    # ================================================================

    def get_robot_cell(
        self,
        pose,
        map_msg,
        free_mask
    ):

        if pose is None:

            return None

        gx, gy = self.world_to_grid(
            pose.position.x,
            pose.position.y,
            map_msg
        )

        height, width = (
            free_mask.shape
        )

        if not self.valid_cell(
            gx,
            gy,
            width,
            height
        ):

            return None

        if free_mask[
            gy,
            gx
        ]:

            return gx, gy

        # ------------------------------------------------------------
        # Search nearby free cell
        # ------------------------------------------------------------

        search_radius = 10

        for radius in range(
            1,
            search_radius + 1
        ):

            for dy in range(
                -radius,
                radius + 1
            ):

                for dx in range(
                    -radius,
                    radius + 1
                ):

                    nx = gx + dx
                    ny = gy + dy

                    if not self.valid_cell(
                        nx,
                        ny,
                        width,
                        height
                    ):

                        continue

                    if free_mask[
                        ny,
                        nx
                    ]:

                        return nx, ny

        return None

    # ================================================================
    # Multi-source Dijkstra
    # ================================================================

    def geodesic_voronoi(
        self,
        free_mask,
        robot1_cell,
        robot2_cell
    ):

        height, width = (
            free_mask.shape
        )

        # ------------------------------------------------------------
        # Distance
        # ------------------------------------------------------------

        distance = np.full(
            (
                height,
                width
            ),
            np.inf,
            dtype=np.float32
        )

        # ------------------------------------------------------------
        # Labels
        #
        # 0 = not assigned
        # 1 = Robot 1
        # 2 = Robot 2
        # ------------------------------------------------------------

        labels = np.zeros(
            (
                height,
                width
            ),
            dtype=np.uint8
        )

        queue = []

        # ------------------------------------------------------------
        # Robot 1
        # ------------------------------------------------------------

        gx, gy = robot1_cell

        distance[
            gy,
            gx
        ] = 0.0

        labels[
            gy,
            gx
        ] = 1

        heapq.heappush(
            queue,
            (
                0.0,
                1,
                gx,
                gy
            )
        )

        # ------------------------------------------------------------
        # Robot 2
        # ------------------------------------------------------------

        gx, gy = robot2_cell

        if labels[
            gy,
            gx
        ] == 0:

            distance[
                gy,
                gx
            ] = 0.0

            labels[
                gy,
                gx
            ] = 2

            heapq.heappush(
                queue,
                (
                    0.0,
                    2,
                    gx,
                    gy
                )
            )

        # ------------------------------------------------------------
        # Dijkstra
        # ------------------------------------------------------------

        processed = 0

        while queue:

            current_distance, robot_id, gx, gy = (
                heapq.heappop(queue)
            )

            if current_distance != distance[
                gy,
                gx
            ]:

                continue

            processed += 1

            if processed % 100000 == 0:

                rospy.loginfo(
                    "Dijkstra processed: %d cells",
                    processed
                )

            # --------------------------------------------------------
            # 4-connected neighbours
            # --------------------------------------------------------

            neighbors = (
                (gx + 1, gy),
                (gx - 1, gy),
                (gx, gy + 1),
                (gx, gy - 1)
            )

            for nx, ny in neighbors:

                if not self.valid_cell(
                    nx,
                    ny,
                    width,
                    height
                ):

                    continue

                if not free_mask[
                    ny,
                    nx
                ]:

                    continue

                new_distance = (
                    current_distance +
                    1.0
                )

                if new_distance < distance[
                    ny,
                    nx
                ]:

                    distance[
                        ny,
                        nx
                    ] = new_distance

                    labels[
                        ny,
                        nx
                    ] = robot_id

                    heapq.heappush(
                        queue,
                        (
                            new_distance,
                            robot_id,
                            nx,
                            ny
                        )
                    )

        rospy.loginfo(
            "Dijkstra finished. "
            "Processed: %d cells",
            processed
        )

        return labels

    # ================================================================
    # Create region map
    # ================================================================

    @staticmethod
    def create_region_map(
        source_map,
        labels
    ):

        height = source_map.info.height
        width = source_map.info.width

        region_array = np.full(
            (
                height,
                width
            ),
            -1,
            dtype=np.int8
        )

        # Robot 1
        region_array[
            labels == 1
        ] = 0

        # Robot 2
        region_array[
            labels == 2
        ] = 100

        msg = OccupancyGrid()

        msg.header.stamp = rospy.Time.now()

        msg.header.frame_id = (
            source_map.header.frame_id
        )

        msg.info = source_map.info

        msg.data = (
            region_array
            .flatten()
            .tolist()
        )

        return msg

    # ================================================================
    # Individual robot region
    # ================================================================

    @staticmethod
    def create_single_region(
        source_map,
        labels,
        robot_id
    ):

        height = source_map.info.height
        width = source_map.info.width

        array = np.full(
            (
                height,
                width
            ),
            -1,
            dtype=np.int8
        )

        # Other free regions = 0
        array[
            labels > 0
        ] = 0

        # Own region = 100
        array[
            labels == robot_id
        ] = 100

        msg = OccupancyGrid()

        msg.header.stamp = rospy.Time.now()

        msg.header.frame_id = (
            source_map.header.frame_id
        )

        msg.info = source_map.info

        msg.data = (
            array
            .flatten()
            .tolist()
        )

        return msg

    # ================================================================
    # Allocate
    # ================================================================

    def allocate(
        self,
        covered_map,
        pose1,
        pose2
    ):

        width = covered_map.info.width
        height = covered_map.info.height

        map_array = np.asarray(
            covered_map.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ------------------------------------------------------------
        # 0 and 100 are free space.
        # -1 is obstacle / unknown.
        # ------------------------------------------------------------

        free_mask = (
            map_array >= 0
        )

        free_count = int(
            np.count_nonzero(
                free_mask
            )
        )

        rospy.loginfo(
            "Free-space cells: %d",
            free_count
        )

        # ------------------------------------------------------------
        # Robot positions
        # ------------------------------------------------------------

        robot1_cell = (
            self.get_robot_cell(
                pose1,
                covered_map,
                free_mask
            )
        )

        robot2_cell = (
            self.get_robot_cell(
                pose2,
                covered_map,
                free_mask
            )
        )

        if robot1_cell is None:

            rospy.logerr(
                "Robot1 is not inside free space."
            )

            return None

        if robot2_cell is None:

            rospy.logerr(
                "Robot2 is not inside free space."
            )

            return None

        rospy.loginfo(
            "Robot1 cell: (%d, %d)",
            robot1_cell[0],
            robot1_cell[1]
        )

        rospy.loginfo(
            "Robot2 cell: (%d, %d)",
            robot2_cell[0],
            robot2_cell[1]
        )

        # ============================================================
        # Geodesic Voronoi
        # ============================================================

        labels = self.geodesic_voronoi(
            free_mask,
            robot1_cell,
            robot2_cell
        )

        # ============================================================
        # Statistics
        # ============================================================

        robot1_count = int(
            np.count_nonzero(
                labels == 1
            )
        )

        robot2_count = int(
            np.count_nonzero(
                labels == 2
            )
        )

        total = (
            robot1_count +
            robot2_count
        )

        if total > 0:

            robot1_percent = (
                100.0 *
                robot1_count /
                total
            )

            robot2_percent = (
                100.0 *
                robot2_count /
                total
            )

        else:

            robot1_percent = 0.0
            robot2_percent = 0.0

        rospy.loginfo(
            "=================================================="
        )

        rospy.loginfo(
            "Region allocation finished"
        )

        rospy.loginfo(
            "Robot1: %d cells (%.2f%%)",
            robot1_count,
            robot1_percent
        )

        rospy.loginfo(
            "Robot2: %d cells (%.2f%%)",
            robot2_count,
            robot2_percent
        )

        rospy.loginfo(
            "=================================================="
        )

        return labels

    # ================================================================
    # Timer
    # ================================================================

    def timer_callback(
        self,
        event
    ):

        with self.lock:

            covered_map = self.covered_map

            pose1 = self.robot1_pose
            pose2 = self.robot2_pose

        # ------------------------------------------------------------
        # Wait for map
        # ------------------------------------------------------------

        if covered_map is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for /covered_map ..."
            )

            return

        # ------------------------------------------------------------
        # Wait for robots
        # ------------------------------------------------------------

        if pose1 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for Robot1 ..."
            )

            return

        if pose2 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for Robot2 ..."
            )

            return

        # ============================================================
        # Initial allocation
        # ============================================================

        if not self.region_assigned:

            rospy.loginfo(
                "Starting initial region allocation..."
            )

            labels = self.allocate(
                covered_map,
                pose1,
                pose2
            )

            if labels is None:

                return

            self.region_labels = labels

            self.region_assigned = True

            # --------------------------------------------------------
            # Publish
            # --------------------------------------------------------

            region_map = (
                self.create_region_map(
                    covered_map,
                    labels
                )
            )

            robot1_region = (
                self.create_single_region(
                    covered_map,
                    labels,
                    1
                )
            )

            robot2_region = (
                self.create_single_region(
                    covered_map,
                    labels,
                    2
                )
            )

            self.region_map_pub.publish(
                region_map
            )

            self.robot1_region_pub.publish(
                robot1_region
            )

            self.robot2_region_pub.publish(
                robot2_region
            )

            self.last_robot1_position = (
                pose1.position.x,
                pose1.position.y
            )

            self.last_robot2_position = (
                pose2.position.x,
                pose2.position.y
            )

            rospy.loginfo(
                "Region maps published."
            )

            return

        # ============================================================
        # Optional dynamic reallocation
        # ============================================================

        if self.reallocate_on_robot_move:

            dx1 = (
                pose1.position.x -
                self.last_robot1_position[0]
            )

            dy1 = (
                pose1.position.y -
                self.last_robot1_position[1]
            )

            dx2 = (
                pose2.position.x -
                self.last_robot2_position[0]
            )

            dy2 = (
                pose2.position.y -
                self.last_robot2_position[1]
            )

            distance1 = math.sqrt(
                dx1 * dx1 +
                dy1 * dy1
            )

            distance2 = math.sqrt(
                dx2 * dx2 +
                dy2 * dy2
            )

            if (
                distance1 >= self.min_robot_move
                or
                distance2 >= self.min_robot_move
            ):

                rospy.loginfo(
                    "Robot moved enough. "
                    "Reallocating regions..."
                )

                labels = self.allocate(
                    covered_map,
                    pose1,
                    pose2
                )

                if labels is None:

                    return

                self.region_labels = labels

                region_map = (
                    self.create_region_map(
                        covered_map,
                        labels
                    )
                )

                robot1_region = (
                    self.create_single_region(
                        covered_map,
                        labels,
                        1
                    )
                )

                robot2_region = (
                    self.create_single_region(
                        covered_map,
                        labels,
                        2
                    )
                )

                self.region_map_pub.publish(
                    region_map
                )

                self.robot1_region_pub.publish(
                    robot1_region
                )

                self.robot2_region_pub.publish(
                    robot2_region
                )

                self.last_robot1_position = (
                    pose1.position.x,
                    pose1.position.y
                )

                self.last_robot2_position = (
                    pose2.position.x,
                    pose2.position.y
                )


# ====================================================================
# Main
# ====================================================================

def main():

    try:

        node = RegionAllocator()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()