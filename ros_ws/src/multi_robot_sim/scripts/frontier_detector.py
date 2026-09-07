#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque

import numpy as np
import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class FrontierDetector:

    def __init__(self):

        rospy.init_node("frontier_detector")

        self.map_msg = None

        # =====================================================
        # Parameters
        # =====================================================

        # -----------------------------------------------------
        # Frontier 最小连通区域
        # -----------------------------------------------------

        self.min_frontier_size = rospy.get_param(
            "~min_frontier_size",
            20
        )

        # -----------------------------------------------------
        # Frontier 最大/最小跨度
        #
        # 注意：
        # 不再要求 width 和 height 都 >= 4
        #
        # 因为真实 frontier 很可能：
        #
        #   ################
        #
        # 是一条很长但很薄的区域。
        # -----------------------------------------------------

        self.min_frontier_span = rospy.get_param(
            "~min_frontier_span",
            8
        )

        # -----------------------------------------------------
        # Unknown 连通区域最小面积
        #
        # 这是本次最重要的过滤参数。
        #
        # 389x389 / 0.05m resolution 下：
        #
        # 100 cells = 0.25 m²
        # 200 cells = 0.50 m²
        # 400 cells = 1.00 m²
        #
        # 小于这个面积的 unknown 区域，
        # 大概率是 SLAM 噪点/孔洞。
        # -----------------------------------------------------

        self.min_unknown_region_size = rospy.get_param(
            "~min_unknown_region_size",
            100
        )

        # -----------------------------------------------------
        # Unknown 区域使用的连通方式
        # -----------------------------------------------------

        self.use_diagonal_connectivity = rospy.get_param(
            "~use_diagonal_connectivity",
            True
        )

        # -----------------------------------------------------
        # Free threshold
        # -----------------------------------------------------

        self.free_threshold = rospy.get_param(
            "~free_threshold",
            20
        )

        # -----------------------------------------------------
        # 一个 frontier cell 周围最少 unknown 数量
        # -----------------------------------------------------

        self.min_unknown_neighbors = rospy.get_param(
            "~min_unknown_neighbors",
            2
        )

        # -----------------------------------------------------
        # Frontier marker size
        # -----------------------------------------------------

        self.marker_point_size = rospy.get_param(
            "~marker_point_size",
            0.06
        )

        # -----------------------------------------------------
        # Publish rate
        # -----------------------------------------------------

        self.publish_rate = rospy.get_param(
            "~publish_rate",
            1.0
        )

        # =====================================================
        # Subscriber
        # =====================================================

        rospy.Subscriber(
            "/global_map",
            OccupancyGrid,
            self.map_callback,
            queue_size=1
        )

        # =====================================================
        # Publishers
        # =====================================================

        self.frontier_marker_pub = rospy.Publisher(
            "/frontiers",
            Marker,
            queue_size=1
        )

        self.frontier_points_pub = rospy.Publisher(
            "/frontier_points",
            Marker,
            queue_size=1
        )

        # =====================================================
        # Timer
        # =====================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 / self.publish_rate
            ),
            self.process
        )

        # =====================================================
        # Log
        # =====================================================

        rospy.loginfo(
            "======================================"
        )

        rospy.loginfo(
            "Frontier Detection Node"
        )

        rospy.loginfo(
            "======================================"
        )

        rospy.loginfo(
            "min_frontier_size       = %d cells",
            self.min_frontier_size
        )

        rospy.loginfo(
            "min_frontier_span       = %d cells",
            self.min_frontier_span
        )

        rospy.loginfo(
            "min_unknown_region_size = %d cells",
            self.min_unknown_region_size
        )

        rospy.loginfo(
            "min_unknown_neighbors   = %d",
            self.min_unknown_neighbors
        )

        rospy.loginfo(
            "free_threshold          = %d",
            self.free_threshold
        )

    # =========================================================
    # Map callback
    # =========================================================

    def map_callback(self, msg):

        self.map_msg = msg

    # =========================================================
    # Get connectivity
    # =========================================================

    def get_directions(self):

        if self.use_diagonal_connectivity:

            return [
                (-1, -1),
                (-1, 0),
                (-1, 1),

                (0, -1),
                (0, 1),

                (1, -1),
                (1, 0),
                (1, 1)
            ]

        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

    # =========================================================
    # Find unknown connected components
    # =========================================================

    def find_unknown_components(
        self,
        unknown
    ):
        """
        对 unknown 区域做连通域分析。

        返回：
            labels[y, x] = unknown component ID
            component_sizes[id] = component size

        小的 unknown island 可以直接被认为是 SLAM 噪声。
        """

        height, width = unknown.shape

        labels = np.full(
            (height, width),
            -1,
            dtype=np.int32
        )

        directions = self.get_directions()

        component_sizes = []

        component_id = 0

        for y in range(1, height - 1):

            for x in range(1, width - 1):

                if not unknown[y, x]:
                    continue

                if labels[y, x] != -1:
                    continue

                queue = deque()

                queue.append(
                    (x, y)
                )

                labels[y, x] = component_id

                size = 0

                while queue:

                    cx, cy = queue.popleft()

                    size += 1

                    for dx, dy in directions:

                        nx = cx + dx
                        ny = cy + dy

                        if nx < 1:
                            continue

                        if nx >= width - 1:
                            continue

                        if ny < 1:
                            continue

                        if ny >= height - 1:
                            continue

                        if not unknown[ny, nx]:
                            continue

                        if labels[ny, nx] != -1:
                            continue

                        labels[ny, nx] = component_id

                        queue.append(
                            (nx, ny)
                        )

                component_sizes.append(size)

                component_id += 1

        return labels, component_sizes

    # =========================================================
    # Detect frontier
    # =========================================================

    def detect_frontiers(self, msg):

        width = msg.info.width
        height = msg.info.height

        data = np.asarray(
            msg.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # =====================================================
        # Cell classification
        # =====================================================

        free = (
            (data >= 0) &
            (data < self.free_threshold)
        )

        unknown = (
            data == -1
        )

        # =====================================================
        # STEP 1
        # Find unknown connected regions
        # =====================================================

        unknown_labels, unknown_sizes = (
            self.find_unknown_components(
                unknown
            )
        )

        # =====================================================
        # STEP 2
        # Build valid unknown mask
        # =====================================================

        valid_unknown = np.zeros(
            (height, width),
            dtype=bool
        )

        valid_unknown_component_count = 0

        for component_id, size in enumerate(
            unknown_sizes
        ):

            if (
                size >=
                self.min_unknown_region_size
            ):

                valid_unknown[
                    unknown_labels ==
                    component_id
                ] = True

                valid_unknown_component_count += 1

        # =====================================================
        # STEP 3
        # Frontier candidate
        # =====================================================

        frontier = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        directions = self.get_directions()

        # -----------------------------------------------------
        # 这里不再使用简单的 3x3 any unknown。
        #
        # 必须：
        #
        # 1. 自己是 free
        # 2. 周围存在 valid unknown
        # 3. valid unknown 数量 >= min_unknown_neighbors
        #
        # 这样一个孤立 -1 噪点不会产生 frontier。
        # -----------------------------------------------------

        for y in range(1, height - 1):

            for x in range(1, width - 1):

                if not free[y, x]:
                    continue

                valid_unknown_count = 0

                for dx, dy in directions:

                    nx = x + dx
                    ny = y + dy

                    if valid_unknown[ny, nx]:

                        valid_unknown_count += 1

                if (
                    valid_unknown_count >=
                    self.min_unknown_neighbors
                ):

                    frontier[y, x] = 1

        # =====================================================
        # STEP 4
        # Filter frontier components
        # =====================================================

        filtered_frontier = (
            self.filter_frontier_components(
                frontier
            )
        )

        # =====================================================
        # Debug
        # =====================================================

        rospy.loginfo_throttle(
            5.0,
            "Unknown components: total=%d, "
            "valid=%d",
            len(unknown_sizes),
            valid_unknown_component_count
        )

        return filtered_frontier

    # =========================================================
    # Filter frontier connected components
    # =========================================================

    def filter_frontier_components(
        self,
        frontier
    ):

        height, width = frontier.shape

        visited = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        filtered = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        directions = self.get_directions()

        total_components = 0
        kept_components = 0
        removed_components = 0

        for y in range(1, height - 1):

            for x in range(1, width - 1):

                if frontier[y, x] == 0:
                    continue

                if visited[y, x]:
                    continue

                queue = deque()

                queue.append(
                    (x, y)
                )

                visited[y, x] = 1

                component = []

                min_x = x
                max_x = x

                min_y = y
                max_y = y

                while queue:

                    cx, cy = queue.popleft()

                    component.append(
                        (cx, cy)
                    )

                    min_x = min(
                        min_x,
                        cx
                    )

                    max_x = max(
                        max_x,
                        cx
                    )

                    min_y = min(
                        min_y,
                        cy
                    )

                    max_y = max(
                        max_y,
                        cy
                    )

                    for dx, dy in directions:

                        nx = cx + dx
                        ny = cy + dy

                        if nx < 1:
                            continue

                        if nx >= width - 1:
                            continue

                        if ny < 1:
                            continue

                        if ny >= height - 1:
                            continue

                        if visited[ny, nx]:
                            continue

                        if frontier[ny, nx] == 0:
                            continue

                        visited[ny, nx] = 1

                        queue.append(
                            (nx, ny)
                        )

                # =================================================
                # Statistics
                # =================================================

                total_components += 1

                size = len(component)

                component_width = (
                    max_x - min_x + 1
                )

                component_height = (
                    max_y - min_y + 1
                )

                component_span = max(
                    component_width,
                    component_height
                )

                # =================================================
                # Filtering
                # =================================================

                keep = True

                # -------------------------------------------------
                # Filter 1:
                # area
                # -------------------------------------------------

                if size < self.min_frontier_size:

                    keep = False

                # -------------------------------------------------
                # Filter 2:
                # maximum physical/grid span
                #
                # 不再要求 width AND height。
                # -------------------------------------------------

                if (
                    component_span <
                    self.min_frontier_span
                ):

                    keep = False

                # =================================================
                # Save
                # =================================================

                if keep:

                    kept_components += 1

                    for px, py in component:

                        filtered[
                            py,
                            px
                        ] = 1

                else:

                    removed_components += 1

        # =====================================================
        # Debug
        # =====================================================

        rospy.loginfo_throttle(
            5.0,
            "Frontier components: "
            "total=%d, kept=%d, removed=%d",
            total_components,
            kept_components,
            removed_components
        )

        return filtered

    # =========================================================
    # Convert frontier to world coordinates
    # =========================================================

    def frontier_to_points(
        self,
        frontier,
        msg
    ):

        ys, xs = np.where(
            frontier > 0
        )

        points = []

        resolution = msg.info.resolution

        origin_x = (
            msg.info.origin.position.x
        )

        origin_y = (
            msg.info.origin.position.y
        )

        for y, x in zip(ys, xs):

            px = (
                origin_x +
                (x + 0.5) *
                resolution
            )

            py = (
                origin_y +
                (y + 0.5) *
                resolution
            )

            points.append(
                (px, py)
            )

        return points

    # =========================================================
    # Publish frontier
    # =========================================================

    def publish_frontiers(
        self,
        points,
        msg
    ):

        # =====================================================
        # Red frontier marker
        # =====================================================

        marker = Marker()

        marker.header = msg.header
        marker.header.stamp = rospy.Time.now()

        marker.ns = "frontiers"
        marker.id = 0

        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = (
            self.marker_point_size
        )

        marker.scale.y = (
            self.marker_point_size
        )

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        for x, y in points:

            p = Point()

            p.x = x
            p.y = y
            p.z = 0.05

            marker.points.append(p)

        self.frontier_marker_pub.publish(
            marker
        )

        # =====================================================
        # Green frontier points
        # =====================================================

        points_marker = Marker()

        points_marker.header = msg.header
        points_marker.header.stamp = rospy.Time.now()

        points_marker.ns = "frontier_points"
        points_marker.id = 0

        points_marker.type = Marker.POINTS
        points_marker.action = Marker.ADD

        points_marker.pose.orientation.w = 1.0

        points_marker.scale.x = 0.04
        points_marker.scale.y = 0.04

        points_marker.color.a = 1.0
        points_marker.color.r = 0.0
        points_marker.color.g = 1.0
        points_marker.color.b = 0.0

        for x, y in points:

            p = Point()

            p.x = x
            p.y = y
            p.z = 0.06

            points_marker.points.append(p)

        self.frontier_points_pub.publish(
            points_marker
        )

    # =========================================================
    # Delete markers
    # =========================================================

    def delete_markers(self):

        for publisher, namespace in [
            (
                self.frontier_marker_pub,
                "frontiers"
            ),
            (
                self.frontier_points_pub,
                "frontier_points"
            )
        ]:

            marker = Marker()

            marker.header.frame_id = (
                self.map_msg.header.frame_id
            )

            marker.header.stamp = rospy.Time.now()

            marker.ns = namespace
            marker.id = 0

            marker.action = Marker.DELETE

            publisher.publish(marker)

    # =========================================================
    # Main process
    # =========================================================

    def process(self, event):

        if self.map_msg is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for /global_map ..."
            )

            return

        # =====================================================
        # Detect
        # =====================================================

        frontier = self.detect_frontiers(
            self.map_msg
        )

        # =====================================================
        # Convert
        # =====================================================

        points = self.frontier_to_points(
            frontier,
            self.map_msg
        )

        # =====================================================
        # No frontier
        # =====================================================

        if len(points) == 0:

            rospy.logwarn_throttle(
                5.0,
                "No valid frontier detected."
            )

            self.delete_markers()

            return

        # =====================================================
        # Log
        # =====================================================

        rospy.loginfo(
            "Valid frontier cells: %d",
            len(points)
        )

        # =====================================================
        # Publish
        # =====================================================

        self.publish_frontiers(
            points,
            self.map_msg
        )


# =================================================================
# Main
# =================================================================

def main():

    try:

        node = FrontierDetector()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


# =================================================================
# Entry
# =================================================================

if __name__ == "__main__":

    main()