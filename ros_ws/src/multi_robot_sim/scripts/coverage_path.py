#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid


class CoveragePlanner:

    """
    Generate coverage waypoints for ONE robot.

    Input:
        OccupancyGrid region map

    Region map convention from RegionAllocator:
        -1  : unknown / invalid
         0  : other robot's region
       100  : current robot's region

    Output:
        List of (gx, gy) grid cells.
    """

    def __init__(
        self,
        robot_id,
        robot_width=0.45,
        coverage_width=0.50,
        waypoint_spacing=0.30
    ):

        self.robot_id = robot_id

        self.robot_width = robot_width
        self.coverage_width = coverage_width
        self.waypoint_spacing = waypoint_spacing

        self.region_map = None

        rospy.loginfo(
            "[CoveragePlanner %d] initialized",
            self.robot_id
        )

    # ================================================================
    # Map callback
    # ================================================================

    def update_map(self, msg):

        self.region_map = msg

    # ================================================================
    # Grid validity
    # ================================================================

    @staticmethod
    def valid_cell(gx, gy, width, height):

        return (
            0 <= gx < width and
            0 <= gy < height
        )

    # ================================================================
    # Extract own region
    # ================================================================

    def get_region_mask(self):

        if self.region_map is None:
            return None

        width = self.region_map.info.width
        height = self.region_map.info.height

        array = np.asarray(
            self.region_map.data,
            dtype=np.int16
        ).reshape(height, width)

        # Own region = 100
        region_mask = (
            array == 100
        )

        return region_mask

    # ================================================================
    # Find horizontal segments
    # ================================================================

    def find_horizontal_segments(
        self,
        region_mask,
        gy
    ):

        height, width = region_mask.shape

        if gy < 0 or gy >= height:
            return []

        segments = []

        in_segment = False
        start_x = None

        for gx in range(width):

            inside = region_mask[
                gy,
                gx
            ]

            if inside and not in_segment:

                start_x = gx
                in_segment = True

            elif not inside and in_segment:

                end_x = gx - 1

                if end_x >= start_x:
                    segments.append(
                        (
                            start_x,
                            end_x,
                            gy
                        )
                    )

                in_segment = False

        # Segment reaches map boundary
        if in_segment:

            end_x = width - 1

            if end_x >= start_x:

                segments.append(
                    (
                        start_x,
                        end_x,
                        gy
                    )
                )

        return segments

    # ================================================================
    # Generate scan rows
    # ================================================================

    def generate_scan_rows(
        self,
        region_mask
    ):

        height, width = region_mask.shape

        resolution = (
            self.region_map.info.resolution
        )

        # ------------------------------------------------------------
        # Distance between scan lines
        # ------------------------------------------------------------

        spacing = max(
            self.coverage_width,
            resolution
        )

        row_step = max(
            1,
            int(
                round(
                    spacing /
                    resolution
                )
            )
        )

        # ------------------------------------------------------------
        # Find region bounding box
        # ------------------------------------------------------------

        ys, xs = np.where(
            region_mask
        )

        if len(xs) == 0:
            return []

        min_y = int(np.min(ys))
        max_y = int(np.max(ys))

        rows = list(
            range(
                min_y,
                max_y + 1,
                row_step
            )
        )

        # Ensure last row is included
        if rows[-1] != max_y:

            rows.append(max_y)

        return rows

    # ================================================================
    # Convert segment to waypoint sequence
    # ================================================================

    def segment_waypoints(
        self,
        segment
    ):

        start_x, end_x, gy = segment

        resolution = (
            self.region_map.info.resolution
        )

        spacing = max(
            self.waypoint_spacing,
            resolution
        )

        point_step = max(
            1,
            int(
                round(
                    spacing /
                    resolution
                )
            )
        )

        points = []

        length = end_x - start_x

        if length <= 0:

            return [
                (start_x, gy)
            ]

        x_values = list(
            range(
                start_x,
                end_x + 1,
                point_step
            )
        )

        if x_values[-1] != end_x:
            x_values.append(end_x)

        for x in x_values:

            points.append(
                (x, gy)
            )

        return points

    # ================================================================
    # Generate coverage path
    # ================================================================

    def generate_coverage_path(self):

        region_mask = self.get_region_mask()

        if region_mask is None:

            rospy.logwarn(
                "[CoveragePlanner %d] "
                "No region map.",
                self.robot_id
            )

            return []

        if not np.any(region_mask):

            rospy.logwarn(
                "[CoveragePlanner %d] "
                "Own region is empty.",
                self.robot_id
            )

            return []

        rows = self.generate_scan_rows(
            region_mask
        )

        coverage_points = []

        reverse_direction = False

        # ------------------------------------------------------------
        # Scan every row
        # ------------------------------------------------------------

        for gy in rows:

            segments = (
                self.find_horizontal_segments(
                    region_mask,
                    gy
                )
            )

            if not segments:
                continue

            # --------------------------------------------------------
            # Multiple segments can appear because of obstacles.
            # --------------------------------------------------------

            if reverse_direction:

                segments = list(
                    reversed(segments)
                )

            for segment in segments:

                points = (
                    self.segment_waypoints(
                        segment
                    )
                )

                if reverse_direction:

                    points.reverse()

                coverage_points.extend(
                    points
                )

                reverse_direction = (
                    not reverse_direction
                )

        # ------------------------------------------------------------
        # Remove duplicate consecutive points
        # ------------------------------------------------------------

        result = []

        for point in coverage_points:

            if not result:

                result.append(point)
                continue

            if point != result[-1]:

                result.append(point)

        rospy.loginfo(
            "[CoveragePlanner %d] "
            "Generated %d coverage waypoints.",
            self.robot_id,
            len(result)
        )

        return result

    # ================================================================
    # Grid -> World
    # ================================================================

    def grid_to_world(
        self,
        gx,
        gy
    ):

        if self.region_map is None:
            return None

        resolution = (
            self.region_map.info.resolution
        )

        origin_x = (
            self.region_map.info.origin.position.x
        )

        origin_y = (
            self.region_map.info.origin.position.y
        )

        # Use cell center
        x = (
            origin_x +
            (gx + 0.5) * resolution
        )

        y = (
            origin_y +
            (gy + 0.5) * resolution
        )

        return x, y

    # ================================================================
    # Coverage completion
    # ================================================================

    def coverage_finished(
        self,
        current_index,
        coverage_points
    ):

        if not coverage_points:
            return True

        return (
            current_index >=
            len(coverage_points)
        )