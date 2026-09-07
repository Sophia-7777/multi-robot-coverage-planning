#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid


class FreeSpaceGenerator:

    def __init__(self):

        rospy.init_node(
            "free_space_generator"
        )

        self.free_threshold = rospy.get_param(
            "~free_threshold",
            25
        )

        self.global_map_topic = rospy.get_param(
            "~global_map_topic",
            "/global_map"
        )

        self.free_space_topic = rospy.get_param(
            "~free_space_topic",
            "/free_space"
        )

        self.free_space_pub = rospy.Publisher(
            self.free_space_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True
        )

        self.map_sub = rospy.Subscriber(
            self.global_map_topic,
            OccupancyGrid,
            self.map_callback,
            queue_size=1
        )

        rospy.loginfo(
            "Free Space Generator Started"
        )

        rospy.loginfo(
            "Input  : %s",
            self.global_map_topic
        )

        rospy.loginfo(
            "Output : %s",
            self.free_space_topic
        )

    def map_callback(self, msg):

        # ============================================================
        # OccupancyGrid -> NumPy
        # ============================================================

        map_array = np.asarray(
            msg.data,
            dtype=np.int8
        ).reshape(
            msg.info.height,
            msg.info.width
        )

        # ============================================================
        # Extract Free Space
        # ============================================================

        free_mask = (
            (map_array >= 0) &
            (map_array <= self.free_threshold)
        )

        # ============================================================
        # Remove small non-free noise inside free space
        #
        # 3x3 morphological closing:
        #
        #     dilation -> erosion
        #
        # Used to fill small holes / small gaps in free space.
        # No ROS parameter is changed or added.
        # ============================================================

        # ----------------------------
        # Dilation
        # ----------------------------

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
                    dy:dy + free_mask.shape[0],
                    dx:dx + free_mask.shape[1]
                ]

        # ----------------------------
        # Erosion
        # ----------------------------

        padded = np.pad(
            dilated,
            pad_width=1,
            mode="constant",
            constant_values=False
        )

        closed_free_mask = np.ones_like(
            free_mask,
            dtype=bool
        )

        for dy in range(3):
            for dx in range(3):

                closed_free_mask &= padded[
                    dy:dy + free_mask.shape[0],
                    dx:dx + free_mask.shape[1]
                ]

        # ============================================================
        # Create Free Space Map
        # ============================================================

        free_map = np.full(
            map_array.shape,
            -1,
            dtype=np.int8
        )

        free_map[closed_free_mask] = 0

        # ============================================================
        # Create OccupancyGrid
        # ============================================================

        free_msg = OccupancyGrid()

        free_msg.header.stamp = rospy.Time.now()

        free_msg.header.frame_id = (
            msg.header.frame_id
        )

        free_msg.info = msg.info

        free_msg.data = (
            free_map
            .flatten()
            .tolist()
        )

        # ============================================================
        # Publish
        # ============================================================

        self.free_space_pub.publish(
            free_msg
        )

        # ============================================================
        # Statistics
        # ============================================================

        free_count = np.count_nonzero(
            closed_free_mask
        )

        total_count = (
            msg.info.width *
            msg.info.height
        )

        free_percentage = (
            100.0 *
            free_count /
            max(total_count, 1)
        )

        rospy.loginfo_throttle(
            5.0,
            "Free space: %d cells (%.2f%%)",
            free_count,
            free_percentage
        )


def main():

    try:

        node = FreeSpaceGenerator()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()