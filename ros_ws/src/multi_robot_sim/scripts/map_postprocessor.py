#!/usr/bin/env python3

import rospy
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

import tf2_ros
from tf.transformations import quaternion_matrix


class MapProcessor:

    def __init__(self, robot_name):

        self.robot_name = robot_name

        # ==================================================
        # Topics
        # ==================================================

        self.map_topic = "/{}/map".format(robot_name)
        self.scan_topic = "/{}/scan".format(robot_name)
        self.output_topic = "/{}/map_processed".format(robot_name)

        # ==================================================
        # TF frames
        # ==================================================

        self.map_frame = "{}/map".format(robot_name)

        self.max_range = rospy.get_param(
            '~max_range',
            3.5
        )

        self.map_msg = None

        # ==================================================
        # TF
        # ==================================================

        self.tf_buffer = tf2_ros.Buffer(
            rospy.Duration(10.0)
        )

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        # ==================================================
        # Subscribers
        # ==================================================

        self.map_sub = rospy.Subscriber(
            self.map_topic,
            OccupancyGrid,
            self.map_callback,
            queue_size=1
        )

        self.scan_sub = rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1
        )

        # ==================================================
        # Publisher
        # ==================================================

        self.pub = rospy.Publisher(
            self.output_topic,
            OccupancyGrid,
            queue_size=1
        )

        rospy.loginfo(
            "[%s] processor started",
            self.robot_name
        )

        rospy.loginfo(
            "[%s] SUB map  : %s",
            self.robot_name,
            self.map_topic
        )

        rospy.loginfo(
            "[%s] SUB scan : %s",
            self.robot_name,
            self.scan_topic
        )

        rospy.loginfo(
            "[%s] PUB map  : %s",
            self.robot_name,
            self.output_topic
        )

        rospy.loginfo(
            "[%s] TF map frame : %s",
            self.robot_name,
            self.map_frame
        )

    # ==================================================
    # Map callback
    # ==================================================

    def map_callback(self, msg):

        self.map_msg = msg

    # ==================================================
    # Scan callback
    # ==================================================

    def scan_callback(self, scan):

        if self.map_msg is None:
            return

        self.process(scan)

    # ==================================================
    # Process
    # ==================================================

    def process(self, scan):

        map_msg = self.map_msg

        # ==================================================
        # TF frames
        # ==================================================

        # 地图 frame 明确使用：
        #
        # robot1/map
        # robot2/map
        #
        # 不直接使用 map_msg.header.frame_id，
        # 防止 namespace 重复解析。
        map_frame = self.map_frame

        # Laser frame 从实际 LaserScan 消息获取。
        #
        # 例如：
        # robot1/scan
        # 或
        # robot1/laser
        #
        laser_frame = scan.header.frame_id.strip("/")

        rospy.loginfo_throttle(
            5.0,
            "[%s] TF: %s <- %s",
            self.robot_name,
            map_frame,
            laser_frame
        )

        # ==================================================
        # Lookup TF
        # ==================================================

        try:

            transform = self.tf_buffer.lookup_transform(
                map_frame,
                laser_frame,
                scan.header.stamp,
                rospy.Duration(0.2)
            )

        except Exception as e:

            rospy.logwarn_throttle(
                2.0,
                "[%s] TF error: %s",
                self.robot_name,
                str(e)
            )

            return

        # ==================================================
        # Transform
        # ==================================================

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w

        T = quaternion_matrix(
            [qx, qy, qz, qw]
        )

        # ==================================================
        # Laser origin
        # ==================================================

        laser_x = tx
        laser_y = ty

        resolution = map_msg.info.resolution

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        width = map_msg.info.width
        height = map_msg.info.height

        robot_mx = int(
            math.floor(
                (laser_x - origin_x) /
                resolution
            )
        )

        robot_my = int(
            math.floor(
                (laser_y - origin_y) /
                resolution
            )
        )

        # ==================================================
        # Copy map
        # ==================================================

        new_data = list(map_msg.data)

        # ==================================================
        # Laser angle
        # ==================================================

        angle = scan.angle_min

        # 实际使用的最大距离
        max_distance = min(
            self.max_range,
            scan.range_max
        )

        # ==================================================
        # Every laser beam
        # ==================================================

        for r in scan.ranges:

            # --------------------------------------------------
            # NaN
            # --------------------------------------------------

            if math.isnan(r):

                angle += scan.angle_increment
                continue

            # --------------------------------------------------
            # Invalid minimum range
            # --------------------------------------------------

            if r < scan.range_min:

                angle += scan.angle_increment
                continue

            # --------------------------------------------------
            # No obstacle
            # --------------------------------------------------

            if math.isinf(r) or r >= scan.range_max:

                distance = max_distance
                hit_obstacle = False

            # --------------------------------------------------
            # Beyond processor max range
            # --------------------------------------------------

            elif r > self.max_range:

                distance = max_distance
                hit_obstacle = False

            # --------------------------------------------------
            # Valid obstacle
            # --------------------------------------------------

            else:

                distance = r
                hit_obstacle = True

            # ==================================================
            # Laser frame
            # ==================================================

            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)

            # ==================================================
            # Laser -> Map
            # ==================================================

            mx_world = (
                T[0, 0] * lx +
                T[0, 1] * ly +
                tx
            )

            my_world = (
                T[1, 0] * lx +
                T[1, 1] * ly +
                ty
            )

            # ==================================================
            # World -> Grid
            # ==================================================

            end_mx = int(
                math.floor(
                    (mx_world - origin_x) /
                    resolution
                )
            )

            end_my = int(
                math.floor(
                    (my_world - origin_y) /
                    resolution
                )
            )

            # ==================================================
            # Ray tracing
            # ==================================================

            dx = abs(
                end_mx - robot_mx
            )

            dy = abs(
                end_my - robot_my
            )

            sx = (
                1
                if robot_mx < end_mx
                else -1
            )

            sy = (
                1
                if robot_my < end_my
                else -1
            )

            err = dx - dy

            x = robot_mx
            y = robot_my

            while True:

                # --------------------------------------------------
                # Free space
                # --------------------------------------------------

                if (
                    0 <= x < width and
                    0 <= y < height
                ):

                    if (
                        x != end_mx or
                        y != end_my
                    ):

                        index = (
                            y * width +
                            x
                        )

                        # 不覆盖已经确认的 occupied
                        if new_data[index] != 100:
                            new_data[index] = 0

                # --------------------------------------------------
                # Endpoint
                # --------------------------------------------------

                if (
                    x == end_mx and
                    y == end_my
                ):
                    break

                e2 = 2 * err

                if e2 > -dy:

                    err -= dy
                    x += sx

                if e2 < dx:

                    err += dx
                    y += sy

            # ==================================================
            # Obstacle endpoint
            # ==================================================

            if hit_obstacle:

                if (
                    0 <= end_mx < width and
                    0 <= end_my < height
                ):

                    index = (
                        end_my * width +
                        end_mx
                    )

                    new_data[index] = 100

            angle += scan.angle_increment

        # ==================================================
        # Publish
        # ==================================================

        new_map = OccupancyGrid()

        new_map.header = map_msg.header
        new_map.info = map_msg.info
        new_map.data = new_data

        self.pub.publish(new_map)


# ==================================================
# Main
# ==================================================

def main():

    rospy.init_node(
        'multi_robot_map_processor'
    )

    MapProcessor('robot1')
    MapProcessor('robot2')

    rospy.loginfo(
        "Robot1 + Robot2 map processing active"
    )

    rospy.spin()


if __name__ == '__main__':
    main()