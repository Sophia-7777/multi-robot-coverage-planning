#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import rospy
import numpy as np
import tf2_ros

from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped


class RealTimeMapFusion:
    """
    Multi-Robot OccupancyGrid Map Fusion
    ====================================

    Current version:
        Gazebo Ground Truth is used to establish the initial
        relationship between each SLAM map and global_map.

    TF architecture:

                            global_map
                           /          \
                          /            \
                         v              v
                   robot1/map       robot2/map
                       |                |
                       v                v
                   robot1/odom     robot2/odom
                       |                |
                       v                v
              robot1/base_footprint   robot2/base_footprint


    Important:
        global_map -> robotX/map is treated as a FIXED map alignment
        after initialization.

    Robot movement does NOT change global_map -> robotX/map.

    Gazebo ground truth is only used to initialize the map alignment.

    This is suitable for simulation testing.

    For real robots, the Gazebo section can later be replaced by:
        - map matching
        - scan matching
        - ICP
        - feature matching
        - pose graph optimization
    """

    def __init__(self):

        rospy.init_node(
            "realtime_map_fusion_node"
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

        self.robot1_map_frame = rospy.get_param(
            "~robot1_map_frame",
            "robot1/map"
        )

        self.robot2_map_frame = rospy.get_param(
            "~robot2_map_frame",
            "robot2/map"
        )

        self.robot1_base_frame = rospy.get_param(
            "~robot1_base_frame",
            "robot1/base_footprint"
        )

        self.robot2_base_frame = rospy.get_param(
            "~robot2_base_frame",
            "robot2/base_footprint"
        )

        self.global_frame = rospy.get_param(
            "~global_frame",
            "global_map"
        )

        self.map_topic_1 = rospy.get_param(
            "~map_topic_1",
            "/robot1/map"
        )

        self.map_topic_2 = rospy.get_param(
            "~map_topic_2",
            "/robot2/map"
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/global_map"
        )

        self.publish_rate = rospy.get_param(
            "~publish_rate",
            2.0
        )

        self.occupied_threshold = rospy.get_param(
            "~occupied_threshold",
            65
        )

        self.free_threshold = rospy.get_param(
            "~free_threshold",
            25
        )

        # ============================================================
        # Data
        # ============================================================

        self.map1 = None
        self.map2 = None

        self.robot1_world_pose = None
        self.robot2_world_pose = None

        self.lock = threading.Lock()

        # ============================================================
        # Map alignment
        # ============================================================

        self.T_global_map1 = None
        self.T_global_map2 = None

        self.alignment_initialized = False

        # ============================================================
        # TF listener
        # ============================================================

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(30.0)
        )

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )

        # ============================================================
        # TF broadcaster
        # ============================================================

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ============================================================
        # Subscribers
        # ============================================================

        self.map1_sub = rospy.Subscriber(
            self.map_topic_1,
            OccupancyGrid,
            self.map1_callback,
            queue_size=1
        )

        self.map2_sub = rospy.Subscriber(
            self.map_topic_2,
            OccupancyGrid,
            self.map2_callback,
            queue_size=1
        )

        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback,
            queue_size=1
        )

        # ============================================================
        # Publisher
        # ============================================================

        self.global_map_pub = rospy.Publisher(
            self.output_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True
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

        rospy.loginfo(
            "=================================================="
        )

        rospy.loginfo(
            "Multi-Robot Map Fusion Node Started"
        )

        rospy.loginfo(
            "Robot 1       : %s",
            self.robot1_name
        )

        rospy.loginfo(
            "Robot 2       : %s",
            self.robot2_name
        )

        rospy.loginfo(
            "Map 1         : %s",
            self.map_topic_1
        )

        rospy.loginfo(
            "Map 2         : %s",
            self.map_topic_2
        )

        rospy.loginfo(
            "Global map    : %s",
            self.output_topic
        )

        rospy.loginfo(
            "Global frame  : %s",
            self.global_frame
        )

        rospy.loginfo(
            "=================================================="
        )

    # ================================================================
    # Map callbacks
    # ================================================================

    def map1_callback(self, msg):

        with self.lock:

            self.map1 = msg

    def map2_callback(self, msg):

        with self.lock:

            self.map2 = msg

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

            self.robot1_world_pose = pose1
            self.robot2_world_pose = pose2

    # ================================================================
    # Quaternion -> yaw
    # ================================================================

    @staticmethod
    def quaternion_to_yaw(q):

        siny_cosp = (
            2.0 *
            (
                q.w * q.z +
                q.x * q.y
            )
        )

        cosy_cosp = (
            1.0 -
            2.0 *
            (
                q.y * q.y +
                q.z * q.z
            )
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )

    # ================================================================
    # Yaw -> rotation matrix
    # ================================================================

    @staticmethod
    def yaw_to_matrix(yaw):

        c = math.cos(yaw)
        s = math.sin(yaw)

        T = np.eye(3)

        T[0, 0] = c
        T[0, 1] = -s

        T[1, 0] = s
        T[1, 1] = c

        return T

    # ================================================================
    # Pose -> matrix
    # ================================================================

    @staticmethod
    def pose_to_matrix(pose):

        yaw = RealTimeMapFusion.quaternion_to_yaw(
            pose.orientation
        )

        T = RealTimeMapFusion.yaw_to_matrix(
            yaw
        )

        T[0, 2] = pose.position.x
        T[1, 2] = pose.position.y

        return T

    # ================================================================
    # TransformStamped -> matrix
    # ================================================================

    @staticmethod
    def transform_to_matrix(transform):

        q = transform.rotation

        siny_cosp = (
            2.0 *
            (
                q.w * q.z +
                q.x * q.y
            )
        )

        cosy_cosp = (
            1.0 -
            2.0 *
            (
                q.y * q.y +
                q.z * q.z
            )
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        T = RealTimeMapFusion.yaw_to_matrix(
            yaw
        )

        T[0, 2] = transform.translation.x
        T[1, 2] = transform.translation.y

        return T

    # ================================================================
    # Matrix -> TransformStamped
    # ================================================================

    @staticmethod
    def matrix_to_transform(
        T,
        parent_frame,
        child_frame,
        stamp
    ):

        msg = TransformStamped()

        msg.header.stamp = stamp
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame

        msg.transform.translation.x = float(
            T[0, 2]
        )

        msg.transform.translation.y = float(
            T[1, 2]
        )

        msg.transform.translation.z = 0.0

        yaw = math.atan2(
            T[1, 0],
            T[0, 0]
        )

        msg.transform.rotation.x = 0.0
        msg.transform.rotation.y = 0.0

        msg.transform.rotation.z = math.sin(
            yaw / 2.0
        )

        msg.transform.rotation.w = math.cos(
            yaw / 2.0
        )

        return msg

    # ================================================================
    # Get map -> base TF
    # ================================================================

    def get_map_to_base_matrix(
        self,
        map_frame,
        base_frame
    ):

        try:

            transform = self.tf_buffer.lookup_transform(
                map_frame,
                base_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )

            return self.transform_to_matrix(
                transform.transform
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ):

            rospy.logwarn_throttle(
                5.0,
                "TF unavailable: %s -> %s",
                map_frame,
                base_frame
            )

            return None

    # ================================================================
    # Calculate global -> map
    # ================================================================

    def calculate_global_map_transform(
        self,
        world_base_pose,
        map_frame,
        base_frame
    ):

        if world_base_pose is None:

            return None

        # ------------------------------------------------------------
        # T_world_base
        # ------------------------------------------------------------

        T_world_base = self.pose_to_matrix(
            world_base_pose
        )

        # ------------------------------------------------------------
        # T_map_base
        # ------------------------------------------------------------

        T_map_base = self.get_map_to_base_matrix(
            map_frame,
            base_frame
        )

        if T_map_base is None:

            return None

        # ------------------------------------------------------------
        # T_base_map
        # ------------------------------------------------------------

        T_base_map = np.linalg.inv(
            T_map_base
        )

        # ------------------------------------------------------------
        # T_world_map
        # ------------------------------------------------------------

        T_world_map = np.matmul(
            T_world_base,
            T_base_map
        )

        return T_world_map

    # ================================================================
    # Initialize map alignment
    # ================================================================

    def initialize_alignment(self):

        if self.alignment_initialized:

            return True

        with self.lock:

            pose1 = self.robot1_world_pose
            pose2 = self.robot2_world_pose

        if pose1 is None:

            rospy.loginfo_throttle(
                5.0,
                "Waiting for Robot1 Gazebo pose..."
            )

            return False

        if pose2 is None:

            rospy.loginfo_throttle(
                5.0,
                "Waiting for Robot2 Gazebo pose..."
            )

            return False

        # ------------------------------------------------------------
        # Calculate initial global -> map
        # ------------------------------------------------------------

        T1 = self.calculate_global_map_transform(
            pose1,
            self.robot1_map_frame,
            self.robot1_base_frame
        )

        T2 = self.calculate_global_map_transform(
            pose2,
            self.robot2_map_frame,
            self.robot2_base_frame
        )

        if T1 is None:

            rospy.loginfo_throttle(
                5.0,
                "Waiting for Robot1 map TF..."
            )

            return False

        if T2 is None:

            rospy.loginfo_throttle(
                5.0,
                "Waiting for Robot2 map TF..."
            )

            return False

        # ------------------------------------------------------------
        # Save alignment
        # ------------------------------------------------------------

        self.T_global_map1 = T1.copy()
        self.T_global_map2 = T2.copy()

        self.alignment_initialized = True

        yaw1 = math.atan2(
            T1[1, 0],
            T1[0, 0]
        )

        yaw2 = math.atan2(
            T2[1, 0],
            T2[0, 0]
        )

        rospy.loginfo(
            "=================================================="
        )

        rospy.loginfo(
            "Map alignment initialized"
        )

        rospy.loginfo(
            "Robot1 map:"
        )

        rospy.loginfo(
            "  x = %.3f",
            T1[0, 2]
        )

        rospy.loginfo(
            "  y = %.3f",
            T1[1, 2]
        )

        rospy.loginfo(
            "  yaw = %.3f rad",
            yaw1
        )

        rospy.loginfo(
            "Robot2 map:"
        )

        rospy.loginfo(
            "  x = %.3f",
            T2[0, 2]
        )

        rospy.loginfo(
            "  y = %.3f",
            T2[1, 2]
        )

        rospy.loginfo(
            "  yaw = %.3f rad",
            yaw2
        )

        rospy.loginfo(
            "=================================================="
        )

        return True

    # ================================================================
    # Publish map TF
    # ================================================================

    def publish_map_tfs(self):

        if not self.alignment_initialized:

            return

        stamp = rospy.Time.now()

        tf1 = self.matrix_to_transform(
            self.T_global_map1,
            self.global_frame,
            self.robot1_map_frame,
            stamp
        )

        tf2 = self.matrix_to_transform(
            self.T_global_map2,
            self.global_frame,
            self.robot2_map_frame,
            stamp
        )

        self.tf_broadcaster.sendTransform(
            tf1
        )

        self.tf_broadcaster.sendTransform(
            tf2
        )

    # ================================================================
    # Transform point
    # ================================================================

    @staticmethod
    def transform_point(
        x,
        y,
        T
    ):

        p = np.array(
            [
                x,
                y,
                1.0
            ]
        )

        p_global = np.matmul(
            T,
            p
        )

        return (
            float(p_global[0]),
            float(p_global[1])
        )

    # ================================================================
    # Map origin transform
    # ================================================================

    @staticmethod
    def map_origin_matrix(map_msg):

        origin = map_msg.info.origin

        yaw = RealTimeMapFusion.quaternion_to_yaw(
            origin.orientation
        )

        T = RealTimeMapFusion.yaw_to_matrix(
            yaw
        )

        T[0, 2] = origin.position.x
        T[1, 2] = origin.position.y

        return T

    # ================================================================
    # Map corners
    # ================================================================

    def get_map_corners(
        self,
        map_msg,
        T_global_map
    ):

        resolution = map_msg.info.resolution

        width = map_msg.info.width
        height = map_msg.info.height

        # ------------------------------------------------------------
        # Map origin
        # ------------------------------------------------------------

        T_map_origin = self.map_origin_matrix(
            map_msg
        )

        width_m = width * resolution
        height_m = height * resolution

        local_corners = [
            (0.0, 0.0),
            (width_m, 0.0),
            (0.0, height_m),
            (width_m, height_m)
        ]

        result = []

        for x, y in local_corners:

            point = np.array(
                [
                    x,
                    y,
                    1.0
                ]
            )

            # map-local -> map
            p_map = np.matmul(
                T_map_origin,
                point
            )

            # map -> global
            p_global = np.matmul(
                T_global_map,
                p_map
            )

            result.append(
                (
                    float(p_global[0]),
                    float(p_global[1])
                )
            )

        return result

    # ================================================================
    # Calculate global bounds
    # ================================================================

    def calculate_global_bounds(
        self,
        map1,
        map2,
        T_global_map1,
        T_global_map2
    ):

        corners1 = self.get_map_corners(
            map1,
            T_global_map1
        )

        corners2 = self.get_map_corners(
            map2,
            T_global_map2
        )

        all_corners = (
            corners1 +
            corners2
        )

        min_x = min(
            p[0]
            for p in all_corners
        )

        max_x = max(
            p[0]
            for p in all_corners
        )

        min_y = min(
            p[1]
            for p in all_corners
        )

        max_y = max(
            p[1]
            for p in all_corners
        )

        return (
            min_x,
            max_x,
            min_y,
            max_y
        )

    # ================================================================
    # Insert map
    # ================================================================

    def insert_map(
        self,
        global_grid,
        global_width,
        global_height,
        global_origin_x,
        global_origin_y,
        source_map,
        T_global_map
    ):

        resolution = source_map.info.resolution

        width = source_map.info.width
        height = source_map.info.height

        source_data = np.asarray(
            source_map.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ------------------------------------------------------------
        # Map origin
        # ------------------------------------------------------------

        T_map_origin = self.map_origin_matrix(
            source_map
        )

        # ------------------------------------------------------------
        # Iterate through source map
        # ------------------------------------------------------------

        for row in range(height):

            for col in range(width):

                value = source_data[
                    row,
                    col
                ]

                # ----------------------------------------------------
                # Unknown
                # ----------------------------------------------------

                if value < 0:

                    continue

                # ----------------------------------------------------
                # Cell center in map-local coordinates
                # ----------------------------------------------------

                local_x = (
                    (col + 0.5)
                    * resolution
                )

                local_y = (
                    (row + 0.5)
                    * resolution
                )

                p_local = np.array(
                    [
                        local_x,
                        local_y,
                        1.0
                    ]
                )

                # ----------------------------------------------------
                # map-local -> map
                # ----------------------------------------------------

                p_map = np.matmul(
                    T_map_origin,
                    p_local
                )

                # ----------------------------------------------------
                # map -> global
                # ----------------------------------------------------

                p_global = np.matmul(
                    T_global_map,
                    p_map
                )

                global_x = p_global[0]
                global_y = p_global[1]

                # ----------------------------------------------------
                # global -> cell
                # ----------------------------------------------------

                global_col = int(
                    math.floor(
                        (
                            global_x -
                            global_origin_x
                        )
                        / resolution
                    )
                )

                global_row = int(
                    math.floor(
                        (
                            global_y -
                            global_origin_y
                        )
                        / resolution
                    )
                )

                if (
                    global_col < 0
                    or
                    global_col >= global_width
                    or
                    global_row < 0
                    or
                    global_row >= global_height
                ):

                    continue

                old_value = global_grid[
                    global_row,
                    global_col
                ]

                # ====================================================
                # Occupied
                # ====================================================

                if value >= self.occupied_threshold:

                    global_grid[
                        global_row,
                        global_col
                    ] = 100

                    continue

                # ====================================================
                # Free
                # ====================================================

                if value <= self.free_threshold:

                    if old_value < 0:

                        global_grid[
                            global_row,
                            global_col
                        ] = 0

                    elif old_value <= self.free_threshold:

                        global_grid[
                            global_row,
                            global_col
                        ] = 0

                    continue

                # ====================================================
                # Intermediate probability
                # ====================================================

                if old_value < 0:

                    global_grid[
                        global_row,
                        global_col
                    ] = value

                elif value > old_value:

                    global_grid[
                        global_row,
                        global_col
                    ] = value

    # ================================================================
    # Fuse maps
    # ================================================================

    def fuse_maps(
        self,
        map1,
        map2
    ):

        resolution1 = map1.info.resolution
        resolution2 = map2.info.resolution

        # ------------------------------------------------------------
        # Resolution check
        # ------------------------------------------------------------

        if abs(
            resolution1 -
            resolution2
        ) > 1e-6:

            rospy.logerr_throttle(
                5.0,
                "Robot map resolutions are different: %.4f vs %.4f",
                resolution1,
                resolution2
            )

            return None

        resolution = resolution1

        # ------------------------------------------------------------
        # Global bounds
        # ------------------------------------------------------------

        (
            min_x,
            max_x,
            min_y,
            max_y
        ) = self.calculate_global_bounds(
            map1,
            map2,
            self.T_global_map1,
            self.T_global_map2
        )

        # ------------------------------------------------------------
        # Margin
        # ------------------------------------------------------------

        margin = resolution * 2.0

        min_x -= margin
        min_y -= margin

        max_x += margin
        max_y += margin

        # ------------------------------------------------------------
        # Global dimensions
        # ------------------------------------------------------------

        global_width = int(
            math.ceil(
                (
                    max_x -
                    min_x
                )
                / resolution
            )
        )

        global_height = int(
            math.ceil(
                (
                    max_y -
                    min_y
                )
                / resolution
            )
        )

        # ------------------------------------------------------------
        # Safety
        # ------------------------------------------------------------

        if (
            global_width <= 0
            or
            global_height <= 0
        ):

            rospy.logerr(
                "Invalid global map dimensions."
            )

            return None

        if (
            global_width > 3000
            or
            global_height > 3000
        ):

            rospy.logerr_throttle(
                5.0,
                "Global map too large: %d x %d",
                global_width,
                global_height
            )

            return None

        # ------------------------------------------------------------
        # Allocate unknown map
        # ------------------------------------------------------------

        global_grid = np.full(
            (
                global_height,
                global_width
            ),
            -1,
            dtype=np.int16
        )

        # ------------------------------------------------------------
        # Insert Robot 1
        # ------------------------------------------------------------

        self.insert_map(
            global_grid,
            global_width,
            global_height,
            min_x,
            min_y,
            map1,
            self.T_global_map1
        )

        # ------------------------------------------------------------
        # Insert Robot 2
        # ------------------------------------------------------------

        self.insert_map(
            global_grid,
            global_width,
            global_height,
            min_x,
            min_y,
            map2,
            self.T_global_map2
        )

        # ------------------------------------------------------------
        # Create ROS OccupancyGrid
        # ------------------------------------------------------------

        msg = OccupancyGrid()

        msg.header.stamp = rospy.Time.now()

        msg.header.frame_id = (
            self.global_frame
        )

        msg.info.resolution = resolution

        msg.info.width = global_width

        msg.info.height = global_height

        # ------------------------------------------------------------
        # Global origin
        # ------------------------------------------------------------

        msg.info.origin.position.x = float(
            min_x
        )

        msg.info.origin.position.y = float(
            min_y
        )

        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # ------------------------------------------------------------
        # Data
        # ------------------------------------------------------------

        msg.data = (
            global_grid
            .flatten()
            .astype(np.int8)
            .tolist()
        )

        return msg

    # ================================================================
    # Main timer
    # ================================================================

    def timer_callback(
        self,
        event
    ):

        # ------------------------------------------------------------
        # Get latest data
        # ------------------------------------------------------------

        with self.lock:

            map1 = self.map1
            map2 = self.map2

        # ------------------------------------------------------------
        # Check maps
        # ------------------------------------------------------------

        if map1 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for /robot1/map ..."
            )

            return

        if map2 is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for /robot2/map ..."
            )

            return

        # ------------------------------------------------------------
        # Initialize alignment once
        # ------------------------------------------------------------

        if not self.alignment_initialized:

            success = (
                self.initialize_alignment()
            )

            if not success:

                return

        # ------------------------------------------------------------
        # Publish TF
        # ------------------------------------------------------------

        self.publish_map_tfs()

        # ------------------------------------------------------------
        # Fuse
        # ------------------------------------------------------------

        global_map = self.fuse_maps(
            map1,
            map2
        )

        if global_map is None:

            return

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.global_map_pub.publish(
            global_map
        )

        rospy.loginfo_throttle(
            5.0,
            "Global map published: %d x %d",
            global_map.info.width,
            global_map.info.height
        )


# ====================================================================
# Main
# ====================================================================

def main():

    try:

        node = RealTimeMapFusion()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()