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

    Current version additionally performs:

        Robot1 map
             +
        Robot2 map
             |
             v
        current global_map
             |
             OR
             |
        previous global_map
             |
             v
        final global_map


    OR fusion rule:

        -1 + -1   -> -1
        -1 + 0    -> 0
         0 + -1  -> 0
         0 + 0   -> 0

        -1 + 100  -> 100
        100 + -1  -> 100
        100 + 100 -> 100

         0 + 100  -> 0
        100 + 0   -> 0


    Priority:

        0 > 100 > -1


    Performance:
        Map insertion and previous-map fusion are implemented
        using NumPy vectorized operations instead of Python
        per-cell nested loops.
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
            "robot1/map_processed"
        )

        self.robot2_map_frame = rospy.get_param(
            "~robot2_map_frame",
            "robot2/map_processed"
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
            "/robot1/map_processed"
        )

        self.map_topic_2 = rospy.get_param(
            "~map_topic_2",
            "/robot2/map_processed"
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

        # ============================================================
        # Previous global map
        # ============================================================

        self.previous_global_map = None

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
            "Previous global_map OR fusion: ENABLED"
        )

        rospy.loginfo(
            "NumPy vectorized fusion: ENABLED"
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

        T_map_origin = self.map_origin_matrix(
            map_msg
        )

        width_m = width * resolution
        height_m = height * resolution

        local_corners = np.array(
            [
                [0.0, 0.0, 1.0],
                [width_m, 0.0, 1.0],
                [0.0, height_m, 1.0],
                [width_m, height_m, 1.0]
            ],
            dtype=np.float64
        )

        # map-local -> map
        p_map = (
            T_map_origin @
            local_corners.T
        )

        # map -> global
        p_global = (
            T_global_map @
            p_map
        )

        return [
            (
                float(p_global[0, i]),
                float(p_global[1, i])
            )
            for i in range(4)
        ]

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

        xs = np.array(
            [p[0] for p in all_corners]
        )

        ys = np.array(
            [p[1] for p in all_corners]
        )

        return (
            float(np.min(xs)),
            float(np.max(xs)),
            float(np.min(ys)),
            float(np.max(ys))
        )

    # ================================================================
    # Insert map - NumPy vectorized version
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
        """
        Insert source OccupancyGrid into global_grid.

        This version avoids Python nested loops.

        The complete map is transformed using NumPy matrix
        operations.

        Fusion priority:

            0 > 100 > intermediate > -1
        """

        resolution = source_map.info.resolution

        width = source_map.info.width
        height = source_map.info.height

        # ------------------------------------------------------------
        # Source map
        # ------------------------------------------------------------

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
        # Generate all cell centers
        #
        # Shape:
        #     X -> (height, width)
        #     Y -> (height, width)
        # ------------------------------------------------------------

        cols = (
            np.arange(
                width,
                dtype=np.float64
            )
            + 0.5
        )

        rows = (
            np.arange(
                height,
                dtype=np.float64
            )
            + 0.5
        )

        local_x = cols * resolution
        local_y = rows * resolution

        local_x, local_y = np.meshgrid(
            local_x,
            local_y
        )

        # ------------------------------------------------------------
        # Flatten
        # ------------------------------------------------------------

        x = local_x.ravel()
        y = local_y.ravel()

        values = source_data.ravel()

        # ------------------------------------------------------------
        # Remove unknown cells
        #
        # Unknown (-1) does not need to be inserted.
        # ------------------------------------------------------------

        valid = (
            values >= 0
        )

        if not np.any(valid):

            return

        x = x[valid]
        y = y[valid]
        values = values[valid]

        # ------------------------------------------------------------
        # map-local -> map
        #
        # Since this is 2D homogeneous transformation:
        #
        # x' = R00*x + R01*y + tx
        # y' = R10*x + R11*y + ty
        # ------------------------------------------------------------

        map_x = (
            T_map_origin[0, 0] * x
            +
            T_map_origin[0, 1] * y
            +
            T_map_origin[0, 2]
        )

        map_y = (
            T_map_origin[1, 0] * x
            +
            T_map_origin[1, 1] * y
            +
            T_map_origin[1, 2]
        )

        # ------------------------------------------------------------
        # map -> global
        # ------------------------------------------------------------

        global_x = (
            T_global_map[0, 0] * map_x
            +
            T_global_map[0, 1] * map_y
            +
            T_global_map[0, 2]
        )

        global_y = (
            T_global_map[1, 0] * map_x
            +
            T_global_map[1, 1] * map_y
            +
            T_global_map[1, 2]
        )

        # ------------------------------------------------------------
        # global -> grid cell
        # ------------------------------------------------------------

        global_cols = np.floor(
            (
                global_x -
                global_origin_x
            ) / resolution
        ).astype(
            np.int32
        )

        global_rows = np.floor(
            (
                global_y -
                global_origin_y
            ) / resolution
        ).astype(
            np.int32
        )

        # ------------------------------------------------------------
        # Remove points outside global grid
        # ------------------------------------------------------------

        inside = (
            (global_cols >= 0)
            &
            (global_cols < global_width)
            &
            (global_rows >= 0)
            &
            (global_rows < global_height)
        )

        if not np.any(inside):

            return

        global_cols = global_cols[inside]
        global_rows = global_rows[inside]
        values = values[inside]

        # ------------------------------------------------------------
        # Existing values
        # ------------------------------------------------------------

        old_values = global_grid[
            global_rows,
            global_cols
        ]

        # ============================================================
        # Fusion
        #
        # Priority:
        #
        #     0 > 100 > intermediate > -1
        #
        # ============================================================

        # ------------------------------------------------------------
        # Source free
        #
        # Free has highest priority.
        # ------------------------------------------------------------

        source_free = (
            values <= self.free_threshold
        )

        if np.any(source_free):

            rows_free = global_rows[
                source_free
            ]

            cols_free = global_cols[
                source_free
            ]

            global_grid[
                rows_free,
                cols_free
            ] = 0

        # ------------------------------------------------------------
        # Source occupied
        #
        # If old is already free, keep free.
        # Otherwise occupied becomes 100.
        # ------------------------------------------------------------

        source_occupied = (
            values >= self.occupied_threshold
        )

        if np.any(source_occupied):

            rows_occ = global_rows[
                source_occupied
            ]

            cols_occ = global_cols[
                source_occupied
            ]

            old_occ = global_grid[
                rows_occ,
                cols_occ
            ]

            occupied_result = np.where(
                old_occ == 0,
                0,
                100
            )

            global_grid[
                rows_occ,
                cols_occ
            ] = occupied_result

        # ------------------------------------------------------------
        # Intermediate probability
        # ------------------------------------------------------------

        source_intermediate = (
            (~source_free)
            &
            (~source_occupied)
        )

        if np.any(source_intermediate):

            rows_mid = global_rows[
                source_intermediate
            ]

            cols_mid = global_cols[
                source_intermediate
            ]

            values_mid = values[
                source_intermediate
            ]

            old_mid = global_grid[
                rows_mid,
                cols_mid
            ]

            # Existing free remains free.
            free_mask = (
                old_mid == 0
            )

            # Existing occupied remains occupied.
            occupied_mask = (
                old_mid == 100
            )

            # Unknown gets source value.
            unknown_mask = (
                old_mid < 0
            )

            # Existing intermediate values:
            # keep the larger probability.
            intermediate_mask = (
                (~free_mask)
                &
                (~occupied_mask)
                &
                (~unknown_mask)
            )

            # Unknown
            if np.any(unknown_mask):

                global_grid[
                    rows_mid[unknown_mask],
                    cols_mid[unknown_mask]
                ] = values_mid[
                    unknown_mask
                ]

            # Intermediate
            if np.any(intermediate_mask):

                current_values = global_grid[
                    rows_mid[intermediate_mask],
                    cols_mid[intermediate_mask]
                ]

                new_values = np.maximum(
                    current_values,
                    values_mid[
                        intermediate_mask
                    ]
                )

                global_grid[
                    rows_mid[intermediate_mask],
                    cols_mid[intermediate_mask]
                ] = new_values

    # ================================================================
    # Fuse Robot 1 + Robot 2
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
    # OR current global_map with previous global_map
    # ================================================================

    def merge_global_maps(
        self,
        previous_map,
        current_map
    ):
        """
        Vectorized previous/current global_map fusion.

        Priority:

            0 > 100 > -1

        No Python per-cell nested loop.
        """

        # ============================================================
        # First global_map
        # ============================================================

        if previous_map is None:

            return current_map

        # ============================================================
        # Check resolution
        # ============================================================

        previous_resolution = (
            previous_map.info.resolution
        )

        current_resolution = (
            current_map.info.resolution
        )

        if abs(
            previous_resolution
            -
            current_resolution
        ) > 1e-6:

            rospy.logwarn_throttle(
                5.0,
                "Previous/current global_map resolution "
                "is different."
            )

            return current_map

        resolution = current_resolution

        # ============================================================
        # Current map information
        # ============================================================

        current_width = current_map.info.width
        current_height = current_map.info.height

        current_origin_x = (
            current_map.info.origin.position.x
        )

        current_origin_y = (
            current_map.info.origin.position.y
        )

        current_data = np.asarray(
            current_map.data,
            dtype=np.int16
        ).reshape(
            current_height,
            current_width
        )

        # ============================================================
        # Previous map information
        # ============================================================

        previous_width = previous_map.info.width
        previous_height = previous_map.info.height

        previous_origin_x = (
            previous_map.info.origin.position.x
        )

        previous_origin_y = (
            previous_map.info.origin.position.y
        )

        previous_data = np.asarray(
            previous_map.data,
            dtype=np.int16
        ).reshape(
            previous_height,
            previous_width
        )

        # ============================================================
        # Result starts from current map
        # ============================================================

        result = current_data.copy()

        # ============================================================
        # Calculate previous map bounds
        # ============================================================

        previous_max_x = (
            previous_origin_x
            +
            previous_width * resolution
        )

        previous_max_y = (
            previous_origin_y
            +
            previous_height * resolution
        )

        # ============================================================
        # Current map bounds
        # ============================================================

        current_max_x = (
            current_origin_x
            +
            current_width * resolution
        )

        current_max_y = (
            current_origin_y
            +
            current_height * resolution
        )

        # ============================================================
        # Overlap
        # ============================================================

        overlap_min_x = max(
            previous_origin_x,
            current_origin_x
        )

        overlap_max_x = min(
            previous_max_x,
            current_max_x
        )

        overlap_min_y = max(
            previous_origin_y,
            current_origin_y
        )

        overlap_max_y = min(
            previous_max_y,
            current_max_y
        )

        # ============================================================
        # No overlap
        # ============================================================

        if (
            overlap_min_x >= overlap_max_x
            or
            overlap_min_y >= overlap_max_y
        ):

            rospy.logwarn_throttle(
                5.0,
                "Previous and current global_map "
                "have no spatial overlap."
            )

            return current_map

        # ============================================================
        # Current row range
        # ============================================================

        current_row_start = max(
            0,
            int(
                math.floor(
                    (
                        overlap_min_y
                        -
                        current_origin_y
                    )
                    / resolution
                )
            )
        )

        current_row_end = min(
            current_height,
            int(
                math.ceil(
                    (
                        overlap_max_y
                        -
                        current_origin_y
                    )
                    / resolution
                )
            )
        )

        # ============================================================
        # Current column range
        # ============================================================

        current_col_start = max(
            0,
            int(
                math.floor(
                    (
                        overlap_min_x
                        -
                        current_origin_x
                    )
                    / resolution
                )
            )
        )

        current_col_end = min(
            current_width,
            int(
                math.ceil(
                    (
                        overlap_max_x
                        -
                        current_origin_x
                    )
                    / resolution
                )
            )
        )

        if (
            current_row_start >= current_row_end
            or
            current_col_start >= current_col_end
        ):

            return current_map

        # ============================================================
        # Generate current-grid coordinates
        #
        # No Python nested loop.
        # ============================================================

        current_rows = np.arange(
            current_row_start,
            current_row_end,
            dtype=np.int32
        )

        current_cols = np.arange(
            current_col_start,
            current_col_end,
            dtype=np.int32
        )

        # ------------------------------------------------------------
        # Global coordinate of cell centers
        # ------------------------------------------------------------

        global_y = (
            current_origin_y
            +
            (current_rows.astype(np.float64) + 0.5)
            * resolution
        )

        global_x = (
            current_origin_x
            +
            (current_cols.astype(np.float64) + 0.5)
            * resolution
        )

        # ------------------------------------------------------------
        # Corresponding previous indices
        # ------------------------------------------------------------

        previous_rows = np.floor(
            (
                global_y
                -
                previous_origin_y
            )
            / resolution
        ).astype(
            np.int32
        )

        previous_cols = np.floor(
            (
                global_x
                -
                previous_origin_x
            )
            / resolution
        ).astype(
            np.int32
        )

        # ============================================================
        # Safety clipping
        # ============================================================

        valid_rows = (
            (previous_rows >= 0)
            &
            (previous_rows < previous_height)
        )

        valid_cols = (
            (previous_cols >= 0)
            &
            (previous_cols < previous_width)
        )

        if (
            not np.any(valid_rows)
            or
            not np.any(valid_cols)
        ):

            return current_map

        # ============================================================
        # Since map resolution and origins are aligned,
        # the valid overlap is rectangular.
        #
        # Use the valid index ranges directly.
        # ============================================================

        current_rows_valid = (
            current_rows[valid_rows]
        )

        previous_rows_valid = (
            previous_rows[valid_rows]
        )

        current_cols_valid = (
            current_cols[valid_cols]
        )

        previous_cols_valid = (
            previous_cols[valid_cols]
        )

        # ============================================================
        # Extract previous/current overlap
        # ============================================================

        current_overlap = result[
            np.ix_(
                current_rows_valid,
                current_cols_valid
            )
        ]

        previous_overlap = previous_data[
            np.ix_(
                previous_rows_valid,
                previous_cols_valid
            )
        ]

        # ============================================================
        # OR fusion
        #
        # Priority:
        #
        #       0 > 100 > -1
        #
        # ============================================================

        # ------------------------------------------------------------
        # Free
        #
        # If either map is 0 -> 0
        # ------------------------------------------------------------

        free_mask = (
            (current_overlap == 0)
            |
            (previous_overlap == 0)
        )

        # ------------------------------------------------------------
        # Occupied
        #
        # If no free exists and either map is 100 -> 100
        # ------------------------------------------------------------

        occupied_mask = (
            (~free_mask)
            &
            (
                (current_overlap == 100)
                |
                (previous_overlap == 100)
            )
        )

        # ------------------------------------------------------------
        # Unknown / intermediate
        # ------------------------------------------------------------

        intermediate_mask = (
            (~free_mask)
            &
            (~occupied_mask)
        )

        # ------------------------------------------------------------
        # Build merged result
        # ------------------------------------------------------------

        merged_overlap = current_overlap.copy()

        # Free
        merged_overlap[
            free_mask
        ] = 0

        # Occupied
        merged_overlap[
            occupied_mask
        ] = 100

        # Remaining:
        #
        # -1 + -1 -> -1
        # -1 + value -> value
        # value + -1 -> value
        # value + value -> max(value)
        #
        if np.any(intermediate_mask):

            current_remaining = (
                current_overlap[
                    intermediate_mask
                ]
            )

            previous_remaining = (
                previous_overlap[
                    intermediate_mask
                ]
            )

            merged_remaining = np.where(
                current_remaining < 0,
                previous_remaining,
                np.where(
                    previous_remaining < 0,
                    current_remaining,
                    np.maximum(
                        current_remaining,
                        previous_remaining
                    )
                )
            )

            merged_overlap[
                intermediate_mask
            ] = merged_remaining

        # ============================================================
        # Write back
        # ============================================================

        result[
            np.ix_(
                current_rows_valid,
                current_cols_valid
            )
        ] = merged_overlap

        # ============================================================
        # Create merged OccupancyGrid
        # ============================================================

        merged_map = OccupancyGrid()

        merged_map.header.stamp = rospy.Time.now()

        merged_map.header.frame_id = (
            current_map.header.frame_id
        )

        merged_map.info = current_map.info

        merged_map.data = (
            result
            .flatten()
            .astype(np.int8)
            .tolist()
        )

        return merged_map

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
        # Fuse Robot1 + Robot2
        # ------------------------------------------------------------

        current_global_map = self.fuse_maps(
            map1,
            map2
        )

        if current_global_map is None:

            return

        # ============================================================
        # OR current global_map with previous global_map
        # ============================================================

        with self.lock:

            previous_global_map = (
                self.previous_global_map
            )

        final_global_map = (
            self.merge_global_maps(
                previous_global_map,
                current_global_map
            )
        )

        if final_global_map is None:

            return

        # ============================================================
        # Save final global_map as previous global_map
        # ============================================================

        with self.lock:

            self.previous_global_map = (
                final_global_map
            )

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.global_map_pub.publish(
            final_global_map
        )

        rospy.loginfo_throttle(
            5.0,
            "Global map published: %d x %d",
            final_global_map.info.width,
            final_global_map.info.height
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

