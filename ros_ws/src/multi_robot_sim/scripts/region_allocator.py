#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import heapq

from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates


class RegionAllocator:
    """
    RegionAllocator

    根据 /covered_map 对两个机器人进行区域分配。

    covered_map：

        -1 : 非自由区域 / 未知区域
         0 : 未覆盖的自由区域 -> 需要进行任务分配
       100 : 已覆盖的自由区域 -> 不作为任务分配区域

    分配规则：

        1. 只有 covered_map == 0 的区域最终成为任务。

        2. covered_map == 100 不产生任务。

        3. covered_map == 100 可以作为机器人移动 /
           Dijkstra 传播路径。

        4. covered_map == -1 不允许传播。

        5. 最终 region_map 中：

               -1 -> 非任务区域
                0 -> Robot1 任务
              100 -> Robot2 任务

        6. robot1_region / robot2_region：

               100 -> 自己的任务
                 0 -> 对方的任务
                -1 -> 非任务区域
    """

    def __init__(self):

        rospy.init_node(
            "region_allocator_node"
        )

        # ============================================================
        # Parameters
        # ============================================================

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

        self.robot1_name = rospy.get_param(
            "~robot1_name",
            "robot1"
        )

        self.robot2_name = rospy.get_param(
            "~robot2_name",
            "robot2"
        )

        self.update_rate = rospy.get_param(
            "~update_rate",
            1.0
        )

        self.reallocate_on_robot_move = rospy.get_param(
            "~reallocate_on_robot_move",
            True
        )

        self.min_robot_move = rospy.get_param(
            "~min_robot_move",
            0.5
        )

        self.inflation_radius = rospy.get_param(
            "~inflation_radius",
            0.30
        )

        self.covered_inflation_radius = rospy.get_param(
            "~covered_inflation_radius",
            0.50
        )

        # ============================================================
        # Internal state
        # ============================================================

        self.covered_map = None

        self.robot1_pose = None
        self.robot2_pose = None

        self.last_robot1_pose = None
        self.last_robot2_pose = None

        self.last_map_stamp = None

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
        # Timer
        # ============================================================

        self.timer = rospy.Timer(
            rospy.Duration(
                1.0 /
                max(
                    self.update_rate,
                    1e-6
                )
            ),
            self.timer_callback
        )

        rospy.loginfo(
            "=============================================="
        )

        rospy.loginfo(
            " Region Allocator Started"
        )

        rospy.loginfo(
            "=============================================="
        )

        rospy.loginfo(
            "covered_map topic : %s",
            self.covered_map_topic
        )

        rospy.loginfo(
            "region_map topic  : %s",
            self.region_map_topic
        )

        rospy.loginfo(
            "robot1            : %s",
            self.robot1_name
        )

        rospy.loginfo(
            "robot2            : %s",
            self.robot2_name
        )

        rospy.loginfo(
            "update rate       : %.2f Hz",
            self.update_rate
        )

        rospy.loginfo(
            "inflation radius  : %.2f m",
            self.inflation_radius
        )

        rospy.loginfo(
            "covered inflation : %.2f m",
            self.covered_inflation_radius
        )

        rospy.loginfo(
            "----------------------------------------------"
        )

        rospy.loginfo(
            "Allocation rule:"
        )

        rospy.loginfo(
            "  covered_map == 0   -> TASK"
        )

        rospy.loginfo(
            "  covered_map == 100 -> TRAVERSABLE / NO TASK"
        )

        rospy.loginfo(
            "  covered_map == -1  -> NON-FREE"
        )

        rospy.loginfo(
            "=============================================="
        )

    # ================================================================
    # Callback: covered map
    # ================================================================

    def covered_map_callback(
        self,
        msg
    ):

        self.covered_map = msg

        self.last_map_stamp = msg.header.stamp

    # ================================================================
    # Callback: Gazebo model states
    # ================================================================

    def model_states_callback(
        self,
        msg
    ):

        # ------------------------------------------------------------
        # Robot 1
        # ------------------------------------------------------------

        if self.robot1_name in msg.name:

            idx = msg.name.index(
                self.robot1_name
            )

            pose = msg.pose[idx]

            self.robot1_pose = (
                pose.position.x,
                pose.position.y
            )

        # ------------------------------------------------------------
        # Robot 2
        # ------------------------------------------------------------

        if self.robot2_name in msg.name:

            idx = msg.name.index(
                self.robot2_name
            )

            pose = msg.pose[idx]

            self.robot2_pose = (
                pose.position.x,
                pose.position.y
            )

    # ================================================================
    # Timer callback
    # ================================================================

    def timer_callback(
        self,
        event
    ):

        if self.covered_map is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for covered_map..."
            )

            return

        if self.robot1_pose is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for robot1 pose..."
            )

            return

        if self.robot2_pose is None:

            rospy.logwarn_throttle(
                5.0,
                "Waiting for robot2 pose..."
            )

            return

        # ============================================================
        # 第一次运行
        # ============================================================

        if (
            self.last_robot1_pose is None
            or
            self.last_robot2_pose is None
        ):

            rospy.loginfo(
                "Initial robot positions received, "
                "allocating regions..."
            )

            self.allocate()

            self.last_robot1_pose = (
                self.robot1_pose
            )

            self.last_robot2_pose = (
                self.robot2_pose
            )

            return

        # ============================================================
        # 根据机器人移动距离判断是否重新分配
        # ============================================================

        if self.reallocate_on_robot_move:

            robot1_moved = self.distance(
                self.robot1_pose,
                self.last_robot1_pose
            )

            robot2_moved = self.distance(
                self.robot2_pose,
                self.last_robot2_pose
            )

            if (
                robot1_moved >= self.min_robot_move
                or
                robot2_moved >= self.min_robot_move
            ):

                rospy.loginfo(
                    "Robot moved: "
                    "robot1=%.2f m, "
                    "robot2=%.2f m. "
                    "Reallocating...",
                    robot1_moved,
                    robot2_moved
                )

                self.allocate()

                self.last_robot1_pose = (
                    self.robot1_pose
                )

                self.last_robot2_pose = (
                    self.robot2_pose
                )

    # ================================================================
    # Distance
    # ================================================================

    @staticmethod
    def distance(
        p1,
        p2
    ):

        if p1 is None or p2 is None:
            return float("inf")

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        return np.sqrt(
            dx * dx
            +
            dy * dy
        )

    # ================================================================
    # World -> Grid
    # ================================================================

    def world_to_grid(
        self,
        x,
        y,
        map_msg
    ):
        """
        返回：

            gx = column
            gy = row
        """

        resolution = map_msg.info.resolution

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        gx = int(
            np.floor(
                (x - origin_x) /
                resolution
            )
        )

        gy = int(
            np.floor(
                (y - origin_y) /
                resolution
            )
        )

        return gx, gy

    # ================================================================
    # Grid -> World
    # ================================================================

    def grid_to_world(
        self,
        gx,
        gy,
        map_msg
    ):

        resolution = map_msg.info.resolution

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        x = (
            origin_x
            +
            (gx + 0.5) *
            resolution
        )

        y = (
            origin_y
            +
            (gy + 0.5) *
            resolution
        )

        return x, y

    # ================================================================
    # Check grid valid
    # ================================================================

    @staticmethod
    def is_valid_grid(
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
    # Find nearest traversable cell
    # ================================================================

    def find_nearest_traversable_cell(
        self,
        start_gx,
        start_gy,
        traversable_mask,
        max_radius=20
    ):
        """
        寻找机器人附近最近的可通行栅格。

        covered_map == 0
        covered_map == 100

        都属于 traversable。
        """

        height, width = traversable_mask.shape

        if self.is_valid_grid(
            start_gx,
            start_gy,
            width,
            height
        ):

            if traversable_mask[
                start_gy,
                start_gx
            ]:

                return (
                    start_gx,
                    start_gy
                )

        for radius in range(
            1,
            max_radius + 1
        ):

            for dx in range(
                -radius,
                radius + 1
            ):

                for dy in range(
                    -radius,
                    radius + 1
                ):

                    if (
                        abs(dx) != radius
                        and
                        abs(dy) != radius
                    ):
                        continue

                    gx = start_gx + dx
                    gy = start_gy + dy

                    if not self.is_valid_grid(
                        gx,
                        gy,
                        width,
                        height
                    ):
                        continue

                    if traversable_mask[
                        gy,
                        gx
                    ]:

                        return (
                            gx,
                            gy
                        )

        return None

    # ================================================================
    # Get robot cell
    # ================================================================

    def get_robot_cell(
        self,
        robot_pose,
        map_msg,
        traversable_mask
    ):

        gx, gy = self.world_to_grid(
            robot_pose[0],
            robot_pose[1],
            map_msg
        )

        height, width = traversable_mask.shape

        if self.is_valid_grid(
            gx,
            gy,
            width,
            height
        ):

            if traversable_mask[
                gy,
                gx
            ]:

                return (
                    gx,
                    gy
                )

        return self.find_nearest_traversable_cell(
            gx,
            gy,
            traversable_mask,
            max_radius=20
        )

    # ================================================================
    # Neighbors
    # ================================================================

    @staticmethod
    def get_neighbors(
        x,
        y,
        width,
        height
    ):

        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]

        result = []

        for nx, ny in neighbors:

            if (
                0 <= nx < width
                and
                0 <= ny < height
            ):

                result.append(
                    (
                        nx,
                        ny
                    )
                )

        return result

    # ================================================================
    # Geodesic Voronoi
    # ================================================================

    def geodesic_voronoi(
        self,
        traversable_mask,
        task_mask,
        robot1_cell,
        robot2_cell
    ):
        """
        多源 Dijkstra。

        traversable：

            0   -> 可以传播
            100 -> 可以传播

        task：

            0 -> 最终保留标签
        """

        height, width = traversable_mask.shape

        distances = np.full(
            (height, width),
            np.inf,
            dtype=np.float64
        )

        labels = np.zeros(
            (height, width),
            dtype=np.int8
        )

        heap = []

        # ============================================================
        # Robot 1
        # ============================================================

        if robot1_cell is not None:

            x1, y1 = robot1_cell

            distances[
                y1,
                x1
            ] = 0.0

            labels[
                y1,
                x1
            ] = 1

            heapq.heappush(
                heap,
                (
                    0.0,
                    1,
                    x1,
                    y1
                )
            )

        # ============================================================
        # Robot 2
        # ============================================================

        if robot2_cell is not None:

            x2, y2 = robot2_cell

            if distances[
                y2,
                x2
            ] > 0.0:

                distances[
                    y2,
                    x2
                ] = 0.0

                labels[
                    y2,
                    x2
                ] = 2

                heapq.heappush(
                    heap,
                    (
                        0.0,
                        2,
                        x2,
                        y2
                    )
                )

        # ============================================================
        # Dijkstra
        # ============================================================

        while heap:

            current_dist, robot_id, x, y = heapq.heappop(
                heap
            )

            if current_dist > distances[
                y,
                x
            ]:

                continue

            for nx, ny in self.get_neighbors(
                x,
                y,
                width,
                height
            ):

                if not traversable_mask[
                    ny,
                    nx
                ]:

                    continue

                new_dist = (
                    current_dist
                    +
                    1.0
                )

                old_dist = distances[
                    ny,
                    nx
                ]

                if new_dist < old_dist:

                    distances[
                        ny,
                        nx
                    ] = new_dist

                    labels[
                        ny,
                        nx
                    ] = robot_id

                    heapq.heappush(
                        heap,
                        (
                            new_dist,
                            robot_id,
                            nx,
                            ny
                        )
                    )

                elif new_dist == old_dist:

                    # Robot1 优先
                    if robot_id < labels[
                        ny,
                        nx
                    ]:

                        labels[
                            ny,
                            nx
                        ] = robot_id

                        heapq.heappush(
                            heap,
                            (
                                new_dist,
                                robot_id,
                                nx,
                                ny
                            )
                        )

        # ============================================================
        # 最终只保留 task_mask
        # ============================================================

        labels[
            ~task_mask
        ] = 0

        return labels, distances

    # ================================================================
    # Inflate obstacles
    # ================================================================

    def inflate_obstacles(
        self,
        map_array,
        resolution,
        inflation_radius=None
    ):

        obstacle_mask = (
            map_array < 0
        )

        if inflation_radius is None:
            inflation_radius = self.inflation_radius

        if inflation_radius <= 0.0:

            return obstacle_mask

        radius_cells = int(
            np.ceil(
                inflation_radius /
                resolution
            )
        )

        if radius_cells <= 0:

            return obstacle_mask

        # ============================================================
        # scipy
        # ============================================================

        try:

            from scipy.ndimage import binary_dilation

            yy, xx = np.ogrid[
                -radius_cells:
                radius_cells + 1,
                -radius_cells:
                radius_cells + 1
            ]

            kernel = (
                xx * xx
                +
                yy * yy
                <=
                radius_cells * radius_cells
            )

            inflated = binary_dilation(
                obstacle_mask,
                structure=kernel
            )

            return inflated

        except ImportError:

            rospy.logwarn_throttle(
                10.0,
                "scipy not available, "
                "using slow obstacle inflation."
            )

        # ============================================================
        # Fallback
        # ============================================================

        height, width = obstacle_mask.shape

        inflated = obstacle_mask.copy()

        obstacle_indices = np.argwhere(
            obstacle_mask
        )

        for gy, gx in obstacle_indices:

            y_min = max(
                0,
                gy - radius_cells
            )

            y_max = min(
                height,
                gy + radius_cells + 1
            )

            x_min = max(
                0,
                gx - radius_cells
            )

            x_max = min(
                width,
                gx + radius_cells + 1
            )

            for ny in range(
                y_min,
                y_max
            ):

                for nx in range(
                    x_min,
                    x_max
                ):

                    dx = nx - gx
                    dy = ny - gy

                    if (
                        dx * dx
                        +
                        dy * dy
                        <=
                        radius_cells * radius_cells
                    ):

                        inflated[
                            ny,
                            nx
                        ] = True

        return inflated

    # ================================================================
    # Create region map
    # ================================================================

    def create_region_map(
        self,
        labels,
        task_mask,
        inflated_obstacle_mask,
        original_map
    ):

        region_map = np.full(
            labels.shape,
            -1,
            dtype=np.int8
        )

        robot1_mask = (
            (labels == 1)
            &
            task_mask
            &
            (~inflated_obstacle_mask)
        )

        robot2_mask = (
            (labels == 2)
            &
            task_mask
            &
            (~inflated_obstacle_mask)
        )

        region_map[
            robot1_mask
        ] = 0

        region_map[
            robot2_mask
        ] = 100

        return region_map

    # ================================================================
    # Create single robot region
    # ================================================================

    def create_single_region(
        self,
        labels,
        task_mask,
        robot_id,
        inflated_obstacle_mask
    ):

        region = np.full(
            labels.shape,
            -1,
            dtype=np.int8
        )

        valid_task = (
            task_mask
            &
            (~inflated_obstacle_mask)
        )

        # 所有有效任务区域
        # 先设置成 0
        region[
            valid_task
        ] = 0

        # 自己负责的任务
        # 设置成 100
        region[
            valid_task
            &
            (labels == robot_id)
        ] = 100

        return region

    # ================================================================
    # Publish OccupancyGrid
    # ================================================================

    def publish_grid(
        self,
        data_array,
        source_map,
        publisher
    ):

        msg = OccupancyGrid()

        msg.header = source_map.header

        msg.header.frame_id = (
            source_map.header.frame_id
        )

        msg.info = source_map.info

        msg.data = (
            data_array
            .astype(np.int8)
            .flatten()
            .tolist()
        )

        publisher.publish(
            msg
        )

    # ================================================================
    # Main allocation
    # ================================================================

    def allocate(self):

        if self.covered_map is None:
            return

        if self.robot1_pose is None:
            return

        if self.robot2_pose is None:
            return

        map_msg = self.covered_map

        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution

        # ============================================================
        # Convert map
        # ============================================================

        map_array = np.asarray(
            map_msg.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ============================================================
        # Task mask
        # ============================================================

        task_mask = (
            map_array == 0
        )

        # ============================================================
        # Covered inflation
        # 已覆盖区域 100 向外膨胀，膨胀区域不再作为任务
        # ============================================================

        if self.covered_inflation_radius > 0.0:

            covered_source_map = np.where(
                map_array == 100,
                -1,
                0
            )

            inflated_covered_mask = self.inflate_obstacles(
                covered_source_map,
                resolution,
                self.covered_inflation_radius
            )

            task_mask = (
                task_mask
                &
                (~inflated_covered_mask)
            )

        # ============================================================
        # Traversable mask
        #
        # 0   -> 可以走
        # 100 -> 可以走
        #
        # -1  -> 不可以走
        # ============================================================

        traversable_mask = (
            (map_array == 0)
            |
            (map_array == 100)
        )

        # ============================================================
        # Statistics
        # ============================================================

        task_count = int(
            np.sum(
                task_mask
            )
        )

        covered_count = int(
            np.sum(
                map_array == 100
            )
        )

        obstacle_count = int(
            np.sum(
                map_array < 0
            )
        )

        rospy.loginfo(
            "Map statistics: "
            "task(0)=%d, "
            "covered(100)=%d, "
            "non-free(-1)=%d",
            task_count,
            covered_count,
            obstacle_count
        )

        # ============================================================
        # Robot grid cells
        # ============================================================

        robot1_cell = self.get_robot_cell(
            self.robot1_pose,
            map_msg,
            traversable_mask
        )

        robot2_cell = self.get_robot_cell(
            self.robot2_pose,
            map_msg,
            traversable_mask
        )

        # ============================================================
        # Check
        # ============================================================

        if robot1_cell is None:

            rospy.logwarn(
                "Cannot find traversable cell near "
                "robot1 position (%.2f, %.2f)",
                self.robot1_pose[0],
                self.robot1_pose[1]
            )

            return

        if robot2_cell is None:

            rospy.logwarn(
                "Cannot find traversable cell near "
                "robot2 position (%.2f, %.2f)",
                self.robot2_pose[0],
                self.robot2_pose[1]
            )

            return

        rospy.loginfo(
            "Robot1 grid cell: (%d, %d)",
            robot1_cell[0],
            robot1_cell[1]
        )

        rospy.loginfo(
            "Robot2 grid cell: (%d, %d)",
            robot2_cell[0],
            robot2_cell[1]
        )

        # ============================================================
        # Dijkstra allocation
        # ============================================================

        labels, distances = self.geodesic_voronoi(
            traversable_mask,
            task_mask,
            robot1_cell,
            robot2_cell
        )

        # ============================================================
        # Obstacle inflation
        # ============================================================

        inflated_obstacle_mask = self.inflate_obstacles(
            map_array,
            resolution
        )

        # ============================================================
        # Create region maps
        # ============================================================

        region_map = self.create_region_map(
            labels,
            task_mask,
            inflated_obstacle_mask,
            map_msg
        )

        robot1_region = self.create_single_region(
            labels,
            task_mask,
            robot_id=1,
            inflated_obstacle_mask=inflated_obstacle_mask
        )

        robot2_region = self.create_single_region(
            labels,
            task_mask,
            robot_id=2,
            inflated_obstacle_mask=inflated_obstacle_mask
        )

        # ============================================================
        # Allocation statistics
        # ============================================================

        robot1_task_count = int(
            np.sum(
                region_map == 0
            )
        )

        robot2_task_count = int(
            np.sum(
                region_map == 100
            )
        )

        allocated_count = (
            robot1_task_count
            +
            robot2_task_count
        )

        rospy.loginfo(
            "Allocation result: "
            "robot1=%d cells, "
            "robot2=%d cells, "
            "total=%d/%d",
            robot1_task_count,
            robot2_task_count,
            allocated_count,
            task_count
        )

        # ============================================================
        # Publish
        # ============================================================

        self.publish_grid(
            region_map,
            map_msg,
            self.region_map_pub
        )

        self.publish_grid(
            robot1_region,
            map_msg,
            self.robot1_region_pub
        )

        self.publish_grid(
            robot2_region,
            map_msg,
            self.robot2_region_pub
        )

        rospy.loginfo(
            "Region maps published."
        )


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    try:

        allocator = RegionAllocator()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass