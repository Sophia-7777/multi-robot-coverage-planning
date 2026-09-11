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

    covered_map 的数据定义：
        -1 : 非自由区域 / 未知区域
         0 : 未覆盖的自由区域 -> 需要进行任务分配
       100 : 已覆盖的自由区域 -> 不作为任务分配区域

    分配策略：
        1. 只有 covered_map == 0 的区域最终会被分配
        2. covered_map == 100 的区域不会产生任务
        3. 但是 100 区域允许作为机器人移动/传播路径
        4. 使用多源 Dijkstra，根据机器人到各任务单元的栅格距离进行划分
        5. 两个机器人分别得到自己的区域
    """

    def __init__(self):
        rospy.init_node("region_allocator_node")

        # ============================================================
        # Parameters
        # ============================================================

        # 地图 topic
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

        # 机器人名称
        self.robot1_name = rospy.get_param(
            "~robot1_name",
            "robot1"
        )

        self.robot2_name = rospy.get_param(
            "~robot2_name",
            "robot2"
        )

        # 更新频率
        self.update_rate = rospy.get_param(
            "~update_rate",
            1.0
        )

        # 是否根据机器人移动重新分配
        self.reallocate_on_robot_move = rospy.get_param(
            "~reallocate_on_robot_move",
            True
        )

        # 机器人移动多少距离后重新分配
        self.min_robot_move = rospy.get_param(
            "~min_robot_move",
            0.5
        )

        # 障碍物膨胀半径
        self.inflation_radius = rospy.get_param(
            "~inflation_radius",
            0.30
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
            rospy.Duration(1.0 / self.update_rate),
            self.timer_callback
        )

        rospy.loginfo("==============================================")
        rospy.loginfo(" Region Allocator Started")
        rospy.loginfo("==============================================")
        rospy.loginfo("covered_map topic : %s", self.covered_map_topic)
        rospy.loginfo("region_map topic  : %s", self.region_map_topic)
        rospy.loginfo("robot1            : %s", self.robot1_name)
        rospy.loginfo("robot2            : %s", self.robot2_name)
        rospy.loginfo("update rate       : %.2f Hz", self.update_rate)
        rospy.loginfo("inflation radius  : %.2f m", self.inflation_radius)
        rospy.loginfo("----------------------------------------------")
        rospy.loginfo("Allocation rule:")
        rospy.loginfo("  covered_map == 0   -> TASK")
        rospy.loginfo("  covered_map == 100 -> ALREADY COVERED")
        rospy.loginfo("  covered_map == -1  -> NON-FREE")
        rospy.loginfo("==============================================")

    # ================================================================
    # Callback: covered map
    # ================================================================

    def covered_map_callback(self, msg):
        """
        保存最新 covered_map
        """

        self.covered_map = msg

        self.last_map_stamp = msg.header.stamp

    # ================================================================
    # Callback: Gazebo model states
    # ================================================================

    def model_states_callback(self, msg):
        """
        从 Gazebo ModelStates 中获取两个机器人位置
        """

        # robot1
        if self.robot1_name in msg.name:
            idx = msg.name.index(self.robot1_name)

            pose = msg.pose[idx]

            self.robot1_pose = (
                pose.position.x,
                pose.position.y
            )

        # robot2
        if self.robot2_name in msg.name:
            idx = msg.name.index(self.robot2_name)

            pose = msg.pose[idx]

            self.robot2_pose = (
                pose.position.x,
                pose.position.y
            )

    # ================================================================
    # Timer callback
    # ================================================================

    def timer_callback(self, event):
        """
        定期检查是否需要进行区域重新分配
        """

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

        if self.last_robot1_pose is None or self.last_robot2_pose is None:

            rospy.loginfo(
                "Initial robot positions received, allocating regions..."
            )

            self.allocate()

            self.last_robot1_pose = self.robot1_pose
            self.last_robot2_pose = self.robot2_pose

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
                    "Robot moved: robot1=%.2f m, robot2=%.2f m. "
                    "Reallocating...",
                    robot1_moved,
                    robot2_moved
                )

                self.allocate()

                self.last_robot1_pose = self.robot1_pose
                self.last_robot2_pose = self.robot2_pose

    # ================================================================
    # Distance
    # ================================================================

    def distance(self, p1, p2):
        """
        计算两个二维坐标之间的欧氏距离
        """

        if p1 is None or p2 is None:
            return float("inf")

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        return np.sqrt(dx * dx + dy * dy)

    # ================================================================
    # World -> Grid
    # ================================================================

    def world_to_grid(self, x, y, map_msg):
        """
        世界坐标 -> 栅格坐标

        注意：
        必须考虑 OccupancyGrid 的 origin。
        """

        resolution = map_msg.info.resolution

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        gx = int(np.floor(
            (x - origin_x) / resolution
        ))

        gy = int(np.floor(
            (y - origin_y) / resolution
        ))

        return gx, gy

    # ================================================================
    # Grid -> World
    # ================================================================

    def grid_to_world(self, gx, gy, map_msg):
        """
        栅格坐标 -> 世界坐标
        返回栅格中心点
        """

        resolution = map_msg.info.resolution

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        x = origin_x + (gx + 0.5) * resolution
        y = origin_y + (gy + 0.5) * resolution

        return x, y

    # ================================================================
    # Check grid valid
    # ================================================================

    def is_valid_grid(self, gx, gy, width, height):
        """
        判断栅格坐标是否在地图范围内
        """

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
        寻找距离机器人最近的可通行栅格。

        traversable_mask:
            True  -> 可以作为路径
            False -> 不可通行

        这样即使机器人当前位于 covered_map == 100，
        也可以找到附近的 100 或 0 栅格作为 Dijkstra 起点。
        """

        height, width = traversable_mask.shape

        if self.is_valid_grid(
            start_gx,
            start_gy,
            width,
            height
        ):
            if traversable_mask[start_gy, start_gx]:
                return start_gx, start_gy

        # 螺旋/半径搜索
        for radius in range(1, max_radius + 1):

            for dx in range(-radius, radius + 1):

                for dy in range(-radius, radius + 1):

                    if abs(dx) != radius and abs(dy) != radius:
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

                    if traversable_mask[gy, gx]:

                        return gx, gy

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
        """
        获取机器人当前所在的栅格。

        注意：
        机器人可以位于 covered_map == 100。
        因此这里不是寻找 task cell，
        而是寻找 traversable cell。
        """

        gx, gy = self.world_to_grid(
            robot_pose[0],
            robot_pose[1],
            map_msg
        )

        height, width = traversable_mask.shape

        # 当前栅格可以通行
        if self.is_valid_grid(
            gx,
            gy,
            width,
            height
        ):
            if traversable_mask[gy, gx]:
                return gx, gy

        # 否则搜索附近可通行区域
        return self.find_nearest_traversable_cell(
            gx,
            gy,
            traversable_mask,
            max_radius=20
        )

    # ================================================================
    # Neighbors
    # ================================================================

    def get_neighbors(self, x, y, width, height):
        """
        4-connected grid
        """

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
                result.append((nx, ny))

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

        traversable_mask:
            0 或 100
            都可以作为路径。

        task_mask:
            只有 covered_map == 0。

        最终：
            只有 task_mask 中的栅格才会保留机器人标签。

        labels:
            0 -> 无任务
            1 -> Robot1
            2 -> Robot2
        """

        height, width = traversable_mask.shape

        # 距离
        distances = np.full(
            (height, width),
            np.inf,
            dtype=np.float64
        )

        # 所属机器人
        labels = np.zeros(
            (height, width),
            dtype=np.int8
        )

        # Priority Queue
        heap = []

        # ============================================================
        # Robot 1 source
        # ============================================================

        if robot1_cell is not None:

            x1, y1 = robot1_cell

            distances[y1, x1] = 0.0
            labels[y1, x1] = 1

            heapq.heappush(
                heap,
                (0.0, 1, x1, y1)
            )

        # ============================================================
        # Robot 2 source
        # ============================================================

        if robot2_cell is not None:

            x2, y2 = robot2_cell

            # 如果两个机器人恰好位于同一个栅格
            if distances[y2, x2] > 0.0:

                distances[y2, x2] = 0.0
                labels[y2, x2] = 2

                heapq.heappush(
                    heap,
                    (0.0, 2, x2, y2)
                )

        # ============================================================
        # Dijkstra
        # ============================================================

        while heap:

            current_dist, robot_id, x, y = heapq.heappop(heap)

            # 跳过旧状态
            if current_dist > distances[y, x]:
                continue

            for nx, ny in self.get_neighbors(
                x,
                y,
                width,
                height
            ):

                # 不能穿过非 traversable 区域
                if not traversable_mask[ny, nx]:
                    continue

                new_dist = current_dist + 1.0

                old_dist = distances[ny, nx]

                # 更短路径
                if new_dist < old_dist:

                    distances[ny, nx] = new_dist
                    labels[ny, nx] = robot_id

                    heapq.heappush(
                        heap,
                        (
                            new_dist,
                            robot_id,
                            nx,
                            ny
                        )
                    )

                # 距离相同
                elif new_dist == old_dist:

                    # Robot1 优先
                    if robot_id < labels[ny, nx]:

                        labels[ny, nx] = robot_id

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
        # 非任务区域全部清零
        # ============================================================

        # 这是本次修改最关键的一步。
        #
        # 即使 Dijkstra 穿过 100 区域，
        # 这些区域也不能成为最终任务区域。
        #
        # 因此：
        #
        #     labels[covered_map != 0] = 0
        #
        # 最终只有 covered_map == 0 的区域拥有机器人标签。

        labels[~task_mask] = 0

        return labels, distances

    # ================================================================
    # Inflate obstacles
    # ================================================================

    def inflate_obstacles(
        self,
        map_array,
        resolution
    ):
        """
        对非自由区域进行膨胀。

        对 /covered_map 来说：

            -1 -> 非自由区域 / unknown
             0 -> 未覆盖自由区域
           100 -> 已覆盖自由区域

        因此：
            obstacle_mask = map_array < 0
        """

        obstacle_mask = (
            map_array < 0
        )

        if self.inflation_radius <= 0.0:
            return obstacle_mask

        radius_cells = int(
            np.ceil(
                self.inflation_radius /
                resolution
            )
        )

        if radius_cells <= 0:
            return obstacle_mask

        # ============================================================
        # 优先使用 scipy
        # ============================================================

        try:

            from scipy.ndimage import binary_dilation

            yy, xx = np.ogrid[
                -radius_cells:radius_cells + 1,
                -radius_cells:radius_cells + 1
            ]

            kernel = (
                xx * xx + yy * yy
                <= radius_cells * radius_cells
            )

            inflated = binary_dilation(
                obstacle_mask,
                structure=kernel
            )

            return inflated

        except ImportError:

            rospy.logwarn_throttle(
                10.0,
                "scipy not available, using slow obstacle inflation."
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
                        dx * dx + dy * dy
                        <= radius_cells * radius_cells
                    ):

                        inflated[ny, nx] = True

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
        """
        创建总区域地图。

        输出定义：

            -1 -> 非任务区域
                 包括：
                 * obstacle
                 * unknown
                 * 已覆盖区域 100

             0 -> Robot1 负责的未覆盖区域

            100 -> Robot2 负责的未覆盖区域
        """

        region_map = np.full(
            labels.shape,
            -1,
            dtype=np.int8
        )

        # ============================================================
        # 只有 task_mask == True 才允许输出任务
        # ============================================================

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

        region_map[robot1_mask] = 0
        region_map[robot2_mask] = 100

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
        """
        创建单机器人区域地图。

        对 robot_id：

            100 -> 该机器人负责的任务
              0 -> 另一个机器人负责的任务
             -1 -> 非任务区域

        这样可以继续保持原来节点的接口形式。
        """

        region = np.full(
            labels.shape,
            -1,
            dtype=np.int8
        )

        # ============================================================
        # 只有 task_mask 才是任务
        # ============================================================

        valid_task = (
            task_mask
            &
            (~inflated_obstacle_mask)
        )

        # 所有有效任务区域先设为 0
        region[valid_task] = 0

        # 自己负责的区域设为 100
        region[
            valid_task &
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
        """
        发布 OccupancyGrid
        """

        msg = OccupancyGrid()

        # Header
        msg.header = source_map.header

        # 保持 map frame
        msg.header.frame_id = source_map.header.frame_id

        # Map info
        msg.info = source_map.info

        # 转成 int8
        msg.data = data_array.astype(
            np.int8
        ).flatten().tolist()

        publisher.publish(msg)

    # ================================================================
    # Main allocation
    # ================================================================

    def allocate(self):
        """
        执行一次完整的区域分配。
        """

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
        # Convert map to numpy
        # ============================================================

        map_array = np.asarray(
            map_msg.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ============================================================
        # 核心修改
        # ============================================================

        # ------------------------------------------------------------
        # task_mask
        #
        # 只有 covered_map == 0 才是需要分配的任务。
        #
        # 100 = 已经覆盖
        # -1  = 非自由/未知
        # ------------------------------------------------------------

        task_mask = (
            map_array == 0
        )

        # ------------------------------------------------------------
        # traversable_mask
        #
        # 允许机器人在：
        #
        #     0   未覆盖区域
        #     100 已覆盖区域
        #
        # 中移动。
        #
        # 但是最终只有 0 会被分配。
        # ------------------------------------------------------------

        traversable_mask = (
            (map_array == 0)
            |
            (map_array == 100)
        )

        # ============================================================
        # Statistics
        # ============================================================

        task_count = int(
            np.sum(task_mask)
        )

        covered_count = int(
            np.sum(map_array == 100)
        )

        obstacle_count = int(
            np.sum(map_array < 0)
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
        # Get robot grid cells
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
        # Check robot cells
        # ============================================================

        if robot1_cell is None:

            rospy.logwarn(
                "Cannot find traversable cell near robot1 "
                "position (%.2f, %.2f)",
                self.robot1_pose[0],
                self.robot1_pose[1]
            )

            return

        if robot2_cell is None:

            rospy.logwarn(
                "Cannot find traversable cell near robot2 "
                "position (%.2f, %.2f)",
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
        # Statistics of allocation
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