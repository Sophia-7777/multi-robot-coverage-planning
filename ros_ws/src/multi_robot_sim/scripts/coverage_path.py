#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coverage_path.py

ROS1 Dual Robot Coverage Path Planner

规划结构：

    Robot Current Position
             |
             v
        A* 起点连接
             |
             v
           BCD
             |
             v
       BCD Cells
             |
             v
   双向 Zig-Zag 候选路径
        /           \
       v             v
   Forward        Reverse
        \           /
         v         v
       最短连接代价
             |
             v
       Cell 顺序优化
             |
             v
            A*
             |
             v
      Continuous Path
             |
             v
       Direction Arrows


输入：

    /robot1_region
    /robot2_region

    /gazebo/model_states

输出：

    /robot1_coverage_path
    /robot2_coverage_path

    /robot1_bcd_cells
    /robot2_bcd_cells

    /robot1_path_arrows
    /robot2_path_arrows


RegionAllocator 编码：

    100 = 当前机器人自己的区域
      0 = 其他机器人区域
     -1 = unknown / obstacle / outside


坐标：

    numpy:
        map_array[row, col]

    grid:
        (col, row)

    world:
        x, y


核心优化：

1. 机器人当前位置作为真正规划起点
2. BCD 分解
3. 每个 Cell 生成 Forward / Reverse 两种 Zig-Zag
4. Cell 顺序考虑实际 A* 连接代价
5. 每个 Cell 的进入方向自动选择
6. Cell 之间使用 A*
7. A* 严格限制在当前机器人 Region
8. 最终 Path 自动计算 yaw
9. RViz 增加方向箭头
"""

import heapq
import math

import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

from bcd import BoustrophedonDecomposition


class CoveragePathPlanner:

    def __init__(self):

        rospy.init_node("dual_robot_coverage_path_node")

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

        self.robot1_region_topic = rospy.get_param(
            "~robot1_region_topic",
            "/robot1_region"
        )

        self.robot2_region_topic = rospy.get_param(
            "~robot2_region_topic",
            "/robot2_region"
        )

        self.model_states_topic = rospy.get_param(
            "~model_states_topic",
            "/gazebo/model_states"
        )

        self.robot1_path_topic = rospy.get_param(
            "~robot1_path_topic",
            "/robot1_coverage_path"
        )

        self.robot2_path_topic = rospy.get_param(
            "~robot2_path_topic",
            "/robot2_coverage_path"
        )

        self.robot1_cell_marker_topic = rospy.get_param(
            "~robot1_cell_marker_topic",
            "/robot1_bcd_cells"
        )

        self.robot2_cell_marker_topic = rospy.get_param(
            "~robot2_cell_marker_topic",
            "/robot2_bcd_cells"
        )

        self.robot1_arrow_topic = rospy.get_param(
            "~robot1_arrow_topic",
            "/robot1_path_arrows"
        )

        self.robot2_arrow_topic = rospy.get_param(
            "~robot2_arrow_topic",
            "/robot2_path_arrows"
        )

        # ============================================================
        # Coverage
        # ============================================================

        self.coverage_spacing = rospy.get_param(
            "~coverage_spacing",
            0.25
        )

        self.min_cell_size = rospy.get_param(
            "~min_cell_size",
            1
        )

        self.scan_direction = rospy.get_param(
            "~scan_direction",
            "horizontal"
        )

        # ============================================================
        # A*
        # ============================================================

        self.enable_astar = rospy.get_param(
            "~enable_astar",
            True
        )

        self.astar_max_iterations = rospy.get_param(
            "~astar_max_iterations",
            100000
        )

        # ============================================================
        # Cell ordering
        # ============================================================

        self.enable_cell_order_optimization = rospy.get_param(
            "~enable_cell_order_optimization",
            True
        )

        # 候选 Cell 太多时，A* 两两计算会比较耗时。
        # 这个参数限制每一步最多考虑多少个最近 Cell。
        #
        # 0 = 所有 Cell
        #
        self.max_order_candidates = rospy.get_param(
            "~max_order_candidates",
            0
        )

        # ============================================================
        # Path arrows
        # ============================================================

        self.enable_path_arrows = rospy.get_param(
            "~enable_path_arrows",
            True
        )

        # 每隔多少个 grid point 画一个箭头
        self.arrow_step = rospy.get_param(
            "~arrow_step",
            8
        )

        self.arrow_length = rospy.get_param(
            "~arrow_length",
            0.18
        )

        self.arrow_width = rospy.get_param(
            "~arrow_width",
            0.05
        )

        self.arrow_height = rospy.get_param(
            "~arrow_height",
            0.04
        )

        # ============================================================
        # Data
        # ============================================================

        self.robot1_region_map = None
        self.robot2_region_map = None

        self.robot_positions = {
            1: None,
            2: None
        }

        self.robot1_processing = False
        self.robot2_processing = False

        # ============================================================
        # Subscribers
        # ============================================================

        self.robot1_region_sub = rospy.Subscriber(
            self.robot1_region_topic,
            OccupancyGrid,
            self.robot1_region_callback,
            queue_size=1
        )

        self.robot2_region_sub = rospy.Subscriber(
            self.robot2_region_topic,
            OccupancyGrid,
            self.robot2_region_callback,
            queue_size=1
        )

        self.model_states_sub = rospy.Subscriber(
            self.model_states_topic,
            ModelStates,
            self.model_states_callback,
            queue_size=1
        )

        # ============================================================
        # Publishers
        # ============================================================

        self.robot1_path_pub = rospy.Publisher(
            self.robot1_path_topic,
            Path,
            queue_size=1,
            latch=True
        )

        self.robot2_path_pub = rospy.Publisher(
            self.robot2_path_topic,
            Path,
            queue_size=1,
            latch=True
        )

        self.robot1_cell_marker_pub = rospy.Publisher(
            self.robot1_cell_marker_topic,
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.robot2_cell_marker_pub = rospy.Publisher(
            self.robot2_cell_marker_topic,
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.robot1_arrow_pub = rospy.Publisher(
            self.robot1_arrow_topic,
            MarkerArray,
            queue_size=1,
            latch=True
        )

        self.robot2_arrow_pub = rospy.Publisher(
            self.robot2_arrow_topic,
            MarkerArray,
            queue_size=1,
            latch=True
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo("==========================================")
        rospy.loginfo("Optimized Dual Robot Coverage Planner")
        rospy.loginfo("==========================================")

        rospy.loginfo(
            "Robot1: %s",
            self.robot1_name
        )

        rospy.loginfo(
            "Robot2: %s",
            self.robot2_name
        )

        rospy.loginfo(
            "Coverage spacing: %.3f m",
            self.coverage_spacing
        )

        rospy.loginfo(
            "Scan direction: %s",
            self.scan_direction
        )

        rospy.loginfo(
            "A* enabled: %s",
            self.enable_astar
        )

        rospy.loginfo(
            "Cell order optimization: %s",
            self.enable_cell_order_optimization
        )

        rospy.loginfo(
            "Path arrows: %s",
            self.enable_path_arrows
        )

        rospy.loginfo("==========================================")


    # =================================================================
    # Gazebo Model States
    # =================================================================

    def model_states_callback(self, msg):

        # ------------------------------------------------------------
        # Robot 1
        # ------------------------------------------------------------

        if self.robot1_name in msg.name:

            index = msg.name.index(self.robot1_name)

            pose = msg.pose[index]

            self.robot_positions[1] = (
                pose.position.x,
                pose.position.y
            )

        # ------------------------------------------------------------
        # Robot 2
        # ------------------------------------------------------------

        if self.robot2_name in msg.name:

            index = msg.name.index(self.robot2_name)

            pose = msg.pose[index]

            self.robot_positions[2] = (
                pose.position.x,
                pose.position.y
            )


    # =================================================================
    # Robot 1 callback
    # =================================================================

    def robot1_region_callback(self, msg):

        self.robot1_region_map = msg

        if self.robot1_processing:
            return

        self.robot1_processing = True

        try:

            self.process_robot(
                robot_id=1,
                region_msg=msg
            )

        except Exception as e:

            rospy.logerr(
                "Robot1 processing failed: %s",
                str(e)
            )

        finally:

            self.robot1_processing = False


    # =================================================================
    # Robot 2 callback
    # =================================================================

    def robot2_region_callback(self, msg):

        self.robot2_region_map = msg

        if self.robot2_processing:
            return

        self.robot2_processing = True

        try:

            self.process_robot(
                robot_id=2,
                region_msg=msg
            )

        except Exception as e:

            rospy.logerr(
                "Robot2 processing failed: %s",
                str(e)
            )

        finally:

            self.robot2_processing = False


    # =================================================================
    # Main processing
    # =================================================================

    def process_robot(
        self,
        robot_id,
        region_msg
    ):

        if region_msg is None:
            return

        width = region_msg.info.width
        height = region_msg.info.height
        resolution = region_msg.info.resolution

        if width <= 0 or height <= 0:
            rospy.logwarn(
                "Robot%d: invalid map.",
                robot_id
            )
            return

        if resolution <= 0:
            rospy.logwarn(
                "Robot%d: invalid resolution.",
                robot_id
            )
            return

        # ============================================================
        # OccupancyGrid -> numpy
        # ============================================================

        map_array = np.asarray(
            region_msg.data,
            dtype=np.int16
        ).reshape(
            height,
            width
        )

        # ============================================================
        # Own region
        # ============================================================

        own_region = (
            map_array == 100
        )

        if not np.any(own_region):

            rospy.logwarn(
                "Robot%d: own region is empty.",
                robot_id
            )

            return

        rospy.loginfo(
            "Robot%d: region cells = %d",
            robot_id,
            int(np.count_nonzero(own_region))
        )

        # ============================================================
        # BCD
        # ============================================================

        bcd = BoustrophedonDecomposition(
            own_region,
            min_cell_size=self.min_cell_size
        )

        cells = bcd.decompose()

        rospy.loginfo(
            "Robot%d: BCD generated %d cells.",
            robot_id,
            len(cells)
        )

        if not cells:
            return

        # ============================================================
        # Print cells
        # ============================================================

        for cell in cells:

            rospy.loginfo(
                "Robot%d Cell %d: size=%d "
                "x=[%d,%d] y=[%d,%d]",
                robot_id,
                cell.id,
                cell.size,
                cell.min_col,
                cell.max_col,
                cell.min_row,
                cell.max_row
            )

        # ============================================================
        # Visualization
        # ============================================================

        self.publish_cell_markers(
            robot_id,
            cells,
            region_msg
        )

        # ============================================================
        # Robot current position
        # ============================================================

        robot_world = self.robot_positions.get(
            robot_id
        )

        if robot_world is None:

            rospy.logwarn(
                "Robot%d: Gazebo position not received yet. "
                "Use first Cell as starting point.",
                robot_id
            )

            start_grid = None

        else:

            start_grid = self.world_to_grid(
                robot_world[0],
                robot_world[1],
                region_msg
            )

            rospy.loginfo(
                "Robot%d current world position: "
                "(%.3f, %.3f)",
                robot_id,
                robot_world[0],
                robot_world[1]
            )

            rospy.loginfo(
                "Robot%d current grid position: %s",
                robot_id,
                str(start_grid)
            )

            # --------------------------------------------------------
            # 如果机器人当前 grid 不属于自己的 Region，
            # 找最近的 own-region cell。
            # --------------------------------------------------------

            if not self.valid_grid_point(
                start_grid[0],
                start_grid[1],
                own_region
            ):

                rospy.logwarn(
                    "Robot%d current grid %s is outside own region. "
                    "Searching nearest own-region cell.",
                    robot_id,
                    str(start_grid)
                )

                start_grid = self.find_nearest_free_cell(
                    start_grid,
                    own_region
                )

                rospy.loginfo(
                    "Robot%d corrected start grid: %s",
                    robot_id,
                    str(start_grid)
                )

        # ============================================================
        # Generate two Zig-Zag candidates for every Cell
        # ============================================================

        cell_candidates = {}

        for cell in cells:

            forward_path = self.generate_cell_coverage(
                cell,
                resolution,
                reverse=False
            )

            reverse_path = self.generate_cell_coverage(
                cell,
                resolution,
                reverse=True
            )

            candidates = []

            if forward_path:

                candidates.append({
                    "path": forward_path,
                    "start": forward_path[0],
                    "end": forward_path[-1],
                    "direction": "forward"
                })

            if reverse_path:

                candidates.append({
                    "path": reverse_path,
                    "start": reverse_path[0],
                    "end": reverse_path[-1],
                    "direction": "reverse"
                })

            if candidates:
                cell_candidates[cell.id] = candidates

        if not cell_candidates:

            rospy.logwarn(
                "Robot%d: no valid Cell coverage paths.",
                robot_id
            )

            return

        # ============================================================
        # Cell order + direction optimization
        # ============================================================

        ordered_plan = self.optimize_cell_order(
            cells,
            cell_candidates,
            start_grid,
            own_region
        )

        if not ordered_plan:

            rospy.logwarn(
                "Robot%d: failed to optimize Cell order.",
                robot_id
            )

            return

        rospy.loginfo(
            "Robot%d optimized Cell plan:",
            robot_id
        )

        for item in ordered_plan:

            rospy.loginfo(
                "    Cell %d | %s | "
                "start=%s end=%s",
                item["cell"].id,
                item["direction"],
                str(item["start"]),
                str(item["end"])
            )

        # ============================================================
        # Construct final continuous path
        # ============================================================

        grid_path = []

        # ------------------------------------------------------------
        # 机器人当前位置 -> 第一个 Cell
        # ------------------------------------------------------------

        first_item = ordered_plan[0]

        first_cell_start = first_item["start"]

        if start_grid is not None:

            if start_grid != first_cell_start:

                rospy.loginfo(
                    "Robot%d: A* robot position -> Cell %d",
                    robot_id,
                    first_item["cell"].id
                )

                start_connection = self.connect_cells_astar(
                    start_grid,
                    first_cell_start,
                    own_region
                )

                if not start_connection:

                    rospy.logwarn(
                        "Robot%d: cannot connect current position "
                        "to first Cell.",
                        robot_id
                    )

                    return

                grid_path.extend(
                    start_connection
                )

            else:

                grid_path.append(
                    start_grid
                )

        # ------------------------------------------------------------
        # Cell coverage
        # ------------------------------------------------------------

        previous_end = (
            grid_path[-1]
            if grid_path
            else None
        )

        for index, item in enumerate(ordered_plan):

            cell = item["cell"]

            cell_path = item["path"]

            # --------------------------------------------------------
            # 第一个 Cell
            # --------------------------------------------------------

            if previous_end is None:

                grid_path.extend(
                    cell_path
                )

                previous_end = cell_path[-1]

                continue

            # --------------------------------------------------------
            # 如果当前位置已经是 Cell 起点
            # --------------------------------------------------------

            if previous_end == cell_path[0]:

                grid_path.extend(
                    cell_path[1:]
                )

                previous_end = cell_path[-1]

                continue

            # --------------------------------------------------------
            # A*
            # --------------------------------------------------------

            rospy.loginfo(
                "Robot%d: A* Cell %d -> Cell %d",
                robot_id,
                ordered_plan[index - 1]["cell"].id
                if index > 0 else -1,
                cell.id
            )

            connection = self.connect_cells_astar(
                previous_end,
                cell_path[0],
                own_region
            )

            if not connection:

                rospy.logwarn(
                    "Robot%d: failed to connect to Cell %d.",
                    robot_id,
                    cell.id
                )

                # 不再加入未连接 Cell
                break

            # --------------------------------------------------------
            # A* 连接
            # --------------------------------------------------------

            if len(connection) > 1:

                grid_path.extend(
                    connection[1:]
                )

            # --------------------------------------------------------
            # Cell coverage
            # --------------------------------------------------------

            grid_path.extend(
                cell_path
            )

            previous_end = cell_path[-1]

        # ============================================================
        # Remove duplicates
        # ============================================================

        grid_path = self.remove_duplicate_points(
            grid_path
        )

        # ============================================================
        # Log
        # ============================================================

        rospy.loginfo(
            "Robot%d: final path = %d grid points.",
            robot_id,
            len(grid_path)
        )

        # ============================================================
        # Publish Path
        # ============================================================

        self.publish_path(
            robot_id,
            grid_path,
            region_msg
        )

        # ============================================================
        # Publish arrows
        # ============================================================

        if self.enable_path_arrows:

            self.publish_path_arrows(
                robot_id,
                grid_path,
                region_msg
            )


    # =================================================================
    # Cell order optimization
    # =================================================================

    def optimize_cell_order(
        self,
        cells,
        cell_candidates,
        start_grid,
        free_map
    ):
        """
        Cell 顺序 + Zig-Zag 方向联合优化。

        当前状态：

            current_position

        对每一个未访问 Cell：

            Forward:
                current -> forward.start

            Reverse:
                current -> reverse.start

        分别计算 A* 连接代价。

        选择：

            最小 A* cost

        然后进入该 Cell。

        这比：

            centroid nearest neighbor

        更准确，因为实际考虑了障碍物和 Region 边界。

        """

        remaining = [
            cell for cell in cells
            if cell.id in cell_candidates
        ]

        if not remaining:
            return []

        # ============================================================
        # 起点
        # ============================================================

        if start_grid is None:

            # 没有机器人位置时，
            # 使用最靠近左上角的 Cell。
            current_cell = min(
                remaining,
                key=lambda c: (
                    c.min_row,
                    c.min_col
                )
            )

            # 选择该 Cell 的第一个方向
            candidate = cell_candidates[
                current_cell.id
            ][0]

            plan = [{
                "cell": current_cell,
                "path": candidate["path"],
                "start": candidate["start"],
                "end": candidate["end"],
                "direction": candidate["direction"]
            }]

            remaining.remove(
                current_cell
            )

            current_position = candidate["end"]

        else:

            # --------------------------------------------------------
            # 从机器人当前位置寻找代价最低的第一个 Cell
            # --------------------------------------------------------

            best = None

            for cell in remaining:

                for candidate in cell_candidates[cell.id]:

                    connection = self.connect_cells_astar(
                        start_grid,
                        candidate["start"],
                        free_map
                    )

                    if not connection:
                        continue

                    cost = len(connection)

                    if (
                        best is None
                        or cost < best["cost"]
                    ):

                        best = {
                            "cell": cell,
                            "candidate": candidate,
                            "cost": cost
                        }

            if best is None:

                rospy.logwarn(
                    "Cannot find reachable first Cell."
                )

                return []

            current_cell = best["cell"]
            candidate = best["candidate"]

            plan = [{
                "cell": current_cell,
                "path": candidate["path"],
                "start": candidate["start"],
                "end": candidate["end"],
                "direction": candidate["direction"]
            }]

            remaining.remove(
                current_cell
            )

            current_position = candidate["end"]

        # ============================================================
        # Greedy minimum A* connection
        # ============================================================

        while remaining:

            candidates_to_check = remaining

            # --------------------------------------------------------
            # 可选：先按照 Manhattan 距离筛选候选
            #
            # 避免 Cell 非常多时大量 A*
            # --------------------------------------------------------

            if (
                self.max_order_candidates > 0
                and len(remaining)
                > self.max_order_candidates
            ):

                candidates_to_check = sorted(
                    remaining,
                    key=lambda cell: self.cell_distance(
                        current_position,
                        cell
                    )
                )[:
                    self.max_order_candidates
                ]

            best = None

            # --------------------------------------------------------
            # 对每一个候选 Cell：
            #
            # Forward
            # Reverse
            #
            # 都计算 A*
            # --------------------------------------------------------

            for cell in candidates_to_check:

                for candidate in cell_candidates[cell.id]:

                    connection = self.connect_cells_astar(
                        current_position,
                        candidate["start"],
                        free_map
                    )

                    if not connection:
                        continue

                    connection_cost = (
                        len(connection) - 1
                    )

                    # ------------------------------------------------
                    # 总成本
                    #
                    # 目前以连接长度为主。
                    #
                    # coverage 本身长度基本固定，
                    # 因此优化重点放在：
                    #
                    # Cell -> Cell connector
                    # ------------------------------------------------

                    total_cost = connection_cost

                    if (
                        best is None
                        or total_cost < best["cost"]
                    ):

                        best = {
                            "cell": cell,
                            "candidate": candidate,
                            "connection": connection,
                            "cost": total_cost
                        }

            # --------------------------------------------------------
            # 没有任何可连接 Cell
            # --------------------------------------------------------

            if best is None:

                rospy.logwarn(
                    "No reachable remaining Cell. "
                    "Stop Cell optimization."
                )

                break

            # --------------------------------------------------------
            # 添加最优 Cell
            # --------------------------------------------------------

            cell = best["cell"]
            candidate = best["candidate"]

            plan.append({
                "cell": cell,
                "path": candidate["path"],
                "start": candidate["start"],
                "end": candidate["end"],
                "direction": candidate["direction"]
            })

            remaining.remove(
                cell
            )

            current_position = candidate["end"]

        return plan


    # =================================================================
    # Approximate Cell distance
    # =================================================================

    @staticmethod
    def cell_distance(
        point,
        cell
    ):

        cx, cy = cell.centroid()

        return (
            abs(point[0] - cx)
            +
            abs(point[1] - cy)
        )


    # =================================================================
    # Generate Zig-Zag
    # =================================================================

    def generate_cell_coverage(
        self,
        cell,
        resolution,
        reverse=False
    ):

        if cell.size == 0:
            return []

        if resolution <= 0:
            return []

        spacing_cells = max(
            1,
            int(
                round(
                    self.coverage_spacing
                    /
                    resolution
                )
            )
        )

        path = []

        # ============================================================
        # Horizontal
        # ============================================================

        if self.scan_direction == "horizontal":

            scan_rows = list(
                range(
                    cell.min_row,
                    cell.max_row + 1,
                    spacing_cells
                )
            )

            if (
                scan_rows
                and
                scan_rows[-1]
                != cell.max_row
            ):

                scan_rows.append(
                    cell.max_row
                )

            if reverse:
                scan_rows.reverse()

            for scan_index, row in enumerate(scan_rows):

                columns = sorted(
                    col
                    for r, col in cell.pixels
                    if r == row
                )

                if not columns:
                    continue

                segments = (
                    self.split_continuous_indices(
                        columns
                    )
                )

                # ----------------------------------------------------
                # Zig-Zag
                # ----------------------------------------------------

                if scan_index % 2 == 0:

                    segments_iter = segments

                else:

                    segments_iter = list(
                        reversed(segments)
                    )

                for segment in segments_iter:

                    start_col = segment[0]
                    end_col = segment[-1]

                    if scan_index % 2 == 0:

                        col_range = range(
                            start_col,
                            end_col + 1
                        )

                    else:

                        col_range = range(
                            end_col,
                            start_col - 1,
                            -1
                        )

                    for col in col_range:

                        if (
                            row,
                            col
                        ) in cell.pixels:

                            path.append(
                                (col, row)
                            )

        # ============================================================
        # Vertical
        # ============================================================

        else:

            scan_cols = list(
                range(
                    cell.min_col,
                    cell.max_col + 1,
                    spacing_cells
                )
            )

            if (
                scan_cols
                and
                scan_cols[-1]
                != cell.max_col
            ):

                scan_cols.append(
                    cell.max_col
                )

            if reverse:

                scan_cols.reverse()

            for scan_index, col in enumerate(scan_cols):

                rows = sorted(
                    row
                    for row, c in cell.pixels
                    if c == col
                )

                if not rows:
                    continue

                segments = (
                    self.split_continuous_indices(
                        rows
                    )
                )

                if scan_index % 2 == 0:

                    segments_iter = segments

                else:

                    segments_iter = list(
                        reversed(segments)
                    )

                for segment in segments_iter:

                    start_row = segment[0]
                    end_row = segment[-1]

                    if scan_index % 2 == 0:

                        row_range = range(
                            start_row,
                            end_row + 1
                        )

                    else:

                        row_range = range(
                            end_row,
                            start_row - 1,
                            -1
                        )

                    for row in row_range:

                        if (
                            row,
                            col
                        ) in cell.pixels:

                            path.append(
                                (col, row)
                            )

        return path


    # =================================================================
    # Split continuous indices
    # =================================================================

    @staticmethod
    def split_continuous_indices(values):

        if not values:
            return []

        values = sorted(values)

        segments = []

        current = [
            values[0]
        ]

        for value in values[1:]:

            if value == current[-1] + 1:

                current.append(
                    value
                )

            else:

                segments.append(
                    current
                )

                current = [
                    value
                ]

        segments.append(
            current
        )

        return segments


    # =================================================================
    # A*
    # =================================================================

    def connect_cells_astar(
        self,
        start,
        goal,
        free_map
    ):

        if start is None or goal is None:
            return []

        if start == goal:
            return [start]

        start_col, start_row = start
        goal_col, goal_row = goal

        if not self.valid_grid_point(
            start_col,
            start_row,
            free_map
        ):
            return []

        if not self.valid_grid_point(
            goal_col,
            goal_row,
            free_map
        ):
            return []

        start_node = (
            start_col,
            start_row
        )

        goal_node = (
            goal_col,
            goal_row
        )

        open_set = []

        counter = 0

        start_h = self.astar_heuristic(
            start_node,
            goal_node
        )

        heapq.heappush(
            open_set,
            (
                start_h,
                counter,
                start_node
            )
        )

        came_from = {}

        g_score = {
            start_node: 0.0
        }

        closed_set = set()

        iterations = 0

        # ============================================================
        # A*
        # ============================================================

        while open_set:

            iterations += 1

            if (
                iterations
                >
                self.astar_max_iterations
            ):

                rospy.logwarn(
                    "A*: max iterations reached."
                )

                return []

            _, _, current = heapq.heappop(
                open_set
            )

            if current in closed_set:
                continue

            if current == goal_node:

                return self.reconstruct_astar_path(
                    came_from,
                    current
                )

            closed_set.add(
                current
            )

            current_col, current_row = current

            neighbors = [
                (
                    current_col + 1,
                    current_row
                ),
                (
                    current_col - 1,
                    current_row
                ),
                (
                    current_col,
                    current_row + 1
                ),
                (
                    current_col,
                    current_row - 1
                )
            ]

            for neighbor in neighbors:

                col, row = neighbor

                if not self.valid_grid_point(
                    col,
                    row,
                    free_map
                ):
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = (
                    g_score[current]
                    +
                    1.0
                )

                if (
                    neighbor not in g_score
                    or
                    tentative_g
                    <
                    g_score[neighbor]
                ):

                    came_from[
                        neighbor
                    ] = current

                    g_score[
                        neighbor
                    ] = tentative_g

                    h = self.astar_heuristic(
                        neighbor,
                        goal_node
                    )

                    f = (
                        tentative_g
                        +
                        h
                    )

                    counter += 1

                    heapq.heappush(
                        open_set,
                        (
                            f,
                            counter,
                            neighbor
                        )
                    )

        return []


    # =================================================================
    # A* heuristic
    # =================================================================

    @staticmethod
    def astar_heuristic(
        node,
        goal
    ):

        return (
            abs(
                node[0]
                -
                goal[0]
            )
            +
            abs(
                node[1]
                -
                goal[1]
            )
        )


    # =================================================================
    # Reconstruct A*
    # =================================================================

    @staticmethod
    def reconstruct_astar_path(
        came_from,
        current
    ):

        path = [
            current
        ]

        while current in came_from:

            current = came_from[
                current
            ]

            path.append(
                current
            )

        path.reverse()

        return path


    # =================================================================
    # Grid validation
    # =================================================================

    @staticmethod
    def valid_grid_point(
        col,
        row,
        free_map
    ):

        height, width = free_map.shape

        if col < 0 or col >= width:
            return False

        if row < 0 or row >= height:
            return False

        return bool(
            free_map[
                row,
                col
            ]
        )


    # =================================================================
    # Find nearest own-region cell
    # =================================================================

    @staticmethod
    def find_nearest_free_cell(
        start,
        free_map
    ):

        start_col, start_row = start

        free_cells = np.argwhere(
            free_map
        )

        if len(free_cells) == 0:
            return None

        # np.argwhere:
        #
        # (row, col)
        #

        distances = (
            (
                free_cells[:, 1]
                -
                start_col
            ) ** 2
            +
            (
                free_cells[:, 0]
                -
                start_row
            ) ** 2
        )

        index = int(
            np.argmin(
                distances
            )
        )

        row = int(
            free_cells[index, 0]
        )

        col = int(
            free_cells[index, 1]
        )

        return (
            col,
            row
        )


    # =================================================================
    # Remove duplicate points
    # =================================================================

    @staticmethod
    def remove_duplicate_points(
        path
    ):

        if not path:
            return []

        result = [
            path[0]
        ]

        for point in path[1:]:

            if point != result[-1]:

                result.append(
                    point
                )

        return result


    # =================================================================
    # World -> Grid
    # =================================================================

    @staticmethod
    def world_to_grid(
        x,
        y,
        map_msg
    ):

        resolution = (
            map_msg.info.resolution
        )

        origin_x = (
            map_msg.info.origin
            .position.x
        )

        origin_y = (
            map_msg.info.origin
            .position.y
        )

        gx = int(
            math.floor(
                (
                    x - origin_x
                )
                /
                resolution
            )
        )

        gy = int(
            math.floor(
                (
                    y - origin_y
                )
                /
                resolution
            )
        )

        return (
            gx,
            gy
        )


    # =================================================================
    # Grid -> World
    # =================================================================

    @staticmethod
    def grid_to_world(
        gx,
        gy,
        map_msg
    ):

        resolution = (
            map_msg.info.resolution
        )

        origin_x = (
            map_msg.info.origin
            .position.x
        )

        origin_y = (
            map_msg.info.origin
            .position.y
        )

        x = (
            origin_x
            +
            (gx + 0.5)
            *
            resolution
        )

        y = (
            origin_y
            +
            (gy + 0.5)
            *
            resolution
        )

        return (
            x,
            y
        )


    # =================================================================
    # Calculate yaw
    # =================================================================

    @staticmethod
    def calculate_yaw(
        current_point,
        next_point
    ):

        dx = (
            next_point[0]
            -
            current_point[0]
        )

        dy = (
            next_point[1]
            -
            current_point[1]
        )

        if (
            abs(dx) < 1e-9
            and
            abs(dy) < 1e-9
        ):

            return 0.0

        return math.atan2(
            dy,
            dx
        )


    # =================================================================
    # Yaw quaternion
    # =================================================================

    @staticmethod
    def yaw_to_quaternion(
        yaw
    ):

        half_yaw = (
            yaw * 0.5
        )

        return (
            0.0,
            0.0,
            math.sin(
                half_yaw
            ),
            math.cos(
                half_yaw
            )
        )


    # =================================================================
    # Publish Path
    # =================================================================

    def publish_path(
        self,
        robot_id,
        grid_path,
        map_msg
    ):

        path_msg = Path()

        path_msg.header.stamp = (
            rospy.Time.now()
        )

        path_msg.header.frame_id = (
            map_msg.header.frame_id
        )

        for index, point in enumerate(
            grid_path
        ):

            gx, gy = point

            x, y = self.grid_to_world(
                gx,
                gy,
                map_msg
            )

            pose = PoseStamped()

            pose.header = (
                path_msg.header
            )

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # --------------------------------------------------------
            # Orientation
            # --------------------------------------------------------

            if index < len(grid_path) - 1:

                next_point = (
                    grid_path[index + 1]
                )

            elif index > 0:

                next_point = (
                    grid_path[index - 1]
                )

            else:

                next_point = point

            yaw = self.calculate_yaw(
                point,
                next_point
            )

            qx, qy, qz, qw = (
                self.yaw_to_quaternion(
                    yaw
                )
            )

            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(
                pose
            )

        if robot_id == 1:

            self.robot1_path_pub.publish(
                path_msg
            )

        elif robot_id == 2:

            self.robot2_path_pub.publish(
                path_msg
            )

        rospy.loginfo(
            "Robot%d: published Path: %d poses",
            robot_id,
            len(path_msg.poses)
        )


    # =================================================================
    # Publish final path arrows
    # =================================================================

    def publish_path_arrows(
        self,
        robot_id,
        grid_path,
        map_msg
    ):

        marker_array = MarkerArray()

        namespace = (
            "robot{}_path_arrows"
            .format(robot_id)
        )

        # ============================================================
        # DELETEALL
        # ============================================================

        delete_marker = Marker()

        delete_marker.header.frame_id = (
            map_msg.header.frame_id
        )

        delete_marker.header.stamp = (
            rospy.Time.now()
        )

        delete_marker.ns = namespace

        delete_marker.id = 0

        delete_marker.action = (
            Marker.DELETEALL
        )

        marker_array.markers.append(
            delete_marker
        )

        if len(grid_path) < 2:

            if robot_id == 1:

                self.robot1_arrow_pub.publish(
                    marker_array
                )

            else:

                self.robot2_arrow_pub.publish(
                    marker_array
                )

            return

        # ============================================================
        # Arrow generation
        # ============================================================

        marker_id = 1

        step = max(
            1,
            int(self.arrow_step)
        )

        for index in range(
            0,
            len(grid_path) - 1,
            step
        ):

            p1 = grid_path[index]
            p2 = grid_path[
                min(
                    index + step,
                    len(grid_path) - 1
                )
            ]

            if p1 == p2:
                continue

            x1, y1 = self.grid_to_world(
                p1[0],
                p1[1],
                map_msg
            )

            x2, y2 = self.grid_to_world(
                p2[0],
                p2[1],
                map_msg
            )

            dx = x2 - x1
            dy = y2 - y1

            distance = math.sqrt(
                dx * dx
                +
                dy * dy
            )

            if distance < 1e-6:
                continue

            # --------------------------------------------------------
            # 如果 arrow_length 小于实际 segment，
            # 让箭头保持统一长度。
            # --------------------------------------------------------

            length = min(
                self.arrow_length,
                distance
            )

            ux = dx / distance
            uy = dy / distance

            end_x = (
                x1
                +
                ux * length
            )

            end_y = (
                y1
                +
                uy * length
            )

            # --------------------------------------------------------
            # ARROW marker
            # --------------------------------------------------------

            marker = Marker()

            marker.header.frame_id = (
                map_msg.header.frame_id
            )

            marker.header.stamp = (
                rospy.Time.now()
            )

            marker.ns = namespace

            marker.id = marker_id

            marker_id += 1

            marker.type = Marker.ARROW

            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0

            marker.scale.x = (
                max(
                    0.03,
                    self.arrow_width
                )
            )

            marker.scale.y = (
                max(
                    0.06,
                    self.arrow_width * 2.0
                )
            )

            marker.scale.z = (
                max(
                    0.03,
                    self.arrow_height
                )
            )

            marker.color.a = 1.0

            # --------------------------------------------------------
            # Arrow geometry
            #
            # ROS Marker.ARROW：
            #
            # points[0] = start
            # points[1] = end
            # --------------------------------------------------------

            start_point = Point()

            start_point.x = x1
            start_point.y = y1
            start_point.z = 0.08

            end_point = Point()

            end_point.x = end_x
            end_point.y = end_y
            end_point.z = 0.08

            marker.points.append(
                start_point
            )

            marker.points.append(
                end_point
            )

            marker_array.markers.append(
                marker
            )

        # ============================================================
        # Publish
        # ============================================================

        if robot_id == 1:

            self.robot1_arrow_pub.publish(
                marker_array
            )

        elif robot_id == 2:

            self.robot2_arrow_pub.publish(
                marker_array
            )

        rospy.loginfo(
            "Robot%d: published %d direction arrows.",
            robot_id,
            marker_id - 1
        )


    # =================================================================
    # Publish BCD cells
    # =================================================================

    def publish_cell_markers(
        self,
        robot_id,
        cells,
        map_msg
    ):

        marker_array = MarkerArray()

        namespace = (
            "robot{}_bcd_cells"
            .format(robot_id)
        )

        # ============================================================
        # DELETEALL
        # ============================================================

        delete_marker = Marker()

        delete_marker.header.frame_id = (
            map_msg.header.frame_id
        )

        delete_marker.header.stamp = (
            rospy.Time.now()
        )

        delete_marker.ns = namespace

        delete_marker.id = 0

        delete_marker.action = (
            Marker.DELETEALL
        )

        marker_array.markers.append(
            delete_marker
        )

        resolution = (
            map_msg.info.resolution
        )

        # ============================================================
        # Cell markers
        # ============================================================

        for index, cell in enumerate(
            cells
        ):

            marker = Marker()

            marker.header.frame_id = (
                map_msg.header.frame_id
            )

            marker.header.stamp = (
                rospy.Time.now()
            )

            marker.ns = namespace

            marker.id = index + 1

            marker.type = Marker.CUBE_LIST

            marker.action = Marker.ADD

            marker.scale.x = (
                resolution * 0.8
            )

            marker.scale.y = (
                resolution * 0.8
            )

            marker.scale.z = 0.02

            marker.pose.orientation.w = 1.0

            marker.color.a = 0.35

            marker.color.r = (
                ((index * 37) % 255)
                /
                255.0
            )

            marker.color.g = (
                ((index * 97) % 255)
                /
                255.0
            )

            marker.color.b = (
                ((index * 157) % 255)
                /
                255.0
            )

            for row, col in cell.pixels:

                x, y = self.grid_to_world(
                    col,
                    row,
                    map_msg
                )

                point = Point()

                point.x = x
                point.y = y
                point.z = 0.0

                marker.points.append(
                    point
                )

            marker_array.markers.append(
                marker
            )

            # ========================================================
            # Cell ID
            # ========================================================

            centroid_x, centroid_y = (
                cell.centroid()
            )

            centroid_col = int(
                round(centroid_x)
            )

            centroid_row = int(
                round(centroid_y)
            )

            centroid_col = max(
                0,
                min(
                    centroid_col,
                    map_msg.info.width - 1
                )
            )

            centroid_row = max(
                0,
                min(
                    centroid_row,
                    map_msg.info.height - 1
                )
            )

            x, y = self.grid_to_world(
                centroid_col,
                centroid_row,
                map_msg
            )

            text_marker = Marker()

            text_marker.header.frame_id = (
                map_msg.header.frame_id
            )

            text_marker.header.stamp = (
                rospy.Time.now()
            )

            text_marker.ns = (
                "robot{}_bcd_cell_ids"
                .format(robot_id)
            )

            text_marker.id = (
                10000 + index
            )

            text_marker.type = (
                Marker.TEXT_VIEW_FACING
            )

            text_marker.action = (
                Marker.ADD
            )

            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.15

            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = max(
                0.15,
                resolution * 4.0
            )

            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            text_marker.text = (
                "R{} Cell {}"
                .format(
                    robot_id,
                    cell.id
                )
            )

            marker_array.markers.append(
                text_marker
            )

        # ============================================================
        # Publish
        # ============================================================

        if robot_id == 1:

            self.robot1_cell_marker_pub.publish(
                marker_array
            )

        elif robot_id == 2:

            self.robot2_cell_marker_pub.publish(
                marker_array
            )


# =====================================================================
# Main
# =====================================================================

def main():

    try:

        CoveragePathPlanner()

        rospy.spin()

    except rospy.ROSInterruptException:

        pass


if __name__ == "__main__":

    main()

