#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq
import threading
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

from bcd import BoustrophedonDecomposition


# ============================================================
# Data structures
# ============================================================

@dataclass
class PathSegment:
    """
    一段路径。

    segment_type:
        coverage  : 真正的 Zig-Zag 覆盖段
        connector : 两条覆盖段之间的连接 / 掉头
        start     : 从机器人当前位置到第一条覆盖线
    """

    points: List[Tuple[int, int]]
    segment_type: str


@dataclass
class CoverageCandidate:
    """
    一个 BCD cell 的候选覆盖方案。
    """

    cell_id: int
    angle: float
    phase: float

    path: List[Tuple[int, int]]

    entry: Tuple[int, int]
    exit: Tuple[int, int]

    coverage_length_m: float
    coverage_time: float

    turn_count: int
    turn_time: float

    score: float


# ============================================================
# Main planner
# ============================================================

class CoveragePathPlanner:

    def __init__(self):

        rospy.init_node(
            "coverage_path_planner",
            anonymous=False
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        self.coverage_spacing = rospy.get_param(
            "~coverage_spacing",
            1.4
        )

        self.robot_linear_speed = rospy.get_param(
            "~robot_linear_speed",
            0.40
        )

        self.robot_angular_speed = rospy.get_param(
            "~robot_angular_speed",
            0.80
        )

        self.turn_time_weight = rospy.get_param(
            "~turn_time_weight",
            1.0
        )

        self.min_cell_size = rospy.get_param(
            "~min_cell_size",
            3
        )

        self.max_angle_candidates = rospy.get_param(
            "~max_angle_candidates",
            5
        )

        self.angle_step_deg = rospy.get_param(
            "~angle_step_deg",
            15.0
        )

        self.phase_samples = rospy.get_param(
            "~phase_samples",
            2
        )

        self.candidate_top_k = rospy.get_param(
            "~candidate_top_k",
            4
        )

        self.astar_diagonal = rospy.get_param(
            "~astar_diagonal",
            True
        )

        self.covered_cost = rospy.get_param(
            "~covered_cost",
            5.0
        )

        self.covered_inflation_radius = rospy.get_param(
            "~covered_inflation_radius",
            0.30
        )

        if self.covered_inflation_radius < 0.0:

            rospy.logwarn(
                "[CoveragePlanner] covered_inflation_radius < 0.0, "
                "forcing it to 0.0"
            )

            self.covered_inflation_radius = 0.0

        if self.covered_cost < 1.0:

            rospy.logwarn(
                "[CoveragePlanner] covered_cost < 1.0, "
                "forcing it to 1.0"
            )

            self.covered_cost = 1.0

        self.simplify_connector = rospy.get_param(
            "~simplify_connector",
            False
        )

        # ====================================================
        # 下面几个参数是新增的
        # ====================================================

        # covered_map 最短重新检查周期。
        #
        # 注意：
        # 不是每次 covered_map callback 都规划。
        self.covered_replan_interval = rospy.get_param(
            "~covered_replan_interval",
            2.0
        )

        # 当前 task 有多少比例已经被 covered，
        # 才认为当前 task 明显受到影响。
        #
        # 例如：
        #     0.25
        #
        # 表示当前 task 剩余 coverage 点中，
        # 至少 25% 已经变成 covered，
        # 才触发 local replan。
        self.task_covered_ratio_threshold = rospy.get_param(
            "~task_covered_ratio_threshold",
            0.25
        )

        # 判断 coverage task 是否已经被机器人走过。
        #
        # 当前 coverage path 点附近只要有一定比例 covered，
        # 就认为该点已经完成。
        self.task_point_covered_ratio = rospy.get_param(
            "~task_point_covered_ratio",
            0.80
        )

        # local replan 最少剩余 coverage 点。
        #
        # 如果剩余任务太少，则直接等待完成，
        # 不再浪费计算资源。
        self.min_local_replan_points = rospy.get_param(
            "~min_local_replan_points",
            5
        )

        # local replanning 后，
        # 如果剩余任务太短，直接保留当前路径。
        self.min_remaining_task_points = rospy.get_param(
            "~min_remaining_task_points",
            3
        )

        # planning_period 仍然控制 worker 检查频率。
        self.planning_period = rospy.get_param(
            "~planning_period",
            1.0
        )

        self.scan_band_cells = rospy.get_param(
            "~scan_band_cells",
            0.65
        )

        self.scan_gap_cells = rospy.get_param(
            "~scan_gap_cells",
            2.0
        )

        # ----------------------------------------------------
        # Robot names
        # ----------------------------------------------------

        self.robot_names = {
            1: rospy.get_param(
                "~robot1_name",
                "robot1"
            ),
            2: rospy.get_param(
                "~robot2_name",
                "robot2"
            )
        }

        # ----------------------------------------------------
        # Shared state
        # ----------------------------------------------------

        self.state_lock = threading.RLock()

        self.robot_positions = {
            1: None,
            2: None
        }

        self.robot_regions = {
            1: None,
            2: None
        }

        # 最新 covered_map
        self.covered_map = None

        # ====================================================
        # 新增：
        #
        # covered_map 只设置 dirty，
        # 不直接让 robot_dirty=true。
        # ====================================================

        self.covered_dirty = False

        self.last_covered_check_time = 0.0

        # ----------------------------------------------------
        # Robot dirty
        # ----------------------------------------------------

        self.robot_dirty = {
            1: False,
            2: False
        }

        self.robot_planning = {
            1: False,
            2: False
        }

        # ====================================================
        # 当前正在执行的 task
        #
        # current_task_segments:
        #
        #     当前已经发布给机器人执行的完整任务
        #
        # local replan 会基于这个任务进行。
        # ====================================================

        self.current_task_segments = {
            1: [],
            2: []
        }

        # 当前 task 对应的 map geometry
        self.current_task_map = {
            1: None,
            2: None
        }

        # 当前 task 是否有效
        self.current_task_valid = {
            1: False,
            2: False
        }

        # ----------------------------------------------------
        # Publishers
        # ----------------------------------------------------

        self.path_publishers = {
            1: rospy.Publisher(
                "/robot1_coverage_path",
                Path,
                queue_size=1,
                latch=True
            ),

            2: rospy.Publisher(
                "/robot2_coverage_path",
                Path,
                queue_size=1,
                latch=True
            )
        }

        self.cell_publishers = {
            1: rospy.Publisher(
                "/robot1_bcd_cells",
                MarkerArray,
                queue_size=1,
                latch=True
            ),

            2: rospy.Publisher(
                "/robot2_bcd_cells",
                MarkerArray,
                queue_size=1,
                latch=True
            )
        }

        self.arrow_publishers = {
            1: rospy.Publisher(
                "/robot1_path_arrows",
                MarkerArray,
                queue_size=1,
                latch=True
            ),

            2: rospy.Publisher(
                "/robot2_path_arrows",
                MarkerArray,
                queue_size=1,
                latch=True
            )
        }

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        self.region_subscribers = {
            1: rospy.Subscriber(
                "/robot1_region",
                OccupancyGrid,
                self.robot1_region_callback,
                queue_size=1
            ),

            2: rospy.Subscriber(
                "/robot2_region",
                OccupancyGrid,
                self.robot2_region_callback,
                queue_size=1
            )
        }

        self.covered_map_sub = rospy.Subscriber(
            "/covered_map",
            OccupancyGrid,
            self.covered_map_callback,
            queue_size=1
        )

        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback
        )

        # ----------------------------------------------------
        # A* cache
        # ----------------------------------------------------

        self.astar_cache = {}

        # ----------------------------------------------------
        # Worker
        # ----------------------------------------------------

        self.worker_thread = threading.Thread(
            target=self.planning_worker,
            daemon=True
        )

        self.worker_thread.start()

        rospy.loginfo(
            "[CoveragePlanner] started"
        )

        rospy.loginfo(
            "[CoveragePlanner] covered_cost = %.2f",
            self.covered_cost
        )

        rospy.loginfo(
            "[CoveragePlanner] covered_inflation_radius = %.2f m",
            self.covered_inflation_radius
        )

        rospy.loginfo(
            "[CoveragePlanner] covered_replan_interval = %.2f",
            self.covered_replan_interval
        )

        rospy.loginfo(
            "[CoveragePlanner] task_covered_ratio_threshold = %.2f",
            self.task_covered_ratio_threshold
        )

        rospy.loginfo(
            "[CoveragePlanner] max_angle_candidates = %d",
            self.max_angle_candidates
        )

        rospy.loginfo(
            "[CoveragePlanner] phase_samples = %d",
            self.phase_samples
        )

        rospy.loginfo(
            "[CoveragePlanner] candidate_top_k = %d",
            self.candidate_top_k
        )

    # ========================================================
    # ROS callbacks
    # ========================================================

    def robot1_region_callback(self, msg):

        with self.state_lock:

            self.robot_regions[1] = msg

            # region 变化属于真正的 global planning trigger
            self.robot_dirty[1] = True

            # 原来的 current task 可能已经失效
            self.current_task_valid[1] = False

        rospy.loginfo_throttle(
            2.0,
            "[CoveragePlanner] robot1 region updated"
        )

    def robot2_region_callback(self, msg):

        with self.state_lock:

            self.robot_regions[2] = msg

            self.robot_dirty[2] = True

            self.current_task_valid[2] = False

        rospy.loginfo_throttle(
            2.0,
            "[CoveragePlanner] robot2 region updated"
        )

    def covered_map_callback(self, msg):

        with self.state_lock:

            self.covered_map = msg

            # =================================================
            # 关键修改
            #
            # 不再：
            #
            #     robot_dirty[1] = True
            #     robot_dirty[2] = True
            #
            # 而是只记录：
            #
            #     covered_dirty = True
            #
            # worker 后面检查 current task 是否真的受影响。
            # =================================================

            self.covered_dirty = True

        rospy.loginfo_throttle(
            2.0,
            "[CoveragePlanner] covered_map updated"
        )

    def model_states_callback(self, msg):

        with self.state_lock:

            for robot_id in (1, 2):

                name = self.robot_names[robot_id]

                try:
                    index = msg.name.index(name)
                except ValueError:
                    continue

                pose = msg.pose[index]

                x = pose.position.x
                y = pose.position.y

                yaw = self.quaternion_to_yaw(
                    pose.orientation
                )

                self.robot_positions[robot_id] = (
                    x,
                    y,
                    yaw
                )

    # ========================================================
    # Worker
    # ========================================================

    def planning_worker(self):

        rate = rospy.Rate(
            1.0 / max(
                self.planning_period,
                0.01
            )
        )

        while not rospy.is_shutdown():

            # =================================================
            # Step 1:
            #
            # 先检查 covered_map 是否影响当前 task。
            #
            # 这里只做轻量判断。
            # =================================================

            self.check_covered_task_impact()

            jobs = []

            with self.state_lock:

                for robot_id in (1, 2):

                    if (
                        self.robot_dirty[robot_id]
                        and
                        not self.robot_planning[robot_id]
                        and
                        self.robot_regions[robot_id] is not None
                        and
                        self.covered_map is not None
                    ):

                        self.robot_planning[robot_id] = True

                        region = self.robot_regions[robot_id]
                        covered = self.covered_map

                        self.robot_dirty[robot_id] = False

                        jobs.append(
                            (
                                robot_id,
                                region,
                                covered
                            )
                        )

            # =================================================
            # 不持有 state_lock 做重规划
            # =================================================

            for robot_id, region, covered in jobs:

                try:

                    self.process_robot(
                        robot_id,
                        region,
                        covered
                    )

                except Exception as exc:

                    rospy.logerr(
                        "[CoveragePlanner] robot%d planning failed: %s",
                        robot_id,
                        exc
                    )

                finally:

                    with self.state_lock:

                        self.robot_planning[robot_id] = False

            rate.sleep()

    # ========================================================
    # Check whether covered_map affects current task
    # ========================================================

    def check_covered_task_impact(self):

        now = rospy.get_time()

        with self.state_lock:

            if not self.covered_dirty:
                return

            if (
                now - self.last_covered_check_time
                <
                self.covered_replan_interval
            ):
                return

            covered_msg = self.covered_map

            self.last_covered_check_time = now

            # 这一次变化已经被取出来检查
            self.covered_dirty = False

        if covered_msg is None:
            return

        # ----------------------------------------------------
        # 分别检查两个机器人
        # ----------------------------------------------------

        for robot_id in (1, 2):

            with self.state_lock:

                if self.robot_planning[robot_id]:
                    continue

                if not self.current_task_valid[robot_id]:
                    continue

                segments = list(
                    self.current_task_segments[robot_id]
                )

                region_msg = self.robot_regions[robot_id]

                current_pose = self.robot_positions[robot_id]

            if not segments:
                continue

            if region_msg is None:
                continue

            # ------------------------------------------------
            # geometry mismatch
            # ------------------------------------------------

            if not self.same_map_geometry(
                region_msg,
                covered_msg
            ):
                continue

            affected, remaining_points = (
                self.current_task_is_affected(
                    segments,
                    covered_msg
                )
            )

            if not affected:
                continue

            rospy.loginfo(
                "[CoveragePlanner] robot%d current task "
                "affected by covered_map: remaining=%d",
                robot_id,
                remaining_points
            )

            if current_pose is None:
                continue

            # =================================================
            # Local replan
            # =================================================

            success = self.local_replan_current_task(
                robot_id,
                region_msg,
                covered_msg,
                segments
            )

            if success:

                rospy.loginfo(
                    "[CoveragePlanner] robot%d local replan success",
                    robot_id
                )

            else:

                rospy.logwarn(
                    "[CoveragePlanner] robot%d local replan failed, "
                    "requesting global replan",
                    robot_id
                )

                with self.state_lock:

                    self.robot_dirty[robot_id] = True

    # ========================================================
    # Determine whether current task is affected
    # ========================================================

    def current_task_is_affected(
        self,
        segments,
        covered_msg
    ):
        """
        判断当前 task 是否真的受到 covered_map 影响。

        注意：

        不是简单判断：
            covered_map 更新了
            -> replan

        而是：

            当前剩余 coverage path
                    ↓
            有多少已经 covered
                    ↓
            超过 threshold
                    ↓
            才认为 task 受到影响
        """

        covered = self.occupancy_grid_to_numpy(
            covered_msg
        )

        coverage_points = []

        for segment in segments:

            if segment.segment_type != "coverage":
                continue

            for point in segment.points:

                if point not in coverage_points:

                    coverage_points.append(
                        point
                    )

        if len(coverage_points) < self.min_local_replan_points:

            return (
                False,
                len(coverage_points)
            )

        total = len(
            coverage_points
        )

        covered_count = 0

        for r, c in coverage_points:

            if (
                0 <= r < covered.shape[0]
                and
                0 <= c < covered.shape[1]
                and
                covered[r, c] == 100
            ):

                covered_count += 1

        ratio = (
            covered_count /
            float(max(total, 1))
        )

        rospy.logdebug(
            "[CoveragePlanner] task covered ratio = %.3f",
            ratio
        )

        if (
            ratio >=
            self.task_covered_ratio_threshold
        ):

            return (
                True,
                total - covered_count
            )

        return (
            False,
            total - covered_count
        )

    # ========================================================
    # Local replan
    # ========================================================

    def local_replan_current_task(
        self,
        robot_id,
        region_msg,
        covered_msg,
        old_segments
    ):
        """
        局部重新规划。

        不重新执行：

            BCD
            angle search
            phase search
            candidate generation
            DP

        只做：

            current robot pose
                    ↓
            当前 task 中剩余的 coverage
                    ↓
            找第一个未覆盖点
                    ↓
            A* connector
                    ↓
            保留后面的 task

        这是整个优化的核心。
        """

        with self.state_lock:

            current_pose = self.robot_positions[robot_id]

        if current_pose is None:
            return False

        # ----------------------------------------------------
        # Convert maps
        # ----------------------------------------------------

        region = self.occupancy_grid_to_numpy(
            region_msg
        )

        covered = self.occupancy_grid_to_numpy(
            covered_msg
        )

        own_region = (
            region == 100
        )

        already_covered = (
            covered == 100
        )

        inflated_covered = self.inflate_covered_map(
            already_covered,
            resolution
        )

        # 膨胀区域可以作为 connector 通过，但不会成为新的 coverage task。
        # 同时避免膨胀把 region_map 中的障碍（-1）变成可通行区域。
        inflated_covered &= (
            occupancy >= 0
        )

        connector_map = (
            own_region
            |
            inflated_covered
        )

        path_cost_map = np.ones(
            covered.shape,
            dtype=np.float64
        )

        path_cost_map[
            inflated_covered
        ] = self.covered_cost

        # ----------------------------------------------------
        # Current robot grid
        # ----------------------------------------------------

        start_grid = self.world_to_grid(
            current_pose[0],
            current_pose[1],
            region_msg
        )

        start_grid = self.find_nearest_free_cell(
            start_grid,
            connector_map
        )

        if start_grid is None:
            return False

        # ----------------------------------------------------
        # Build remaining task
        # ----------------------------------------------------

        remaining_segments = []

        found_first_uncovered = False

        for segment in old_segments:

            if segment.segment_type != "coverage":

                # connector/start：
                # 只在还没有找到新的 coverage 起点时忽略。
                #
                # 后续 connector 将由 A* 重新生成。
                continue

            points = segment.points

            if not points:
                continue

            remaining = []

            for point in points:

                r, c = point

                is_covered = (
                    0 <= r < covered.shape[0]
                    and
                    0 <= c < covered.shape[1]
                    and
                    covered[r, c] == 100
                )

                if not found_first_uncovered:

                    if is_covered:
                        continue

                    found_first_uncovered = True

                    remaining.append(
                        point
                    )

                else:

                    remaining.append(
                        point
                    )

            if len(remaining) >= 2:

                remaining_segments.append(
                    PathSegment(
                        points=remaining,
                        segment_type="coverage"
                    )
                )

        # ----------------------------------------------------
        # 没有剩余 coverage
        #
        # 当前 task 已经完成。
        # ----------------------------------------------------

        if not remaining_segments:

            rospy.loginfo(
                "[CoveragePlanner] robot%d current task completed "
                "by covered_map",
                robot_id
            )

            self.publish_empty_path(
                robot_id,
                region_msg.header.frame_id
            )

            with self.state_lock:

                self.current_task_segments[robot_id] = []

                self.current_task_valid[robot_id] = False

                # 当前 task 完成后，
                # 下一次 worker 可以做 global planning。
                self.robot_dirty[robot_id] = True

            return True

        # ----------------------------------------------------
        # Find first remaining point
        # ----------------------------------------------------

        first_target = None

        for segment in remaining_segments:

            if segment.points:

                first_target = segment.points[0]
                break

        if first_target is None:
            return False

        # ----------------------------------------------------
        # Local A*
        # ----------------------------------------------------

        self.astar_cache.clear()

        connector = self.astar(
            start_grid,
            first_target,
            connector_map,
            path_cost_map
        )

        if connector is None:

            rospy.logwarn(
                "[CoveragePlanner] robot%d local connector "
                "cannot reach current task",
                robot_id
            )

            return False

        connector = self.remove_consecutive_duplicates(
            connector
        )

        new_segments = []

        if len(connector) >= 2:

            if self.simplify_connector:

                connector = self.shortcut_path(
                    connector,
                    connector_map
                )

            new_segments.append(
                PathSegment(
                    points=connector,
                    segment_type="start"
                )
            )

        # ----------------------------------------------------
        # Append remaining coverage
        # ----------------------------------------------------

        for segment in remaining_segments:

            new_segments.append(
                PathSegment(
                    points=list(segment.points),
                    segment_type="coverage"
                )
            )

        # ----------------------------------------------------
        # 如果剩余任务太少，
        # 没必要再发布复杂 local path。
        # ----------------------------------------------------

        total_remaining = sum(
            len(segment.points)
            for segment in remaining_segments
        )

        if (
            total_remaining
            <
            self.min_remaining_task_points
        ):

            rospy.loginfo(
                "[CoveragePlanner] robot%d local task almost finished",
                robot_id
            )

        # ----------------------------------------------------
        # Publish local path
        # ----------------------------------------------------

        resolution = region_msg.info.resolution

        origin_x = region_msg.info.origin.position.x
        origin_y = region_msg.info.origin.position.y

        origin_yaw = self.quaternion_to_yaw(
            region_msg.info.origin.orientation
        )

        self.publish_segments(
            robot_id,
            new_segments,
            region_msg,
            resolution,
            origin_x,
            origin_y,
            origin_yaw
        )

        self.publish_path_arrows(
            robot_id,
            new_segments,
            region_msg,
            resolution,
            origin_x,
            origin_y,
            origin_yaw
        )

        # ----------------------------------------------------
        # 保存新的 current task
        # ----------------------------------------------------

        with self.state_lock:

            self.current_task_segments[robot_id] = (
                new_segments
            )

            self.current_task_valid[robot_id] = True

        return True

    # ========================================================
    # Robot planning
    # ========================================================

    def process_robot(
        self,
        robot_id: int,
        region_msg: OccupancyGrid,
        covered_msg: OccupancyGrid
    ):

        self.astar_cache.clear()

        # ----------------------------------------------------
        # Snapshot robot pose
        # ----------------------------------------------------

        with self.state_lock:

            start_pose = self.robot_positions[robot_id]

        if start_pose is None:

            rospy.logwarn(
                "[CoveragePlanner] robot%d pose unavailable",
                robot_id
            )

            return

        # ----------------------------------------------------
        # Check map geometry
        # ----------------------------------------------------

        if not self.same_map_geometry(
            region_msg,
            covered_msg
        ):

            rospy.logerr(
                "[CoveragePlanner] robot%d region map and "
                "covered_map geometry mismatch",
                robot_id
            )

            return

        # ----------------------------------------------------
        # Convert occupancy grids
        # ----------------------------------------------------

        occupancy = self.occupancy_grid_to_numpy(
            region_msg
        )

        covered = self.occupancy_grid_to_numpy(
            covered_msg
        )

        resolution = region_msg.info.resolution

        origin_x = region_msg.info.origin.position.x
        origin_y = region_msg.info.origin.position.y

        origin_yaw = self.quaternion_to_yaw(
            region_msg.info.origin.orientation
        )

        # ----------------------------------------------------
        # Own region
        # ----------------------------------------------------

        own_region = (
            occupancy == 100
        )

        if not np.any(own_region):

            rospy.logwarn(
                "[CoveragePlanner] robot%d has empty region",
                robot_id
            )

            self.publish_empty_path(
                robot_id,
                region_msg.header.frame_id
            )

            return

        already_covered = (
            covered == 100
        )

        inflated_covered = self.inflate_covered_map(
            already_covered,
            resolution
        )

        # 膨胀区域可以作为 connector 通过，但不会成为新的 coverage task。
        # 同时避免膨胀把 region_map 中的障碍（-1）变成可通行区域。
        inflated_covered &= (
            occupancy >= 0
        )

        connector_map = (
            own_region
            |
            inflated_covered
        )

        path_cost_map = np.ones(
            covered.shape,
            dtype=np.float64
        )

        path_cost_map[
            inflated_covered
        ] = self.covered_cost

        # ----------------------------------------------------
        # Convert robot start
        # ----------------------------------------------------

        start_grid = self.world_to_grid(
            start_pose[0],
            start_pose[1],
            region_msg
        )

        start_grid = self.find_nearest_free_cell(
            start_grid,
            connector_map
        )

        if start_grid is None:

            rospy.logwarn(
                "[CoveragePlanner] robot%d cannot find valid start cell",
                robot_id
            )

            return

        rospy.loginfo(
            "[CoveragePlanner] robot%d start grid = (%d,%d)",
            robot_id,
            start_grid[0],
            start_grid[1]
        )

        # ----------------------------------------------------
        # BCD
        # ----------------------------------------------------

        rospy.loginfo(
            "[CoveragePlanner] robot%d BCD...",
            robot_id
        )

        bcd = BoustrophedonDecomposition(
            own_region.astype(np.uint8),
            min_cell_size=self.min_cell_size
        )

        cells = bcd.decompose()

        if not cells:

            rospy.logwarn(
                "[CoveragePlanner] robot%d no BCD cells",
                robot_id
            )

            return

        rospy.loginfo(
            "[CoveragePlanner] robot%d BCD cells = %d",
            robot_id,
            len(cells)
        )

        # ----------------------------------------------------
        # Build cell masks
        # ----------------------------------------------------

        cell_masks = {}

        for cell in cells:

            mask = np.zeros_like(
                own_region,
                dtype=bool
            )

            for r, c in cell.pixels:

                if (
                    0 <= r < mask.shape[0]
                    and
                    0 <= c < mask.shape[1]
                ):

                    mask[r, c] = True

            cell_masks[cell.id] = mask

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        all_candidates = {}

        for cell in cells:

            candidates = self.generate_cell_candidates(
                cell,
                cell_masks[cell.id],
                resolution
            )

            if not candidates:

                rospy.logwarn(
                    "[CoveragePlanner] robot%d cell%d no candidate",
                    robot_id,
                    cell.id
                )

                continue

            all_candidates[cell.id] = candidates

        if not all_candidates:

            rospy.logwarn(
                "[CoveragePlanner] robot%d no coverage candidate",
                robot_id
            )

            return

        # ----------------------------------------------------
        # Order BCD cells
        # ----------------------------------------------------

        ordered_cells = self.order_cells(
            cells,
            start_grid
        )

        ordered_cells = [
            c
            for c in ordered_cells
            if c.id in all_candidates
        ]

        # ----------------------------------------------------
        # Optimize
        # ----------------------------------------------------

        selected = self.optimize_cell_route(
            ordered_cells,
            all_candidates,
            connector_map,
            path_cost_map,
            start_grid,
            resolution
        )

        if selected is None:

            rospy.logwarn(
                "[CoveragePlanner] robot%d route optimization failed",
                robot_id
            )

            return

        # ----------------------------------------------------
        # Build final path
        # ----------------------------------------------------

        segments = self.build_final_segments(
            start_grid,
            selected,
            connector_map,
            path_cost_map
        )

        if not segments:

            rospy.logwarn(
                "[CoveragePlanner] robot%d final path empty",
                robot_id
            )

            return

        # ====================================================
        # 重要：
        #
        # 删除原来的：
        #
        #     robot moved > 0.30m
        #     -> discard old path
        #
        # 这一逻辑会导致：
        #
        #     planning 15s
        #         ↓
        #     robot 移动 1m
        #         ↓
        #     discard
        #         ↓
        #     dirty
        #         ↓
        #     再规划
        #
        # 现在不再因为机器人正常移动而丢弃整个规划。
        # ====================================================

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.publish_segments(
            robot_id,
            segments,
            region_msg,
            resolution,
            origin_x,
            origin_y,
            origin_yaw
        )

        self.publish_bcd_cells(
            robot_id,
            cells,
            region_msg
        )

        self.publish_path_arrows(
            robot_id,
            segments,
            region_msg,
            resolution,
            origin_x,
            origin_y,
            origin_yaw
        )

        # ====================================================
        # 保存 current task
        # ====================================================

        with self.state_lock:

            self.current_task_segments[robot_id] = (
                segments
            )

            self.current_task_map[robot_id] = (
                region_msg
            )

            self.current_task_valid[robot_id] = True

        rospy.loginfo(
            "[CoveragePlanner] robot%d path published",
            robot_id
        )

    # ========================================================
    # Map geometry
    # ========================================================

    @staticmethod
    def same_map_geometry(
        a,
        b
    ):

        if a.info.width != b.info.width:
            return False

        if a.info.height != b.info.height:
            return False

        if not math.isclose(
            a.info.resolution,
            b.info.resolution,
            rel_tol=1e-6,
            abs_tol=1e-9
        ):
            return False

        if not math.isclose(
            a.info.origin.position.x,
            b.info.origin.position.x,
            abs_tol=1e-6
        ):
            return False

        if not math.isclose(
            a.info.origin.position.y,
            b.info.origin.position.y,
            abs_tol=1e-6
        ):
            return False

        if not math.isclose(
            a.info.origin.position.z,
            b.info.origin.position.z,
            abs_tol=1e-6
        ):
            return False

        if not math.isclose(
            a.info.origin.orientation.x,
            b.info.origin.orientation.x,
            abs_tol=1e-6
        ):
            return False

        if not math.isclose(
            a.info.origin.orientation.y,
            b.info.origin.orientation.y,
            abs_tol=1e-6
        ):
            return False

        if not math.isclose(
            a.info.origin.orientation.z,
            b.info.origin.orientation.z,
            abs_tol=1e-6
        ):
            return False

        if not math.isclose(
            a.info.origin.orientation.w,
            b.info.origin.orientation.w,
            abs_tol=1e-6
        ):
            return False

        return True

    # ========================================================
    # Covered map inflation
    # ========================================================

    def inflate_covered_map(
        self,
        covered_mask,
        resolution
    ):
        """
        对 covered_map == 100 的区域进行圆形膨胀。

        膨胀后的区域用于 connector_map，并统一赋予 covered_cost。
        不改变 own_region，因此不会增加新的 BCD coverage task。
        """

        if self.covered_inflation_radius <= 0.0:
            return covered_mask.copy()

        radius_cells = int(
            math.ceil(
                self.covered_inflation_radius /
                max(float(resolution), 1e-9)
            )
        )

        if radius_cells <= 0:
            return covered_mask.copy()

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

            return binary_dilation(
                covered_mask,
                structure=kernel
            )

        except ImportError:
            rospy.logwarn_throttle(
                10.0,
                "[CoveragePlanner] scipy not available, "
                "using slow covered inflation."
            )

        inflated = covered_mask.copy()
        rows, cols = covered_mask.shape

        covered_indices = np.argwhere(covered_mask)

        for r, c in covered_indices:

            r_min = max(0, r - radius_cells)
            r_max = min(rows, r + radius_cells + 1)
            c_min = max(0, c - radius_cells)
            c_max = min(cols, c + radius_cells + 1)

            for nr in range(r_min, r_max):
                for nc in range(c_min, c_max):

                    dr = nr - r
                    dc = nc - c

                    if (
                        dr * dr
                        +
                        dc * dc
                        <=
                        radius_cells * radius_cells
                    ):
                        inflated[nr, nc] = True

        return inflated

    # ========================================================
    # Occupancy conversion
    # ========================================================

    def occupancy_grid_to_numpy(
        self,
        msg
    ):

        h = msg.info.height
        w = msg.info.width

        data = np.asarray(
            msg.data,
            dtype=np.int16
        )

        expected = h * w

        if data.size != expected:

            raise ValueError(
                "OccupancyGrid data size mismatch: "
                "%d != %d"
                %
                (
                    data.size,
                    expected
                )
            )

        return data.reshape(
            (h, w)
        )

    # ========================================================
    # Coordinate conversion
    # ========================================================

    def world_to_grid(
        self,
        x,
        y,
        map_msg
    ):

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        origin_yaw = self.quaternion_to_yaw(
            map_msg.info.origin.orientation
        )

        dx = x - origin_x
        dy = y - origin_y

        cos_yaw = math.cos(
            origin_yaw
        )

        sin_yaw = math.sin(
            origin_yaw
        )

        local_x = (
            cos_yaw * dx
            +
            sin_yaw * dy
        )

        local_y = (
            -sin_yaw * dx
            +
            cos_yaw * dy
        )

        col = int(
            math.floor(
                local_x /
                map_msg.info.resolution
            )
        )

        row = int(
            math.floor(
                local_y /
                map_msg.info.resolution
            )
        )

        return (
            row,
            col
        )

    def grid_to_world(
        self,
        row,
        col,
        map_msg
    ):

        resolution = map_msg.info.resolution

        local_x = (
            col + 0.5
        ) * resolution

        local_y = (
            row + 0.5
        ) * resolution

        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        origin_yaw = self.quaternion_to_yaw(
            map_msg.info.origin.orientation
        )

        cos_yaw = math.cos(
            origin_yaw
        )

        sin_yaw = math.sin(
            origin_yaw
        )

        world_x = (
            origin_x
            +
            cos_yaw * local_x
            -
            sin_yaw * local_y
        )

        world_y = (
            origin_y
            +
            sin_yaw * local_x
            +
            cos_yaw * local_y
        )

        return (
            world_x,
            world_y
        )

    # ========================================================
    # Quaternion
    # ========================================================

    @staticmethod
    def quaternion_to_yaw(q):

        sin_yaw = (
            2.0 *
            (
                q.w * q.z
                +
                q.x * q.y
            )
        )

        cos_yaw = (
            1.0
            -
            2.0 *
            (
                q.y * q.y
                +
                q.z * q.z
            )
        )

        return math.atan2(
            sin_yaw,
            cos_yaw
        )

    # ========================================================
    # Find nearest free
    # ========================================================

    def find_nearest_free_cell(
        self,
        start,
        free_map
    ):

        if start is None:
            return None

        rows, cols = free_map.shape

        sr, sc = start

        sr = max(
            0,
            min(rows - 1, sr)
        )

        sc = max(
            0,
            min(cols - 1, sc)
        )

        if free_map[sr, sc]:

            return (
                sr,
                sc
            )

        visited = set()

        queue = [
            (sr, sc)
        ]

        visited.add(
            (sr, sc)
        )

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1)
        ]

        head = 0

        while head < len(queue):

            r, c = queue[head]
            head += 1

            for dr, dc in directions:

                nr = r + dr
                nc = c + dc

                if (
                    nr < 0
                    or nr >= rows
                    or nc < 0
                    or nc >= cols
                ):
                    continue

                if (
                    nr,
                    nc
                ) in visited:
                    continue

                if free_map[nr, nc]:

                    return (
                        nr,
                        nc
                    )

                visited.add(
                    (
                        nr,
                        nc
                    )
                )

                queue.append(
                    (
                        nr,
                        nc
                    )
                )

        return None

    # ========================================================
    # Generate angle candidates
    # ========================================================

    def generate_angle_candidates(
        self,
        cell
    ):

        pixels = np.asarray(
            cell.pixels,
            dtype=float
        )

        if len(pixels) < 2:

            return [
                0.0
            ]

        center = pixels.mean(
            axis=0
        )

        centered = pixels - center

        cov = np.cov(
            centered.T
        )

        try:

            eigenvalues, eigenvectors = np.linalg.eigh(
                cov
            )

            direction = eigenvectors[
                :,
                np.argmax(eigenvalues)
            ]

            theta_pca = math.atan2(
                direction[0],
                direction[1]
            )

        except Exception:

            theta_pca = 0.0

        angles = []

        for k in range(
            -2,
            3
        ):

            angle = (
                theta_pca
                +
                math.radians(
                    self.angle_step_deg
                ) * k
            )

            angles.append(
                self.normalize_angle(
                    angle
                )
            )

        angles.extend(
            [
                0.0,
                math.pi / 2.0,
                math.pi / 4.0,
                -math.pi / 4.0
            ]
        )

        unique = []

        for a in angles:

            if not any(
                abs(
                    self.angle_difference(
                        a,
                        b
                    )
                ) < math.radians(3.0)
                for b in unique
            ):

                unique.append(a)

        if len(unique) > self.max_angle_candidates:

            unique = sorted(
                unique,
                key=lambda a:
                abs(
                    self.angle_difference(
                        a,
                        theta_pca
                    )
                )
            )

            unique = unique[
                :self.max_angle_candidates
            ]

        return unique

    # ========================================================
    # Generate phases
    # ========================================================

    def generate_phases(
        self,
        spacing_cells
    ):

        spacing_cells = max(
            1.0,
            spacing_cells
        )

        if self.phase_samples <= 1:

            return [
                0.0
            ]

        phases = []

        for i in range(
            self.phase_samples
        ):

            phases.append(
                (
                    i /
                    float(self.phase_samples)
                )
                *
                spacing_cells
            )

        return phases

    # ========================================================
    # Generate cell candidates
    # ========================================================

    def generate_cell_candidates(
        self,
        cell,
        cell_mask,
        resolution
    ):

        spacing_cells = (
            self.coverage_spacing /
            max(
                resolution,
                1e-6
            )
        )

        spacing_cells = max(
            1.0,
            spacing_cells
        )

        angles = self.generate_angle_candidates(
            cell
        )

        phases = self.generate_phases(
            spacing_cells
        )

        candidates = []

        for angle in angles:

            for phase in phases:

                result = self.generate_rotated_coverage(
                    cell_mask,
                    angle,
                    phase,
                    spacing_cells
                )

                if result is None:
                    continue

                path, coverage_segments = result

                if len(path) < 2:
                    continue

                entry = path[0]
                exit_ = path[-1]

                length_cells = self.path_length_grid(
                    path
                )

                coverage_length_m = (
                    length_cells *
                    resolution
                )

                turn_count = max(
                    0,
                    len(coverage_segments) - 1
                )

                turn_angle = self.coverage_turn_angle(
                    coverage_segments
                )

                turn_time = (
                    turn_angle /
                    max(
                        self.robot_angular_speed,
                        1e-6
                    )
                    *
                    self.turn_time_weight
                )

                coverage_time = (
                    coverage_length_m /
                    max(
                        self.robot_linear_speed,
                        1e-6
                    )
                    +
                    turn_time
                )

                score = (
                    coverage_time
                    +
                    0.05 *
                    turn_count
                )

                candidates.append(
                    CoverageCandidate(
                        cell_id=cell.id,
                        angle=angle,
                        phase=phase,
                        path=path,
                        entry=entry,
                        exit=exit_,
                        coverage_length_m=coverage_length_m,
                        coverage_time=coverage_time,
                        turn_count=turn_count,
                        turn_time=turn_time,
                        score=score
                    )
                )

        candidates.sort(
            key=lambda x: x.score
        )

        top_k = max(
            1,
            int(self.candidate_top_k)
        )

        return candidates[
            :min(
                top_k,
                len(candidates)
            )
        ]

    # ========================================================
    # TRUE Zip-Zap scanline generator
    # ========================================================

    def generate_rotated_coverage(
        self,
        cell_mask,
        angle,
        phase,
        spacing_cells
    ):

        pixels = np.argwhere(
            cell_mask
        )

        if len(pixels) == 0:
            return None

        center = pixels.mean(
            axis=0
        )

        st = math.sin(
            angle
        )

        ct = math.cos(
            angle
        )

        u = np.array([
            st,
            ct
        ])

        v = np.array([
            ct,
            -st
        ])

        centered = (
            pixels.astype(float)
            -
            center
        )

        sweep = centered @ u
        normal = centered @ v

        n_min = float(
            normal.min()
        )

        n_max = float(
            normal.max()
        )

        if n_max - n_min < 0.5:

            return None

        first_lane = (
            math.floor(
                (
                    n_min - phase
                ) /
                spacing_cells
            )
            *
            spacing_cells
            +
            phase
        )

        lanes = []

        current = first_lane

        while current <= (
            n_max + 1e-6
        ):

            lanes.append(
                current
            )

            current += spacing_cells

        scan_segments = []

        for lane_index, lane_n in enumerate(lanes):

            selected = np.where(
                np.abs(
                    normal - lane_n
                )
                <=
                self.scan_band_cells
            )[0]

            if len(selected) == 0:
                continue

            selected = selected[
                np.argsort(
                    sweep[selected]
                )
            ]

            runs = []

            current_run = [
                selected[0]
            ]

            for k in range(
                1,
                len(selected)
            ):

                a_idx = selected[
                    k - 1
                ]

                b_idx = selected[
                    k
                ]

                a = pixels[
                    a_idx
                ]

                b = pixels[
                    b_idx
                ]

                distance = math.hypot(
                    float(a[0] - b[0]),
                    float(a[1] - b[1])
                )

                if distance > self.scan_gap_cells:

                    if len(current_run) > 0:

                        runs.append(
                            current_run
                        )

                    current_run = [
                        b_idx
                    ]

                else:

                    current_run.append(
                        b_idx
                    )

            if current_run:
                runs.append(
                    current_run
                )

            for run in runs:

                if len(run) < 2:
                    continue

                points = [
                    tuple(
                        pixels[i]
                    )
                    for i in run
                ]

                start = points[0]
                end = points[-1]

                if start == end:
                    continue

                scan_segments.append(
                    (
                        lane_index,
                        start,
                        end
                    )
                )

        if not scan_segments:
            return None

        scan_segments.sort(
            key=lambda x: (
                x[0],
                x[1][0] + x[1][1]
            )
        )

        final_path = []

        coverage_segments = []

        previous_end = None

        for segment_index, (
            lane_index,
            start,
            end
        ) in enumerate(scan_segments):

            if segment_index % 2 == 0:

                seg_start = start
                seg_end = end

            else:

                seg_start = end
                seg_end = start

            if previous_end is not None:

                connector = self.astar(
                    previous_end,
                    seg_start,
                    cell_mask
                )

                if connector is None:
                    return None

                if len(connector) > 1:

                    if final_path:

                        final_path.extend(
                            connector[1:]
                        )

                    else:

                        final_path.extend(
                            connector
                        )

            else:

                final_path.append(
                    seg_start
                )

            coverage = self.raster_line(
                seg_start,
                seg_end
            )

            if not self.path_is_free(
                coverage,
                cell_mask
            ):

                coverage = self.astar(
                    seg_start,
                    seg_end,
                    cell_mask
                )

                if coverage is None:
                    return None

            if final_path:

                if coverage[0] == final_path[-1]:

                    final_path.extend(
                        coverage[1:]
                    )

                else:

                    final_path.extend(
                        coverage
                    )

            else:

                final_path.extend(
                    coverage
                )

            coverage_segments.append(
                coverage
            )

            previous_end = seg_end

        final_path = self.remove_consecutive_duplicates(
            final_path
        )

        if len(final_path) < 2:
            return None

        return (
            final_path,
            coverage_segments
        )

    # ========================================================
    # Raster line
    # ========================================================

    def raster_line(
        self,
        p0,
        p1
    ):

        r0, c0 = p0
        r1, c1 = p1

        dr = abs(
            r1 - r0
        )

        dc = abs(
            c1 - c0
        )

        sr = (
            1
            if r0 < r1
            else -1
        )

        sc = (
            1
            if c0 < c1
            else -1
        )

        r = r0
        c = c0

        result = [
            (
                r,
                c
            )
        ]

        err = dc - dr

        while not (
            r == r1
            and c == c1
        ):

            e2 = 2 * err

            if e2 > -dr:

                err -= dr
                c += sc

            if e2 < dc:

                err += dc
                r += sr

            result.append(
                (
                    r,
                    c
                )
            )

        return result

    # ========================================================
    # Weighted A*
    # ========================================================

    def astar(
        self,
        start,
        goal,
        free_map,
        cost_map=None
    ):

        if start == goal:

            return [
                start
            ]

        rows, cols = free_map.shape

        sr, sc = start
        gr, gc = goal

        if not (
            0 <= sr < rows
            and
            0 <= sc < cols
            and
            0 <= gr < rows
            and
            0 <= gc < cols
        ):
            return None

        if not free_map[sr, sc]:
            return None

        if not free_map[gr, gc]:
            return None

        if cost_map is None:

            cost_map_key = None

        else:

            if cost_map.shape != free_map.shape:

                raise ValueError(
                    "cost_map shape mismatch"
                )

            cost_map_key = id(
                cost_map
            )

        cache_key = (
            id(free_map),
            cost_map_key,
            start,
            goal
        )

        cached = self.astar_cache.get(
            cache_key
        )

        if cached is not None:

            return list(
                cached
            )

        open_heap = []

        g_cost = {
            start: 0.0
        }

        parent = {}

        counter = 0

        heapq.heappush(
            open_heap,
            (
                self.heuristic(
                    start,
                    goal
                ),
                counter,
                start
            )
        )

        closed = set()

        if self.astar_diagonal:

            directions = (
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (-1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (1, 1, math.sqrt(2))
            )

        else:

            directions = (
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0)
            )

        heuristic = self.heuristic
        reconstruct_path = self.reconstruct_path

        while open_heap:

            _, _, current = heapq.heappop(
                open_heap
            )

            if current in closed:
                continue

            if current == goal:

                path = reconstruct_path(
                    parent,
                    current
                )

                self.astar_cache[
                    cache_key
                ] = tuple(path)

                return path

            closed.add(
                current
            )

            cr, cc = current

            for dr, dc, move_cost in directions:

                nr = cr + dr
                nc = cc + dc

                if not (
                    0 <= nr < rows
                    and
                    0 <= nc < cols
                ):
                    continue

                if not free_map[nr, nc]:
                    continue

                if dr != 0 and dc != 0:

                    if not (
                        free_map[cr, nc]
                        and
                        free_map[nr, cc]
                    ):
                        continue

                neighbor = (
                    nr,
                    nc
                )

                if neighbor in closed:
                    continue

                if cost_map is None:

                    terrain_cost = 1.0

                else:

                    terrain_cost = max(
                        1.0,
                        float(
                            cost_map[nr, nc]
                        )
                    )

                tentative_g = (
                    g_cost[current]
                    +
                    move_cost *
                    terrain_cost
                )

                old_g = g_cost.get(
                    neighbor,
                    float("inf")
                )

                if tentative_g < old_g:

                    parent[neighbor] = current

                    g_cost[neighbor] = tentative_g

                    counter += 1

                    f = (
                        tentative_g
                        +
                        heuristic(
                            neighbor,
                            goal
                        )
                    )

                    heapq.heappush(
                        open_heap,
                        (
                            f,
                            counter,
                            neighbor
                        )
                    )

        return None

    # ========================================================
    # Heuristic
    # ========================================================

    @staticmethod
    def heuristic(
        a,
        b
    ):

        dr = a[0] - b[0]
        dc = a[1] - b[1]

        return math.hypot(
            dr,
            dc
        )

    # ========================================================
    # Reconstruct path
    # ========================================================

    @staticmethod
    def reconstruct_path(
        parent,
        current
    ):

        path = [
            current
        ]

        while current in parent:

            current = parent[
                current
            ]

            path.append(
                current
            )

        path.reverse()

        return path

    # ========================================================
    # Path validation
    # ========================================================

    @staticmethod
    def path_is_free(
        path,
        free_map
    ):

        rows, cols = free_map.shape

        for r, c in path:

            if not (
                0 <= r < rows
                and
                0 <= c < cols
            ):
                return False

            if not free_map[r, c]:
                return False

        return True

    # ========================================================
    # Cell ordering
    # ========================================================

    def order_cells(
        self,
        cells,
        start_grid
    ):

        remaining = list(
            cells
        )

        ordered = []

        current = start_grid

        while remaining:

            best = min(
                remaining,
                key=lambda cell:
                self.distance_to_cell(
                    current,
                    cell
                )
            )

            ordered.append(
                best
            )

            centroid = best.centroid()

            current = (
                int(round(centroid[0])),
                int(round(centroid[1]))
            )

            remaining.remove(
                best
            )

        return ordered

    @staticmethod
    def distance_to_cell(
        point,
        cell
    ):

        centroid = cell.centroid()

        return math.hypot(
            point[0] - centroid[0],
            point[1] - centroid[1]
        )

    # ========================================================
    # Optimize candidate selection
    # ========================================================

    def optimize_cell_route(
        self,
        ordered_cells,
        all_candidates,
        free_map,
        cost_map,
        start_grid,
        resolution
    ):

        if not ordered_cells:
            return []

        dp = {}
        parent = {}

        first_cell = ordered_cells[0]

        for i, candidate in enumerate(
            all_candidates[
                first_cell.id
            ]
        ):

            connector = self.astar(
                start_grid,
                candidate.entry,
                free_map,
                cost_map
            )

            if connector is None:
                continue

            connector_time = self.path_time(
                connector,
                resolution,
                cost_map
            )

            total = (
                connector_time
                +
                candidate.coverage_time
            )

            dp[
                (
                    first_cell.id,
                    i
                )
            ] = total

        if not dp:
            return None

        for cell_index in range(
            1,
            len(ordered_cells)
        ):

            cell = ordered_cells[
                cell_index
            ]

            new_dp = {}

            candidates = all_candidates[
                cell.id
            ]

            prev_cell = ordered_cells[
                cell_index - 1
            ]

            prev_candidates = all_candidates[
                prev_cell.id
            ]

            for j, current_candidate in enumerate(
                candidates
            ):

                best_cost = float(
                    "inf"
                )

                best_prev = None

                for i, prev_candidate in enumerate(
                    prev_candidates
                ):

                    key = (
                        prev_cell.id,
                        i
                    )

                    if key not in dp:
                        continue

                    connector = self.astar(
                        prev_candidate.exit,
                        current_candidate.entry,
                        free_map,
                        cost_map
                    )

                    if connector is None:
                        continue

                    connector_time = self.path_time(
                        connector,
                        resolution,
                        cost_map
                    )

                    cost = (
                        dp[key]
                        +
                        connector_time
                        +
                        current_candidate.coverage_time
                    )

                    if cost < best_cost:

                        best_cost = cost

                        best_prev = key

                if best_prev is not None:

                    new_key = (
                        cell.id,
                        j
                    )

                    new_dp[
                        new_key
                    ] = best_cost

                    parent[
                        new_key
                    ] = best_prev

            dp = new_dp

            if not dp:
                return None

        final_key = min(
            dp,
            key=dp.get
        )

        selected = []

        key = final_key

        for index in range(
            len(ordered_cells) - 1,
            -1,
            -1
        ):

            cell = ordered_cells[
                index
            ]

            candidate_index = key[1]

            candidate = all_candidates[
                cell.id
            ][
                candidate_index
            ]

            selected.append(
                candidate
            )

            if index > 0:

                key = parent[
                    key
                ]

        selected.reverse()

        return selected

    # ========================================================
    # Build final segmented path
    # ========================================================

    def build_final_segments(
        self,
        start_grid,
        selected,
        free_map,
        cost_map
    ):

        if not selected:
            return []

        segments = []

        previous = start_grid

        for index, candidate in enumerate(
            selected
        ):

            connector = self.astar(
                previous,
                candidate.entry,
                free_map,
                cost_map
            )

            if connector is None:
                return []

            connector = self.remove_consecutive_duplicates(
                connector
            )

            if len(connector) >= 2:

                segment_type = (
                    "start"
                    if index == 0
                    else "connector"
                )

                if self.simplify_connector:

                    connector = self.shortcut_path(
                        connector,
                        free_map
                    )

                segments.append(
                    PathSegment(
                        points=connector,
                        segment_type=segment_type
                    )
                )

            coverage = self.remove_consecutive_duplicates(
                candidate.path
            )

            if len(coverage) >= 2:

                segments.append(
                    PathSegment(
                        points=coverage,
                        segment_type="coverage"
                    )
                )

            previous = candidate.exit

        return segments

    # ========================================================
    # Connector shortcut
    # ========================================================

    def shortcut_path(
        self,
        path,
        free_map
    ):

        if len(path) <= 2:
            return path

        result = [
            path[0]
        ]

        anchor = 0

        while anchor < len(path) - 1:

            best = anchor + 1

            for candidate_index in range(
                len(path) - 1,
                anchor,
                -1
            ):

                line = self.raster_line(
                    path[anchor],
                    path[candidate_index]
                )

                if self.path_is_free(
                    line,
                    free_map
                ):

                    best = candidate_index
                    break

            result.append(
                path[best]
            )

            anchor = best

        return result

    # ========================================================
    # Publish path
    # ========================================================

    def publish_segments(
        self,
        robot_id,
        segments,
        map_msg,
        resolution,
        origin_x,
        origin_y,
        origin_yaw
    ):

        msg = Path()

        msg.header = map_msg.header
        msg.header.stamp = rospy.Time.now()

        flattened = []

        for segment in segments:

            for point in segment.points:

                if (
                    not flattened
                    or
                    point != flattened[-1]
                ):

                    flattened.append(
                        point
                    )

        for i, point in enumerate(
            flattened
        ):

            row, col = point

            x, y = self.grid_to_world(
                row,
                col,
                map_msg
            )

            pose = PoseStamped()

            pose.header = msg.header

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if i + 1 < len(flattened):

                nr, nc = flattened[
                    i + 1
                ]

                nx, ny = self.grid_to_world(
                    nr,
                    nc,
                    map_msg
                )

                yaw = math.atan2(
                    ny - y,
                    nx - x
                )

            elif i > 0:

                pr, pc = flattened[
                    i - 1
                ]

                px, py = self.grid_to_world(
                    pr,
                    pc,
                    map_msg
                )

                yaw = math.atan2(
                    y - py,
                    x - px
                )

            else:

                yaw = 0.0

            pose.pose.orientation.z = math.sin(
                yaw / 2.0
            )

            pose.pose.orientation.w = math.cos(
                yaw / 2.0
            )

            msg.poses.append(
                pose
            )

        self.path_publishers[
            robot_id
        ].publish(
            msg
        )

    # ========================================================
    # Empty path
    # ========================================================

    def publish_empty_path(
        self,
        robot_id,
        frame_id
    ):

        msg = Path()

        msg.header.frame_id = frame_id
        msg.header.stamp = rospy.Time.now()

        self.path_publishers[
            robot_id
        ].publish(
            msg
        )

    # ========================================================
    # BCD visualization
    # ========================================================

    def publish_bcd_cells(
        self,
        robot_id,
        cells,
        map_msg
    ):

        marker_array = MarkerArray()

        for index, cell in enumerate(
            cells
        ):

            marker = Marker()

            marker.header = map_msg.header
            marker.header.stamp = rospy.Time.now()

            marker.ns = (
                "robot%d_bcd"
                %
                robot_id
            )

            marker.id = index

            marker.type = Marker.CUBE_LIST

            marker.action = Marker.ADD

            marker.scale.x = (
                map_msg.info.resolution
            )

            marker.scale.y = (
                map_msg.info.resolution
            )

            marker.scale.z = 0.02

            marker.color.a = 0.25

            for row, col in cell.pixels:

                x, y = self.grid_to_world(
                    row,
                    col,
                    map_msg
                )

                p = Point()

                p.x = x
                p.y = y
                p.z = 0.0

                marker.points.append(
                    p
                )

            marker_array.markers.append(
                marker
            )

        self.cell_publishers[
            robot_id
        ].publish(
            marker_array
        )

    # ========================================================
    # Path arrows
    # ========================================================

    def publish_path_arrows(
        self,
        robot_id,
        segments,
        map_msg,
        resolution,
        origin_x,
        origin_y,
        origin_yaw
    ):

        marker_array = MarkerArray()

        flattened = []

        for segment in segments:

            flattened.extend(
                segment.points
            )

        flattened = self.remove_consecutive_duplicates(
            flattened
        )

        step = max(
            1,
            int(
                0.5 /
                max(
                    resolution,
                    1e-6
                )
            )
        )

        for i in range(
            0,
            len(flattened) - 1,
            step
        ):

            r1, c1 = flattened[i]
            r2, c2 = flattened[i + 1]

            x1, y1 = self.grid_to_world(
                r1,
                c1,
                map_msg
            )

            x2, y2 = self.grid_to_world(
                r2,
                c2,
                map_msg
            )

            marker = Marker()

            marker.header = map_msg.header
            marker.header.stamp = rospy.Time.now()

            marker.ns = (
                "robot%d_path_arrows"
                %
                robot_id
            )

            marker.id = i

            marker.type = Marker.ARROW

            marker.action = Marker.ADD

            marker.scale.x = 0.25
            marker.scale.y = 0.05
            marker.scale.z = 0.05

            marker.points = []

            p1 = Point()

            p1.x = x1
            p1.y = y1
            p1.z = 0.05

            p2 = Point()

            p2.x = x2
            p2.y = y2
            p2.z = 0.05

            marker.points.append(
                p1
            )

            marker.points.append(
                p2
            )

            marker.color.a = 0.8

            marker_array.markers.append(
                marker
            )

        self.arrow_publishers[
            robot_id
        ].publish(
            marker_array
        )

    # ========================================================
    # Timing
    # ========================================================

    def path_time(
        self,
        path,
        resolution,
        cost_map=None
    ):

        if len(path) < 2:
            return 0.0

        weighted_distance_m = 0.0

        resolution = float(
            resolution
        )

        linear_speed = max(
            self.robot_linear_speed,
            1e-6
        )

        if cost_map is None:

            for i in range(
                len(path) - 1
            ):

                r1, c1 = path[i]
                r2, c2 = path[i + 1]

                move_cells = math.hypot(
                    r2 - r1,
                    c2 - c1
                )

                weighted_distance_m += (
                    move_cells *
                    resolution
                )

        else:

            for i in range(
                len(path) - 1
            ):

                r1, c1 = path[i]
                r2, c2 = path[i + 1]

                move_cells = math.hypot(
                    r2 - r1,
                    c2 - c1
                )

                move_m = (
                    move_cells *
                    resolution
                )

                terrain_cost = max(
                    1.0,
                    float(
                        cost_map[r2, c2]
                    )
                )

                weighted_distance_m += (
                    move_m *
                    terrain_cost
                )

        return (
            weighted_distance_m /
            linear_speed
        )

    # ========================================================
    # Path length
    # ========================================================

    @staticmethod
    def path_length_grid(
        path
    ):

        if len(path) < 2:
            return 0.0

        total = 0.0

        for i in range(
            len(path) - 1
        ):

            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            total += math.hypot(
                r2 - r1,
                c2 - c1
            )

        return total

    # ========================================================
    # Coverage turn angle
    # ========================================================

    def coverage_turn_angle(
        self,
        segments
    ):

        if len(segments) < 2:
            return 0.0

        total = 0.0

        for i in range(
            len(segments) - 1
        ):

            a = segments[i]
            b = segments[i + 1]

            if len(a) < 2:
                continue

            if len(b) < 2:
                continue

            v1 = (
                a[-1][0] - a[-2][0],
                a[-1][1] - a[-2][1]
            )

            v2 = (
                b[1][0] - b[0][0],
                b[1][1] - b[0][1]
            )

            angle1 = math.atan2(
                v1[0],
                v1[1]
            )

            angle2 = math.atan2(
                v2[0],
                v2[1]
            )

            total += abs(
                self.angle_difference(
                    angle2,
                    angle1
                )
            )

        return total

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def remove_consecutive_duplicates(
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

    @staticmethod
    def normalize_angle(
        angle
    ):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle <= -math.pi:
            angle += 2.0 * math.pi

        return angle

    @staticmethod
    def angle_difference(
        a,
        b
    ):

        return CoveragePathPlanner.normalize_angle(
            a - b
        )

    # ========================================================
    # Shutdown
    # ========================================================

    def shutdown(self):

        rospy.loginfo(
            "[CoveragePlanner] shutting down"
        )


# ============================================================
# Main
# ============================================================

def main():

    planner = CoveragePathPlanner()

    rospy.on_shutdown(
        planner.shutdown
    )

    rospy.spin()


if __name__ == "__main__":

    main()