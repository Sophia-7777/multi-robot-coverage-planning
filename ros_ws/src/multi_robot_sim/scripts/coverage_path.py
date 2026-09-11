#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq
import threading
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
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
            0.60
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
            7
        )

        self.angle_step_deg = rospy.get_param(
            "~angle_step_deg",
            15.0
        )

        self.phase_samples = rospy.get_param(
            "~phase_samples",
            3
        )

        self.astar_diagonal = rospy.get_param(
            "~astar_diagonal",
            True
        )

        self.simplify_connector = rospy.get_param(
            "~simplify_connector",
            True
        )

        self.replan_start_threshold = rospy.get_param(
            "~replan_start_threshold",
            0.30
        )

        self.planning_period = rospy.get_param(
            "~planning_period",
            0.05
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

        self.robot_dirty = {
            1: False,
            2: False
        }

        self.robot_planning = {
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

        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback,
            queue_size=1
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

    # ========================================================
    # ROS callbacks
    # ========================================================

    def robot1_region_callback(self, msg):

        with self.state_lock:
            self.robot_regions[1] = msg
            self.robot_dirty[1] = True

        rospy.loginfo_throttle(
            2.0,
            "[CoveragePlanner] robot1 region updated"
        )

    def robot2_region_callback(self, msg):

        with self.state_lock:
            self.robot_regions[2] = msg
            self.robot_dirty[2] = True

        rospy.loginfo_throttle(
            2.0,
            "[CoveragePlanner] robot2 region updated"
        )

    def model_states_callback(self, msg):

        # 这里只做轻量级位置更新。
        # 绝对不能在这里做 BCD / A* / 路径优化。

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
            1.0 / self.planning_period
        )

        while not rospy.is_shutdown():

            jobs = []

            with self.state_lock:

                for robot_id in (1, 2):

                    if (
                        self.robot_dirty[robot_id]
                        and
                        not self.robot_planning[robot_id]
                        and
                        self.robot_regions[robot_id] is not None
                    ):

                        self.robot_planning[robot_id] = True

                        region = self.robot_regions[robot_id]

                        self.robot_dirty[robot_id] = False

                        jobs.append(
                            (
                                robot_id,
                                region
                            )
                        )

            # 不持有 state_lock 做重规划
            for robot_id, region in jobs:

                try:

                    self.process_robot(
                        robot_id,
                        region
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
    # Robot planning
    # ========================================================

    def process_robot(
        self,
        robot_id: int,
        region_msg: OccupancyGrid
    ):

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
        # Convert occupancy grid
        # ----------------------------------------------------

        occupancy = self.occupancy_grid_to_numpy(
            region_msg
        )

        resolution = region_msg.info.resolution

        origin_x = region_msg.info.origin.position.x
        origin_y = region_msg.info.origin.position.y

        origin_yaw = self.quaternion_to_yaw(
            region_msg.info.origin.orientation
        )

        # ----------------------------------------------------
        # Own region = 100
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

        # ----------------------------------------------------
        # Convert robot start to grid
        # ----------------------------------------------------

        start_grid = self.world_to_grid(
            start_pose[0],
            start_pose[1],
            region_msg
        )

        start_grid = self.find_nearest_free_cell(
            start_grid,
            own_region
        )

        if start_grid is None:

            rospy.logwarn(
                "[CoveragePlanner] robot%d cannot find valid start cell",
                robot_id
            )

            return

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
                mask[r, c] = True

            cell_masks[cell.id] = mask

        # ----------------------------------------------------
        # Generate candidate paths
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
            c for c in ordered_cells
            if c.id in all_candidates
        ]

        # ----------------------------------------------------
        # Select best candidate per cell
        # ----------------------------------------------------

        selected = self.optimize_cell_route(
            ordered_cells,
            all_candidates,
            own_region,
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
        # Build segmented final path
        # ----------------------------------------------------

        segments = self.build_final_segments(
            start_grid,
            selected,
            own_region
        )

        if not segments:

            rospy.logwarn(
                "[CoveragePlanner] robot%d final path empty",
                robot_id
            )

            return

        # ----------------------------------------------------
        # Validate robot hasn't moved during planning
        # ----------------------------------------------------

        with self.state_lock:

            current_pose = self.robot_positions[robot_id]

        if current_pose is None:
            return

        movement = math.hypot(
            current_pose[0] - start_pose[0],
            current_pose[1] - start_pose[1]
        )

        if movement > self.replan_start_threshold:

            rospy.logwarn(
                "[CoveragePlanner] robot%d moved %.2fm during planning, discard old path",
                robot_id,
                movement
            )

            with self.state_lock:
                self.robot_dirty[robot_id] = True

            return

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

        rospy.loginfo(
            "[CoveragePlanner] robot%d path published",
            robot_id
        )

    # ========================================================
    # Occupancy conversion
    # ========================================================

    def occupancy_grid_to_numpy(
        self,
        msg: OccupancyGrid
    ):

        h = msg.info.height
        w = msg.info.width

        data = np.asarray(
            msg.data,
            dtype=np.int16
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

        while queue:

            r, c = queue.pop(0)

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
                    (nr, nc)
                )

                queue.append(
                    (nr, nc)
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

        # PCA
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

            # row/col -> x/y
            theta_pca = math.atan2(
                direction[0],
                direction[1]
            )

        except Exception:

            theta_pca = 0.0

        angles = []

        # PCA 附近
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

        # 常用方向
        angles.extend(
            [
                0.0,
                math.pi / 2.0,
                math.pi / 4.0,
                -math.pi / 4.0
            ]
        )

        # 去重
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

        # 最多保留几个
        if len(unique) > self.max_angle_candidates:

            # 优先 PCA
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

                # 实际 heading change
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

        # 每个 cell 只保留最好的 K 个
        candidates.sort(
            key=lambda x: x.score
        )

        return candidates[
            :min(
                8,
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
        """
        真正的 Boustrophedon / Zip-Zap 路径。

        核心逻辑：

            lane 0:  ---------------->
            connector:              |
            lane 1:  <----------------
            connector:              |
            lane 2:  ---------------->

        注意：
        这里不再把所有 pixel 按最近 lane 排序。
        每一条 lane 都是真正的连续扫描段。
        """

        pixels = np.argwhere(
            cell_mask
        )

        if len(pixels) == 0:
            return None

        center = pixels.mean(
            axis=0
        )

        # row/col 坐标
        #
        # sweep direction:
        #     u = (sin(theta), cos(theta))
        #
        # normal:
        #     v = (cos(theta), -sin(theta))
        #
        # 这样 theta=0 时，扫描方向为 col 正方向。

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

        # ----------------------------------------------------
        # Generate lane coordinates
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Extract scanline runs
        # ----------------------------------------------------

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

            # 按扫描方向排序
            selected = selected[
                np.argsort(
                    sweep[selected]
                )
            ]

            # ------------------------------------------------
            # Split discontinuous runs
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Create segments
            # ------------------------------------------------

            for run in runs:

                if len(run) < 2:

                    # 太短的单点不作为独立覆盖线
                    continue

                points = [
                    tuple(
                        pixels[i]
                    )
                    for i in run
                ]

                # 压缩成真正的 scanline 两端
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

        # ----------------------------------------------------
        # 排序
        # ----------------------------------------------------

        scan_segments.sort(
            key=lambda x: (
                x[0],
                x[1][0] + x[1][1]
            )
        )

        # ----------------------------------------------------
        # 构造 Zip-Zap
        # ----------------------------------------------------

        final_path = []

        coverage_segments = []

        previous_end = None

        for segment_index, (
            lane_index,
            start,
            end
        ) in enumerate(scan_segments):

            # ------------------------------------------------
            # Alternate direction
            # ------------------------------------------------

            if segment_index % 2 == 0:

                seg_start = start
                seg_end = end

            else:

                seg_start = end
                seg_end = start

            # ------------------------------------------------
            # Connector
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Coverage line
            # ------------------------------------------------

            coverage = self.raster_line(
                seg_start,
                seg_end
            )

            # 如果直线不完全在 cell 中，
            # 用 A* 生成安全路径。
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

        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

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
    # A*
    # ========================================================

    def astar(
        self,
        start,
        goal,
        free_map
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

        cache_key = (
            id(free_map),
            start,
            goal
        )

        if cache_key in self.astar_cache:

            return list(
                self.astar_cache[
                    cache_key
                ]
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

            directions = [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (-1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (1, 1, math.sqrt(2))
            ]

        else:

            directions = [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0)
            ]

        while open_heap:

            _, _, current = heapq.heappop(
                open_heap
            )

            if current in closed:
                continue

            if current == goal:

                path = self.reconstruct_path(
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

                # 禁止 diagonal corner cutting
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

                tentative_g = (
                    g_cost[current]
                    +
                    move_cost
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
                        self.heuristic(
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
        start_grid,
        resolution
    ):

        if not ordered_cells:
            return []

        # ----------------------------------------------------
        # Dynamic programming over candidate states
        # ----------------------------------------------------

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
                free_map
            )

            if connector is None:
                continue

            connector_time = self.path_time(
                connector,
                resolution
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

        # ----------------------------------------------------
        # Remaining cells
        # ----------------------------------------------------

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
                        free_map
                    )

                    if connector is None:
                        continue

                    connector_time = self.path_time(
                        connector,
                        resolution
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

        # ----------------------------------------------------
        # Backtrack
        # ----------------------------------------------------

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
        free_map
    ):

        if not selected:
            return []

        segments = []

        previous = start_grid

        for index, candidate in enumerate(
            selected
        ):

            # ------------------------------------------------
            # Connector / start connection
            # ------------------------------------------------

            connector = self.astar(
                previous,
                candidate.entry,
                free_map
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

            # ------------------------------------------------
            # COVERAGE
            #
            # 这里绝对不能 shortcut。
            # ------------------------------------------------

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

            # 不指定固定颜色
            marker.color.a = 0.25

            for row, col in cell.pixels:

                x, y = self.grid_to_world(
                    row,
                    col,
                    map_msg
                )

                from geometry_msgs.msg import Point

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

        for i in range(
            0,
            len(flattened) - 1,
            max(
                1,
                int(
                    0.5 /
                    max(
                        resolution,
                        1e-6
                    )
                )
            )
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

            from geometry_msgs.msg import Point

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
        resolution
    ):

        if len(path) < 2:
            return 0.0

        distance_cells = (
            self.path_length_grid(
                path
            )
        )

        distance_m = (
            distance_cells *
            resolution
        )

        return (
            distance_m /
            max(
                self.robot_linear_speed,
                1e-6
            )
        )

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