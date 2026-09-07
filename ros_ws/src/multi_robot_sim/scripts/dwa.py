#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import numpy as np


class DWAPlanner:

    """
    Dynamic Window Approach for differential-drive robot.

    State:
        [x, y, yaw, v, omega]

    Output:
        [v, omega]

    The planner considers:
        1. global path tracking
        2. heading
        3. obstacle clearance
        4. velocity
        5. other robot collision
    """

    def __init__(
        self,
        max_speed=0.6,
        min_speed=0.0,
        max_yaw_rate=1.2,
        max_accel=0.8,
        max_yaw_accel=1.5,

        v_resolution=0.05,
        yaw_rate_resolution=0.10,

        dt=0.1,
        predict_time=1.5,

        robot_radius=0.25,
        other_robot_radius=0.25,

        obstacle_cost_weight=2.0,
        path_cost_weight=1.5,
        heading_cost_weight=1.0,
        velocity_cost_weight=0.5,
        robot_cost_weight=5.0
    ):

        self.max_speed = max_speed
        self.min_speed = min_speed

        self.max_yaw_rate = max_yaw_rate

        self.max_accel = max_accel
        self.max_yaw_accel = max_yaw_accel

        self.v_resolution = v_resolution
        self.yaw_rate_resolution = (
            yaw_rate_resolution
        )

        self.dt = dt
        self.predict_time = predict_time

        self.robot_radius = robot_radius
        self.other_robot_radius = (
            other_robot_radius
        )

        self.obstacle_cost_weight = (
            obstacle_cost_weight
        )

        self.path_cost_weight = (
            path_cost_weight
        )

        self.heading_cost_weight = (
            heading_cost_weight
        )

        self.velocity_cost_weight = (
            velocity_cost_weight
        )

        self.robot_cost_weight = (
            robot_cost_weight
        )

    # ================================================================
    # Dynamic window
    # ================================================================

    def dynamic_window(
        self,
        current_v,
        current_omega
    ):

        v_min = max(
            self.min_speed,
            current_v -
            self.max_accel * self.dt
        )

        v_max = min(
            self.max_speed,
            current_v +
            self.max_accel * self.dt
        )

        omega_min = max(
            -self.max_yaw_rate,
            current_omega -
            self.max_yaw_accel * self.dt
        )

        omega_max = min(
            self.max_yaw_rate,
            current_omega +
            self.max_yaw_accel * self.dt
        )

        return (
            v_min,
            v_max,
            omega_min,
            omega_max
        )

    # ================================================================
    # Motion model
    # ================================================================

    @staticmethod
    def motion(
        state,
        v,
        omega,
        dt
    ):

        x, y, yaw, _, _ = state

        x += (
            v *
            math.cos(yaw) *
            dt
        )

        y += (
            v *
            math.sin(yaw) *
            dt
        )

        yaw += (
            omega *
            dt
        )

        yaw = (
            math.atan2(
                math.sin(yaw),
                math.cos(yaw)
            )
        )

        return [
            x,
            y,
            yaw,
            v,
            omega
        ]

    # ================================================================
    # Trajectory prediction
    # ================================================================

    def predict_trajectory(
        self,
        state,
        v,
        omega
    ):

        trajectory = []

        current = list(state)

        time = 0.0

        while time <= self.predict_time:

            trajectory.append(
                current[:]
            )

            current = self.motion(
                current,
                v,
                omega,
                self.dt
            )

            time += self.dt

        return trajectory

    # ================================================================
    # Nearest path point
    # ================================================================

    @staticmethod
    def nearest_path_point(
        x,
        y,
        global_path
    ):

        if not global_path:
            return None

        best_point = None
        best_distance = float("inf")

        for point in global_path:

            px, py = point

            dx = px - x
            dy = py - y

            distance = (
                dx * dx +
                dy * dy
            )

            if distance < best_distance:

                best_distance = distance
                best_point = point

        return best_point

    # ================================================================
    # Path tracking cost
    # ================================================================

    def path_cost(
        self,
        trajectory,
        global_path
    ):

        if not global_path:
            return float("inf")

        total = 0.0

        for state in trajectory:

            x = state[0]
            y = state[1]

            point = (
                self.nearest_path_point(
                    x,
                    y,
                    global_path
                )
            )

            if point is None:
                continue

            px, py = point

            dx = px - x
            dy = py - y

            total += math.sqrt(
                dx * dx +
                dy * dy
            )

        return (
            total /
            max(1, len(trajectory))
        )

    # ================================================================
    # Heading cost
    # ================================================================

    def heading_cost(
        self,
        trajectory,
        global_path
    ):

        if not global_path:
            return 0.0

        state = trajectory[-1]

        x = state[0]
        y = state[1]
        yaw = state[2]

        target = (
            self.nearest_path_point(
                x,
                y,
                global_path
            )
        )

        if target is None:
            return 0.0

        tx, ty = target

        target_yaw = math.atan2(
            ty - y,
            tx - x
        )

        error = math.atan2(
            math.sin(
                target_yaw - yaw
            ),
            math.cos(
                target_yaw - yaw
            )
        )

        return abs(error)

    # ================================================================
    # Map obstacle cost
    # ================================================================

    def obstacle_cost(
        self,
        trajectory,
        occupancy_grid
    ):

        if occupancy_grid is None:
            return 0.0

        height, width = (
            occupancy_grid.shape
        )

        min_distance = float("inf")

        for state in trajectory:

            x = state[0]
            y = state[1]

            # The manager provides world -> grid
            gx, gy = (
                self.world_to_grid(
                    x,
                    y
                )
            )

            if (
                gx < 0 or
                gx >= width or
                gy < 0 or
                gy >= height
            ):

                return float("inf")

            # --------------------------------------------------------
            # Direct collision check around robot
            # --------------------------------------------------------

            radius_cells = max(
                1,
                int(
                    math.ceil(
                        self.robot_radius /
                        self.map_resolution
                    )
                )
            )

            for dy in range(
                -radius_cells,
                radius_cells + 1
            ):

                for dx in range(
                    -radius_cells,
                    radius_cells + 1
                ):

                    nx = gx + dx
                    ny = gy + dy

                    if (
                        nx < 0 or
                        nx >= width or
                        ny < 0 or
                        ny >= height
                    ):

                        return float("inf")

                    if occupancy_grid[
                        ny,
                        nx
                    ] < 0:

                        return float("inf")

            # --------------------------------------------------------
            # Distance to nearest obstacle
            # --------------------------------------------------------

            search_radius = 8

            for dy in range(
                -search_radius,
                search_radius + 1
            ):

                for dx in range(
                    -search_radius,
                    search_radius + 1
                ):

                    nx = gx + dx
                    ny = gy + dy

                    if (
                        nx < 0 or
                        nx >= width or
                        ny < 0 or
                        ny >= height
                    ):

                        continue

                    if occupancy_grid[
                        ny,
                        nx
                    ] < 0:

                        distance = math.sqrt(
                            (
                                dx *
                                self.map_resolution
                            ) ** 2 +
                            (
                                dy *
                                self.map_resolution
                            ) ** 2
                        )

                        min_distance = min(
                            min_distance,
                            distance
                        )

        if min_distance == float("inf"):
            return 0.0

        return 1.0 / max(
            min_distance,
            0.05
        )

    # ================================================================
    # Other robot collision cost
    # ================================================================

    def robot_collision_cost(
        self,
        trajectory,
        other_robot
    ):

        if other_robot is None:
            return 0.0

        ox, oy = other_robot

        min_distance = float("inf")

        collision_distance = (
            self.robot_radius +
            self.other_robot_radius
        )

        for state in trajectory:

            x = state[0]
            y = state[1]

            dx = x - ox
            dy = y - oy

            distance = math.sqrt(
                dx * dx +
                dy * dy
            )

            min_distance = min(
                min_distance,
                distance
            )

            if distance <= collision_distance:

                return float("inf")

        return 1.0 / max(
            min_distance,
            0.05
        )

    # ================================================================
    # Velocity cost
    # ================================================================

    def velocity_cost(
        self,
        v
    ):

        return (
            self.max_speed - v
        )

    # ================================================================
    # Set map information
    # ================================================================

    def set_map_info(
        self,
        resolution,
        origin_x,
        origin_y
    ):

        self.map_resolution = (
            resolution
        )

        self.origin_x = origin_x
        self.origin_y = origin_y

    # ================================================================
    # World -> Grid
    # ================================================================

    def world_to_grid(
        self,
        x,
        y
    ):

        gx = int(
            math.floor(
                (
                    x -
                    self.origin_x
                ) /
                self.map_resolution
            )
        )

        gy = int(
            math.floor(
                (
                    y -
                    self.origin_y
                ) /
                self.map_resolution
            )
        )

        return gx, gy

    # ================================================================
    # Main DWA
    # ================================================================

    def plan(
        self,
        state,
        global_path,
        occupancy_grid,
        other_robot=None
    ):

        if not global_path:

            return (
                0.0,
                0.0
            )

        current_v = state[3]
        current_omega = state[4]

        (
            v_min,
            v_max,
            omega_min,
            omega_max
        ) = self.dynamic_window(
            current_v,
            current_omega
        )

        best_cost = float("inf")

        best_v = 0.0
        best_omega = 0.0

        # ------------------------------------------------------------
        # Candidate velocities
        # ------------------------------------------------------------

        v_values = np.arange(
            v_min,
            v_max +
            self.v_resolution * 0.5,
            self.v_resolution
        )

        omega_values = np.arange(
            omega_min,
            omega_max +
            self.yaw_rate_resolution * 0.5,
            self.yaw_rate_resolution
        )

        for v in v_values:

            for omega in omega_values:

                trajectory = (
                    self.predict_trajectory(
                        state,
                        float(v),
                        float(omega)
                    )
                )

                obstacle_cost = (
                    self.obstacle_cost(
                        trajectory,
                        occupancy_grid
                    )
                )

                if math.isinf(
                    obstacle_cost
                ):

                    continue

                robot_cost = (
                    self.robot_collision_cost(
                        trajectory,
                        other_robot
                    )
                )

                if math.isinf(
                    robot_cost
                ):

                    continue

                path_cost = (
                    self.path_cost(
                        trajectory,
                        global_path
                    )
                )

                heading_cost = (
                    self.heading_cost(
                        trajectory,
                        global_path
                    )
                )

                velocity_cost = (
                    self.velocity_cost(
                        float(v)
                    )
                )

                total_cost = (

                    self.obstacle_cost_weight *
                    obstacle_cost

                    +

                    self.robot_cost_weight *
                    robot_cost

                    +

                    self.path_cost_weight *
                    path_cost

                    +

                    self.heading_cost_weight *
                    heading_cost

                    +

                    self.velocity_cost_weight *
                    velocity_cost
                )

                if total_cost < best_cost:

                    best_cost = total_cost

                    best_v = float(v)
                    best_omega = float(omega)

        return (
            best_v,
            best_omega
        )