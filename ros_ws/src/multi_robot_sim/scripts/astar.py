#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import heapq
import rospy

import numpy as np


class AStarPlanner:

    """
    Grid-based A* planner.

    Static obstacle:
        OccupancyGrid value < 0

    Optional allowed_mask:
        True  -> robot is allowed to use this cell
        False -> forbidden

    This allows each robot to stay inside its allocated region.
    """

    def __init__(
        self,
        allow_diagonal=False
    ):

        self.allow_diagonal = (
            allow_diagonal
        )

    # ================================================================
    # Heuristic
    # ================================================================

    @staticmethod
    def heuristic(
        x1,
        y1,
        x2,
        y2
    ):

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        return dx + dy

    # ================================================================
    # Valid cell
    # ================================================================

    @staticmethod
    def valid_cell(
        x,
        y,
        width,
        height
    ):

        return (
            0 <= x < width and
            0 <= y < height
        )

    # ================================================================
    # Neighbours
    # ================================================================

    def get_neighbors(
        self,
        x,
        y
    ):

        neighbors = [
            (x + 1, y, 1.0),
            (x - 1, y, 1.0),
            (x, y + 1, 1.0),
            (x, y - 1, 1.0)
        ]

        if self.allow_diagonal:

            diagonal_cost = math.sqrt(2.0)

            neighbors.extend([
                (
                    x + 1,
                    y + 1,
                    diagonal_cost
                ),
                (
                    x + 1,
                    y - 1,
                    diagonal_cost
                ),
                (
                    x - 1,
                    y + 1,
                    diagonal_cost
                ),
                (
                    x - 1,
                    y - 1,
                    diagonal_cost
                )
            ])

        return neighbors

    # ================================================================
    # Plan
    # ================================================================

    def plan(
        self,
        occupancy_array,
        start,
        goal,
        allowed_mask=None
    ):

        if occupancy_array is None:
            return []

        height, width = (
            occupancy_array.shape
        )

        sx, sy = start
        gx, gy = goal

        # ------------------------------------------------------------
        # Check start / goal
        # ------------------------------------------------------------

        if not self.valid_cell(
            sx,
            sy,
            width,
            height
        ):

            rospy.logwarn(
                "[A*] Start outside map."
            )

            return []

        if not self.valid_cell(
            gx,
            gy,
            width,
            height
        ):

            rospy.logwarn(
                "[A*] Goal outside map."
            )

            return []

        # ------------------------------------------------------------
        # Occupancy condition
        # ------------------------------------------------------------

        def traversable(x, y):

            if occupancy_array[
                y,
                x
            ] < 0:

                return False

            if allowed_mask is not None:

                if not allowed_mask[
                    y,
                    x
                ]:

                    return False

            return True

        if not traversable(sx, sy):

            rospy.logwarn(
                "[A*] Start is not traversable."
            )

            return []

        if not traversable(gx, gy):

            rospy.logwarn(
                "[A*] Goal is not traversable."
            )

            return []

        # ------------------------------------------------------------
        # Same point
        # ------------------------------------------------------------

        if start == goal:

            return [start]

        # ------------------------------------------------------------
        # A*
        # ------------------------------------------------------------

        open_set = []

        g_cost = {}

        parent = {}

        start_key = (
            sx,
            sy
        )

        goal_key = (
            gx,
            gy
        )

        g_cost[
            start_key
        ] = 0.0

        h = self.heuristic(
            sx,
            sy,
            gx,
            gy
        )

        heapq.heappush(
            open_set,
            (
                h,
                0.0,
                sx,
                sy
            )
        )

        closed = set()

        expanded = 0

        while open_set:

            (
                f,
                current_g,
                x,
                y
            ) = heapq.heappop(
                open_set
            )

            current = (
                x,
                y
            )

            if current in closed:
                continue

            closed.add(current)

            expanded += 1

            # --------------------------------------------------------
            # Goal reached
            # --------------------------------------------------------

            if current == goal_key:

                path = []

                node = current

                while node != start_key:

                    path.append(node)

                    node = parent[node]

                path.append(start_key)

                path.reverse()

                rospy.logdebug(
                    "[A*] Path found. "
                    "Length=%d, expanded=%d",
                    len(path),
                    expanded
                )

                return path

            # --------------------------------------------------------
            # Expand neighbours
            # --------------------------------------------------------

            for nx, ny, move_cost in (
                self.get_neighbors(x, y)
            ):

                if not self.valid_cell(
                    nx,
                    ny,
                    width,
                    height
                ):

                    continue

                if not traversable(
                    nx,
                    ny
                ):

                    continue

                neighbor = (
                    nx,
                    ny
                )

                if neighbor in closed:
                    continue

                tentative_g = (
                    current_g +
                    move_cost
                )

                old_g = g_cost.get(
                    neighbor,
                    float("inf")
                )

                if tentative_g < old_g:

                    g_cost[
                        neighbor
                    ] = tentative_g

                    parent[
                        neighbor
                    ] = current

                    h = self.heuristic(
                        nx,
                        ny,
                        gx,
                        gy
                    )

                    f = (
                        tentative_g +
                        h
                    )

                    heapq.heappush(
                        open_set,
                        (
                            f,
                            tentative_g,
                            nx,
                            ny
                        )
                    )

        rospy.logwarn(
            "[A*] No path found: "
            "(%d,%d) -> (%d,%d)",
            sx,
            sy,
            gx,
            gy
        )

        return []

    # ================================================================
    # Path length
    # ================================================================

    @staticmethod
    def path_length(
        path
    ):

        if len(path) < 2:
            return 0.0

        length = 0.0

        for i in range(
            len(path) - 1
        ):

            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            dx = x2 - x1
            dy = y2 - y1

            length += math.sqrt(
                dx * dx +
                dy * dy
            )

        return length