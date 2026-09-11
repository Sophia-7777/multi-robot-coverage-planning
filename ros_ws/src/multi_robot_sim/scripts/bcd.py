#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ============================================================
# BCD Cell
# ============================================================

@dataclass
class BCDCell:

    id: int

    pixels: List[Tuple[int, int]]

    size: int

    min_row: int
    max_row: int

    min_col: int
    max_col: int

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    def centroid(self):

        if not self.pixels:

            return (
                0.0,
                0.0
            )

        rows = [
            p[0]
            for p in self.pixels
        ]

        cols = [
            p[1]
            for p in self.pixels
        ]

        return (
            float(
                sum(rows)
            )
            /
            len(rows),

            float(
                sum(cols)
            )
            /
            len(cols)
        )

    # --------------------------------------------------------

    def bounds(self):

        return (
            self.min_row,
            self.max_row,
            self.min_col,
            self.max_col
        )

    # --------------------------------------------------------

    def get_rows(self):

        return sorted(
            set(
                r
                for r, c in self.pixels
            )
        )

    # --------------------------------------------------------

    def get_cols(self):

        return sorted(
            set(
                c
                for r, c in self.pixels
            )
        )

    # --------------------------------------------------------

    def pixels_in_row(
        self,
        row
    ):

        return sorted(
            c
            for r, c in self.pixels
            if r == row
        )

    # --------------------------------------------------------

    def pixels_in_col(
        self,
        col
    ):

        return sorted(
            r
            for r, c in self.pixels
            if c == col
        )


# ============================================================
# Boustrophedon Decomposition
# ============================================================

class BoustrophedonDecomposition:

    def __init__(
        self,
        occupancy_map,
        min_cell_size=1
    ):

        self.occupancy_map = (
            np.asarray(
                occupancy_map
            ).astype(bool)
        )

        self.min_cell_size = max(
            1,
            int(
                min_cell_size
            )
        )

        self.rows, self.cols = (
            self.occupancy_map.shape
        )

        self.cells = []

    # ========================================================
    # Main
    # ========================================================

    def decompose(self):

        """
        Raster BCD。

        对于 coverage planning，
        我们将自由区域按垂直扫描方向分解。

        这里的核心目标是：

        1. 保证每个 cell 内连通
        2. 不把不同障碍两侧的区域错误合并
        3. 给 coverage planner 提供稳定的局部区域
        """

        free = self.occupancy_map

        visited = np.zeros(
            free.shape,
            dtype=bool
        )

        cells = []

        cell_id = 0

        # ----------------------------------------------------
        # 先做连通区域分解
        # ----------------------------------------------------

        for r in range(
            self.rows
        ):

            for c in range(
                self.cols
            ):

                if not free[r, c]:
                    continue

                if visited[r, c]:
                    continue

                pixels = self.flood_fill(
                    r,
                    c,
                    free,
                    visited
                )

                if len(pixels) < self.min_cell_size:
                    continue

                rows = [
                    p[0]
                    for p in pixels
                ]

                cols = [
                    p[1]
                    for p in pixels
                ]

                cell = BCDCell(
                    id=cell_id,
                    pixels=pixels,
                    size=len(pixels),
                    min_row=min(rows),
                    max_row=max(rows),
                    min_col=min(cols),
                    max_col=max(cols)
                )

                cells.append(
                    cell
                )

                cell_id += 1

        # ----------------------------------------------------
        # 更细的 BCD split
        # ----------------------------------------------------
        #
        # 对简单矩形区域，一个连通 cell 就够。
        #
        # 对具有复杂拓扑的区域，
        # 根据每一行的连续区间进行 split。
        #
        # 这里采用稳定的 row-interval 方法。
        #

        refined = []

        refined_id = 0

        for cell in cells:

            subcells = self.split_by_row_intervals(
                cell
            )

            for pixels in subcells:

                if len(pixels) < self.min_cell_size:
                    continue

                rows = [
                    p[0]
                    for p in pixels
                ]

                cols = [
                    p[1]
                    for p in pixels
                ]

                refined.append(
                    BCDCell(
                        id=refined_id,
                        pixels=pixels,
                        size=len(pixels),
                        min_row=min(rows),
                        max_row=max(rows),
                        min_col=min(cols),
                        max_col=max(cols)
                    )
                )

                refined_id += 1

        self.cells = refined

        return refined

    # ========================================================
    # Flood fill
    # ========================================================

    def flood_fill(
        self,
        start_r,
        start_c,
        free,
        visited
    ):

        queue = [
            (
                start_r,
                start_c
            )
        ]

        visited[
            start_r,
            start_c
        ] = True

        pixels = []

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        while queue:

            r, c = queue.pop()

            pixels.append(
                (
                    r,
                    c
                )
            )

            for dr, dc in directions:

                nr = r + dr
                nc = c + dc

                if not (
                    0 <= nr < self.rows
                    and
                    0 <= nc < self.cols
                ):
                    continue

                if visited[nr, nc]:
                    continue

                if not free[nr, nc]:
                    continue

                visited[nr, nc] = True

                queue.append(
                    (
                        nr,
                        nc
                    )
                )

        return pixels

    # ========================================================
    # Row interval split
    # ========================================================

    def split_by_row_intervals(
        self,
        cell
    ):

        """
        根据每一行连续 free interval 建立局部区域。

        这是一个 raster-friendly 的 BCD 实现。

        对典型 coverage map：

            ###########
            #         #
            #         #
            #    ######
            #         #
            ###########

        可以避免把完全不相连的区域作为一个 coverage cell。
        """

        pixel_set = set(
            cell.pixels
        )

        rows = sorted(
            set(
                r
                for r, c in cell.pixels
            )
        )

        if not rows:
            return []

        # ----------------------------------------------------
        # 建立每一行的 interval
        # ----------------------------------------------------

        row_intervals = {}

        for r in rows:

            cols = sorted(
                c
                for rr, c in cell.pixels
                if rr == r
            )

            intervals = []

            if cols:

                start = cols[0]
                prev = cols[0]

                for c in cols[1:]:

                    if c == prev + 1:

                        prev = c

                    else:

                        intervals.append(
                            (
                                start,
                                prev
                            )
                        )

                        start = c
                        prev = c

                intervals.append(
                    (
                        start,
                        prev
                    )
                )

            row_intervals[r] = intervals

        # ----------------------------------------------------
        # 如果所有行只有一个 interval，
        # 当前 cell 已经足够简单。
        # ----------------------------------------------------

        if all(
            len(row_intervals[r]) <= 1
            for r in rows
        ):

            return [
                list(
                    cell.pixels
                )
            ]

        # ----------------------------------------------------
        # General connected components
        # over pixels.
        #
        # 这里再次使用 4-connectivity，
        # 保证 split 后每个 cell 都是真正连通的。
        # ----------------------------------------------------

        visited = set()

        components = []

        for p in cell.pixels:

            if p in visited:
                continue

            queue = [
                p
            ]

            visited.add(
                p
            )

            component = []

            while queue:

                current = queue.pop()

                component.append(
                    current
                )

                r, c = current

                for dr, dc in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1)
                ):

                    neighbor = (
                        r + dr,
                        c + dc
                    )

                    if neighbor in visited:
                        continue

                    if neighbor not in pixel_set:
                        continue

                    visited.add(
                        neighbor
                    )

                    queue.append(
                        neighbor
                    )

            components.append(
                component
            )

        return components


# ============================================================
# Optional compatibility helpers
# ============================================================

def build_bcd(
    occupancy_map,
    min_cell_size=1
):

    planner = BoustrophedonDecomposition(
        occupancy_map,
        min_cell_size=min_cell_size
    )

    return planner.decompose()