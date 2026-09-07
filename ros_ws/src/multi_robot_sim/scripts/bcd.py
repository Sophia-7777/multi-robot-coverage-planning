#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bcd.py

Boustrophedon Cell Decomposition (BCD)

输入：
    2D numpy bool array
        True  -> free / 可通行
        False -> obstacle / 不可通行

输出：
    BCDCell 列表

设计目标：
    1. 正确检测 BCD 的 split / merge 拓扑事件
    2. 每个最终 Cell 拥有唯一 ID
    3. 与 coverage_path.py 当前接口兼容
    4. 支持后续在 Cell 内生成 Zig-Zag 路径
    5. 支持 Cell 之间使用 A* 连接

坐标约定：
    numpy:
        array[row, col]

    row:
        y 方向，向下增加

    col:
        x 方向，向右增加

    BCDCell.centroid():
        返回 (x=col, y=row)
"""

import numpy as np


class BCDCell(object):
    """
    BCD 单元。

    pixels:
        set((row, col), ...)

    id:
        唯一 Cell ID
    """

    def __init__(self, cell_id, pixels=None):
        self.id = int(cell_id)

        if pixels is None:
            pixels = set()

        self.pixels = set(pixels)

        self._update_properties()

    def _update_properties(self):
        """根据 pixels 更新 Cell 属性。"""

        self.size = len(self.pixels)

        if self.size == 0:
            self.min_row = None
            self.max_row = None
            self.min_col = None
            self.max_col = None
            return

        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]

        self.min_row = min(rows)
        self.max_row = max(rows)

        self.min_col = min(cols)
        self.max_col = max(cols)

    def add_pixel(self, row, col):
        """添加一个像素。"""

        self.pixels.add((int(row), int(col)))

        self._update_properties()

    def add_pixels(self, pixels):
        """批量添加像素。"""

        self.pixels.update(pixels)

        self._update_properties()

    def contains(self, row, col):
        """判断某个像素是否属于当前 Cell。"""

        return (int(row), int(col)) in self.pixels

    def centroid(self):
        """
        返回 Cell 几何中心。

        返回：
            (x, y)
            即：
            (col, row)
        """

        if not self.pixels:
            return (0.0, 0.0)

        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]

        return (
            float(np.mean(cols)),
            float(np.mean(rows))
        )

    def bounds(self):
        """
        返回边界：

            (min_col, min_row, max_col, max_row)
        """

        if self.size == 0:
            return (None, None, None, None)

        return (
            self.min_col,
            self.min_row,
            self.max_col,
            self.max_row
        )

    def get_rows(self):
        """
        获取当前 Cell 中所有 row。

        返回：
            sorted list
        """

        return sorted(set(row for row, col in self.pixels))

    def get_cols(self):
        """
        获取当前 Cell 中所有 col。

        返回：
            sorted list
        """

        return sorted(set(col for row, col in self.pixels))

    def pixels_in_row(self, row):
        """
        获取指定 row 中属于当前 Cell 的所有 col。
        """

        return sorted(
            col
            for r, col in self.pixels
            if r == row
        )

    def pixels_in_col(self, col):
        """
        获取指定 col 中属于当前 Cell 的所有 row。
        """

        return sorted(
            row
            for row, c in self.pixels
            if c == col
        )

    def __repr__(self):
        return (
            "BCDCell("
            "id={}, "
            "size={}, "
            "bounds=({}, {}, {}, {})"
            ")"
        ).format(
            self.id,
            self.size,
            self.min_col,
            self.min_row,
            self.max_col,
            self.max_row
        )


class BoustrophedonDecomposition(object):
    """
    Boustrophedon Cell Decomposition。

    核心思想：

        按 row 扫描地图。

        每一行先找到连续的 free interval：

            ####.....####....
                 ↑
              interval

        然后比较当前 row 和上一 row 的 interval。

        拓扑关系：

        1. 0 -> 1
           新 Cell

        2. 1 -> 1
           Cell 延续

        3. 1 -> N
           Split
           一个区域分裂成多个区域

        4. N -> 1
           Merge
           多个区域合并成一个区域

        5. N -> M
           复杂拓扑变化
    """

    def __init__(self, occupancy_map, min_cell_size=1):
        """
        参数：

        occupancy_map:
            2D numpy array

            True:
                free

            False:
                obstacle

        min_cell_size:
            最小 Cell 像素数量。
        """

        if occupancy_map is None:
            raise ValueError("occupancy_map cannot be None")

        self.map = np.asarray(
            occupancy_map,
            dtype=bool
        )

        if self.map.ndim != 2:
            raise ValueError(
                "occupancy_map must be a 2D array"
            )

        self.height, self.width = self.map.shape

        self.min_cell_size = max(
            1,
            int(min_cell_size)
        )

        # 下一个 Cell ID
        self.next_cell_id = 0

    # ============================================================
    # Cell ID
    # ============================================================

    def _new_cell(self):
        """创建一个新的 Cell。"""

        cell = BCDCell(
            self.next_cell_id
        )

        self.next_cell_id += 1

        return cell

    # ============================================================
    # Interval
    # ============================================================

    @staticmethod
    def _intervals_overlap(interval1, interval2):
        """
        判断两个闭区间是否重叠。

        interval:
            (start_col, end_col)

        例如：

            [2, 5]
            [5, 8]

        在像素网格中：
            col=5 是同一个 free pixel

        所以认为 overlap。
        """

        start1, end1 = interval1
        start2, end2 = interval2

        return not (
            end1 < start2 or
            end2 < start1
        )

    @staticmethod
    def _interval_contains(interval1, interval2):
        """
        判断 interval1 是否完全包含 interval2。
        """

        start1, end1 = interval1
        start2, end2 = interval2

        return (
            start1 <= start2 and
            end1 >= end2
        )

    # ============================================================
    # 找一行中的连续 free interval
    # ============================================================

    def _find_intervals(self, row):
        """
        找出某一行中所有连续 free 区间。

        例如：

            0 0 1 1 1 0 0 1 1

        返回：

            [(2, 4), (7, 8)]
        """

        intervals = []

        in_interval = False
        start_col = None

        for col in range(self.width):

            free = bool(self.map[row, col])

            if free and not in_interval:

                # 开始新的 interval
                start_col = col
                in_interval = True

            elif not free and in_interval:

                # 当前 interval 结束
                intervals.append(
                    (start_col, col - 1)
                )

                in_interval = False
                start_col = None

        # 如果这一行最后仍处于 interval
        if in_interval:

            intervals.append(
                (start_col, self.width - 1)
            )

        return intervals

    # ============================================================
    # Interval pixels
    # ============================================================

    @staticmethod
    def _interval_pixels(row, interval):
        """
        将 interval 转换成像素集合。
        """

        start_col, end_col = interval

        return set(
            (row, col)
            for col in range(
                start_col,
                end_col + 1
            )
        )

    # ============================================================
    # 找 predecessor
    # ============================================================

    def _find_predecessors(
        self,
        current_interval,
        previous_intervals
    ):
        """
        找当前 interval 与上一行哪些 interval 重叠。

        返回：
            predecessor index 列表
        """

        predecessors = []

        for index, previous_interval in enumerate(
            previous_intervals
        ):

            if self._intervals_overlap(
                current_interval,
                previous_interval
            ):
                predecessors.append(index)

        return predecessors

    # ============================================================
    # 主 BCD
    # ============================================================

    def decompose(self):
        """
        执行 BCD。

        返回：

            List[BCDCell]

        注意：

        返回的 Cell 是最终 BCD 分解结果，
        每一个 Cell 都具有唯一 ID。
        """

        # --------------------------------------------------------
        # 没有 free space
        # --------------------------------------------------------

        if not np.any(self.map):

            return []

        # --------------------------------------------------------
        # 最终 Cell
        #
        # 注意：
        #
        # 我们不会简单地把一个 lineage 永久保留。
        #
        # 在 split / merge 拓扑事件处，
        # 当前 Cell 会结束，并创建新的 Cell。
        #
        # 这样最终得到的 Cell 才是真正用于
        # coverage planning 的局部 BCD cell。
        # --------------------------------------------------------

        final_cells = []

        # 当前活动 Cell
        #
        # interval:
        #       (start_col, end_col)
        #
        # cell:
        #       BCDCell
        #
        active = []

        # 上一行的 intervals
        previous_intervals = []

        for row in range(self.height):

            current_intervals = self._find_intervals(row)

            # ====================================================
            # 当前行没有 free space
            # ====================================================

            if not current_intervals:

                # 所有 active Cell 到这里结束
                final_cells.extend(
                    item["cell"]
                    for item in active
                )

                active = []
                previous_intervals = []

                continue

            # ====================================================
            # 第一行 / 上一行没有 free interval
            # ====================================================

            if not previous_intervals:

                new_active = []

                for current_interval in current_intervals:

                    cell = self._new_cell()

                    cell.add_pixels(
                        self._interval_pixels(
                            row,
                            current_interval
                        )
                    )

                    new_active.append(
                        {
                            "interval": current_interval,
                            "cell": cell
                        }
                    )

                active = new_active
                previous_intervals = current_intervals

                continue

            # ====================================================
            # 建立 predecessor / successor 图
            # ====================================================

            current_predecessors = []

            for current_interval in current_intervals:

                predecessors = self._find_predecessors(
                    current_interval,
                    previous_intervals
                )

                current_predecessors.append(
                    predecessors
                )

            # ----------------------------------------------------
            # 反向关系：
            #
            # 一个 previous interval
            # 被几个 current interval 使用
            # ----------------------------------------------------

            previous_successors = [
                []
                for _ in previous_intervals
            ]

            for current_index, predecessors in enumerate(
                current_predecessors
            ):

                for previous_index in predecessors:

                    previous_successors[
                        previous_index
                    ].append(
                        current_index
                    )

            # ====================================================
            # 构造下一轮 active
            # ====================================================

            new_active = []

            # 记录本轮哪些旧 Cell 已经被结束
            finished_previous = set()

            # ====================================================
            # 逐个处理 current interval
            # ====================================================

            for current_index, current_interval in enumerate(
                current_intervals
            ):

                predecessors = current_predecessors[
                    current_index
                ]

                # ------------------------------------------------
                # Case 1:
                #
                # 没有 predecessor
                #
                # 新区域产生
                # ------------------------------------------------

                if len(predecessors) == 0:

                    cell = self._new_cell()

                    cell.add_pixels(
                        self._interval_pixels(
                            row,
                            current_interval
                        )
                    )

                    new_active.append(
                        {
                            "interval": current_interval,
                            "cell": cell
                        }
                    )

                    continue

                # ------------------------------------------------
                # Case 2:
                #
                # 一个 predecessor
                # ------------------------------------------------

                if len(predecessors) == 1:

                    previous_index = predecessors[0]

                    previous_item = active[
                        previous_index
                    ]

                    previous_cell = previous_item["cell"]

                    # 这个 previous interval 是否 split？
                    successors = previous_successors[
                        previous_index
                    ]

                    # ============================================
                    # 2A:
                    #
                    # 1 -> 1
                    #
                    # 正常延续
                    # ============================================

                    if len(successors) == 1:

                        previous_cell.add_pixels(
                            self._interval_pixels(
                                row,
                                current_interval
                            )
                        )

                        new_active.append(
                            {
                                "interval": current_interval,
                                "cell": previous_cell
                            }
                        )

                    # ============================================
                    # 2B:
                    #
                    # 1 -> N
                    #
                    # Split
                    #
                    # 旧 Cell 到这里结束。
                    #
                    # 每个新 branch 都创建新的 Cell。
                    # ============================================

                    else:

                        finished_previous.add(
                            previous_index
                        )

                        cell = self._new_cell()

                        cell.add_pixels(
                            self._interval_pixels(
                                row,
                                current_interval
                            )
                        )

                        new_active.append(
                            {
                                "interval": current_interval,
                                "cell": cell
                            }
                        )

                    continue

                # ------------------------------------------------
                # Case 3:
                #
                # 多个 predecessor
                #
                # N -> 1
                #
                # Merge
                # ------------------------------------------------

                # 所有 predecessor 对应的旧 Cell
                predecessor_items = [
                    active[index]
                    for index in predecessors
                ]

                # 这些旧 Cell 都结束
                for previous_index in predecessors:

                    finished_previous.add(
                        previous_index
                    )

                # 创建新的 Cell
                #
                # 不直接沿用某一个旧 Cell，
                # 避免 merge 后把多个拓扑区域继续当成
                # 一个旧 Cell。
                #
                merged_cell = self._new_cell()

                merged_cell.add_pixels(
                    self._interval_pixels(
                        row,
                        current_interval
                    )
                )

                new_active.append(
                    {
                        "interval": current_interval,
                        "cell": merged_cell
                    }
                )

            # ====================================================
            # 处理旧 Cell
            #
            # 没有 successor 的 Cell 直接结束
            # ====================================================

            for previous_index, previous_item in enumerate(
                active
            ):

                if previous_index in finished_previous:
                    final_cells.append(
                        previous_item["cell"]
                    )
                    continue

                if len(
                    previous_successors[
                        previous_index
                    ]
                ) == 0:

                    final_cells.append(
                        previous_item["cell"]
                    )

            # ====================================================
            # 更新 active
            # ====================================================

            active = new_active

            previous_intervals = current_intervals

        # ========================================================
        # 扫描结束
        # ========================================================

        for item in active:

            final_cells.append(
                item["cell"]
            )

        # ========================================================
        # 过滤过小 Cell
        # ========================================================

        final_cells = [
            cell
            for cell in final_cells
            if cell.size >= self.min_cell_size
        ]

        # ========================================================
        # 按照空间位置排序
        #
        # 这一步不是 BCD 算法本身，
        # 只是为了让 RViz / coverage planner
        # 中的 Cell ID 更稳定、更容易观察。
        # ========================================================

        final_cells.sort(
            key=lambda cell: (
                cell.min_row
                if cell.min_row is not None
                else 999999,

                cell.min_col
                if cell.min_col is not None
                else 999999
            )
        )

        # ========================================================
        # 重新编号
        #
        # 保证：
        #
        # Cell 0
        # Cell 1
        # Cell 2
        # ...
        #
        # 唯一且连续。
        # ========================================================

        for new_id, cell in enumerate(final_cells):

            cell.id = new_id

        return final_cells


# ================================================================
# 调试 / 测试
# ================================================================

def print_cells(cells):
    """
    打印 BCD Cell 信息。
    """

    print("")
    print("========== BCD Result ==========")
    print("Total cells:", len(cells))

    for cell in cells:

        print(
            "Cell {:3d}: size={:5d}, "
            "bounds=({}, {}, {}, {}), "
            "centroid=({:.2f}, {:.2f})".format(
                cell.id,
                cell.size,
                cell.min_col,
                cell.min_row,
                cell.max_col,
                cell.max_row,
                cell.centroid()[0],
                cell.centroid()[1]
            )
        )

    print("================================")
    print("")


def test_bcd():
    """
    BCD 测试。

    测试地图：

        0 = obstacle
        1 = free

    中间制造一个 split：

        ###########
        ##1111111##
        ##1111111##
        ##1111111##
        ##111###11##
        ##111###11##
        ##111###11##
        ##1111111##
        ##1111111##
        ###########

    拓扑：

        1
        |
        1
        |
        1
       / \
      2   3
       \ /
        4
        |
        4

    注意：

    BCD Cell 并不是简单的连通区域编号。
    Split / Merge 会产生新的 Cell。
    """

    test_map = np.zeros(
        (10, 11),
        dtype=bool
    )

    # 上方一个完整区域
    test_map[1:4, 2:9] = True

    # 中间发生 split
    test_map[4:7, 2:5] = True
    test_map[4:7, 7:9] = True

    # 下方重新 merge
    test_map[7:9, 2:9] = True

    print("")
    print("Test map:")
    print("")

    for row in range(
        test_map.shape[0]
    ):

        line = ""

        for col in range(
            test_map.shape[1]
        ):

            if test_map[row, col]:
                line += "█"
            else:
                line += " "

        print(line)

    # ------------------------------------------------------------
    # BCD
    # ------------------------------------------------------------

    bcd = BoustrophedonDecomposition(
        test_map,
        min_cell_size=1
    )

    cells = bcd.decompose()

    # ------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------

    print_cells(cells)

    # ------------------------------------------------------------
    # 输出每个 Cell 的像素
    # ------------------------------------------------------------

    for cell in cells:

        print(
            "Cell {} pixels:".format(
                cell.id
            )
        )

        rows = sorted(
            set(
                row
                for row, col
                in cell.pixels
            )
        )

        for row in rows:

            cols = sorted(
                col
                for r, col
                in cell.pixels
                if r == row
            )

            print(
                "  row {:2d}: {}".format(
                    row,
                    cols
                )
            )

        print("")


if __name__ == "__main__":
    test_bcd()