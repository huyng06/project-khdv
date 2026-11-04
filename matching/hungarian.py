"""Chiến lược 2: Batch Matching với thuật toán Hungarian (Kuhn-Munkres).

Ý tưởng: thay vì xử lý từng yêu cầu một như Greedy, ta xét CẢ LÔ yêu cầu cùng
lúc, dựng ma trận chi phí (hàng = yêu cầu, cột = tài xế) và giải bài toán gán
tối ưu toàn cục. Nhờ đó tránh được lỗi của Greedy: một khách "cướp" tài xế gần
nhất rồi đẩy khách khác sang tài xế rất xa.

Mục tiêu tối ưu (theo thứ tự ưu tiên):
1. Phục vụ NHIỀU khách nhất có thể (chỉ ghép các cặp hợp lệ).
2. Trong các phương án đó, tổng chi phí tài xế -> điểm đón NHỎ nhất.

Lưu ý khi số khách > số xe: các yêu cầu được chọn là những yêu cầu cho tổng
chi phí đón nhỏ nhất, KHÔNG xét ai đặt trước (khác Greedy).

Cửa sổ gom lô (ví dụ mỗi 5 giây) do vòng lặp mô phỏng quyết định; `match()`
chỉ xử lý nhóm yêu cầu được đưa vào.

Độ phức tạp: O(R*D) lần gọi cost_fn để dựng ma trận, thêm O(k^2 * m) để giải
(k = min(R, D), m = max(R, D)). Cài đặt thuần Python, không cần scipy.
"""
from __future__ import annotations

from math import isfinite
from typing import List, Sequence

from .base import Matcher
from .models import CostFn, Driver, MatchResult, Request

_INF = float("inf")


def solve_assignment(cost: Sequence[Sequence[float]]) -> List[int]:
    """Bài toán gán tối ưu (Hungarian) trên ma trận chi phí hình chữ nhật.

    Trả về danh sách `cols` độ dài = số hàng: `cols[i]` là cột được gán cho hàng i,
    hoặc -1 nếu hàng i không được gán (xảy ra khi số hàng > số cột). Số cặp được
    gán luôn bằng min(số hàng, số cột) và tổng chi phí nhỏ nhất.

    Mọi phần tử phải là số hữu hạn (cho phép số âm). Độ phức tạp O(k^2 * m).
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    for row in cost:
        if len(row) != m:
            raise ValueError("Ma trận chi phí phải là hình chữ nhật")
        if not all(isfinite(x) for x in row):
            raise ValueError("Ma trận chi phí chỉ được chứa số hữu hạn")
    if n == 0 or m == 0:
        return [-1] * n

    if n > m:  # thuật toán cần số hàng <= số cột: chuyển vị rồi đổi chiều kết quả
        transposed = [[cost[i][j] for i in range(n)] for j in range(m)]
        col_to_row = solve_assignment(transposed)       # độ dài m, mỗi phần tử là cột (= hàng gốc)
        rows = [-1] * n
        for orig_col, orig_row in enumerate(col_to_row):
            rows[orig_row] = orig_col
        return rows

    # n <= m: thuật toán thế năng (potentials) với chỉ số bắt đầu từ 1.
    u = [0.0] * (n + 1)          # thế năng của hàng
    v = [0.0] * (m + 1)          # thế năng của cột
    match_of_col = [0] * (m + 1)  # match_of_col[j] = hàng đang gán cho cột j (0 = trống)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        match_of_col[0] = i
        j0 = 0
        min_to = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:                                   # tìm đường tăng
            used[j0] = True
            i0, delta, j1 = match_of_col[j0], _INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < min_to[j]:
                        min_to[j], way[j] = cur, j0
                    if min_to[j] < delta:
                        delta, j1 = min_to[j], j
            for j in range(m + 1):
                if used[j]:
                    u[match_of_col[j]] += delta
                    v[j] -= delta
                else:
                    min_to[j] -= delta
            j0 = j1
            if match_of_col[j0] == 0:
                break
        while j0:                                     # lật đường tăng
            j1 = way[j0]
            match_of_col[j0] = match_of_col[j1]
            j0 = j1

    rows = [-1] * n
    for j in range(1, m + 1):
        if match_of_col[j]:
            rows[match_of_col[j] - 1] = j - 1
    return rows


class HungarianMatcher(Matcher):
    """Ghép theo lô bằng Hungarian. Tham số `max_pickup_cost` kế thừa từ Matcher."""

    def _match(self, requests: list[Request], drivers: list[Driver],
               cost_fn: CostFn) -> MatchResult:
        # Sắp xếp cố định để kết quả không phụ thuộc thứ tự danh sách đầu vào.
        reqs = sorted(requests, key=lambda r: (r.time, r.id))
        drvs = sorted(drivers, key=lambda d: d.id)

        # None = cặp không hợp lệ (quá xa, không đi được, không đủ chỗ).
        costs = [[self.pickup_cost(d, r, cost_fn) for d in drvs] for r in reqs]
        feasible = [c for row in costs for c in row if c is not None]
        if not feasible:
            return self.build_result([], requests, drivers)

        # Cặp không hợp lệ bị phạt `big` đủ lớn để thuật toán luôn ưu tiên
        # phục vụ nhiều khách hơn, rồi mới đến chi phí thấp hơn.
        k = min(len(reqs), len(drvs))
        big = k * max(feasible) + 1.0
        matrix = [[big if c is None else c for c in row] for row in costs]

        assignments = []
        for i, j in enumerate(solve_assignment(matrix)):
            if j >= 0 and costs[i][j] is not None:     # bỏ các cặp bị phạt
                assignments.append(self.single_assignment(drvs[j], reqs[i], cost_fn))
        return self.build_result(assignments, requests, drivers)
