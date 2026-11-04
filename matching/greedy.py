"""Chiến lược 1: Greedy Nearest Matching (ghép tham lam).

Ý tưởng: xử lý yêu cầu theo thứ tự đặt xe (ai đặt trước được phục vụ trước),
mỗi yêu cầu nhận tài xế rảnh GẦN NHẤT (chi phí tài xế -> điểm đón nhỏ nhất)
trong số các tài xế hợp lệ, rồi tài xế đó bị loại khỏi danh sách.

Quy tắc xác định (kết quả không phụ thuộc thứ tự phần tử trong danh sách đầu vào):
- Yêu cầu được xử lý theo (thời điểm đặt, id).
- Hai tài xế cùng chi phí thì chọn tài xế có id nhỏ hơn.

Độ phức tạp: O(R * D) lần gọi cost_fn (R = số yêu cầu, D = số tài xế).

Hạn chế đã biết: chọn tối ưu cục bộ từng yêu cầu nên tổng chi phí có thể tệ hơn
phương án tối ưu toàn cục (xem HungarianMatcher). Đây là baseline để so sánh.
"""
from __future__ import annotations

from math import inf
from typing import Optional

from .base import Matcher
from .models import CostFn, Driver, MatchResult, Request


class GreedyMatcher(Matcher):
    """Ghép tham lam theo tài xế gần nhất. Tham số `max_pickup_cost` kế thừa từ Matcher."""

    def _match(self, requests: list[Request], drivers: list[Driver],
               cost_fn: CostFn) -> MatchResult:
        pool = list(drivers)          # tài xế còn rảnh; không sửa danh sách đầu vào
        assignments = []

        for request in sorted(requests, key=lambda r: (r.time, r.id)):
            best: Optional[Driver] = None
            best_cost = inf
            for driver in pool:
                cost = self.pickup_cost(driver, request, cost_fn)
                if cost is None:
                    continue
                if best is None or (cost, driver.id) < (best_cost, best.id):
                    best, best_cost = driver, cost

            if best is not None:
                assignments.append(self.single_assignment(best, request, cost_fn))
                pool.remove(best)

        return self.build_result(assignments, requests, drivers)
