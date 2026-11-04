"""Chiến lược 4: Hungarian Ride-Pooling Matching (Hungary gom lô + Ghép chuyến chung).

Kết hợp thuật toán Hungarian tối ưu hóa toàn cục phép ghép ban đầu,
sau đó áp dụng cơ chế insertion heuristic để ghép thêm yêu cầu vào xe đang bận
hoặc các xe đã được giao lộ trình trong cùng đợt gom lô.

Quy trình:
1. Bước 1: Sử dụng Hungarian Algorithm (Kuhn-Munkres) tìm phép ghép 1-1 tối ưu toàn cục
   giữa tập yêu cầu và tập tài xế rảnh, tối thiểu hóa tổng chi phí đón khách ban đầu.
2. Bước 2: Với các yêu cầu chưa được ghép (khi số yêu cầu > số xe rảnh), thực hiện
   chèn tối ưu (insertion heuristic) vào lộ trình của các tài xế đã được ghép ở Bước 1.
   Mỗi điểm chèn phải thỏa mãn:
     - Sức chứa xe (capacity)
     - Giới hạn đi vòng tối đa (max_detour_ratio)
     - Chi phí đón tối đa (max_pickup_cost)
     - Thứ tự đón trước trả sau
3. Bước 3: Thu thập kết quả và trả về MatchResult chuẩn.
"""
from __future__ import annotations

from math import inf
from typing import Dict, List, Optional

from .base import Matcher
from .hungarian import HungarianMatcher
from .models import (Assignment, CostFn, Driver, MatchResult, Request,
                     Stop, StopType)
from .pooling import check_route


class HungarianPoolingMatcher(Matcher):
    """Ghép theo lô bằng Hungarian kết hợp ghép chuyến chung (Ride-Pooling)."""

    def __init__(self, max_pickup_cost: float = inf, max_detour_ratio: float = 1.5,
                 max_requests_per_driver: Optional[int] = None):
        super().__init__(max_pickup_cost)
        if max_detour_ratio < 1:
            raise ValueError("max_detour_ratio phải >= 1 (đi thẳng đã bằng 1)")
        if max_requests_per_driver is not None and max_requests_per_driver < 1:
            raise ValueError("max_requests_per_driver phải >= 1 hoặc None")
        self.max_detour_ratio = max_detour_ratio
        self.max_requests_per_driver = max_requests_per_driver
        self._hungarian = HungarianMatcher(max_pickup_cost=max_pickup_cost)

    def _match(self, requests: list[Request], drivers: list[Driver],
               cost_fn: CostFn) -> MatchResult:
        # Bước 1: Chạy Hungarian để ghép 1-1 tối ưu toàn cục
        base_result = self._hungarian._match(requests, drivers, cost_fn)

        by_id: Dict[str, Request] = {r.id: r for r in requests}
        driver_by_id: Dict[str, Driver] = {d.id: d for d in drivers}

        # Lưu trữ lộ trình của các tài xế
        routes: Dict[str, List[Stop]] = {d.id: [] for d in drivers}
        route_costs: Dict[str, float] = {d.id: 0.0 for d in drivers}
        request_counts: Dict[str, int] = {d.id: 0 for d in drivers}
        used_drivers: List[str] = []

        for asgn in base_result.assignments:
            routes[asgn.driver_id] = list(asgn.stops)
            route_costs[asgn.driver_id] = asgn.cost
            request_counts[asgn.driver_id] = len(asgn.request_ids)
            used_drivers.append(asgn.driver_id)

        # Bước 2: Các yêu cầu chưa ghép được -> cố gắng pool vào các tài xế đã có lộ trình
        unmatched_ids = set(base_result.unmatched_requests)
        unmatched_reqs = [r for r in requests if r.id in unmatched_ids]

        limit = self.max_requests_per_driver

        for request in sorted(unmatched_reqs, key=lambda r: (r.time, r.id)):
            pickup = Stop(request.id, StopType.PICKUP, request.pickup)
            dropoff = Stop(request.id, StopType.DROPOFF, request.dropoff)
            best = None

            # Thử chèn vào xe đã được giao ở Bước 1
            for driver_id in used_drivers:
                driver = driver_by_id[driver_id]
                if request.passengers > driver.capacity:
                    continue
                if limit is not None and request_counts[driver.id] >= limit:
                    continue

                route = routes[driver.id]
                for i in range(len(route) + 1):
                    with_pickup = route[:i] + [pickup] + route[i:]
                    for j in range(i + 1, len(with_pickup) + 1):
                        candidate = with_pickup[:j] + [dropoff] + with_pickup[j:]
                        new_cost = check_route(driver, candidate, by_id, cost_fn,
                                               self.max_pickup_cost, self.max_detour_ratio)
                        if new_cost is None:
                            continue
                        key = (new_cost - route_costs[driver.id], driver.id, i, j)
                        if best is None or key < best[0]:
                            best = (key, driver, candidate, new_cost)

            # Nếu còn tài xế chưa dùng, cũng có thể mở chuyến mới cho yêu cầu này
            unused_drivers = [d for d in drivers if d.id not in used_drivers]
            for driver in unused_drivers:
                if request.passengers > driver.capacity:
                    continue
                candidate = [pickup, dropoff]
                new_cost = check_route(driver, candidate, by_id, cost_fn,
                                       self.max_pickup_cost, self.max_detour_ratio)
                if new_cost is None:
                    continue
                key = (new_cost, driver.id, 0, 1)
                if best is None or key < best[0]:
                    best = (key, driver, candidate, new_cost)

            if best is not None:
                _, driver, candidate, new_cost = best
                if not routes[driver.id]:
                    used_drivers.append(driver.id)
                routes[driver.id] = candidate
                route_costs[driver.id] = new_cost
                request_counts[driver.id] += 1

        assignments = [Assignment(did, routes[did], route_costs[did]) for did in used_drivers]
        return self.build_result(assignments, requests, drivers)
