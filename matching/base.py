"""Lớp cơ sở cho mọi chiến lược ghép cặp.

Các trường hợp biên dùng chung (không có tài xế, không có yêu cầu, id trùng,
tài xế bận) được xử lý một lần ở `match()`. Lớp con chỉ cần cài `_match()`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf, isfinite
from typing import Dict, Optional, Sequence, Tuple

from .models import (Assignment, CostFn, Driver, Location, MatchResult, Request,
                     Stop, StopType, euclidean, route_cost)


def memoize_cost_fn(cost_fn: CostFn) -> CostFn:
    """Bọc cost_fn để mỗi cặp (a, b) chỉ tính một lần (dùng trong phạm vi một lần match())."""
    cache: Dict[Tuple[Location, Location], float] = {}

    def cached(a: Location, b: Location) -> float:
        key = (a, b)
        if key not in cache:
            cache[key] = cost_fn(a, b)
        return cache[key]

    return cached


class Matcher(ABC):
    def __init__(self, max_pickup_cost: float = inf):
        """max_pickup_cost: chi phí tối đa từ tài xế đến điểm đón.
        Cặp nào vượt ngưỡng (hoặc cost = inf) không được ghép."""
        self.max_pickup_cost = max_pickup_cost

    # ------------------------------------------------------------ API công khai
    def match(self, requests: Sequence[Request], drivers: Sequence[Driver],
              cost_fn: CostFn = euclidean) -> MatchResult:
        self._check_unique_ids(requests, drivers)
        free = [d for d in drivers if d.available]

        if not requests or not free:
            return MatchResult(
                unmatched_requests=[r.id for r in requests],
                idle_drivers=[d.id for d in free],
            )

        cost_fn = memoize_cost_fn(cost_fn)        # mỗi cặp điểm chỉ tính một lần trong lần ghép này

        # Yêu cầu có chặng đón -> trả không đi được (ví dụ đường một chiều) thì không ai
        # phục vụ được: loại trước, để không giao khách đó cho tài xế rồi kẹt giữa đường.
        servable = [r for r in requests if isfinite(cost_fn(r.pickup, r.dropoff))]
        if not servable:
            return MatchResult(
                unmatched_requests=[r.id for r in requests],
                idle_drivers=[d.id for d in free],
            )

        result = self._match(servable, free, cost_fn)
        self._check_output(result, servable, free)
        served = set(result.matched_request_ids)
        result.unmatched_requests = [r.id for r in requests if r.id not in served]
        return result

    # ------------------------------------------------------------ lớp con cài đặt
    @abstractmethod
    def _match(self, requests: list[Request], drivers: list[Driver],
               cost_fn: CostFn) -> MatchResult:
        """Nhận danh sách KHÔNG rỗng, tài xế đều rảnh, mọi yêu cầu đều có chặng đón -> trả đi được.
        `cost_fn` đã được ghi nhớ. Trả về MatchResult."""

    # ------------------------------------------------------------ tiện ích cho lớp con
    def pickup_cost(self, driver: Driver, request: Request,
                    cost_fn: CostFn) -> Optional[float]:
        """Chi phí tài xế -> điểm đón, hoặc None nếu cặp không hợp lệ.

        Không hợp lệ khi: khách đông hơn sức chứa, chi phí không hữu hạn
        (inf/nan), hoặc vượt `max_pickup_cost`. Chỉ gọi cost_fn đúng một lần
        (cost_fn có thể đắt, ví dụ Dijkstra trên đồ thị đường).
        """
        if request.passengers > driver.capacity:
            return None
        cost = cost_fn(driver.location, request.pickup)
        if isfinite(cost) and cost <= self.max_pickup_cost:
            return cost
        return None

    def is_feasible(self, driver: Driver, request: Request, cost_fn: CostFn) -> bool:
        """Cặp (tài xế, yêu cầu) có hợp lệ không: đủ chỗ và trong ngưỡng đón."""
        return self.pickup_cost(driver, request, cost_fn) is not None

    @staticmethod
    def single_assignment(driver: Driver, request: Request, cost_fn: CostFn) -> Assignment:
        """Dựng Assignment 1 khách (dùng cho Greedy và Hungarian)."""
        stops = [Stop(request.id, StopType.PICKUP, request.pickup),
                 Stop(request.id, StopType.DROPOFF, request.dropoff)]
        return Assignment(driver.id, stops, route_cost(driver.location, stops, cost_fn))

    @staticmethod
    def build_result(assignments: list[Assignment], requests: Sequence[Request],
                     drivers: Sequence[Driver]) -> MatchResult:
        """Tự suy ra danh sách chưa ghép / tài xế rảnh từ các assignment."""
        used_req = {rid for a in assignments for rid in a.request_ids}
        used_drv = {a.driver_id for a in assignments}
        return MatchResult(
            assignments=assignments,
            unmatched_requests=[r.id for r in requests if r.id not in used_req],
            idle_drivers=[d.id for d in drivers if d.id not in used_drv],
        )

    # ------------------------------------------------------------ kiểm tra
    @staticmethod
    def _check_unique_ids(requests, drivers):
        for name, items in (("request", requests), ("driver", drivers)):
            ids = [x.id for x in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Có {name} id bị trùng")

    @staticmethod
    def _check_output(result: MatchResult, requests, free_drivers):
        """Bắt lỗi của lớp con: mỗi yêu cầu/tài xế xuất hiện đúng một lần."""
        matched = result.matched_request_ids
        if len(matched) != len(set(matched)):
            raise AssertionError("Một yêu cầu bị giao cho nhiều tài xế")
        if set(matched) | set(result.unmatched_requests) != {r.id for r in requests} \
                or set(matched) & set(result.unmatched_requests):
            raise AssertionError("Yêu cầu không khớp giữa assignments và unmatched")
        used = [a.driver_id for a in result.assignments]
        if len(used) != len(set(used)):
            raise AssertionError("Một tài xế xuất hiện trong nhiều assignment")
        if set(used) | set(result.idle_drivers) != {d.id for d in free_drivers} \
                or set(used) & set(result.idle_drivers):
            raise AssertionError("Tài xế không khớp giữa assignments và idle_drivers")
