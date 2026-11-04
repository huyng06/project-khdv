"""
Dịch vụ Điều phối & Ghép cặp (Matching & Dispatch Service).
Đóng gói 4 chiến lược ghép cặp cốt lõi:
  - S1: Greedy Nearest Matching
  - S2: Hungarian Batch Matching
  - S3: Greedy Ride-Pooling
  - S4: Hungarian Ride-Pooling
Giao tiếp thông qua DTO chuẩn hóa và EventBus.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

from matching import (
    GreedyMatcher,
    HungarianMatcher,
    PoolingMatcher,
    HungarianPoolingMatcher,
    Driver as AlgoDriver,
    Request as AlgoRequest,
)
from services.base import ServiceResult, event_bus, EventType
from services.map_service.service import MapService
from services.matching_service.contracts import (
    AssignmentDTO,
    DispatchResultDTO,
    DispatchStrategy,
)
from core_engine.entities import Driver, Request


class MatchingService:
    def __init__(self, map_service: MapService):
        self.map_service = map_service
        self._matchers = {
            DispatchStrategy.GREEDY: GreedyMatcher(max_pickup_cost=3000.0),
            DispatchStrategy.HUNGARIAN: HungarianMatcher(max_pickup_cost=3000.0),
            DispatchStrategy.POOLING: PoolingMatcher(max_pickup_cost=3000.0, max_detour_ratio=1.5),
            DispatchStrategy.HUNGARIAN_POOLING: HungarianPoolingMatcher(max_pickup_cost=3000.0, max_detour_ratio=1.5),
        }
        self.active_strategy = DispatchStrategy.GREEDY

    def set_strategy(self, strategy: DispatchStrategy):
        if strategy in self._matchers:
            self.active_strategy = strategy

    def get_cost_function(self) -> Callable[[int, int], float]:
        """Tạo hàm chi phí kết nối với MapService có lưu cache."""
        return self.map_service.get_distance

    def dispatch(
        self,
        idle_drivers: List[Driver],
        pending_requests: List[Request],
        strategy: Optional[DispatchStrategy] = None
    ) -> ServiceResult[DispatchResultDTO]:
        """
        Thực hiện một chu kỳ ghép cặp giữa xe rảnh và yêu cầu chờ.
        """
        strat = strategy or self.active_strategy
        matcher = self._matchers.get(strat, self._matchers[DispatchStrategy.GREEDY])

        if not idle_drivers or not pending_requests:
            return ServiceResult.ok(DispatchResultDTO(
                strategy=strat.value,
                assignments=[],
                matched_count=0,
                unmatched_count=len(pending_requests),
                execution_time_ms=0.0
            ))

        start_t = time.perf_counter()

        # 1. Chuyển đổi sang format Input của Matching Library
        algo_drivers = [
            AlgoDriver(
                id=d.driver_id,
                location=d.current_node,
                capacity=d.capacity,
                available=True
            )
            for d in idle_drivers
        ]

        algo_requests = [
            AlgoRequest(
                id=r.request_id,
                pickup=r.origin_node,
                dropoff=r.destination_node,
                time=r.request_time,
                passengers=r.num_passengers
            )
            for r in pending_requests
        ]

        # 2. Gọi thuật toán ghép cặp
        cost_fn = self.get_cost_function()
        match_result = matcher.match(algo_requests, algo_drivers, cost_fn)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # 3. Chuyển đổi kết quả sang AssignmentDTO chuẩn
        assignments_dto: List[AssignmentDTO] = []
        for asgn in match_result.assignments:
            stops = [(s.request_id, s.type.value) for s in asgn.stops]
            assignments_dto.append(AssignmentDTO(
                driver_id=asgn.driver_id,
                request_ids=asgn.request_ids,
                stops=stops,
                cost=asgn.cost
            ))

        result_dto = DispatchResultDTO(
            strategy=strat.value,
            assignments=assignments_dto,
            matched_count=len(match_result.matched_request_ids),
            unmatched_count=len(match_result.unmatched_requests),
            execution_time_ms=round(elapsed_ms, 2)
        )

        # Phát sự kiện điều phối
        event_bus.publish(EventType.DISPATCH_BATCH_EXECUTED, {
            "strategy": strat.value,
            "matched": len(match_result.matched_request_ids),
            "unmatched": len(match_result.unmatched_requests),
            "time_ms": elapsed_ms
        })

        return ServiceResult.ok(result_dto)
