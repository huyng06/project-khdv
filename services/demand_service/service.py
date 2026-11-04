"""
Dịch vụ Phát sinh Nhu cầu Đi lại (Demand Service).
Quản lý:
- Quá trình Poisson phát sinh yêu cầu theo không-thời gian
- Lựa chọn điểm đón/trả ngẫu nhiên hoặc theo cụm nhu cầu thực tế
- Phát sinh sự kiện REQUEST_CREATED qua EventBus
"""
from __future__ import annotations

import random
from typing import List, Optional
import numpy as np

from core_engine.entities import Request
from services.base import ServiceResult, event_bus, EventType
from services.map_service.service import MapService
from services.demand_service.contracts import DemandConfigDTO, TripRequestDTO


class DemandService:
    def __init__(self, map_service: MapService, config: Optional[DemandConfigDTO] = None):
        self.map_service = map_service
        self.config = config or DemandConfigDTO()
        self.req_counter = 0
        self.next_arrival_time = 0.0
        self.reset(self.config.seed)

    def reset(self, seed: Optional[int] = 42):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.req_counter = 0
        self.next_arrival_time = np.random.exponential(1.0 / max(self.config.arrival_rate, 1e-5))

    def update_rate(self, new_rate: float):
        self.config.arrival_rate = max(0.001, new_rate)

    def generate_tick_requests(self, current_time: float) -> List[Request]:
        """Sinh tất cả yêu cầu có thời điểm xuất hiện <= current_time."""
        new_requests: List[Request] = []

        while current_time >= self.next_arrival_time:
            self.req_counter += 1
            origin = self.map_service.get_random_node()
            destination = self.map_service.get_random_node()
            while destination == origin:
                destination = self.map_service.get_random_node()

            req = Request(
                request_id=f"R_{self.req_counter:04d}",
                origin_node=origin,
                destination_node=destination,
                request_time=self.next_arrival_time,
                max_wait_time=self.config.max_wait_time,
                num_passengers=1,
            )
            new_requests.append(req)

            # Phát sự kiện qua EventBus
            event_bus.publish(EventType.REQUEST_CREATED, {
                "id": req.request_id,
                "origin": req.origin_node,
                "destination": req.destination_node,
                "time": req.request_time,
            })

            inter_arrival = np.random.exponential(1.0 / max(self.config.arrival_rate, 1e-5))
            self.next_arrival_time += inter_arrival

        return new_requests
