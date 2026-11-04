import numpy as np
import random
from typing import List, Optional
from core_engine.entities import Request
from core_engine.map_network import RealRoadMap

class DemandGenerator:
    def __init__(self, road_map: RealRoadMap, arrival_rate: float = 0.03, seed: Optional[int] = 42):
        """
        arrival_rate: Cường độ Poisson (lambda - số yêu cầu trung bình xuất hiện trong 1 giây).
        """
        self.road_map = road_map
        self.arrival_rate = arrival_rate
        self.req_counter = 0

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Tính thời điểm yêu cầu đầu tiên xuất hiện
        self.next_arrival_time = np.random.exponential(1.0 / self.arrival_rate)

    def tick(self, current_time: float) -> List[Request]:
        """
        Gọi ở mỗi bước mô phỏng. Trả về danh sách yêu cầu mới vừa xuất hiện tại current_time.
        """
        new_requests = []

        # Sinh tất cả yêu cầu có thời điểm xuất hiện <= current_time
        while current_time >= self.next_arrival_time:
            self.req_counter += 1
            origin_node = self.road_map.get_random_node()
            destination_node = self.road_map.get_random_node()

            while destination_node == origin_node:
                destination_node = self.road_map.get_random_node()

            req = Request(
                request_id=f"R_{self.req_counter:04d}",
                origin_node=origin_node,
                destination_node=destination_node,
                request_time=self.next_arrival_time,
                max_wait_time=300.0  # 5 phút kiên nhẫn
            )
            new_requests.append(req)

            # Tính mốc thời gian cho yêu cầu kế tiếp
            inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
            self.next_arrival_time += inter_arrival_time

        return new_requests