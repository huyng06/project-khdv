"""Chiến lược 3: Ride-Pooling Matching (ghép chuyến chung).

Ý tưởng: một tài xế có thể chở NHIỀU khách cùng lúc. Mỗi tài xế có một lộ trình
(danh sách điểm đón/trả có thứ tự). Với mỗi yêu cầu mới, thử CHÈN điểm đón và
điểm trả vào mọi vị trí hợp lệ trong lộ trình của từng tài xế (không đổi thứ tự
các điểm đã có), rồi chọn phương án làm TĂNG CHI PHÍ ÍT NHẤT (insertion heuristic).
Tài xế chưa có khách có lộ trình rỗng, nên "mở chuyến mới" cũng là một lựa chọn.

Ràng buộc của một lộ trình hợp lệ (kiểm tra bởi `check_route`):
1. Mỗi khách được đón trước khi trả, đúng một lần.
2. Số khách trên xe không bao giờ vượt `Driver.capacity`.
3. Chi phí đi từ vị trí tài xế đến điểm đón của mỗi khách (theo lộ trình thực tế)
   không vượt `max_pickup_cost`.
4. Mỗi khách không bị đi vòng quá `max_detour_ratio` lần quãng đi thẳng
   (chi phí từ lúc lên xe đến lúc xuống xe <= ratio * chi phí đi thẳng).

Quy tắc xác định (kết quả không phụ thuộc thứ tự phần tử đầu vào):
- Yêu cầu được xử lý theo (thời điểm đặt, id).
- Hai phương án bằng chi phí tăng thêm thì chọn tài xế id nhỏ hơn, rồi vị trí chèn sớm hơn
  (khoá so sánh là bộ (chi phí tăng thêm, id tài xế, vị trí đón, vị trí trả), luôn phân định được).

Hạn chế đã biết (đây là heuristic, không đảm bảo tối ưu toàn cục):
- Xử lý lần lượt từng yêu cầu nên kết quả phụ thuộc thứ tự xử lý.
- Chỉ chèn thêm điểm dừng, không sắp xếp lại các điểm đã có.
- Mọi tài xế bắt đầu từ vị trí hiện tại và đang trống (tài xế `available=False` bị loại),
  nên chưa chèn thêm khách vào xe đang chạy dở.

Độ phức tạp: mỗi cặp (yêu cầu, tài xế) thử O(L^2) vị trí chèn, mỗi vị trí kiểm tra
O(L) với L là số điểm dừng của lộ trình (nhỏ vì bị giới hạn bởi sức chứa).
`cost_fn` được ghi nhớ trong `Matcher.match()`, mỗi cặp điểm chỉ tính một lần.
"""
from __future__ import annotations

from math import inf, isfinite
from typing import Dict, List, Mapping, Optional, Sequence

from .base import Matcher
from .models import (Assignment, CostFn, Driver, Location, MatchResult, Request,
                     Stop, StopType)

_EPS = 1e-9   # sai số số học khi so sánh với ngưỡng đi vòng


def check_route(driver: Driver, stops: Sequence[Stop], requests: Mapping[str, Request],
                cost_fn: CostFn, max_pickup_cost: float = inf,
                max_detour_ratio: float = inf) -> Optional[float]:
    """Kiểm tra lộ trình `stops` của `driver`; trả về tổng chi phí hoặc None nếu vi phạm.

    Tổng chi phí tính từ vị trí hiện tại của tài xế qua lần lượt các điểm dừng
    (cùng định nghĩa với `route_cost` và `Assignment.cost`). Lộ trình rỗng có chi phí 0.
    """
    cur: Location = driver.location
    total = 0.0
    load = 0
    ride: Dict[str, float] = {}      # khách đang trên xe -> chi phí đã đi kể từ lúc đón
    finished = set()

    for stop in stops:
        leg = cost_fn(cur, stop.location)
        if not isfinite(leg):
            return None
        total += leg
        for rid in ride:
            ride[rid] += leg
        cur = stop.location

        request = requests[stop.request_id]
        if stop.type == StopType.PICKUP:
            if stop.request_id in ride or stop.request_id in finished:
                return None                                  # đón hai lần
            if total > max_pickup_cost:
                return None
            load += request.passengers
            if load > driver.capacity:
                return None
            ride[stop.request_id] = 0.0
        else:
            if stop.request_id not in ride:
                return None                                  # trả trước khi đón, hoặc trả hai lần
            if max_detour_ratio != inf:
                direct = cost_fn(request.pickup, request.dropoff)
                if not isfinite(direct) or ride[stop.request_id] > max_detour_ratio * direct + _EPS:
                    return None
            load -= request.passengers
            del ride[stop.request_id]
            finished.add(stop.request_id)

    return None if ride else total                           # còn khách chưa được trả


class PoolingMatcher(Matcher):
    """Ghép chuyến chung bằng insertion heuristic.

    Tham số:
        max_pickup_cost: chi phí tối đa từ tài xế đến điểm đón của MỖI khách, tính theo
            lộ trình thực tế (khách lên sau phải chờ lâu hơn). Với lộ trình một khách,
            ý nghĩa giống hệt Greedy và Hungarian.
        max_detour_ratio: mỗi khách đi trên xe tối đa bấy nhiêu lần quãng đi thẳng
            (>= 1; mặc định 1.5; `inf` = không giới hạn).
        max_requests_per_driver: số yêu cầu tối đa mỗi tài xế nhận trong một lần ghép
            (None = chỉ bị giới hạn bởi sức chứa; 1 = không ghép chung).
    """

    def __init__(self, max_pickup_cost: float = inf, max_detour_ratio: float = 1.5,
                 max_requests_per_driver: Optional[int] = None):
        super().__init__(max_pickup_cost)
        if max_detour_ratio < 1:
            raise ValueError("max_detour_ratio phải >= 1 (đi thẳng đã bằng 1)")
        if max_requests_per_driver is not None and max_requests_per_driver < 1:
            raise ValueError("max_requests_per_driver phải >= 1 hoặc None")
        self.max_detour_ratio = max_detour_ratio
        self.max_requests_per_driver = max_requests_per_driver

    def _match(self, requests: list[Request], drivers: list[Driver],
               cost_fn: CostFn) -> MatchResult:
        cost = cost_fn                                       # đã được ghi nhớ trong Matcher.match()
        by_id = {r.id: r for r in requests}
        routes: Dict[str, List[Stop]] = {d.id: [] for d in drivers}
        route_costs: Dict[str, float] = {d.id: 0.0 for d in drivers}
        request_counts: Dict[str, int] = {d.id: 0 for d in drivers}
        used: List[str] = []                                 # tài xế theo thứ tự nhận khách đầu tiên
        limit = self.max_requests_per_driver

        for request in sorted(requests, key=lambda r: (r.time, r.id)):
            pickup = Stop(request.id, StopType.PICKUP, request.pickup)
            dropoff = Stop(request.id, StopType.DROPOFF, request.dropoff)
            best = None                                      # (khoá so sánh, tài xế, lộ trình mới, chi phí mới)

            for driver in drivers:
                if request.passengers > driver.capacity:
                    continue
                if limit is not None and request_counts[driver.id] >= limit:
                    continue
                route = routes[driver.id]
                for i in range(len(route) + 1):              # vị trí chèn điểm đón
                    with_pickup = route[:i] + [pickup] + route[i:]
                    for j in range(i + 1, len(with_pickup) + 1):   # điểm trả luôn sau điểm đón
                        candidate = with_pickup[:j] + [dropoff] + with_pickup[j:]
                        new_cost = check_route(driver, candidate, by_id, cost,
                                               self.max_pickup_cost, self.max_detour_ratio)
                        if new_cost is None:
                            continue
                        key = (new_cost - route_costs[driver.id], driver.id, i, j)
                        if best is None or key < best[0]:
                            best = (key, driver, candidate, new_cost)

            if best is not None:
                _, driver, candidate, new_cost = best
                if not routes[driver.id]:
                    used.append(driver.id)
                routes[driver.id] = candidate
                route_costs[driver.id] = new_cost
                request_counts[driver.id] += 1

        assignments = [Assignment(did, routes[did], route_costs[did]) for did in used]
        return self.build_result(assignments, requests, drivers)
