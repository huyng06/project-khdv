"""Các kiểu dữ liệu chuẩn dùng chung cho mọi thuật toán ghép cặp.

Quy ước thống nhất cho cả nhóm:
- Vị trí (`Location`) là toạ độ tuple (x, y) / (lat, lon), HOẶC mã nút bản đồ (int/str),
  ví dụ node id của OpenStreetMap. Thuật toán không tự diễn giải vị trí: chỉ truyền
  nguyên xi cho `cost_fn`, nên vị trí phải hashable. Dùng toạ độ thì dùng euclidean/
  haversine_km; dùng node id thì phải tự cung cấp `cost_fn` (ví dụ Dijkstra).
- Mọi id (yêu cầu, tài xế) là chuỗi (str).
- Hàm chi phí `cost_fn(a, b)` trả về chi phí đi từ a đến b (km hoặc phút).
  Có thể bất đối xứng (cost(a, b) != cost(b, a)); trả về math.inf nếu không đi được.
  Phải "thuần": cùng đầu vào luôn cho cùng kết quả trong một lần gọi match().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import asin, cos, hypot, inf, radians, sin, sqrt
from typing import Callable, List, Sequence, Tuple, Union

Location = Union[Tuple[float, float], int, str]
Point = Location          # tên cũ, giữ lại để code cũ không bị lỗi import
CostFn = Callable[[Location, Location], float]


# ---------------------------------------------------------------- hàm chi phí mẫu
def euclidean(a: Location, b: Location) -> float:
    """Khoảng cách đường chim bay trên mặt phẳng. Dùng để test và mô phỏng nhanh.

    Với vị trí không phải toạ độ (node id): 0 nếu trùng nhau, ngược lại inf (không biết khoảng cách).
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        return hypot(a[0] - b[0], a[1] - b[1])
    return 0.0 if a == b else inf


def haversine_km(a: Location, b: Location) -> float:
    """Khoảng cách trên mặt cầu (km). a, b là (lat, lon) tính bằng độ.

    Với vị trí không phải toạ độ (node id): 0 nếu trùng nhau, ngược lại inf.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
        h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
        return 2 * 6371.0088 * asin(sqrt(h))
    return 0.0 if a == b else inf


# ---------------------------------------------------------------- đầu vào
@dataclass(frozen=True)
class Request:
    """Một yêu cầu đặt xe."""
    id: str
    pickup: Location
    dropoff: Location
    time: float = 0.0          # thời điểm đặt (giây hoặc phút, nhóm thống nhất)
    passengers: int = 1

    def __post_init__(self):
        if self.passengers < 1:
            raise ValueError(f"Request {self.id}: passengers phải >= 1")


@dataclass(frozen=True)
class Driver:
    """Một tài xế."""
    id: str
    location: Location
    capacity: int = 4          # số khách tối đa trên xe
    available: bool = True     # False = đang bận, không được xét ghép

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError(f"Driver {self.id}: capacity phải >= 1")


# ---------------------------------------------------------------- đầu ra
class StopType(str, Enum):
    PICKUP = "pickup"
    DROPOFF = "dropoff"


@dataclass(frozen=True)
class Stop:
    """Một điểm dừng trong lộ trình của tài xế."""
    request_id: str
    type: StopType
    location: Location


def route_cost(start: Location, stops: Sequence[Stop], cost_fn: CostFn) -> float:
    """Tổng chi phí đi từ `start` qua lần lượt các điểm dừng."""
    total, cur = 0.0, start
    for s in stops:
        total += cost_fn(cur, s.location)
        cur = s.location
    return total


@dataclass
class Assignment:
    """Lộ trình giao cho một tài xế.

    Greedy/Hungarian: stops = [PICKUP, DROPOFF] của đúng 1 yêu cầu.
    Pooling: stops là chuỗi có thứ tự gồm nhiều điểm đón/trả đan xen.
    `cost` = tổng chi phí lộ trình tính từ vị trí hiện tại của tài xế.
    """
    driver_id: str
    stops: List[Stop]
    cost: float

    @property
    def request_ids(self) -> List[str]:
        """Các yêu cầu được phục vụ, theo thứ tự đón."""
        return [s.request_id for s in self.stops if s.type == StopType.PICKUP]


@dataclass
class MatchResult:
    """Kết quả chuẩn của mọi thuật toán."""
    assignments: List[Assignment] = field(default_factory=list)
    unmatched_requests: List[str] = field(default_factory=list)  # id yêu cầu chưa ghép được
    idle_drivers: List[str] = field(default_factory=list)        # id tài xế rảnh nhưng không được giao

    @property
    def total_cost(self) -> float:
        return sum(a.cost for a in self.assignments)

    @property
    def matched_request_ids(self) -> List[str]:
        return [rid for a in self.assignments for rid in a.request_ids]
