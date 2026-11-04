from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

class RequestState(Enum):
    WAITING = "WAITING"         # Đang chờ ghép chuyến
    MATCHED = "MATCHED"         # Đã ghép tài xế, tài xế đang tới đón
    PICKED_UP = "PICKED_UP"     # Khách đã lên xe
    CANCELLED = "CANCELLED"     # Khách hủy do chờ quá lâu
    COMPLETED = "COMPLETED"     # Chuyến đi hoàn thành

class DriverState(Enum):
    IDLE = "IDLE"               # Rảnh rỗi
    EN_ROUTE_PICKUP = "EN_ROUTE_PICKUP" # Đang đi đón khách
    OCCUPIED = "OCCUPIED"       # Đang chở khách

@dataclass
class Request:
    request_id: str
    origin_node: int        # Node ID điểm đón
    destination_node: int   # Node ID điểm trả
    request_time: float
    max_wait_time: float = 600.0  # Thời gian chờ tối đa trước khi hủy (giây)
    num_passengers: int = 1
    
    # Trạng thái vận hành
    status: RequestState = RequestState.WAITING
    matched_driver_id: Optional[str] = None
    pickup_time: Optional[float] = None
    completion_time: Optional[float] = None

@dataclass
class Driver:
    driver_id: str
    current_node: int
    speed: float = 10.0  # Vận tốc m/s (~36 km/h)
    capacity: int = 4
    
    # Trạng thái vận hành
    status: DriverState = DriverState.IDLE
    assigned_requests: List[Request] = field(default_factory=list)
    current_target: Optional[Tuple[float, float]] = None

    # Thuộc tính simulator sử dụng (khai báo để không bị AttributeError khi tài xế chưa nhận khách)
    assigned_request: Optional[Request] = None
    current_route: List[int] = field(default_factory=list)
    stop_queue: List[tuple] = field(default_factory=list)  # [(Request, "pickup"|"dropoff", node)] khi ghép chuyến chung

    # Đo lường hiệu suất
    total_busy_time: float = 0.0          # Tổng thời gian bận (đón khách hoặc chở khách), giây
    total_passenger_time: float = 0.0     # Thời gian đang có khách trên xe, giây
    total_distance: float = 0.0           # Tổng quãng đường di chuyển (mét)
    total_passenger_distance: float = 0.0 # Quãng đường chở khách (mét)
