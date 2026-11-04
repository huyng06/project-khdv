"""Contracts và DTO cho Dịch vụ Bản đồ & Định tuyến (Map & Routing Service)."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class NodeDTO:
    id: int
    lat: float
    lon: float

@dataclass
class RouteRequestDTO:
    origin_node: int
    destination_node: int

@dataclass
class RouteResponseDTO:
    path_nodes: List[int]
    distance_meters: float
    geometry_coords: List[Tuple[float, float]] = field(default_factory=list)
    travel_time_seconds: Optional[float] = None
