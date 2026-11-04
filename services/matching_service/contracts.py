"""Contracts cho Dịch vụ Điều phối & Ghép cặp (Matching & Dispatch Service)."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Any, Optional

class DispatchStrategy(str, Enum):
    GREEDY = "greedy"
    HUNGARIAN = "hungarian"
    POOLING = "pooling"
    HUNGARIAN_POOLING = "hungarian_pooling"

@dataclass
class AssignmentDTO:
    driver_id: str
    request_ids: List[str]
    stops: List[Tuple[str, str]] # [(req_id, "pickup"|"dropoff")]
    cost: float

@dataclass
class DispatchResultDTO:
    strategy: str
    assignments: List[AssignmentDTO]
    matched_count: int
    unmatched_count: int
    execution_time_ms: float
