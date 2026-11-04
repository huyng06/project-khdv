"""
Lớp nền tảng cho Kiến trúc Hướng Dịch vụ (Service-Oriented Architecture - SOA).
Bao gồm:
- BaseService: Lớp dịch vụ cơ sở
- ServiceResult: Kết quả thực thi chuẩn mực của dịch vụ (Data, Success, Error)
- EventBus: Trục thông điệp nội bộ (In-memory Pub/Sub Event Bus) cho giao tiếp lỏng lẻo giữa các Service
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

T = TypeVar("T")

@dataclass
class ServiceResult(Generic[T]):
    """Chuẩn hóa dữ liệu trả về giữa các service."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, **metadata) -> ServiceResult[T]:
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> ServiceResult[T]:
        return cls(success=False, error=error, metadata=metadata)


class EventType(str, Enum):
    REQUEST_CREATED = "REQUEST_CREATED"
    REQUEST_MATCHED = "REQUEST_MATCHED"
    REQUEST_PICKED_UP = "REQUEST_PICKED_UP"
    REQUEST_COMPLETED = "REQUEST_COMPLETED"
    REQUEST_CANCELLED = "REQUEST_CANCELLED"
    DRIVER_STATUS_CHANGED = "DRIVER_STATUS_CHANGED"
    DISPATCH_BATCH_EXECUTED = "DISPATCH_BATCH_EXECUTED"
    TICK_COMPLETED = "TICK_COMPLETED"


class EventBus:
    """Event Bus hỗ trợ giao tiếp bất đồng bộ và đồng bộ giữa các dịch vụ."""
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Any], Any]]] = {
            et: [] for et in EventType
        }

    def subscribe(self, event_type: EventType, handler: Callable[[Any], Any]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: EventType, payload: Any):
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(payload)
                except Exception as e:
                    print(f"[EventBus] Lỗi khi xử lý sự kiện {event_type}: {e}")


# Singleton Event Bus toàn hệ thống
event_bus = EventBus()
