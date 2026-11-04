"""Unit test cho HungarianPoolingMatcher."""
import pytest
from math import inf

from matching import (Driver, HungarianPoolingMatcher, Request, StopType,
                      euclidean)

P, D = StopType.PICKUP, StopType.DROPOFF


def req(i, pickup=(0, 0), dropoff=(10, 0), **kw):
    return Request(f"r{i}", pickup, dropoff, **kw)


def drv(i, loc=(0, 0), **kw):
    return Driver(f"d{i}", loc, **kw)


def test_hungarian_pooling_co_ban():
    matcher = HungarianPoolingMatcher()
    drivers = [drv(1, (0, 0))]
    requests = [req(1, (1, 0), (5, 0)), req(2, (2, 0), (6, 0))]

    result = matcher.match(requests, drivers, euclidean)
    assert len(result.assignments) == 1
    asgn = result.assignments[0]
    assert asgn.driver_id == "d1"
    # Tài xế d1 phục vụ cả 2 khách (ghép chuyến chung)
    assert set(asgn.request_ids) == {"r1", "r2"}
    assert len(result.unmatched_requests) == 0


def test_hungarian_pooling_gioi_han_so_khach_moi_xe():
    # Giới hạn mỗi tài xế chỉ nhận tối đa 1 yêu cầu
    matcher = HungarianPoolingMatcher(max_requests_per_driver=1)
    drivers = [drv(1, (0, 0), capacity=4)]
    requests = [req(1, (1, 0), (5, 0)), req(2, (2, 0), (6, 0))]

    result = matcher.match(requests, drivers, euclidean)
    assert len(result.assignments) == 1
    assert len(result.assignments[0].request_ids) == 1
    assert len(result.unmatched_requests) == 1


def test_hungarian_pooling_suc_chua_dong_thoi():
    # Xe chở tối đa 1 khách, 2 khách đón cùng vị trí trả cùng vị trí nhưng đặt max_pickup_cost nhỏ
    # để không kịp phục vụ người thứ 2
    matcher = HungarianPoolingMatcher(max_pickup_cost=4.0)
    drivers = [drv(1, (0, 0), capacity=1)]
    requests = [req(1, (1, 0), (10, 0)), req(2, (2, 0), (10, 0))]

    # d1 đón r1 tại (1,0) t=1. Đến (10,0) t=10. r2 đón tại (2,0) thì t đón r2 theo thứ tự sequential là 1+9+8=18 > 4.0
    # nếu đón r2 trước khi trả r1 thì load=2 > capacity 1
    result = matcher.match(requests, drivers, euclidean)
    assert len(result.assignments) == 1
    assert len(result.assignments[0].request_ids) == 1
    assert len(result.unmatched_requests) == 1


def test_hungarian_pooling_nhieu_tai_xe():
    matcher = HungarianPoolingMatcher()
    drivers = [drv(1, (0, 0)), drv(2, (100, 100))]
    requests = [
        req(1, (1, 0), (5, 0)),
        req(2, (2, 0), (6, 0)),
        req(3, (101, 100), (105, 100)),
    ]

    result = matcher.match(requests, drivers, euclidean)
    assert len(result.assignments) == 2
    by_driver = {a.driver_id: set(a.request_ids) for a in result.assignments}
    assert by_driver["d1"] == {"r1", "r2"}
    assert by_driver["d2"] == {"r3"}
    assert len(result.unmatched_requests) == 0


def test_hungarian_pooling_danh_sach_rong():
    matcher = HungarianPoolingMatcher()
    res = matcher.match([], [drv(1)], euclidean)
    assert res.assignments == []
    assert res.idle_drivers == ["d1"]

    res2 = matcher.match([req(1)], [], euclidean)
    assert res2.assignments == []
    assert res2.unmatched_requests == ["r1"]
