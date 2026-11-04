"""Test cho models và lớp Matcher. Dùng một matcher giả để kiểm tra phần dùng chung."""
from math import inf

import pytest

from matching import (Assignment, Driver, Matcher, MatchResult, Request, Stop,
                      StopType, euclidean, haversine_km, route_cost)


class FirstFitMatcher(Matcher):
    """Matcher giả: ghép lần lượt yêu cầu i với tài xế i (nếu hợp lệ)."""
    def _match(self, requests, drivers, cost_fn):
        out = []
        for r, d in zip(requests, drivers):
            if self.is_feasible(d, r, cost_fn):
                out.append(self.single_assignment(d, r, cost_fn))
        return self.build_result(out, requests, drivers)


class BrokenMatcher(Matcher):
    """Cố tình dùng một tài xế cho hai yêu cầu để test bộ kiểm tra đầu ra."""
    def _match(self, requests, drivers, cost_fn):
        out = [self.single_assignment(drivers[0], r, cost_fn) for r in requests]
        return MatchResult(out, [], [d.id for d in drivers[1:]])


def req(i, pickup=(0, 0), dropoff=(1, 0), **kw):
    return Request(f"r{i}", pickup, dropoff, **kw)


def drv(i, loc=(0, 0), **kw):
    return Driver(f"d{i}", loc, **kw)


# ---------------------------------------------------------------- models
def test_request_va_driver_khong_hop_le():
    with pytest.raises(ValueError):
        Request("r", (0, 0), (1, 1), passengers=0)
    with pytest.raises(ValueError):
        Driver("d", (0, 0), capacity=0)


def test_haversine_mot_do_kinh_do_tai_xich_dao():
    assert haversine_km((0, 0), (0, 1)) == pytest.approx(111.19, rel=1e-2)


def test_route_cost():
    stops = [Stop("r1", StopType.PICKUP, (3, 4)), Stop("r1", StopType.DROPOFF, (3, 8))]
    assert route_cost((0, 0), stops, euclidean) == pytest.approx(5 + 4)
    assert route_cost((0, 0), [], euclidean) == 0


def test_route_cost_bat_doi_xung():
    f = lambda a, b: 1.0 if a < b else 10.0
    stops = [Stop("r", StopType.PICKUP, (5, 0))]
    assert route_cost((0, 0), stops, f) == 1.0
    assert route_cost((9, 0), stops, f) == 10.0


def test_assignment_request_ids_theo_thu_tu_don():
    stops = [Stop("a", StopType.PICKUP, (0, 0)), Stop("b", StopType.PICKUP, (1, 0)),
             Stop("a", StopType.DROPOFF, (2, 0)), Stop("b", StopType.DROPOFF, (3, 0))]
    assert Assignment("d", stops, 3).request_ids == ["a", "b"]


# ---------------------------------------------------------------- các trường hợp biên dùng chung
def test_khong_co_tai_xe():
    res = FirstFitMatcher().match([req(1), req(2)], [])
    assert res.assignments == [] and res.unmatched_requests == ["r1", "r2"]
    assert res.idle_drivers == [] and res.total_cost == 0


def test_khong_co_yeu_cau():
    res = FirstFitMatcher().match([], [drv(1), drv(2)])
    assert res.assignments == [] and res.unmatched_requests == []
    assert res.idle_drivers == ["d1", "d2"]


def test_tai_xe_ban_bi_loai():
    res = FirstFitMatcher().match([req(1)], [drv(1, available=False)])
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == []


def test_id_trung_bi_tu_choi():
    with pytest.raises(ValueError):
        FirstFitMatcher().match([req(1), req(1)], [drv(1)])
    with pytest.raises(ValueError):
        FirstFitMatcher().match([req(1)], [drv(1), drv(1)])


def test_nhieu_khach_hon_xe():
    res = FirstFitMatcher().match([req(1), req(2), req(3)], [drv(1), drv(2)])
    assert len(res.assignments) == 2
    assert res.unmatched_requests == ["r3"]


def test_nhieu_xe_hon_khach():
    res = FirstFitMatcher().match([req(1)], [drv(1), drv(2), drv(3)])
    assert len(res.assignments) == 1 and res.idle_drivers == ["d2", "d3"]


def test_vuot_nguong_don_khong_duoc_ghep():
    m = FirstFitMatcher(max_pickup_cost=2.0)
    res = m.match([req(1, pickup=(10, 0))], [drv(1)])
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == ["d1"]


def test_cost_inf_khong_duoc_ghep():
    res = FirstFitMatcher().match([req(1)], [drv(1)], cost_fn=lambda a, b: inf)
    assert res.unmatched_requests == ["r1"]


def test_khach_dong_hon_suc_chua():
    res = FirstFitMatcher().match([req(1, passengers=5)], [drv(1, capacity=4)])
    assert res.unmatched_requests == ["r1"]


def test_tong_chi_phi_va_lo_trinh_don_le():
    res = FirstFitMatcher().match([req(1, (3, 4), (3, 8))], [drv(1, (0, 0))])
    a = res.assignments[0]
    assert [s.type for s in a.stops] == [StopType.PICKUP, StopType.DROPOFF]
    assert res.total_cost == pytest.approx(9.0)


def test_bo_kiem_tra_dau_ra_bat_loi_lop_con():
    with pytest.raises(AssertionError):
        BrokenMatcher().match([req(1), req(2)], [drv(1), drv(2)])
