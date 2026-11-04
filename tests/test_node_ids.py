"""Test khi vị trí là mã nút bản đồ (int/str) thay vì toạ độ, đúng như cách nhóm tích hợp với OpenStreetMap.

Đồ thị nhỏ có đường MỘT CHIỀU và một nút "ngõ cụt" (nút 9: vào được, không ra được).
"""
from math import inf

import pytest

from matching import (Driver, GreedyMatcher, HungarianMatcher, PoolingMatcher,
                      Request, euclidean, haversine_km)

# khoảng cách ngắn nhất (mét) giữa các nút; thiếu cặp nào = không có đường đi
DIST = {
    (1, 2): 100, (2, 1): 100, (2, 3): 100, (3, 2): 100, (1, 3): 200, (3, 1): 200,
    (1, 4): 300, (4, 1): 300, (2, 4): 250, (4, 2): 250, (3, 4): 150, (4, 3): 150,
    (1, 9): 400, (2, 9): 300, (3, 9): 200, (4, 9): 500,       # vào được nút 9...
}                                                              # ...nhưng không có (9, x): không ra được


def graph_cost(a, b):
    if a == b:
        return 0.0
    return float(DIST.get((a, b), inf))


ALL = [GreedyMatcher, HungarianMatcher, PoolingMatcher]


@pytest.mark.parametrize("matcher_cls", ALL)
def test_node_id_so_nguyen(matcher_cls):
    drivers = [Driver("d1", 1), Driver("d2", 3)]
    requests = [Request("r1", 2, 4, time=0), Request("r2", 3, 1, time=1)]
    res = matcher_cls().match(requests, drivers, graph_cost)
    assert sorted(res.matched_request_ids) == ["r1", "r2"]
    assert res.unmatched_requests == [] and res.total_cost > 0


@pytest.mark.parametrize("matcher_cls", ALL)
def test_node_id_chuoi(matcher_cls):
    cost = lambda a, b: 0.0 if a == b else 10.0
    res = matcher_cls().match([Request("r1", "A", "B")], [Driver("d1", "C")], cost)
    assert res.matched_request_ids == ["r1"]


@pytest.mark.parametrize("matcher_cls", ALL)
def test_diem_tra_khong_di_toi_duoc_thi_khong_ghep(matcher_cls):
    """Khách r1 đón ở nút 9 nhưng nút 9 là ngõ cụt: không có đường từ 9 ra ngoài.
    Ghép cho tài xế sẽ làm tài xế kẹt giữa đường, nên phải để khách chưa ghép."""
    drivers = [Driver("d1", 2)]
    requests = [Request("r1", 9, 1, time=0), Request("r2", 3, 1, time=1)]
    res = matcher_cls().match(requests, drivers, graph_cost)
    assert res.matched_request_ids == ["r2"]
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == []
    assert res.total_cost != inf


@pytest.mark.parametrize("matcher_cls", ALL)
def test_chi_co_khach_khong_phuc_vu_duoc(matcher_cls):
    res = matcher_cls().match([Request("r1", 9, 1)], [Driver("d1", 2)], graph_cost)
    assert res.assignments == [] and res.unmatched_requests == ["r1"] and res.idle_drivers == ["d1"]


@pytest.mark.parametrize("matcher_cls", ALL)
def test_tai_xe_dung_dung_noi_khach_tra_ve_0(matcher_cls):
    res = matcher_cls().match([Request("r1", 2, 3)], [Driver("d1", 2)], graph_cost)
    assert res.matched_request_ids == ["r1"] and res.assignments[0].cost == pytest.approx(100)


@pytest.mark.parametrize("matcher_cls", ALL)
def test_ket_qua_hu_han_khong_co_chi_phi_vo_cuc(matcher_cls):
    drivers = [Driver(f"d{i}", i % 4 + 1) for i in range(3)]
    requests = [Request(f"r{i}", (i % 4) + 1, ((i + 1) % 4) + 1, time=i) for i in range(6)]
    res = matcher_cls().match(requests, drivers, graph_cost)
    assert all(a.cost != inf for a in res.assignments)


def test_ham_mau_voi_node_id():
    assert euclidean(5, 5) == 0.0 and euclidean(5, 6) == inf
    assert haversine_km("a", "a") == 0.0 and haversine_km("a", "b") == inf
    assert euclidean((0, 0), (3, 4)) == pytest.approx(5)


@pytest.mark.parametrize("matcher_cls", ALL)
def test_moi_cap_diem_chi_tinh_mot_lan_ke_ca_khi_goi_ba_thuat_toan(matcher_cls):
    from collections import Counter
    calls = Counter()
    def cost(a, b):
        calls[(a, b)] += 1
        return graph_cost(a, b)
    matcher_cls().match([Request("r1", 2, 4), Request("r2", 3, 1)], [Driver("d1", 1), Driver("d2", 3)], cost)
    assert max(calls.values()) == 1
