"""Unit test cho check_route và PoolingMatcher."""
import random
from collections import Counter
from itertools import permutations
from math import inf, nan

import pytest

from matching import (Driver, GreedyMatcher, PoolingMatcher, Request, Stop,
                      StopType, check_route, euclidean, route_cost)

P, D = StopType.PICKUP, StopType.DROPOFF


def req(i, pickup=(0, 0), dropoff=(10, 0), **kw):
    return Request(f"r{i}", pickup, dropoff, **kw)


def drv(i, loc=(0, 0), **kw):
    return Driver(f"d{i}", loc, **kw)


def stops(requests, order):
    """order: [("r1", P), ("r2", P), ("r2", D), ("r1", D)] -> list[Stop]."""
    by = {r.id: r for r in requests}
    return [Stop(rid, t, by[rid].pickup if t == P else by[rid].dropoff) for rid, t in order]


def shape(assignment):
    return [(s.request_id, s.type) for s in assignment.stops]


def by_driver(result):
    return {a.driver_id: a for a in result.assignments}


# ---- bộ kiểm tra độc lập (không dùng check_route) để soi kết quả của PoolingMatcher
def simulate(driver, stop_list, by_id, cost_fn, ratio=inf, max_pickup=inf):
    """Chi phí lộ trình, hoặc None nếu vi phạm. Viết lại từ đầu, khác cách làm của check_route."""
    t, cur, load = 0.0, driver.location, 0
    picked_at, picked, dropped = {}, set(), set()
    for s in stop_list:
        t += cost_fn(cur, s.location)
        cur = s.location
        r = by_id[s.request_id]
        if s.type == P:
            if s.request_id in picked or t > max_pickup + 1e-9:
                return None
            picked.add(s.request_id)
            picked_at[s.request_id] = t
            load += r.passengers
            if load > driver.capacity:
                return None
        else:
            if s.request_id not in picked or s.request_id in dropped:
                return None
            dropped.add(s.request_id)
            load -= r.passengers
            if ratio != inf and t - picked_at[s.request_id] > ratio * cost_fn(r.pickup, r.dropoff) + 1e-6:
                return None
    return t if picked == dropped else None


def assert_valid(result, requests, drivers, cost_fn=euclidean, ratio=inf, max_pickup=inf, max_requests=None):
    by_id = {r.id: r for r in requests}
    drv_by_id = {d.id: d for d in drivers}
    matched = [rid for a in result.assignments for rid in a.request_ids]
    assert len(matched) == len(set(matched))
    assert sorted(matched + result.unmatched_requests) == sorted(by_id)
    for a in result.assignments:
        d = drv_by_id[a.driver_id]
        assert d.available
        c = simulate(d, a.stops, by_id, cost_fn, ratio, max_pickup)
        assert c is not None, f"lộ trình của {a.driver_id} vi phạm ràng buộc"
        assert a.cost == pytest.approx(c)
        assert a.cost == pytest.approx(route_cost(d.location, a.stops, cost_fn))
        if max_requests is not None:
            assert len(a.request_ids) <= max_requests


# ================================================================ check_route
def test_lo_trinh_rong_chi_phi_bang_0():
    assert check_route(drv(1), [], {}, euclidean) == 0.0


def test_mot_khach_chi_phi_bang_don_cong_di():
    r = req(1, (3, 4), (3, 10))
    c = check_route(drv(1, (0, 0)), stops([r], [("r1", P), ("r1", D)]), {"r1": r}, euclidean)
    assert c == pytest.approx(5 + 6)


def test_khach_tra_truoc_khi_don_bi_loai():
    r = req(1)
    assert check_route(drv(1), stops([r], [("r1", D), ("r1", P)]), {"r1": r}, euclidean) is None


def test_don_ma_khong_tra_hoac_tra_ma_khong_don():
    r = req(1)
    assert check_route(drv(1), stops([r], [("r1", P)]), {"r1": r}, euclidean) is None
    assert check_route(drv(1), stops([r], [("r1", D)]), {"r1": r}, euclidean) is None


def test_don_hai_lan_hoac_tra_hai_lan_bi_loai():
    r = req(1)
    assert check_route(drv(1), stops([r], [("r1", P), ("r1", P), ("r1", D)]), {"r1": r}, euclidean) is None
    assert check_route(drv(1), stops([r], [("r1", P), ("r1", D), ("r1", D)]), {"r1": r}, euclidean) is None


def test_suc_chua_xet_tai_moi_thoi_diem():
    r1, r2 = req(1, passengers=3), req(2, (1, 0), (9, 0), passengers=2)
    rs = {"r1": r1, "r2": r2}
    chung = stops([r1, r2], [("r1", P), ("r2", P), ("r2", D), ("r1", D)])      # 5 khách cùng lúc
    lan_luot = stops([r1, r2], [("r1", P), ("r1", D), ("r2", P), ("r2", D)])   # tối đa 3 khách
    assert check_route(drv(1, capacity=4), chung, rs, euclidean) is None
    assert check_route(drv(1, capacity=4), lan_luot, rs, euclidean) is not None
    assert check_route(drv(1, capacity=5), chung, rs, euclidean) is not None


def test_khach_dong_hon_suc_chua_cua_xe():
    r = req(1, passengers=5)
    assert check_route(drv(1, capacity=4), stops([r], [("r1", P), ("r1", D)]), {"r1": r}, euclidean) is None


def test_nguong_don_tinh_theo_lo_trinh_thuc_te():
    r1, r2 = req(1, (1, 0), (10, 0)), req(2, (20, 0), (30, 0))
    rs = {"r1": r1, "r2": r2}
    st = stops([r1, r2], [("r1", P), ("r1", D), ("r2", P), ("r2", D)])   # r2 chờ đến t = 20
    assert check_route(drv(1), st, rs, euclidean, max_pickup_cost=20) == pytest.approx(30)
    assert check_route(drv(1), st, rs, euclidean, max_pickup_cost=19.99) is None


def test_di_thang_thi_dat_ratio_1():
    r1, r2 = req(1, (1, 0), (10, 0)), req(2, (2, 0), (20, 0))
    rs = {"r1": r1, "r2": r2}
    st = stops([r1, r2], [("r1", P), ("r2", P), ("r1", D), ("r2", D)])
    # r1: 1 -> 2 -> 10 = 9 (thẳng 9); r2: 2 -> 10 -> 20 = 18 (thẳng 18): không ai đi vòng.
    assert check_route(drv(1), st, rs, euclidean, max_detour_ratio=1.0) == pytest.approx(20)


def test_gioi_han_di_vong_dung_bien():
    # r1 đi 2.5 + 12.5 = 15 so với quãng thẳng 10: tỉ lệ đúng bằng 1.5.
    r1, r2 = req(1, (0, 0), (10, 0)), req(2, (-2.5, 0), (20, 0))
    rs = {"r1": r1, "r2": r2}
    st = stops([r1, r2], [("r1", P), ("r2", P), ("r1", D), ("r2", D)])
    assert check_route(drv(1), st, rs, euclidean, max_detour_ratio=1.5) is not None
    assert check_route(drv(1), st, rs, euclidean, max_detour_ratio=1.49) is None


def test_di_vong_vuot_gioi_han():
    r1, r2 = req(1, (0, 0), (10, 0)), req(2, (0, 5), (10, 5))
    rs = {"r1": r1, "r2": r2}
    st = stops([r1, r2], [("r1", P), ("r2", P), ("r1", D), ("r2", D)])
    # r1 đi (0,0)->(0,5)->(10,0)->... = 5 + 11.18 = 16.18 so với thẳng 10
    assert check_route(drv(1, (0, 0)), st, rs, euclidean, max_detour_ratio=1.5) is None
    assert check_route(drv(1, (0, 0)), st, rs, euclidean, max_detour_ratio=2.0) is not None
    assert check_route(drv(1, (0, 0)), st, rs, euclidean, max_detour_ratio=inf) is not None


def test_cho_khach_co_quang_thang_bang_0():
    r = req(1, (5, 5), (5, 5))
    st = stops([r], [("r1", P), ("r1", D)])
    assert check_route(drv(1, (0, 0)), st, {"r1": r}, euclidean, max_detour_ratio=1.5) is not None


def test_chang_khong_di_duoc_bi_loai():
    r = req(1)
    st = stops([r], [("r1", P), ("r1", D)])
    assert check_route(drv(1), st, {"r1": r}, lambda a, b: inf) is None
    assert check_route(drv(1), st, {"r1": r}, lambda a, b: nan) is None


def test_chi_phi_bat_doi_xung_tinh_dung_chieu():
    f = lambda a, b: 1.0 if a[0] < b[0] else 10.0
    r = req(1, (5, 0), (8, 0))
    assert check_route(drv(1, (0, 0)), stops([r], [("r1", P), ("r1", D)]), {"r1": r}, f) == 2.0
    assert check_route(drv(1, (9, 0)), stops([r], [("r1", P), ("r1", D)]), {"r1": r}, f) == 11.0


# ================================================================ PoolingMatcher: hành vi cốt lõi
def test_hai_khach_cung_huong_di_chung_mot_xe():
    requests = [req(1, (1, 0), (10, 0), time=0), req(2, (2, 0), (9, 0), time=1)]
    drivers = [drv(1, (0, 0)), drv(2, (0, 50))]
    res = PoolingMatcher().match(requests, drivers)
    assert len(res.assignments) == 1 and res.idle_drivers == ["d2"]
    a = res.assignments[0]
    assert a.driver_id == "d1"
    assert shape(a) == [("r1", P), ("r2", P), ("r2", D), ("r1", D)]
    assert res.total_cost == pytest.approx(10)
    assert GreedyMatcher().match(requests, drivers).total_cost > 60      # chạy riêng tốn hơn nhiều
    assert_valid(res, requests, drivers, ratio=1.5)


def test_khach_nguoc_huong_khong_ghep_chung_khi_co_xe_khac():
    requests = [req(1, (0, 0), (10, 0), time=0), req(2, (0, 5), (-10, 5), time=1)]
    res = PoolingMatcher().match(requests, [drv(1), drv(2)])
    assert len(res.assignments) == 2
    assert all(len(a.request_ids) == 1 for a in res.assignments)


def test_mot_xe_phuc_vu_noi_tiep_khi_khong_the_di_chung():
    requests = [req(1, (0, 0), (10, 0), time=0), req(2, (0, 5), (-10, 5), time=1)]
    res = PoolingMatcher().match(requests, [drv(1)])
    a = res.assignments[0]
    assert shape(a) == [("r1", P), ("r1", D), ("r2", P), ("r2", D)]
    assert res.unmatched_requests == []
    assert_valid(res, requests, [drv(1)], ratio=1.5)


def test_nguong_don_chan_khach_phai_cho_qua_lau():
    requests = [req(1, (0, 0), (10, 0), time=0), req(2, (0, 5), (-10, 5), time=1)]
    res = PoolingMatcher(max_pickup_cost=15).match(requests, [drv(1)])
    assert res.matched_request_ids == ["r1"] and res.unmatched_requests == ["r2"]


def test_mot_xe_cho_duoc_nhieu_khach_hon_so_xe():
    requests = [req(1, (1, 0), (10, 0), time=0), req(2, (2, 0), (9, 0), time=1),
                req(3, (3, 0), (8, 0), time=2)]
    res = PoolingMatcher().match(requests, [drv(1)])
    assert len(res.assignments) == 1 and res.unmatched_requests == []
    assert shape(res.assignments[0]) == [("r1", P), ("r2", P), ("r3", P),
                                         ("r3", D), ("r2", D), ("r1", D)]
    assert res.total_cost == pytest.approx(10)


def test_chon_theo_chi_phi_tang_them_chu_khong_phai_tong_chi_phi():
    """d1 đã có lộ trình tốn 100 nhưng nhận thêm r2 chỉ tăng 0; d2 rảnh mở chuyến mới tốn 30.
    Heuristic so chi phí TĂNG THÊM nên ghép r2 vào d1 (đó là cách tiết kiệm tổng quãng đường)."""
    def cost(a, b):
        return inf if (a == (70, 10) and b == (50, 0)) else euclidean(a, b)   # d2 không đến được r1
    requests = [req(1, (50, 0), (100, 0), time=0), req(2, (70, 0), (90, 0), time=1)]
    drivers = [drv(1, (0, 0)), drv(2, (70, 10))]
    res = PoolingMatcher().match(requests, drivers, cost)
    assert by_driver(res).keys() == {"d1"} and res.idle_drivers == ["d2"]
    assert by_driver(res)["d1"].request_ids == ["r1", "r2"]
    assert res.total_cost == pytest.approx(100)


def test_gioi_han_so_yeu_cau_moi_xe():
    requests = [req(1, (1, 0), (10, 0), time=0), req(2, (2, 0), (9, 0), time=1),
                req(3, (3, 0), (8, 0), time=2)]
    res = PoolingMatcher(max_requests_per_driver=2).match(requests, [drv(1)])
    assert res.matched_request_ids == ["r1", "r2"] and res.unmatched_requests == ["r3"]


def test_suc_chua_xe_chan_viec_ghep_chung():
    requests = [req(1, (1, 0), (10, 0), passengers=2, time=0),
                req(2, (2, 0), (9, 0), passengers=2, time=1),
                req(3, (3, 0), (8, 0), passengers=2, time=2)]
    drivers = [drv(1, capacity=4)]
    res = PoolingMatcher(max_pickup_cost=12).match(requests, drivers)
    assert res.matched_request_ids == ["r1", "r2"] and res.unmatched_requests == ["r3"]
    assert_valid(res, requests, drivers, ratio=1.5, max_pickup=12)


def test_khong_ghep_chung_neu_xe_hai_khach_dong_hon_suc_chua():
    res = PoolingMatcher().match([req(1, passengers=5)], [drv(1, capacity=4)])
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == ["d1"]


# ================================================================ trường hợp biên theo yêu cầu của thầy
def test_khong_co_tai_xe():
    res = PoolingMatcher().match([req(1), req(2)], [])
    assert res.assignments == [] and res.unmatched_requests == ["r1", "r2"]


def test_khong_co_yeu_cau():
    res = PoolingMatcher().match([], [drv(1), drv(2)])
    assert res.assignments == [] and res.idle_drivers == ["d1", "d2"]


def test_khong_co_ca_hai():
    res = PoolingMatcher().match([], [])
    assert res.assignments == [] and res.unmatched_requests == [] and res.idle_drivers == []


def test_nhieu_khach_hon_xe_khach_roi_rac_khong_ghep_duoc():
    requests = [req(i, (i * 100, 0), (i * 100 + 10, 0), time=i) for i in range(1, 5)]
    res = PoolingMatcher(max_pickup_cost=50).match(requests, [drv(1, (100, 0)), drv(2, (200, 0))])
    assert sorted(res.matched_request_ids) == ["r1", "r2"]
    assert res.unmatched_requests == ["r3", "r4"]


def test_nhieu_xe_hon_khach():
    res = PoolingMatcher().match([req(1, (5, 0), (9, 0))], [drv(1, (0, 0)), drv(2, (4, 0)), drv(3, (9, 0))])
    assert by_driver(res).keys() == {"d2"} and res.idle_drivers == ["d1", "d3"]


def test_tai_xe_ban_bi_bo_qua():
    res = PoolingMatcher().match([req(1)], [drv(1, (0, 1), available=False), drv(2, (9, 0))])
    assert by_driver(res).keys() == {"d2"}


def test_chang_khong_di_duoc_thi_khach_khong_duoc_ghep():
    # d1 -> điểm đón của r2 không đi được (inf): r2 chỉ có thể do d2 phục vụ.
    def cost(a, b):
        return inf if (a == (0, 0) and b == (50, 0)) else euclidean(a, b)
    requests = [req(1, (1, 0), (10, 0)), req(2, (50, 0), (60, 0))]
    res = PoolingMatcher().match(requests, [drv(1, (0, 0)), drv(2, (45, 0))], cost)
    assert by_driver(res)["d2"].request_ids == ["r2"]
    assert by_driver(res)["d1"].request_ids == ["r1"]


def test_moi_cap_deu_khong_di_duoc():
    res = PoolingMatcher().match([req(1)], [drv(1)], lambda a, b: inf)
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == ["d1"]


# ================================================================ chi phí bất đối xứng
def test_chi_phi_bat_doi_xung_chon_thu_tu_toi_uu():
    """Một xe, hai khách: heuristic xét đủ 6 cách đan xen nên phải bằng vét cạn."""
    def cost(a, b):
        return euclidean(a, b) * (3 if b[0] < a[0] else 1)     # đi sang trái đắt gấp 3
    r1, r2 = req(1, (5, 0), (8, 0), time=0), req(2, (2, 0), (6, 0), time=1)
    driver = drv(1, (0, 0))
    by_id = {"r1": r1, "r2": r2}
    best = min(
        c for order in permutations([("r1", P), ("r1", D), ("r2", P), ("r2", D)])
        if (c := simulate(driver, stops([r1, r2], list(order)), by_id, cost)) is not None
    )
    res = PoolingMatcher(max_detour_ratio=inf).match([r1, r2], [driver], cost)
    assert len(res.assignments) == 1 and len(res.assignments[0].request_ids) == 2
    assert res.total_cost == pytest.approx(best)


# ================================================================ tính chất chung
def test_ket_qua_khong_phu_thuoc_thu_tu_dau_vao():
    rng = random.Random(3)
    requests = [req(i, (rng.uniform(0, 30), rng.uniform(0, 30)), (rng.uniform(0, 30), rng.uniform(0, 30)),
                    time=rng.uniform(0, 5)) for i in range(14)]
    drivers = [drv(i, (rng.uniform(0, 30), rng.uniform(0, 30))) for i in range(5)]
    shapes = lambda res: {a.driver_id: shape(a) for a in res.assignments}
    expected = shapes(PoolingMatcher().match(requests, drivers))
    for seed in range(5):
        r2, d2 = requests[:], drivers[:]
        random.Random(seed).shuffle(r2)
        random.Random(seed + 9).shuffle(d2)
        assert shapes(PoolingMatcher().match(r2, d2)) == expected


def test_hoa_chon_tai_xe_id_nho_hon():
    res = PoolingMatcher().match([req(1, (0, 0), (5, 0))], [drv(2, (3, 0)), drv(1, (0, 3))])
    assert by_driver(res).keys() == {"d1"}


def test_khong_sua_danh_sach_dau_vao():
    requests, drivers = [req(2, time=2), req(1, time=1)], [drv(2), drv(1)]
    PoolingMatcher().match(requests, drivers)
    assert [r.id for r in requests] == ["r2", "r1"] and [d.id for d in drivers] == ["d2", "d1"]


def test_moi_cap_diem_chi_tinh_chi_phi_mot_lan():
    calls = Counter()
    def cost(a, b):
        calls[(a, b)] += 1
        return euclidean(a, b)
    rng = random.Random(8)
    requests = [req(i, (rng.randint(0, 20), rng.randint(0, 20)), (rng.randint(0, 20), rng.randint(0, 20)),
                    time=i) for i in range(12)]
    PoolingMatcher().match(requests, [drv(i, (rng.randint(0, 20), rng.randint(0, 20))) for i in range(4)], cost)
    assert calls and max(calls.values()) == 1


def test_tham_so_khong_hop_le():
    with pytest.raises(ValueError):
        PoolingMatcher(max_detour_ratio=0.9)
    with pytest.raises(ValueError):
        PoolingMatcher(max_requests_per_driver=0)


@pytest.mark.parametrize("seed", range(15))
def test_gioi_han_1_yeu_cau_moi_xe_giong_greedy(seed):
    """Không ghép chung + chặng trả bằng chặng đón (chi phí 0) thì phải ra đúng như Greedy."""
    rng = random.Random(seed)
    requests = []
    for i in range(rng.randint(1, 20)):
        p = (rng.uniform(0, 60), rng.uniform(0, 60))
        requests.append(req(i, p, p, time=rng.uniform(0, 30), passengers=rng.randint(1, 3)))
    drivers = [drv(i, (rng.uniform(0, 60), rng.uniform(0, 60)), capacity=rng.randint(2, 4))
               for i in range(rng.randint(1, 20))]
    pairs = lambda res: {a.driver_id: a.request_ids for a in res.assignments}
    pooled = PoolingMatcher(max_pickup_cost=30, max_requests_per_driver=1).match(requests, drivers)
    greedy = GreedyMatcher(max_pickup_cost=30).match(requests, drivers)
    assert pairs(pooled) == pairs(greedy)
    assert pooled.unmatched_requests == greedy.unmatched_requests


@pytest.mark.parametrize("seed", range(20))
def test_ngau_nhien_moi_lo_trinh_deu_hop_le(seed):
    rng = random.Random(seed)
    requests = [req(i, (rng.uniform(0, 40), rng.uniform(0, 40)), (rng.uniform(0, 40), rng.uniform(0, 40)),
                    time=rng.uniform(0, 20), passengers=rng.randint(1, 3)) for i in range(rng.randint(1, 40))]
    drivers = [drv(i, (rng.uniform(0, 40), rng.uniform(0, 40)), capacity=rng.randint(2, 6),
                   available=rng.random() > 0.15) for i in range(rng.randint(1, 12))]
    ratio, max_pickup = rng.choice([1.2, 1.5, 2.0]), rng.choice([inf, 40.0, 80.0])
    max_req = rng.choice([None, 2, 3])
    res = PoolingMatcher(max_pickup, ratio, max_req).match(requests, drivers)
    assert_valid(res, requests, drivers, ratio=ratio, max_pickup=max_pickup, max_requests=max_req)


def test_du_lieu_lon_chay_nhanh():
    rng = random.Random(1)
    requests = [req(i, (rng.uniform(0, 100), rng.uniform(0, 100)), (rng.uniform(0, 100), rng.uniform(0, 100)),
                    time=rng.uniform(0, 60)) for i in range(150)]
    drivers = [drv(i, (rng.uniform(0, 100), rng.uniform(0, 100))) for i in range(60)]
    res = PoolingMatcher().match(requests, drivers)
    assert_valid(res, requests, drivers, ratio=1.5)
    assert len(res.matched_request_ids) > 60      # ghép chung phục vụ được nhiều khách hơn số xe
