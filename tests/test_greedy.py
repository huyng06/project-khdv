"""Unit test cho GreedyMatcher."""
import random
from math import inf, nan

import pytest

from matching import Driver, GreedyMatcher, Request, StopType, euclidean


def req(i, pickup=(0, 0), dropoff=None, **kw):
    return Request(f"r{i}", pickup, dropoff if dropoff is not None else pickup, **kw)


def drv(i, loc=(0, 0), **kw):
    return Driver(f"d{i}", loc, **kw)


def pairs(result):
    """{driver_id: [request_id, ...]} để so sánh kết quả gọn."""
    return {a.driver_id: a.request_ids for a in result.assignments}


# ---------------------------------------------------------------- hành vi cốt lõi
def test_chon_tai_xe_gan_nhat():
    res = GreedyMatcher().match([req(1, (0, 0))], [drv(1, (5, 0)), drv(2, (1, 0)), drv(3, (9, 0))])
    assert pairs(res) == {"d2": ["r1"]}
    assert res.idle_drivers == ["d1", "d3"]


def test_yeu_cau_dat_truoc_duoc_uu_tien():
    # Cả hai khách cùng muốn d1 (gần nhất); r2 đặt sớm hơn nên d1 thuộc về r2.
    requests = [req(1, (0, 0), time=5), req(2, (0, 0), time=1)]
    res = GreedyMatcher().match(requests, [drv(1, (1, 0)), drv(2, (50, 0))])
    assert pairs(res) == {"d1": ["r2"], "d2": ["r1"]}


def test_ket_qua_khong_phu_thuoc_thu_tu_dau_vao():
    rng = random.Random(7)
    requests = [req(i, (rng.uniform(0, 20), rng.uniform(0, 20)), time=rng.uniform(0, 5))
                for i in range(15)]
    drivers = [drv(i, (rng.uniform(0, 20), rng.uniform(0, 20))) for i in range(10)]
    expected = pairs(GreedyMatcher().match(requests, drivers))
    for seed in range(5):
        r2, d2 = requests[:], drivers[:]
        random.Random(seed).shuffle(r2)
        random.Random(seed + 100).shuffle(d2)
        assert pairs(GreedyMatcher().match(r2, d2)) == expected


def test_hai_tai_xe_cach_deu_chon_id_nho_hon():
    res = GreedyMatcher().match([req(1, (0, 0))], [drv(2, (3, 0)), drv(1, (0, 3))])
    assert pairs(res) == {"d1": ["r1"]}


def test_hai_yeu_cau_cung_thoi_diem_xu_ly_theo_id():
    res = GreedyMatcher().match([req(2, (0, 0), time=1), req(1, (0, 0), time=1)], [drv(1, (1, 0))])
    assert pairs(res) == {"d1": ["r1"]}
    assert res.unmatched_requests == ["r2"]


# ---------------------------------------------------------------- trường hợp biên
def test_khong_co_tai_xe():
    res = GreedyMatcher().match([req(1), req(2)], [])
    assert res.assignments == [] and res.unmatched_requests == ["r1", "r2"]


def test_khong_co_yeu_cau():
    res = GreedyMatcher().match([], [drv(1), drv(2)])
    assert res.assignments == [] and res.idle_drivers == ["d1", "d2"]


def test_khong_co_ca_hai():
    res = GreedyMatcher().match([], [])
    assert res.assignments == [] and res.unmatched_requests == [] and res.idle_drivers == []


def test_nhieu_khach_hon_xe_khach_dat_sau_bi_bo_lai():
    requests = [req(1, time=1), req(2, time=2), req(3, time=3)]
    res = GreedyMatcher().match(requests, [drv(1), drv(2)])
    assert len(res.assignments) == 2
    assert res.unmatched_requests == ["r3"]
    assert res.idle_drivers == []


def test_nhieu_xe_hon_khach():
    res = GreedyMatcher().match([req(1)], [drv(1), drv(2), drv(3)])
    assert len(res.assignments) == 1 and len(res.idle_drivers) == 2


def test_tai_xe_ban_bi_bo_qua_du_gan_nhat():
    res = GreedyMatcher().match([req(1, (0, 0))],
                                [drv(1, (0, 1), available=False), drv(2, (9, 0))])
    assert pairs(res) == {"d2": ["r1"]}


def test_tat_ca_tai_xe_ban():
    res = GreedyMatcher().match([req(1)], [drv(1, available=False)])
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == []


# ---------------------------------------------------------------- ràng buộc
def test_vuot_nguong_don_bi_loai():
    m = GreedyMatcher(max_pickup_cost=5)
    res = m.match([req(1, (0, 0))], [drv(1, (6, 0)), drv(2, (4, 0))])
    assert pairs(res) == {"d2": ["r1"]}


def test_nguong_bang_chinh_xac_van_hop_le():
    res = GreedyMatcher(max_pickup_cost=5).match([req(1, (0, 0))], [drv(1, (5, 0))])
    assert pairs(res) == {"d1": ["r1"]}


def test_tat_ca_tai_xe_ngoai_nguong():
    res = GreedyMatcher(max_pickup_cost=1).match([req(1, (0, 0))], [drv(1, (5, 0))])
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == ["d1"]


def test_suc_chua_xe_gan_nhat_khong_du_chon_xe_xa_hon():
    res = GreedyMatcher().match([req(1, (0, 0), passengers=3)],
                                [drv(1, (1, 0), capacity=2), drv(2, (8, 0), capacity=4)])
    assert pairs(res) == {"d2": ["r1"]}


def test_khach_dong_hon_moi_xe_khong_chiem_tai_xe():
    requests = [req(1, passengers=6, time=0), req(2, passengers=1, time=1)]
    res = GreedyMatcher().match(requests, [drv(1, capacity=4)])
    assert pairs(res) == {"d1": ["r2"]}
    assert res.unmatched_requests == ["r1"]


# ---------------------------------------------------------------- hàm chi phí đặc biệt
def test_cost_inf_cho_mot_so_cap_thi_bo_qua_cap_do():
    def cost(a, b):
        return inf if a == (1, 0) else euclidean(a, b)   # d1 không đi được
    res = GreedyMatcher().match([req(1, (0, 0))], [drv(1, (1, 0)), drv(2, (7, 0))], cost)
    assert pairs(res) == {"d2": ["r1"]}


def test_cost_nan_bi_xem_la_khong_hop_le():
    res = GreedyMatcher().match([req(1)], [drv(1)], cost_fn=lambda a, b: nan)
    assert res.unmatched_requests == ["r1"]


def test_chi_phi_bat_doi_xung_dung_chieu_tai_xe_den_diem_don():
    # Đi từ d1 -> điểm đón rẻ hơn dù d2 gần hơn theo đường chim bay.
    def cost(a, b):
        if a == (10, 0) and b == (0, 0):
            return 1.0      # d1: xa nhưng đường thông
        if a == (2, 0) and b == (0, 0):
            return 50.0     # d2: gần nhưng tắc
        return euclidean(a, b)
    res = GreedyMatcher().match([req(1, (0, 0))], [drv(1, (10, 0)), drv(2, (2, 0))], cost)
    assert pairs(res) == {"d1": ["r1"]}


# ---------------------------------------------------------------- đầu ra
def test_lo_trinh_va_chi_phi():
    res = GreedyMatcher().match([req(1, (3, 4), (3, 10))], [drv(1, (0, 0))])
    a = res.assignments[0]
    assert [(s.type, s.location) for s in a.stops] == [
        (StopType.PICKUP, (3, 4)), (StopType.DROPOFF, (3, 10))]
    assert a.cost == pytest.approx(5 + 6)
    assert res.total_cost == pytest.approx(11)


def test_assignments_theo_thu_tu_xu_ly():
    requests = [req(1, time=3), req(2, time=1), req(3, time=2)]
    res = GreedyMatcher().match(requests, [drv(1), drv(2), drv(3)])
    assert res.matched_request_ids == ["r2", "r3", "r1"]


def test_khong_sua_danh_sach_dau_vao():
    requests = [req(2, time=2), req(1, time=1)]
    drivers = [drv(1), drv(2)]
    GreedyMatcher().match(requests, drivers)
    assert [r.id for r in requests] == ["r2", "r1"]
    assert [d.id for d in drivers] == ["d1", "d2"]


# ---------------------------------------------------------------- hạn chế đã biết của Greedy
def test_greedy_khong_toi_uu_toan_cuc():
    """Ví dụ kinh điển: r1 (đến trước) lấy d1, đẩy r2 sang d2 ở rất xa.
    Phương án tối ưu là r1->d2 (6) + r2->d1 (1) = 7, nhưng Greedy ra 4 + 11 = 15.
    Test này ghi lại hạn chế và là lý do cần Hungarian."""
    drivers = [drv(1, (0, 0)), drv(2, (10, 0))]
    requests = [req(1, (4, 0), time=0), req(2, (-1, 0), time=1)]
    res = GreedyMatcher().match(requests, drivers)
    assert pairs(res) == {"d1": ["r1"], "d2": ["r2"]}
    assert res.total_cost == pytest.approx(15)


# ---------------------------------------------------------------- dữ liệu lớn ngẫu nhiên
@pytest.mark.parametrize("n_req,n_drv", [(200, 50), (50, 200), (100, 100)])
def test_ngau_nhien_so_cap_bang_min_va_khong_trung(n_req, n_drv):
    rng = random.Random(n_req * 1000 + n_drv)
    requests = [req(i, (rng.uniform(0, 100), rng.uniform(0, 100)), time=rng.uniform(0, 60))
                for i in range(n_req)]
    drivers = [drv(i, (rng.uniform(0, 100), rng.uniform(0, 100))) for i in range(n_drv)]
    res = GreedyMatcher().match(requests, drivers)    # Matcher tự kiểm tra tính nhất quán
    assert len(res.assignments) == min(n_req, n_drv)
    assert len(res.unmatched_requests) == max(0, n_req - n_drv)
    assert len(res.idle_drivers) == max(0, n_drv - n_req)
