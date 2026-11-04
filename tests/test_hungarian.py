"""Unit test cho solve_assignment (thuật toán lõi) và HungarianMatcher."""
import random
from itertools import permutations
from math import inf, nan

import pytest

from matching import Driver, GreedyMatcher, HungarianMatcher, Request, euclidean
from matching.hungarian import solve_assignment


def req(i, pickup=(0, 0), dropoff=None, **kw):
    return Request(f"r{i}", pickup, dropoff if dropoff is not None else pickup, **kw)


def drv(i, loc=(0, 0), **kw):
    return Driver(f"d{i}", loc, **kw)


def pairs(result):
    return {a.driver_id: a.request_ids for a in result.assignments}


def total(matrix, cols):
    return sum(matrix[i][j] for i, j in enumerate(cols) if j >= 0)


def brute_force(matrix):
    """Chi phí tối ưu bằng vét cạn (chỉ dùng cho ma trận nhỏ)."""
    n, m = len(matrix), len(matrix[0])
    if n <= m:
        return min(sum(matrix[i][c] for i, c in enumerate(p)) for p in permutations(range(m), n))
    return min(sum(matrix[r][j] for j, r in enumerate(p)) for p in permutations(range(n), m))


# ================================================================ solve_assignment
def test_vi_du_3x3_quen_thuoc():
    m = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    cols = solve_assignment(m)
    assert total(m, cols) == 5
    assert sorted(cols) == [0, 1, 2]


def test_1x1():
    assert solve_assignment([[7]]) == [0]


def test_ma_tran_rong():
    assert solve_assignment([]) == []
    assert solve_assignment([[], []]) == [-1, -1]


def test_it_hang_hon_cot_moi_hang_deu_duoc_gan():
    m = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8]]
    cols = solve_assignment(m)
    assert all(c >= 0 for c in cols) and len(set(cols)) == 3
    assert total(m, cols) == brute_force(m) == 9


def test_nhieu_hang_hon_cot_hang_thua_bi_bo_lai():
    m = [[5, 9], [1, 8], [7, 2], [6, 6]]
    cols = solve_assignment(m)
    assert cols.count(-1) == 2
    assert sorted(c for c in cols if c >= 0) == [0, 1]
    assert total(m, cols) == brute_force(m) == 3


def test_so_am():
    m = [[-5, 2], [3, -4]]
    assert total(m, solve_assignment(m)) == -9


def test_cac_phan_tu_bang_nhau_van_tra_hoan_vi_hop_le():
    cols = solve_assignment([[1] * 4 for _ in range(4)])
    assert sorted(cols) == [0, 1, 2, 3]


def test_ma_tran_bat_doi_xung_khong_vuong():
    m = [[1, 100, 100, 100, 100], [100, 100, 1, 100, 100]]
    assert solve_assignment(m) == [0, 2]


@pytest.mark.parametrize("bad", [[[1, 2], [3]], [[1, inf]], [[nan, 1]]])
def test_ma_tran_khong_hop_le_bi_tu_choi(bad):
    with pytest.raises(ValueError):
        solve_assignment(bad)


def test_doi_chieu_vet_can_ngau_nhien():
    rng = random.Random(2024)
    for _ in range(300):
        n, m = rng.randint(1, 6), rng.randint(1, 6)
        mat = [[rng.randint(0, 20) for _ in range(m)] for _ in range(n)]
        cols = solve_assignment(mat)
        assert len(cols) == n
        assigned = [c for c in cols if c >= 0]
        assert len(assigned) == len(set(assigned)) == min(n, m)
        assert total(mat, cols) == brute_force(mat)


def test_doi_chieu_vet_can_so_thuc():
    rng = random.Random(11)
    for _ in range(100):
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        mat = [[rng.uniform(-10, 10) for _ in range(m)] for _ in range(n)]
        assert total(mat, solve_assignment(mat)) == pytest.approx(brute_force(mat))


@pytest.mark.parametrize("n,m", [(60, 60), (40, 70), (70, 40)])
def test_doi_chieu_scipy(n, m):
    np = pytest.importorskip("numpy")
    opt = pytest.importorskip("scipy.optimize")
    rng = random.Random(n * 100 + m)
    mat = [[rng.uniform(0, 100) for _ in range(m)] for _ in range(n)]
    r, c = opt.linear_sum_assignment(np.array(mat))
    expected = sum(mat[i][j] for i, j in zip(r, c))
    assert total(mat, solve_assignment(mat)) == pytest.approx(expected)


# ================================================================ HungarianMatcher: trường hợp biên
def test_khong_co_tai_xe():
    res = HungarianMatcher().match([req(1), req(2)], [])
    assert res.assignments == [] and res.unmatched_requests == ["r1", "r2"]


def test_khong_co_yeu_cau():
    res = HungarianMatcher().match([], [drv(1), drv(2)])
    assert res.assignments == [] and res.idle_drivers == ["d1", "d2"]


def test_khong_co_ca_hai():
    res = HungarianMatcher().match([], [])
    assert res.assignments == [] and res.unmatched_requests == [] and res.idle_drivers == []


def test_nhieu_khach_hon_xe_chon_cac_khach_cho_tong_chi_phi_nho_nhat():
    # 1 xe, 2 khách: chọn khách gần hơn dù đặt sau (khác Greedy, vốn ưu tiên đặt trước).
    requests = [req(1, (9, 0), time=0), req(2, (1, 0), time=1)]
    res = HungarianMatcher().match(requests, [drv(1, (0, 0))])
    assert pairs(res) == {"d1": ["r2"]}
    assert res.unmatched_requests == ["r1"] and res.idle_drivers == []


def test_nhieu_khach_hon_xe_so_luong():
    requests = [req(i, (i, 0)) for i in range(1, 8)]
    res = HungarianMatcher().match(requests, [drv(1), drv(2), drv(3)])
    assert len(res.assignments) == 3 and len(res.unmatched_requests) == 4
    assert sorted(res.matched_request_ids) == ["r1", "r2", "r3"]   # 3 khách gần nhất


def test_nhieu_xe_hon_khach():
    res = HungarianMatcher().match([req(1, (5, 0))], [drv(1, (0, 0)), drv(2, (4, 0)), drv(3, (9, 0))])
    assert pairs(res) == {"d2": ["r1"]}
    assert res.idle_drivers == ["d1", "d3"]


def test_tai_xe_ban_bi_bo_qua():
    res = HungarianMatcher().match([req(1)], [drv(1, (0, 1), available=False), drv(2, (9, 0))])
    assert pairs(res) == {"d2": ["r1"]}


# ================================================================ ma trận chi phí bất cân xứng
def test_ma_tran_chu_nhat_nhieu_xe_hon_khach():
    drivers = [drv(i, (i * 10, 0)) for i in range(1, 6)]            # 5 xe
    requests = [req(1, (21, 0)), req(2, (39, 0))]                    # 2 khách
    res = HungarianMatcher().match(requests, drivers)
    assert pairs(res) == {"d2": ["r1"], "d4": ["r2"]}
    assert res.idle_drivers == ["d1", "d3", "d5"]


def test_ma_tran_chu_nhat_nhieu_khach_hon_xe():
    drivers = [drv(1, (10, 0)), drv(2, (20, 0))]                     # 2 xe
    requests = [req(i, (i * 10 + 1, 0)) for i in range(1, 6)]        # 5 khách
    res = HungarianMatcher().match(requests, drivers)
    assert pairs(res) == {"d1": ["r1"], "d2": ["r2"]}
    assert res.unmatched_requests == ["r3", "r4", "r5"]


def test_chi_phi_bat_doi_xung_dung_chieu_tai_xe_den_diem_don():
    def cost(a, b):
        if a == (10, 0) and b == (0, 0):
            return 1.0
        if a == (2, 0) and b == (0, 0):
            return 50.0
        return euclidean(a, b)
    res = HungarianMatcher().match([req(1, (0, 0))], [drv(1, (10, 0)), drv(2, (2, 0))], cost)
    assert pairs(res) == {"d1": ["r1"]}


def test_chi_phi_bat_doi_xung_nhieu_cap():
    # Ma trận (hàng r1, r2; cột d1, d2): [[1, 2], [2, 9]]
    # Greedy ghép r1-d1 rồi r2-d2 = 10; tối ưu là r1-d2 + r2-d1 = 4.
    table = {((0, 0), (1, 0)): 1, ((0, 0), (2, 0)): 2,
             ((5, 0), (1, 0)): 2, ((5, 0), (2, 0)): 9}
    cost = lambda a, b: table.get((a, b), euclidean(a, b))
    drivers, requests = [drv(1, (0, 0)), drv(2, (5, 0))], [req(1, (1, 0), time=0), req(2, (2, 0), time=1)]
    res = HungarianMatcher().match(requests, drivers, cost)
    assert pairs(res) == {"d2": ["r1"], "d1": ["r2"]}
    assert res.total_cost == pytest.approx(4)
    assert GreedyMatcher().match(requests, drivers, cost).total_cost == pytest.approx(10)


def test_uu_tien_phuc_vu_nhieu_khach_hon_re_hon():
    # d2 không đến được r1; ghép r2-d1 (rẻ, =2) sẽ bỏ r1. Phải chọn r1-d1 + r2-d2 (=101).
    table = {((0, 0), (1, 0)): 1, ((0, 0), (2, 0)): 2,
             ((10, 0), (1, 0)): inf, ((10, 0), (2, 0)): 100}
    cost = lambda a, b: table.get((a, b), euclidean(a, b))
    res = HungarianMatcher().match([req(1, (1, 0)), req(2, (2, 0))],
                                   [drv(1, (0, 0)), drv(2, (10, 0))], cost)
    assert pairs(res) == {"d1": ["r1"], "d2": ["r2"]}
    assert res.total_cost == pytest.approx(101)


# ================================================================ ràng buộc
def test_vuot_nguong_don_bi_loai():
    res = HungarianMatcher(max_pickup_cost=5).match([req(1, (0, 0))], [drv(1, (6, 0)), drv(2, (4, 0))])
    assert pairs(res) == {"d2": ["r1"]}


def test_tat_ca_cap_khong_hop_le_khong_loi():
    res = HungarianMatcher(max_pickup_cost=1).match([req(1), req(2, (50, 0))], [drv(1, (9, 9))])
    assert res.assignments == [] and res.unmatched_requests == ["r1", "r2"]
    assert res.idle_drivers == ["d1"]


def test_cost_inf_hoac_nan_cho_moi_cap():
    for bad in (lambda a, b: inf, lambda a, b: nan):
        res = HungarianMatcher().match([req(1)], [drv(1)], bad)
        assert res.unmatched_requests == ["r1"] and res.idle_drivers == ["d1"]


def test_suc_chua_xe_gan_khong_du_chon_xe_khac():
    res = HungarianMatcher().match([req(1, (0, 0), passengers=3)],
                                   [drv(1, (1, 0), capacity=2), drv(2, (8, 0), capacity=4)])
    assert pairs(res) == {"d2": ["r1"]}


def test_khach_dong_hon_moi_xe_khong_chiem_tai_xe():
    res = HungarianMatcher().match([req(1, passengers=6), req(2)], [drv(1, capacity=4)])
    assert pairs(res) == {"d1": ["r2"]} and res.unmatched_requests == ["r1"]


# ================================================================ so với Greedy
def test_sua_loi_cua_greedy_vi_du_kinh_dien():
    """Cùng dữ liệu với test_greedy_khong_toi_uu_toan_cuc: Greedy ra 15, Hungarian ra 7."""
    drivers = [drv(1, (0, 0)), drv(2, (10, 0))]
    requests = [req(1, (4, 0), time=0), req(2, (-1, 0), time=1)]
    res = HungarianMatcher().match(requests, drivers)
    assert pairs(res) == {"d2": ["r1"], "d1": ["r2"]}
    assert res.total_cost == pytest.approx(7)


@pytest.mark.parametrize("seed", range(25))
def test_khong_bao_gio_te_hon_greedy_khi_ghep_het_khach(seed):
    rng = random.Random(seed)
    n_drv = rng.randint(3, 25)
    n_req = rng.randint(1, n_drv)                       # đủ xe cho mọi khách
    requests = [req(i, (rng.uniform(0, 50), rng.uniform(0, 50)), time=rng.uniform(0, 30))
                for i in range(n_req)]
    drivers = [drv(i, (rng.uniform(0, 50), rng.uniform(0, 50))) for i in range(n_drv)]
    g = GreedyMatcher().match(requests, drivers)
    h = HungarianMatcher().match(requests, drivers)
    assert len(h.assignments) == len(g.assignments) == n_req
    assert h.total_cost <= g.total_cost + 1e-9          # dropoff = pickup nên cost = chi phí đón


# ================================================================ tính chất chung
def test_ket_qua_khong_phu_thuoc_thu_tu_dau_vao():
    rng = random.Random(5)
    requests = [req(i, (rng.uniform(0, 20), rng.uniform(0, 20)), time=rng.uniform(0, 5)) for i in range(12)]
    drivers = [drv(i, (rng.uniform(0, 20), rng.uniform(0, 20))) for i in range(9)]
    expected = pairs(HungarianMatcher().match(requests, drivers))
    for seed in range(5):
        r2, d2 = requests[:], drivers[:]
        random.Random(seed).shuffle(r2)
        random.Random(seed + 50).shuffle(d2)
        assert pairs(HungarianMatcher().match(r2, d2)) == expected


def test_hoa_hai_tai_xe_cach_deu_chon_id_nho_hon():
    res = HungarianMatcher().match([req(1, (0, 0))], [drv(2, (3, 0)), drv(1, (0, 3))])
    assert pairs(res) == {"d1": ["r1"]}


def test_lo_trinh_va_chi_phi():
    res = HungarianMatcher().match([req(1, (3, 4), (3, 10))], [drv(1, (0, 0))])
    a = res.assignments[0]
    assert [s.location for s in a.stops] == [(3, 4), (3, 10)]
    assert a.cost == pytest.approx(11) and res.total_cost == pytest.approx(11)


def test_khong_sua_danh_sach_dau_vao():
    requests, drivers = [req(2, time=2), req(1, time=1)], [drv(2), drv(1)]
    HungarianMatcher().match(requests, drivers)
    assert [r.id for r in requests] == ["r2", "r1"] and [d.id for d in drivers] == ["d2", "d1"]


def test_so_lan_goi_cost_fn():
    from collections import Counter
    calls = Counter()
    def cost(a, b):
        calls[(a, b)] += 1
        return euclidean(a, b)
    requests = [req(i, (i, 0), (i, 5)) for i in range(6)]
    drivers = [drv(i, (0, i)) for i in range(4)]
    HungarianMatcher().match(requests, drivers, cost)
    # R*D lần dựng ma trận + R lần kiểm tra chặng đón -> trả; lộ trình dùng lại kết quả đã ghi nhớ.
    assert sum(calls.values()) == 6 * 4 + 6
    assert max(calls.values()) == 1


@pytest.mark.parametrize("n_req,n_drv", [(80, 30), (30, 80), (60, 60)])
def test_ngau_nhien_toi_uu_va_dung_so_cap(n_req, n_drv):
    rng = random.Random(n_req * 1000 + n_drv)
    requests = [req(i, (rng.uniform(0, 100), rng.uniform(0, 100))) for i in range(n_req)]
    drivers = [drv(i, (rng.uniform(0, 100), rng.uniform(0, 100))) for i in range(n_drv)]
    res = HungarianMatcher().match(requests, drivers)      # Matcher tự kiểm tra nhất quán
    assert len(res.assignments) == min(n_req, n_drv)

    np = pytest.importorskip("numpy")
    opt = pytest.importorskip("scipy.optimize")
    m = np.array([[euclidean(d.location, r.pickup) for d in drivers] for r in requests])
    r_idx, c_idx = opt.linear_sum_assignment(m)
    assert res.total_cost == pytest.approx(m[r_idx, c_idx].sum())


# ================================================================ bổ sung: các điểm dễ sai
def test_cap_khong_hop_le_khong_duoc_ghep_ke_ca_khi_ma_tran_vuong():
    """Số khách = số xe nên Hungarian luôn gán đủ cặp trên ma trận; cặp bị phạt phải bị lọc ra."""
    drivers = [drv(1, (0, 0)), drv(2, (100, 100))]
    requests = [req(1, (1, 0)), req(2, (200, 200))]
    res = HungarianMatcher(max_pickup_cost=5).match(requests, drivers)
    assert pairs(res) == {"d1": ["r1"]}
    assert res.unmatched_requests == ["r2"] and res.idle_drivers == ["d2"]


def test_hoa_nhieu_phuong_an_toi_uu_cho_ket_qua_on_dinh_theo_id():
    """Hai phương án tối ưu bằng nhau: kết quả phải cố định, không phụ thuộc thứ tự đầu vào."""
    requests = [req(1, (1, 0)), req(2, (1, 0))]
    drivers = [drv(1, (0, 0)), drv(2, (2, 0))]
    expected = {"d1": ["r1"], "d2": ["r2"]}
    for rs in (requests, requests[::-1]):
        for ds in (drivers, drivers[::-1]):
            assert pairs(HungarianMatcher().match(rs, ds)) == expected
