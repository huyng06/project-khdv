import random
import sys
from functools import partial
from typing import Dict

import numpy as np

from core_engine.entities import Driver
from core_engine.map_network import RealRoadMap
from core_engine.demand_generator import DemandGenerator
from core_engine.simulator import SimulationEngine
from visualize_map import generate_interactive_html_map

from matching import (
    GreedyMatcher,
    HungarianMatcher,
    PoolingMatcher,
    HungarianPoolingMatcher,
)
from adapter import run_matcher_on_sim

MATCHERS = {
    "greedy": (GreedyMatcher, "S1: Greedy (Tham lam gần nhất)"),
    "hungarian": (HungarianMatcher, "S2: Hungarian (Gom lô tối ưu toàn cục)"),
    "pooling": (PoolingMatcher, "S3: Greedy Pooling (Tham lam + Ghép chuyến)"),
    "greedy_pool": (PoolingMatcher, "S3: Greedy Pooling (Tham lam + Ghép chuyến)"),
    "hungarian_pooling": (HungarianPoolingMatcher, "S4: Hungarian Pooling (Hungary + Ghép chuyến)"),
    "hungarian_pool": (HungarianPoolingMatcher, "S4: Hungarian Pooling (Hungary + Ghép chuyến)"),
}


def run_single_simulation(
    road_map: RealRoadMap,
    matcher_cls,
    label: str,
    num_drivers: int = 8,
    arrival_rate: float = 0.03,
    total_time: float = 1800.0,
    seed: int = 42,
    dt: float = 1.0,
) -> dict:
    """Chạy mô phỏng đơn lẻ cho một chiến lược ghép cặp."""
    random.seed(seed)
    np.random.seed(seed)

    # 1. Khởi tạo tài xế
    drivers = [
        Driver(driver_id=f"D_{i+1:02d}", current_node=road_map.get_random_node())
        for i in range(num_drivers)
    ]

    # 2. Khởi tạo DemandGenerator & Simulator
    generator = DemandGenerator(road_map, arrival_rate=arrival_rate, seed=seed)
    sim = SimulationEngine(road_map, drivers, default_speed=8.33)  # ~30 km/h

    matcher = matcher_cls(max_pickup_cost=3000.0)
    matching_callback = partial(run_matcher_on_sim, matcher)

    current_sim_time = 0.0
    all_requests = []

    print(f"\n--- BẮT ĐẦU MÔ PHỎNG: {label} ---")

    while current_sim_time < total_time:
        new_requests = generator.tick(current_sim_time)
        if new_requests:
            sim.add_requests(new_requests)
            all_requests.extend(new_requests)

        sim.step(dt, matching_function=matching_callback)
        current_sim_time += dt

    kpis = sim.get_kpis(len(all_requests))
    kpis["label"] = label
    kpis["drivers"] = drivers
    kpis["requests"] = all_requests
    kpis["sim"] = sim

    # In báo cáo
    print("\n================ TỔNG KẾT MÔ PHỎNG ================")
    print(f"Chiến lược:                   {label}")
    print(f"Tổng số yêu cầu đã phát sinh: {kpis['total_requests']}")
    print(f"Chuyến hoàn thành (COMPLETED): {kpis['completed_requests']} ({kpis['completion_rate']}%)")
    print(f"Chuyến bị hủy (CANCELLED):    {kpis['cancelled_requests']} ({kpis['cancellation_rate']}%)")
    print(f"Chuyến chưa hoàn thành:       {kpis['active_requests'] + kpis['pending_requests']}")
    print(f"Thời gian chờ đón trung bình:  {kpis['avg_wait_time']} giây")
    print(f"Hiệu suất sử dụng tài xế:     {kpis['driver_utilization']}%")
    print(f"Tổng quãng đường di chuyển:   {kpis['total_distance_km']} km")
    print("===================================================")

    return kpis


def run_benchmark_all(road_map: RealRoadMap):
    """Chạy và so sánh cả 4 chiến lược ghép cặp trên cùng một kịch bản nhu cầu."""
    strategies = [
        ("greedy", GreedyMatcher, "S1: Greedy"),
        ("hungarian", HungarianMatcher, "S2: Hungarian"),
        ("pooling", PoolingMatcher, "S3: Greedy+Pool"),
        ("hungarian_pooling", HungarianPoolingMatcher, "S4: HU+Pool"),
    ]

    results = []
    for key, cls, name in strategies:
        res = run_single_simulation(
            road_map=road_map,
            matcher_cls=cls,
            label=name,
            num_drivers=8,
            arrival_rate=0.03,
            total_time=1800.0,
            seed=42,
        )
        results.append(res)

    print("\n" + "=" * 90)
    print("             BẢNG SO SÁNH 4 CHIẾN LƯỢC GHÉP CHUYẾN (HÀ NỘI - HOÀN KIẾM)")
    print("=" * 90)
    header = f"{'Chiến lược':<20} | {'Tổng':<6} | {'Xong':<6} | {'Hủy':<5} | {'Tỷ lệ hủy':<10} | {'Đón TB (s)':<11} | {'Hiệu suất':<10} | {'Tổng km':<8}"
    print(header)
    print("-" * 90)
    for r in results:
        row = f"{r['label']:<20} | {r['total_requests']:<6} | {r['completed_requests']:<6} | {r['cancelled_requests']:<5} | {r['cancellation_rate']:>8.1f}% | {r['avg_wait_time']:>9.1f}s | {r['driver_utilization']:>8.1f}% | {r['total_distance_km']:>7.1f}"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    matcher_name = sys.argv[1].lower() if len(sys.argv) > 1 else "greedy"

    valid_choices = list(MATCHERS.keys()) + ["compare", "benchmark", "all"]
    if matcher_name not in valid_choices:
        sys.exit(f"Thuật toán không hợp lệ: {matcher_name}.\nChọn một trong: {valid_choices}")

    # 1. Tải bản đồ giao thông thực tế Hà Nội
    road_map = RealRoadMap(place_name="Hoan Kiem, Hanoi, Vietnam")

    if matcher_name in ("compare", "benchmark", "all"):
        run_benchmark_all(road_map)
    else:
        matcher_cls, label = MATCHERS[matcher_name]
        result = run_single_simulation(
            road_map=road_map,
            matcher_cls=matcher_cls,
            label=label,
            num_drivers=8,
            arrival_rate=0.03,
            total_time=1800.0,
            seed=42,
        )

        # Xuất bản đồ tương tác Web HTML cho lần chạy
        generate_interactive_html_map(
            road_map,
            result["drivers"],
            result["requests"],
            output_file="simulation_map.html",
        )
