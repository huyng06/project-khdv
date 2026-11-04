"""
FastAPI Server cho Ride-Hailing Platform Simulation (Hà Nội - Hoàn Kiếm)
Hỗ trợ cả 4 chiến lược ghép cặp:
  S1: Greedy (Tham lam gần nhất)
  S2: Hungarian (Gom lô tối ưu toàn cục)
  S3: Greedy Pooling (Ghép chuyến chung)
  S4: Hungarian Pooling (Hungary gom lô + Ghép chuyến)

Expose REST API & WebSocket streaming thời gian thực cho Web UI.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Thêm thư mục hiện tại vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core_engine.entities import Driver, Request, DriverState, RequestState
from core_engine.map_network import RealRoadMap
from core_engine.demand_generator import DemandGenerator
from core_engine.simulator import SimulationEngine
from adapter import run_matcher_on_sim, get_cost_fn
from matching import (
    GreedyMatcher,
    HungarianMatcher,
    PoolingMatcher,
    HungarianPoolingMatcher,
)

# ============================================================
# Cấu hình Matchers
# ============================================================
MATCHERS = {
    "greedy": (lambda: GreedyMatcher(max_pickup_cost=3000.0), "S1: Greedy (Tham lam)"),
    "hungarian": (lambda: HungarianMatcher(max_pickup_cost=3000.0), "S2: Hungarian (Gom lô tối ưu)"),
    "pooling": (lambda: PoolingMatcher(max_pickup_cost=3000.0, max_detour_ratio=1.5), "S3: Greedy Pooling (Ghép chuyến)"),
    "greedy_pool": (lambda: PoolingMatcher(max_pickup_cost=3000.0, max_detour_ratio=1.5), "S3: Greedy Pooling (Ghép chuyến)"),
    "hungarian_pooling": (lambda: HungarianPoolingMatcher(max_pickup_cost=3000.0, max_detour_ratio=1.5), "S4: Hungarian Pooling (Hungary + Ghép chuyến)"),
    "hungarian_pool": (lambda: HungarianPoolingMatcher(max_pickup_cost=3000.0, max_detour_ratio=1.5), "S4: Hungarian Pooling (Hungary + Ghép chuyến)"),
}

# ============================================================
# Trạng thái toàn cục
# ============================================================
road_map: Optional[RealRoadMap] = None
map_nodes_cache: Optional[List[dict]] = None

sim_state = {
    "running": False,
    "paused": False,
    "strategy": "greedy",
    "strategy_name": "S1: Greedy",
    "num_drivers": 10,
    "arrival_rate": 0.04,
    "speed_multiplier": 2.0,
    "current_time": 0.0,
}

active_connections: Set[WebSocket] = set()
sim_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
pause_event = threading.Event()
sim_engine_ref: Optional[SimulationEngine] = None
all_requests_ref: List[Request] = []

STATIC_DIR = os.path.join(CURRENT_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global road_map, map_nodes_cache
    print("\n[Server] Đang nạp bản đồ giao thông Hoàn Kiếm, Hà Nội...")
    try:
        road_map = RealRoadMap(place_name="Hoan Kiem, Hanoi, Vietnam")
        # Cache danh sách tọa độ nodes để gửi cho frontend
        map_nodes_cache = []
        for node in road_map.nodes:
            lat, lon = road_map.get_node_latlon(node)
            map_nodes_cache.append({"id": node, "lat": lat, "lon": lon})
        print(f"[Server] ✓ Bản đồ sẵn sàng: {len(map_nodes_cache)} giao lộ.")
    except Exception as e:
        print(f"[Server] ⚠ Lỗi nạp bản đồ: {e}")
        road_map = None
    yield

app = FastAPI(title="Ride-Hailing Platform - Hanoi Simulator", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Snapshot Builder
# ============================================================
def build_snapshot(sim: SimulationEngine, all_requests: List[Request]) -> dict:
    """Tạo gói dữ liệu đồng bộ thời gian thực gửi qua WebSocket cho Web UI."""
    driver_data = []
    idle_count = 0
    pickup_count = 0
    occupied_count = 0

    for d in sim.drivers.values():
        lat, lon = sim.get_driver_latlon(d.driver_id)
        status_val = d.status.value
        if status_val == DriverState.IDLE.value:
            idle_count += 1
        elif status_val == DriverState.EN_ROUTE_PICKUP.value:
            pickup_count += 1
        else:
            occupied_count += 1

        route_coords = []
        if len(d.current_route) > 1:
            route_coords = [sim.map.get_node_latlon(node) for node in d.current_route[:8]]

        passengers_onboard = []
        if d.assigned_request:
            passengers_onboard.append(d.assigned_request.request_id)
        if d.stop_queue:
            for r, kind, _ in d.stop_queue:
                if kind == "dropoff" and r.request_id not in passengers_onboard:
                    passengers_onboard.append(r.request_id)

        # Vận tốc di chuyển thực tế (km/h)
        # Khi IDLE (dừng đỗ): 0 km/h; khi di chuyển (đón/chở khách): default_speed (~30 km/h)
        current_speed_kmh = 0.0 if status_val == DriverState.IDLE.value else round(sim.default_speed * 3.6, 1)

        driver_data.append({
            "id": d.driver_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "status": status_val,
            "speed_kmh": current_speed_kmh,
            "passengers": passengers_onboard,
            "route": route_coords,
            "busy_time": round(d.total_busy_time, 1),
            "distance_km": round(d.total_distance / 1000.0, 2),
        })

    pending_data = []
    for r in sim.pending_requests:
        lat, lon = sim.map.get_node_latlon(r.origin_node)
        dlat, dlon = sim.map.get_node_latlon(r.destination_node)
        pending_data.append({
            "id": r.request_id,
            "pickup_lat": round(lat, 6),
            "pickup_lon": round(lon, 6),
            "dropoff_lat": round(dlat, 6),
            "dropoff_lon": round(dlon, 6),
            "wait_time": round(sim.current_time - r.request_time, 1),
        })

    active_data = []
    for r in sim.active_requests.values():
        lat, lon = sim.map.get_node_latlon(r.origin_node)
        dlat, dlon = sim.map.get_node_latlon(r.destination_node)
        active_data.append({
            "id": r.request_id,
            "driver_id": r.matched_driver_id,
            "pickup_lat": round(lat, 6),
            "pickup_lon": round(lon, 6),
            "dropoff_lat": round(dlat, 6),
            "dropoff_lon": round(dlon, 6),
            "status": r.status.value,
        })

    kpis = sim.get_kpis(len(all_requests))

    return {
        "type": "tick",
        "time": round(sim.current_time, 1),
        "drivers": driver_data,
        "pending": pending_data,
        "active": active_data,
        "fleet": {
            "idle": idle_count,
            "pickup": pickup_count,
            "occupied": occupied_count,
            "total": len(sim.drivers),
        },
        "kpis": kpis,
    }


# ============================================================
# WebSocket Broadcast
# ============================================================
async def broadcast(message: dict):
    if not active_connections:
        return
    text = json.dumps(message, ensure_ascii=False)
    dead = set()
    for ws in list(active_connections):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


# ============================================================
# Vòng lặp mô phỏng chạy ngầm (Background Simulation Thread)
# ============================================================
def simulation_worker(strategy_key: str, num_drivers: int, arrival_rate: float, speed_mult: float, seed: int = 42):
    global road_map, sim_engine_ref, all_requests_ref

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event.clear()
    pause_event.clear()
    sim_state["running"] = True
    sim_state["paused"] = False

    matcher_factory, label = MATCHERS.get(strategy_key, MATCHERS["greedy"])
    matcher = matcher_factory()
    matching_callback = partial(run_matcher_on_sim, matcher)

    # Seed tái lập
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

    drivers = [
        Driver(driver_id=f"D_{i+1:02d}", current_node=road_map.get_random_node())
        for i in range(num_drivers)
    ]

    generator = DemandGenerator(road_map, arrival_rate=arrival_rate, seed=seed)
    sim = SimulationEngine(road_map, drivers, default_speed=8.33)  # 30 km/h
    sim_engine_ref = sim
    all_requests_ref = []

    dt = 1.0
    total_time = 3600.0  # Tối đa 1 tiếng mô phỏng
    last_broadcast_sim_time = 0.0

    loop.run_until_complete(broadcast({
        "type": "sim_started",
        "strategy": strategy_key,
        "strategy_name": label,
        "num_drivers": num_drivers,
        "arrival_rate": arrival_rate,
    }))

    while sim.current_time < total_time and not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.1)
            continue

        # Sinh yêu cầu mới
        new_reqs = generator.tick(sim.current_time)
        if new_reqs:
            sim.add_requests(new_reqs)
            all_requests_ref.extend(new_reqs)
            for r in new_reqs:
                plat, plon = road_map.get_node_latlon(r.origin_node)
                dlat, dlon = road_map.get_node_latlon(r.destination_node)
                loop.run_until_complete(broadcast({
                    "type": "event_new_request",
                    "id": r.request_id,
                    "pickup": [plat, plon],
                    "dropoff": [dlat, dlon],
                    "time": round(r.request_time, 1),
                }))

        # Bước mô phỏng
        prev_completed = len(sim.completed_requests)
        prev_cancelled = len(sim.cancelled_requests)

        sim.step(dt, matching_function=matching_callback)

        # Bắn sự kiện chuyến hoàn thành / bị hủy nếu có thay đổi
        if len(sim.completed_requests) > prev_completed:
            for r in sim.completed_requests[prev_completed:]:
                loop.run_until_complete(broadcast({
                    "type": "event_trip_completed",
                    "id": r.request_id,
                    "driver_id": r.matched_driver_id,
                }))
        if len(sim.cancelled_requests) > prev_cancelled:
            for r in sim.cancelled_requests[prev_cancelled:]:
                loop.run_until_complete(broadcast({
                    "type": "event_trip_cancelled",
                    "id": r.request_id,
                }))

        # Gửi snapshot mỗi 1.0 giây thời gian mô phỏng
        if sim.current_time - last_broadcast_sim_time >= 1.0:
            snapshot = build_snapshot(sim, all_requests_ref)
            loop.run_until_complete(broadcast(snapshot))
            last_broadcast_sim_time = sim.current_time

        # Điều khiển tốc độ thực
        current_mult = sim_state["speed_multiplier"]
        real_sleep = max(0.005, (dt / max(current_mult, 0.1)) * 0.05)
        time.sleep(real_sleep)

    # Kết thúc
    final_snapshot = build_snapshot(sim, all_requests_ref)
    final_snapshot["type"] = "sim_finished"
    loop.run_until_complete(broadcast(final_snapshot))

    sim_state["running"] = False
    sim_state["paused"] = False
    loop.close()


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
async def get_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "ok", "app": "Ride-Hailing Simulator Hanoi"})


@app.get("/api/status")
async def get_status():
    return {
        "map_loaded": road_map is not None,
        "nodes_count": len(map_nodes_cache) if map_nodes_cache else 0,
        "running": sim_state["running"],
        "paused": sim_state["paused"],
        "strategy": sim_state["strategy"],
        "speed_multiplier": sim_state["speed_multiplier"],
        "place": "Hoàn Kiếm, Hà Nội",
    }


@app.get("/api/map-data")
async def get_map_data():
    if not map_nodes_cache:
        return JSONResponse({"error": "Bản đồ chưa sẵn sàng"}, status_code=503)
    return {
        "center": [21.0285, 105.8542],
        "zoom": 14,
        "nodes": map_nodes_cache,
    }


@app.post("/api/start")
async def start_simulation(payload: dict):
    global sim_thread
    if sim_state["running"]:
        return JSONResponse({"error": "Mô phỏng đang chạy"}, status_code=400)
    if not road_map:
        return JSONResponse({"error": "Bản đồ chưa được tải"}, status_code=503)

    strategy = payload.get("strategy", "greedy")
    num_drivers = int(payload.get("num_drivers", 10))
    arrival_rate = float(payload.get("arrival_rate", 0.04))
    speed_mult = float(payload.get("speed_multiplier", 2.0))

    if strategy not in MATCHERS:
        strategy = "greedy"

    sim_state.update({
        "strategy": strategy,
        "strategy_name": MATCHERS[strategy][1],
        "num_drivers": num_drivers,
        "arrival_rate": arrival_rate,
        "speed_multiplier": speed_mult,
    })

    sim_thread = threading.Thread(
        target=simulation_worker,
        args=(strategy, num_drivers, arrival_rate, speed_mult, int(time.time())),
        daemon=True,
    )
    sim_thread.start()
    return {"status": "started", "strategy": strategy}


@app.post("/api/stop")
async def stop_simulation():
    if not sim_state["running"]:
        return {"status": "already_stopped"}
    stop_event.set()
    return {"status": "stopping"}


@app.post("/api/pause")
async def pause_simulation():
    if not sim_state["running"]:
        return {"status": "not_running"}
    if pause_event.is_set():
        pause_event.clear()
        sim_state["paused"] = False
        return {"status": "resumed"}
    else:
        pause_event.set()
        sim_state["paused"] = True
        return {"status": "paused"}


@app.post("/api/set-speed")
async def set_speed(payload: dict):
    mult = float(payload.get("multiplier", 1.0))
    sim_state["speed_multiplier"] = max(0.5, min(mult, 20.0))
    return {"speed_multiplier": sim_state["speed_multiplier"]}


@app.post("/api/benchmark")
async def run_benchmark_api(payload: dict):
    """Chạy benchmark 4 chiến lược trên server và trả về kết quả định lượng so sánh."""
    if sim_state["running"]:
        return JSONResponse({"error": "Vui lòng dừng mô phỏng hiện tại trước khi chạy benchmark"}, status_code=400)
    if not road_map:
        return JSONResponse({"error": "Bản đồ chưa sẵn sàng"}, status_code=503)

    num_drivers = int(payload.get("num_drivers", 8))
    arrival_rate = float(payload.get("arrival_rate", 0.03))
    sim_seconds = float(payload.get("sim_seconds", 900.0))  # 15 phút mô phỏng nhanh
    seed = int(payload.get("seed", 42))

    from main import run_single_simulation
    strategies = [
        ("greedy", GreedyMatcher, "S1: Greedy"),
        ("hungarian", HungarianMatcher, "S2: Hungarian"),
        ("pooling", PoolingMatcher, "S3: Greedy+Pool"),
        ("hungarian_pooling", HungarianPoolingMatcher, "S4: HU+Pool"),
    ]

    results = []
    for key, cls, name in strategies:
        kpis = run_single_simulation(
            road_map=road_map,
            matcher_cls=cls,
            label=name,
            num_drivers=num_drivers,
            arrival_rate=arrival_rate,
            total_time=sim_seconds,
            seed=seed,
            dt=1.0,
        )
        results.append({
            "key": key,
            "name": name,
            "total_requests": kpis["total_requests"],
            "completed_requests": kpis["completed_requests"],
            "cancelled_requests": kpis["cancelled_requests"],
            "completion_rate": kpis["completion_rate"],
            "cancellation_rate": kpis["cancellation_rate"],
            "avg_wait_time": kpis["avg_wait_time"],
            "driver_utilization": kpis["driver_utilization"],
            "total_distance_km": kpis["total_distance_km"],
        })

    return {"results": results}


@app.websocket("/ws/sim")
async def websocket_simulation(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "map_loaded": road_map is not None,
            "running": sim_state["running"],
            "strategy": sim_state["strategy"],
        }))
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        active_connections.discard(websocket)
    except Exception:
        active_connections.discard(websocket)


if __name__ == "__main__":
    print("=" * 70)
    print("  RIDE-HAILING SIMULATOR - HỆ THỐNG ĐIỀU PHỐI GỌI XE HÀ NỘI")
    print("  Truy cập giao diện: http://localhost:8000")
    print("=" * 70)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
