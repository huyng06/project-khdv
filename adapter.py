from math import inf
from typing import Dict, List, Tuple

from matching.base import Matcher
from matching.models import Driver as AlgoDriver, Request as AlgoRequest
from core_engine.entities import Driver, Request
from core_engine.map_network import RealRoadMap


def create_osm_cost_fn(road_map: RealRoadMap):
    """
    Tạo hàm cost_fn tính khoảng cách thực tế (mét) giữa 2 Node trên bản đồ OpenStreetMap.
    Có lưu Cache để không phải tính lại Dijkstra nhiều lần cho cùng một cặp Node.
    """
    cost_cache: Dict[Tuple[int, int], float] = {}

    def cost_fn(node_a: int, node_b: int) -> float:
        if node_a == node_b:
            return 0.0

        pair = (node_a, node_b)
        if pair in cost_cache:
            return cost_cache[pair]

        path = road_map.get_shortest_path(node_a, node_b)
        if not path:
            cost_cache[pair] = inf
            return inf

        dist = road_map.get_path_distance(path)
        cost_cache[pair] = dist
        return dist

    return cost_fn


def get_cost_fn(road_map: RealRoadMap):
    """Mỗi bản đồ chỉ tạo cost_fn MỘT lần và dùng lại ở mọi bước mô phỏng.

    Trước đây cost_fn (và cache của nó) được tạo mới ở mỗi lần gọi matcher, tức mỗi giây mô phỏng,
    nên cache bị bỏ đi liên tục và Dijkstra bị tính lại cho cùng một cặp node hàng nghìn lần.
    Đồ thị đường không đổi trong suốt mô phỏng nên giữ cache lâu dài là an toàn.
    """
    if not hasattr(road_map, "_osm_cost_fn"):
        road_map._osm_cost_fn = create_osm_cost_fn(road_map)
    return road_map._osm_cost_fn


def run_matcher_on_sim(
    matcher: Matcher,
    idle_drivers: List[Driver],
    pending_requests: List[Request],
    road_map: RealRoadMap,
    current_time: float
) -> List[Tuple[str, str]]:
    """
    Hàm Wrapper bọc thuật toán để truyền trực tiếp vào `sim.step()`.

    Trả về: Danh sách các cặp [(driver_id, request_id), ...]
    Với Greedy/Hungarian phần tử là (driver_id, request_id). Với Pooling (một tài xế nhiều khách)
    phần tử là (driver_id, [(request_id, "pickup"|"dropoff"), ...]) theo đúng thứ tự lộ trình.
    """
    algo_drivers = [
        AlgoDriver(
            id=d.driver_id,
            location=d.current_node,
            capacity=d.capacity,                 # lấy theo tài xế thật, không gán cứng 4
            available=True
        )
        for d in idle_drivers
    ]

    algo_requests = [
        AlgoRequest(
            id=r.request_id,
            pickup=r.origin_node,
            dropoff=r.destination_node,
            time=r.request_time,
            passengers=r.num_passengers          # lấy theo yêu cầu thật, không gán cứng 1
        )
        for r in pending_requests
    ]

    cost_fn = get_cost_fn(road_map)
    match_result = matcher.match(algo_requests, algo_drivers, cost_fn)

    sim_matches = []
    for assignment in match_result.assignments:
        ids = assignment.request_ids
        if len(ids) == 1:                       # Greedy / Hungarian: mỗi tài xế một khách
            sim_matches.append((assignment.driver_id, ids[0]))
        else:                                   # Pooling: giữ nguyên thứ tự đón/trả
            stops = [(s.request_id, s.type.value) for s in assignment.stops]
            sim_matches.append((assignment.driver_id, stops))

    return sim_matches
