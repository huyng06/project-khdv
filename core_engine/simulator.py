from typing import List, Dict, Callable, Optional, Tuple
from core_engine.entities import Driver, Request, RequestState, DriverState
from core_engine.map_network import RealRoadMap

class SimulationEngine:
    def __init__(self, road_map: RealRoadMap, drivers: List[Driver], default_speed: float = 8.33):
        """
        road_map: Đồ thị bản đồ thực tế từ OpenStreetMap.
        drivers: Danh sách tài xế tham gia mô phỏng.
        default_speed: Vận tốc di chuyển mặc định (m/s). 8.33 m/s ~ 30 km/h trong nội thành.
        """
        self.map = road_map
        self.drivers = {d.driver_id: d for d in drivers}
        self.pending_requests: List[Request] = []
        self.active_requests: Dict[str, Request] = {}
        self.completed_requests: List[Request] = []
        self.cancelled_requests: List[Request] = []
        self.current_time = 0.0
        self.default_speed = default_speed
        
        # Lưu quãng đường đã đi được trên đoạn đường (u, v) hiện tại của từng tài xế
        self.driver_edge_progress: Dict[str, float] = {d.driver_id: 0.0 for d in drivers}

    def add_requests(self, new_requests: List[Request]):
        self.pending_requests.extend(new_requests)

    def step(self, dt: float, matching_function: Optional[Callable] = None):
        """
        Tiến hành 1 bước mô phỏng (dt giây).
        """
        self.current_time += dt

        # 1. Hủy chuyến nếu khách hàng chờ gán xe quá thời hạn max_wait_time
        remaining_pending = []
        for req in self.pending_requests:
            if self.current_time - req.request_time > req.max_wait_time:
                req.status = RequestState.CANCELLED
                self.cancelled_requests.append(req)
            else:
                remaining_pending.append(req)
        self.pending_requests = remaining_pending

        # 2. Thực thi thuật toán ghép cặp của Thành viên 2 (nếu có)
        if matching_function and self.pending_requests:
            idle_drivers = [d for d in self.drivers.values() if d.status == DriverState.IDLE]
            if idle_drivers:
                matches = matching_function(idle_drivers, self.pending_requests, self.map, self.current_time)
                
                for driver_id, req_id in matches:
                    if isinstance(req_id, (list, tuple)):    # ghép chuyến chung: lộ trình nhiều điểm dừng
                        self._assign_stops(driver_id, req_id)
                        continue
                    driver = self.drivers.get(driver_id)
                    req = next((r for r in self.pending_requests if r.request_id == req_id), None)
                    if not driver or not req:
                        continue

                    # Thiết lập trạng thái ghép chuyến
                    driver.status = DriverState.EN_ROUTE_PICKUP
                    driver.assigned_request = req
                    
                    # Tìm tuyến đường ngắn nhất từ vị trí hiện tại tới điểm đón
                    path_to_pickup = self.map.get_shortest_path(driver.current_node, req.origin_node)
                    driver.current_route = path_to_pickup
                    
                    req.status = RequestState.MATCHED
                    req.matched_driver_id = driver_id
                    
                    self.active_requests[req.request_id] = req
                    self.pending_requests.remove(req)
                    self.driver_edge_progress[driver_id] = 0.0

        # 3. Cập nhật vị trí di chuyển và đo lường thời gian bận của tài xế
        for driver in self.drivers.values():
            if driver.status != DriverState.IDLE:
                driver.total_busy_time += dt
                if driver.status == DriverState.OCCUPIED:
                    driver.total_passenger_time += dt

            if driver.status == DriverState.IDLE or not driver.current_route:
                continue

            distance_to_move = self.default_speed * dt
            self._move_driver_along_route(driver, distance_to_move)

    def _move_driver_along_route(self, driver: Driver, distance_to_move: float):
        """
        Duyệt chuyển động của tài xế qua các giao lộ (Node-by-Node) trên tuyến đường.
        """
        remaining_move = distance_to_move

        while remaining_move > 0 and len(driver.current_route) > 1:
            u = driver.current_route[0]
            v = driver.current_route[1]

            # Lấy chiều dài (mét) của đoạn đường giữa 2 giao lộ u và v
            edge_data = self.map.graph.get_edge_data(u, v)
            if edge_data:
                # Trường hợp MultiDiGraph lấy cạnh đầu tiên
                first_key = list(edge_data.keys())[0]
                edge_len = edge_data[first_key].get('length', 50.0)
            else:
                edge_len = 50.0

            current_progress = self.driver_edge_progress[driver.driver_id]
            dist_needed_to_v = edge_len - current_progress

            if remaining_move >= dist_needed_to_v:
                # Xe đi qua điểm v thành công
                remaining_move -= dist_needed_to_v
                driver.current_node = v
                driver.current_route.pop(0)  # Loại bỏ node u vừa đi qua
                self.driver_edge_progress[driver.driver_id] = 0.0
            else:
                # Xe đang di chuyển dở trên đoạn (u, v)
                self.driver_edge_progress[driver.driver_id] += remaining_move
                remaining_move = 0.0

        # Đo quãng đường thực tế đã di chuyển trong tick này
        actual_moved = distance_to_move - remaining_move
        driver.total_distance += actual_moved
        if driver.status == DriverState.OCCUPIED:
            driver.total_passenger_distance += actual_moved

        # Khi danh sách tuyến đường chỉ còn 1 node -> Đã tới điểm mục tiêu
        if len(driver.current_route) <= 1:
            self._on_reach_destination(driver)

    def _on_reach_destination(self, driver: Driver):
        if driver.stop_queue:                       # đang chạy lộ trình nhiều điểm dừng
            self._on_reach_stop(driver)
            return
        req = driver.assigned_request
        if not req:
            driver.status = DriverState.IDLE
            driver.current_route = []
            return

        if driver.status == DriverState.EN_ROUTE_PICKUP:
            # Vừa tới điểm đón -> Chuyển sang chở khách tới điểm trả
            driver.status = DriverState.OCCUPIED
            req.status = RequestState.PICKED_UP
            req.pickup_time = self.current_time
            
            # Tính đường đi từ điểm đón tới điểm trả
            route_to_dest = self.map.get_shortest_path(req.origin_node, req.destination_node)
            driver.current_route = route_to_dest
            self.driver_edge_progress[driver.driver_id] = 0.0
            
        elif driver.status == DriverState.OCCUPIED:
            # Vừa tới điểm trả -> Hoàn thành chuyến xe
            req.status = RequestState.COMPLETED
            req.completion_time = self.current_time
            self.completed_requests.append(req)
            if req.request_id in self.active_requests:
                del self.active_requests[req.request_id]
            
            driver.status = DriverState.IDLE
            driver.assigned_request = None
            driver.current_route = []
            self.driver_edge_progress[driver.driver_id] = 0.0

    # ------------------------------------------------------------------ ghép chuyến chung (Ride-Pooling)
    def _assign_stops(self, driver_id: str, stops: list):
        """stops: [(request_id, "pickup"|"dropoff"), ...] theo đúng thứ tự thuật toán đã chọn."""
        driver = self.drivers.get(driver_id)
        pending = {r.request_id: r for r in self.pending_requests}
        if not driver or not stops or driver.status != DriverState.IDLE \
                or any(rid not in pending for rid, _ in stops):
            return

        driver.stop_queue = [
            (pending[rid], kind, pending[rid].origin_node if kind == "pickup" else pending[rid].destination_node)
            for rid, kind in stops
        ]
        driver.assigned_request = None
        driver.status = DriverState.EN_ROUTE_PICKUP
        for rid in {rid for rid, _ in stops}:
            req = pending[rid]
            req.status = RequestState.MATCHED
            req.matched_driver_id = driver_id
            self.active_requests[rid] = req
            self.pending_requests.remove(req)
        self._head_to_next_stop(driver)

    def _head_to_next_stop(self, driver: Driver):
        _, _, node = driver.stop_queue[0]
        driver.current_route = self.map.get_shortest_path(driver.current_node, node)
        self.driver_edge_progress[driver.driver_id] = 0.0

    def _on_reach_stop(self, driver: Driver):
        req, kind, _ = driver.stop_queue.pop(0)
        if kind == "pickup":
            req.status = RequestState.PICKED_UP
            req.pickup_time = self.current_time
        else:
            req.status = RequestState.COMPLETED
            req.completion_time = self.current_time
            self.completed_requests.append(req)
            self.active_requests.pop(req.request_id, None)

        if not driver.stop_queue:
            driver.status = DriverState.IDLE
            driver.current_route = []
            self.driver_edge_progress[driver.driver_id] = 0.0
            return
        onboard = any(r.status == RequestState.PICKED_UP for r, _, _ in driver.stop_queue)
        driver.status = DriverState.OCCUPIED if onboard else DriverState.EN_ROUTE_PICKUP
        self._head_to_next_stop(driver)

    # ------------------------------------------------------------------ Hỗ trợ Web UI & Đo lường KPI
    def get_driver_latlon(self, driver_id: str) -> Tuple[float, float]:
        """Lấy tọa độ chính xác thời gian thực của tài xế (nội suy nếu đang di chuyển giữa 2 node)."""
        driver = self.drivers.get(driver_id)
        if not driver:
            return (21.0285, 105.8542)
        if len(driver.current_route) >= 2:
            u = driver.current_route[0]
            v = driver.current_route[1]
            lat_u, lon_u = self.map.get_node_latlon(u)
            lat_v, lon_v = self.map.get_node_latlon(v)
            edge_data = self.map.graph.get_edge_data(u, v)
            first_key = list(edge_data.keys())[0] if edge_data else None
            edge_len = edge_data[first_key].get('length', 50.0) if first_key else 50.0
            progress = self.driver_edge_progress.get(driver_id, 0.0)
            fraction = min(max(progress / edge_len, 0.0), 1.0) if edge_len > 0 else 0.0
            return (lat_u + (lat_v - lat_u) * fraction, lon_u + (lon_v - lon_u) * fraction)
        return self.map.get_node_latlon(driver.current_node)

    def get_kpis(self, total_generated: Optional[int] = None) -> dict:
        """Tính toán đầy đủ các chỉ số KPI theo yêu cầu của đề tài."""
        total = total_generated if total_generated is not None else (
            len(self.completed_requests) + len(self.cancelled_requests) + len(self.active_requests) + len(self.pending_requests)
        )
        completed = len(self.completed_requests)
        cancelled = len(self.cancelled_requests)
        
        waits = [r.pickup_time - r.request_time for r in self.completed_requests if r.pickup_time is not None]
        avg_wait = sum(waits) / len(waits) if waits else 0.0
        
        cancel_rate = (cancelled / total * 100.0) if total > 0 else 0.0
        completion_rate = (completed / total * 100.0) if total > 0 else 0.0
        
        # Driver utilization: tổng thời gian bận / (số tài xế * tổng thời gian mô phỏng)
        num_drivers = len(self.drivers)
        max_possible_time = num_drivers * self.current_time if self.current_time > 0 and num_drivers > 0 else 1.0
        total_busy = sum(d.total_busy_time for d in self.drivers.values())
        utilization = min(100.0, (total_busy / max_possible_time) * 100.0) if max_possible_time > 0 else 0.0
        
        total_dist_km = sum(d.total_distance for d in self.drivers.values()) / 1000.0
        pax_dist_km = sum(d.total_passenger_distance for d in self.drivers.values()) / 1000.0
        
        return {
            "total_requests": total,
            "completed_requests": completed,
            "cancelled_requests": cancelled,
            "active_requests": len(self.active_requests),
            "pending_requests": len(self.pending_requests),
            "cancellation_rate": round(cancel_rate, 2),
            "completion_rate": round(completion_rate, 2),
            "avg_wait_time": round(avg_wait, 2),
            "driver_utilization": round(utilization, 2),
            "total_distance_km": round(total_dist_km, 2),
            "passenger_distance_km": round(pax_dist_km, 2),
            "current_time": round(self.current_time, 1),
        }
