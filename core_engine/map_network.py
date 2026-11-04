import osmnx as ox
import networkx as nx
import random
from typing import Tuple, List, Optional

class RealRoadMap:
    def __init__(self, place_name: str = "Hoan Kiem, Hanoi, Vietnam", network_type: str = "drive"):
        """
        Tải bản đồ giao thông thực tế từ OpenStreetMap.
        Các ví dụ place_name:
        - "Hoan Kiem, Hanoi, Vietnam"
        - "District 1, Ho Chi Minh City, Vietnam"
        """
        print(f"Đang tải bản đồ giao thông cho: {place_name}...")
        # 1. Tải đồ thị dạng tọa độ địa lý (Lat, Lon)
        self.graph_latlon = ox.graph_from_place(place_name, network_type=network_type)
        
        # 2. Chuyển đổi đồ thị sang hệ tọa độ chiếu UTM (đơn vị đo: mét) để tính khoảng cách chuẩn xác
        self.graph = ox.project_graph(self.graph_latlon)
        
        # 3. Thêm thông tin vận tốc tối đa và thời gian di chuyển ước tính cho từng cạnh
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)
        
        # Danh sách tất cả các nút (giao lộ) trên bản đồ
        self.nodes = list(self.graph.nodes())
        print(f"-> Tải bản đồ thành công! Tổng số giao lộ (Nodes): {len(self.nodes)}, Số đoạn đường (Edges): {len(self.graph.edges())}")

    def get_random_node(self) -> int:
        """Lấy ngẫu nhiên một nút giao thông làm điểm đón/trả khách."""
        return random.choice(self.nodes)

    def get_shortest_path(self, source_node: int, target_node: int) -> List[int]:
        """Tìm danh sách các nút giao lộ trên đường đi ngắn nhất giữa 2 điểm."""
        try:
            return nx.shortest_path(self.graph, source=source_node, target=target_node, weight='length')
        except nx.NetworkXNoPath:
            return []

    def get_path_distance(self, path_nodes: List[int]) -> float:
        """Tính tổng khoảng cách di chuyển (met) của một tuyến đường."""
        if len(path_nodes) < 2:
            return 0.0
        return sum(
            self.graph.edges[path_nodes[i], path_nodes[i+1], 0]['length']
            for i in range(len(path_nodes) - 1)
        )

    def get_node_latlon(self, node_id: int) -> Tuple[float, float]:
        """Lấy tọa độ (Vĩ độ - Latitude, Kinh độ - Longitude) để vẽ lên bản đồ Folium."""
        node_data = self.graph_latlon.nodes[node_id]
        return (node_data['y'], node_data['x'])  # (lat, lon)
    
    def get_route_geometry(self, path_nodes: List[int]) -> List[Tuple[float, float]]:
        """
        Trích xuất danh sách tọa độ (lat, lon) chi tiết bám theo từng khúc uốn lượn của đường phố.
        """
        if not path_nodes or len(path_nodes) < 2:
            return []

        route_coords = []
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            # Lấy dữ liệu cạnh giữa nút u và v trên đồ thị gốc (Lat/Lon)
            edge_data = self.graph_latlon.get_edge_data(u, v)
            if edge_data:
                # Lấy cạnh đầu tiên trong MultiDiGraph
                first_key = list(edge_data.keys())[0]
                data = edge_data[first_key]
                
                # Nếu cạnh có thuộc tính geometry (đường cong chi tiết)
                if 'geometry' in data:
                    xs, ys = data['geometry'].xy
                    # xs là lon, ys là lat -> đổi thành (lat, lon)
                    segment_coords = list(zip(ys, xs))
                    route_coords.extend(segment_coords[:-1]) # Bỏ điểm cuối để không trùng với nút tiếp theo
                else:
                    route_coords.append(self.get_node_latlon(u))
            else:
                route_coords.append(self.get_node_latlon(u))

        # Thêm tọa độ nút cuối cùng
        route_coords.append(self.get_node_latlon(path_nodes[-1]))
        return route_coords