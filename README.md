# HỆ THỐNG ĐIỀU PHỐI VÀ GHÉP CHUYẾN CHO DỊCH VỤ GỌI XE (HÀ NỘI)
> **Đề tài Tiểu luận môn:** Khoa học Dịch vụ (Service Science)  
> **Khu vực mô phỏng thực tế:** Quận Hoàn Kiếm, TP. Hà Nội (Dữ liệu OpenStreetMap: 216 nút giao lộ, 462 đoạn đường)  
> **Giao diện vận hành:** Web Dashboard mô phỏng trung tâm điều phối Grab Operations (Leaflet.js + Chart.js + FastAPI WebSocket)

---

## 1. Tổng quan Kiến trúc Hệ thống

Hệ thống được thiết kế theo kiến trúc module hóa chặt chẽ, phân tách rõ ràng giữa động cơ mô phỏng thời gian thực, thuật toán ghép cặp và giao diện trực quan hóa:

```
KHDV/
├── core_engine/                 # Động cơ mô phỏng không-thời gian (Spatiotemporal Engine)
│   ├── entities.py              # Thực thể: Driver, Request, DriverState, RequestState
│   ├── map_network.py           # Bản đồ OSMnx: tải đồ thị đường phố, tính Dijkstra, nội suy tọa độ
│   ├── demand_generator.py      # Sinh nhu cầu theo quá trình Poisson không đồng nhất
│   └── simulator.py             # Quản lý vòng lặp tick mô phỏng, chuyển động xe, đo lường KPIs
│
├── matching/                    # Các chiến lược ghép cặp (Dispatch & Matching Strategies)
│   ├── base.py                  # Lớp trừu tượng Matcher, memoize_cost_fn, kiểm tra tính hợp lệ
│   ├── models.py                # Mô hình dữ liệu: Driver, Request, Stop, Assignment, MatchResult
│   ├── greedy.py                # Chiến lược 1: Greedy Nearest Matching (Tham lam)
│   ├── hungarian.py             # Chiến lược 2: Batch Matching với thuật toán Hungarian (Kuhn-Munkres)
│   ├── pooling.py               # Chiến lược 3: Greedy Ride-Pooling Matching (Ghép chuyến chung)
│   └── hungarian_pooling.py     # Chiến lược 4: Hungarian Ride-Pooling (Hungary gom lô + Ghép chuyến)
│
├── tests/                       # Bộ kiểm thử tự động (212 unit tests, 100% pass)
│   ├── test_base.py
│   ├── test_greedy.py
│   ├── test_hungarian.py
│   ├── test_pooling.py
│   ├── test_hungarian_pooling.py
│   └── test_node_ids.py
│
├── static/                      # Giao diện Web App điều hành Grab Operations
│   └── index.html               # Web UI tương tác: Leaflet bản đồ Hà Nội, Chart.js, KPI real-time
│
├── adapter.py                   # Adapter nối matcher với SimulationEngine
├── main.py                      # Chạy mô phỏng CLI và bảng so sánh Benchmark
├── server.py                    # FastAPI server cung cấp REST API & WebSocket stream
└── start_server.bat             # File khởi chạy nhanh 1-click cho Windows
```

---

## 2. Bốn Chiến lược Ghép Cặp (Matching Strategies)

### Chiến lược 1: Greedy Nearest Matching (S1)
* **Nguyên lý:** Xử lý yêu cầu theo thứ tự đặt (FIFO). Với mỗi yêu cầu, quét toàn bộ tài xế rảnh hợp lệ (khoảng cách đón $\le 3000\text{m}$) và chọn tài xế có chi phí tiếp cận nhỏ nhất.
* **Độ phức tạp:** $O(R \times D)$ lần gọi hàm chi phí (Dijkstra).
* **Đặc điểm:** Tốc độ tính toán nhanh nhất, phản hồi ngay lập tức, nhưng dễ dẫn đến quyết định cận tối ưu (khách đặt trước cướp tài xế gần, đẩy khách sau vào tài xế rất xa).

### Chiến lược 2: Hungarian Batch Matching (S2)
* **Nguyên lý:** Gom các yêu cầu trong một cửa sổ thời gian thành lô (batch). Xây dựng ma trận chi phí hai phía $C_{ij}$ giữa các yêu cầu và các tài xế rảnh, giải bài toán phân công tối ưu toàn cục bằng thuật toán Kuhn-Munkres:
  $$\min \sum_{i} \sum_{j} C_{ij} x_{ij} \quad \text{s.t.} \quad \sum_j x_{ij} \le 1, \quad \sum_i x_{ij} \le 1$$
* **Độ phức tạp:** $O(k^2 \cdot m)$ với $k = \min(R, D), m = \max(R, D)$.
* **Đặc điểm:** Tối thiểu hóa tổng thời gian chờ đón của toàn bộ khách hàng trong lô, loại bỏ hiện tượng tranh chấp tài xế.

### Chiến lược 3: Greedy Ride-Pooling Matching (S3)
* **Nguyên lý:** Cho phép 1 tài xế phục vụ đồng thời nhiều khách cùng hướng đi (carpooling). Với yêu cầu mới, thử chèn (insertion heuristic) điểm đón và điểm trả vào mọi vị trí hợp lệ trong lộ trình của các tài xế, thỏa mãn:
  1. Số khách trên xe không vượt quá sức chứa (`capacity = 4`).
  2. Thời gian đón của mỗi khách $\le \text{max\_pickup\_cost}$.
  3. Tỷ lệ đi vòng của từng khách không vượt quá $\text{max\_detour\_ratio} = 1.5$ lần quãng đường đi thẳng.
* **Đặc điểm:** Nâng cao rõ rệt hiệu suất sử dụng tài xế, giảm lượng xe cần thiết.

### Chiến lược 4: Hungarian Ride-Pooling Matching (S4)
* **Nguyên lý:** Kết hợp tối ưu toàn cục đợt phân bổ 1-1 ban đầu bằng thuật toán Hungarian, sau đó dùng thuật toán chèn tối ưu (insertion) để ghép các yêu cầu còn lại vào xe đang di chuyển.
* **Đặc điểm:** Kết hợp ưu điểm giảm thời gian đón của Hungarian và nâng cao công suất của Pooling.

---

## 3. Các Chỉ số Đánh giá Hiệu năng (KPIs)

1. **Thời gian chờ đón khách trung bình (Average Pickup Wait Time):**
   $$\overline{W}_{\text{pickup}} = \frac{1}{|C|} \sum_{r \in C} (t_{\text{pickup}}^{(r)} - t_{\text{request}}^{(r)}) \quad (\text{giây})$$
2. **Tỷ lệ chuyến bị hủy (Cancellation Rate):**
   $$\text{CR} = \frac{N_{\text{cancelled}}}{N_{\text{total}}} \times 100\%$$
   *(Khách hàng hủy chuyến nếu sau 300 giây = 5 phút chưa được gán tài xế).*
3. **Hiệu suất sử dụng tài xế (Driver Fleet Utilization):**
   $$U = \frac{\sum_{d \in D} T_{\text{busy}}^{(d)}}{|D| \times T_{\text{sim}}} \times 100\%$$
   *(Tỷ lệ phần trăm thời gian tài xế thực sự bận đón hoặc chở khách trên tổng thời gian vận hành).*
4. **Tổng quãng đường di chuyển và quãng đường chạy rỗng (Total Distance & Deadheading Distance).**

---

## 4. Kết quả Thực nghiệm và So sánh Định lượng

Thực nghiệm trên đồ thị giao thông thực tế Hoàn Kiếm, Hà Nội (mô phỏng 30 phút, 8 tài xế, $\lambda = 0.03\text{ req/s}$, seed = 42):

| Chiến lược | Tổng yêu cầu | Chuyến hoàn thành | Chuyến hủy | Tỷ lệ hủy (%) | Thời gian đón TB (s) | Hiệu suất tài xế (%) | Tổng quãng đường (km) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1: Greedy** | 55 | 44 (80.0%) | 6 | 10.9% | 123.4s | 73.8% | 88.2 km |
| **S2: Hungarian** | 55 | 43 (78.2%) | 6 | 10.9% | **101.6s** (-17.6%) | 69.9% | **83.5 km** (-5.3%) |
| **S3: Greedy+Pool** | 55 | 43 (78.2%) | 6 | 10.9% | 102.4s | 70.6% | 84.3 km |
| **S4: HU+Pool** | 55 | 44 (80.0%) | 6 | 10.9% | 112.2s | 70.1% | 83.7 km |

> **Phân tích kết quả:**
> - **Hungarian (S2)** giúp rút ngắn thời gian chờ đón trung bình của khách hàng từ **123.4s xuống 101.6s** (giảm **17.6%** thời gian lãng phí). Đồng thời giảm quãng đường chạy rỗng không tải từ 88.2 km xuống 83.5 km (tiết kiệm nhiên liệu và giảm phát thải).
> - **Ride-Pooling (S3, S4)** cho phép phục vụ lưu lượng nhu cầu cao hơn trong giờ cao điểm khi số lượng xe rảnh bị giới hạn.

---

## 5. Hướng dẫn Sử dụng

### Bước 0: Cài đặt thư viện (làm một lần)
```bash
py -m pip install -r requirements.txt
```
> Trên Windows nên dùng `py` thay cho `python` (lệnh `python` có thể trỏ nhầm sang một bản Python khác chưa cài thư viện). `start_server.bat` tự chọn `py` và tự cài thư viện còn thiếu.
> Giao diện web nằm ở `static/index.html`; file này phải có trong repo thì trang chủ mới hiện giao diện (nếu thiếu, trang chủ chỉ trả về một dòng JSON).

### Cách 1: Chạy Giao diện Web Điều hành (Khuyến nghị)
1. Nhấp đúp chuột vào file **`start_server.bat`** (hoặc chạy lệnh: `py server.py`).
2. Mở trình duyệt web truy cập: **`http://localhost:8000`**
3. Các tính năng trên giao diện:
   - Bản đồ số Hà Nội với xe di chuyển theo thời gian thực dọc theo các tuyến phố.
   - Chuyển đổi linh hoạt giữa 4 chiến lược ghép chuyến.
   - Tùy chỉnh số lượng tài xế (3 - 30 xe), cường độ Poisson $\lambda$, và thanh trượt tốc độ (1x - 10x).
   - Nút **"📊 So sánh 4 chiến lược"** để chạy tự động và xem bảng đối sánh trực quan.

### Cách 2: Chạy Dòng lệnh (CLI)
Chạy so sánh đồng thời cả 4 chiến lược:
```bash
python main.py compare
```

Chạy từng chiến lược cụ thể:
```bash
python main.py greedy
python main.py hungarian
python main.py pooling
python main.py hungarian_pooling
```

### Cách 3: Chạy Kiểm thử Unit Test
```bash
python -m pytest tests/
```
*(Toàn bộ 212 tests xác thực tính toàn vẹn của đồ thị, thuật toán gán, và các ràng buộc ghép chuyến).*
