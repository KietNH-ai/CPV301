# Giải Thích Chi Tiết Hệ Thống Tự Lái Bản Đồ FH4 (UK Left-Hand Traffic) & Các Nâng Cấp Dự Án CPV301

Tài liệu này tổng hợp đầy đủ những gì đã được thiết kế, sửa lỗi và nâng cấp trong dự án **CPV301 — Forza Horizon 4 Lane Detection & UK Autonomous Navigation** để các thành viên trong nhóm dễ dàng nắm bắt kiến trúc và chạy thực nghiệm.

---

## 1. Tổng Quan 6 Hạng Mục Đã Hoàn Thiện

1. **Tạo cấu trúc thư mục bản đồ `fh4_map/`**:
   - `fh4_map/raw/`: Chứa ảnh bản đồ gốc của Forza Horizon 4 (`fh4_uk_map.png` — Vương Quốc Anh / Edinburgh / Cao tốc M68).
   - `fh4_map/processing/`: Chứa kết quả xử lý bản đồ gồm mặt nạ đường giao thông (`road_mask.png`), khung xương đường (`road_skeleton.png`), đồ thị nút giao thông (`road_graph_overlay.png`, `road_graph.json`) và bản đồ lộ trình A* đã đánh dấu điểm đến (`planned_route_map.png`).
2. **Hệ thống Phân tích Bản đồ, Đánh dấu Điểm đến & Tự lái theo Luật Giao thông Vương Quốc Anh (UK Left-Hand Traffic)**:
   - Xây dựng [`src/map/map_analyzer.py`](../src/map/map_analyzer.py) (`FH4MapAnalyzer`) và [`src/map/navigator.py`](../src/map/navigator.py) (`FH4MapNavigator`).
   - Hỗ trợ người dùng đánh dấu điểm đến trên bản đồ bằng click chuột trực tiếp (`--pick-dest`), theo mã nút địa danh (`--dest-node 12`), tọa độ ảnh (`--dest-pixel U V`) hoặc tọa độ thế giới (`--dest-world X Z`).
   - Giải mã tọa độ thực 3D của xe (`PositionX`, `PositionY`, `PositionZ`, `Yaw`, `Pitch`, `Roll`) từ gói tin UDP 324 bytes trong [`test_fh4/fh4_telemetry.py`](../test_fh4/fh4_telemetry.py) và [`test_fh4/FH4_controler.py`](../test_fh4/FH4_controler.py).
   - Phân loại làn đường chuẩn Vương Quốc Anh (đi bên trái — **Left-Hand Traffic**) trong [`src/control/pure_pursuit.py`](../src/control/pure_pursuit.py) (`classify_uk_lanes()`): tách biệt mép đường bên trái (`uk_left_edge`), vạch phân cách giữa đường (`uk_center_divider`) và làn xe ngược chiều bên phải (`uk_oncoming_right_edge`), giữ xe bám tâm **làn bên trái** và điều chỉnh quỹ đạo khi có chỉ dẫn rẽ trái/phải/vào bùng binh chiều kim đồng hồ từ bản đồ A*.
3. **Nâng cấp [`notebooks/01_pre_processing.ipynb`](../notebooks/01_pre_processing.ipynb) & Xuất ra `data/processing/`**:
   - Cải tiến quy trình tiền xử lý lên **6 bước chất lượng cao**:
     1. Cân bằng sáng thích nghi cục bộ **CLAHE** trên kênh L (không gian LAB) + làm mượt Gaussian.
     2. Phát hiện biên **Canny** kết hợp mặt nạ màu **HLS White/Yellow Lane Mask**.
     3. Cắt vùng quan tâm **Polygon ROI** vươn cao để nhìn rõ góc cua phía xa.
     4. Lọc nhiễu đốm nhỏ (**Connected Components Blob Removal**) + **Morphological Closing/Dilation**.
     5. Phát hiện đoạn thẳng **HoughLinesP** có trọng số độ dài + tinh chỉnh theo điểm ảnh vạch trắng + nội suy đường cong **B-Spline** bậc 3.
     6. Phân loại làn đường **UK Left-Hand Traffic** và tính đường dẫn tâm làn trái.
   - Tự động xử lý hàng loạt qua [`src/preprocessing/batch_processor.py`](../src/preprocessing/batch_processor.py) và xuất toàn bộ ảnh kết quả (`highlighted/`) cùng bộ dữ liệu huấn luyện YOLO-seg (`train/`, `valid/`) vào thư mục **`data/processing/`**.
4. **Nâng cấp [`notebooks/02_train.ipynb`](../notebooks/02_train.ipynb) — Huấn luyện từ `data/processing/` & Xuất ra `data/output/`**:
   - Sửa lỗi xung đột NumPy 2.x / PyTorch và lỗi cố định `device=0` khi chạy trên máy không có GPU CUDA (nay tự động chọn `0` hoặc `'cpu'`).
   - Cấu hình [`config/data.yaml`](../config/data.yaml) đọc dữ liệu trực tiếp từ **`data/processing/`**.
   - Lưu toàn bộ kết quả huấn luyện (`best.pt`, `yolo_train/`) và ảnh dự đoán kiểm thử kèm phân loại làn UK (`predictions/`) vào **`data/output/`** thông qua [`src/evaluation/train_and_export.py`](../src/evaluation/train_and_export.py).
5. **Đặt toàn bộ Biểu đồ (`.png`) và Bảng số liệu (`.csv`) ngay tại Thư mục gốc (`ROOT_DIR`)**:
   - Tất cả 5 file CSV số liệu và 3 file biểu đồ tổng hợp đều được lưu trực tiếp ở thư mục gốc `CPV301/` (đồng thời có bản sao trong thư mục con tương ứng).
6. **Kiểm thử tự động (Unit Tests)**:
   - Mở rộng bộ test lên **21/21 bài kiểm thử `pytest` PASSED 100%** (bao gồm cả kiểm thử đồ thị bản đồ FH4, định tuyến A*, phân loại làn trái UK, tiền xử lý 6 bước và vòng lặp điều khiển `main.py`).

---

## 2. Chi Tiết Cơ Chế Tự Lái Kết Hợp Bản Đồ `fh4_map`, `fh4_telemetry` & Làn Đường Nước Anh (UK)

### 2.1. Trích xuất Đồ thị Đường đi từ Bản đồ (`src/map/map_analyzer.py`)
- Ảnh bản đồ gốc đặt tại `fh4_map/raw/fh4_uk_map.png`.
- `FH4MapAnalyzer.extract_road_mask()` tách các tuyến cao tốc (M68), quốc lộ và đường nhánh bằng phân ngưỡng không gian màu HSV kết hợp phép đóng hình thái học (`MORPH_CLOSE`).
- `FH4MapAnalyzer.skeletonize_roads()` làm mảnh mặt nạ đường về khung xương 1-pixel (`road_skeleton.png`).
- `FH4MapAnalyzer.build_road_graph()` xây dựng đồ thị gồm **17 nút giao/địa danh chính** (`RoadNode`, từ *Horizon Festival Site* tới *Edinburgh City Centre*) và **25 tuyến đường kết nối** (`RoadEdge`), quy đổi song song giữa tọa độ ảnh `(x_px, y_px)` và tọa độ thế giới trong game `(PositionX, PositionZ)`.

### 2.2. Đánh dấu Điểm đến & Tìm đường A* (`src/map/navigator.py`)
- Người dùng chọn điểm đến qua hàm `navigator.set_destination(...)` hoặc cửa sổ tương tác `navigator.select_destination_interactive()`.
- Thuật toán **A\* (`_astar_shortest_path`)** tìm lộ trình có chi phí thời gian di chuyển nhỏ nhất (ưu tiên đường cao tốc/quốc lộ có tốc độ cho phép cao).
- Tại mỗi mốc lộ trình (`waypoint`), hệ thống sinh chỉ dẫn lái xe theo **luật giao thông Vương Quốc Anh**:
  - `KEEP_LEFT_LANE_UK`: Giữ làn bên trái.
  - `ROUNDABOUT_CLOCKWISE_KEEP_LEFT`: Vào vòng xuyến (bùng binh) theo **chiều kim đồng hồ** (bên trái).
  - `TURN_LEFT_INTO_UK_LEFT_LANE`: Rẽ trái ôm sát làn trái.
  - `TURN_RIGHT_CROSS_TRAFFIC_TO_UK_LEFT_LANE`: Rẽ phải qua làn đối diện để nhập vào làn trái của đường mới.
  - `ARRIVED_PULL_OVER_LEFT`: Tới đích, tấp vào lề trái và dừng xe an toàn.

### 2.3. Phân loại Làn đường chuẩn Anh Quốc (`src/control/pure_pursuit.py`)
Khác với giao thông đi bên phải, tại Vương Quốc Anh (bối cảnh của Forza Horizon 4), xe luôn chạy ở **làn bên trái**:
- Khi phát hiện **>= 3 vạch kẻ đường** (đường 2 chiều nhiều làn):
  - Vạch ngoài cùng bên trái là **mép lề trái (`uk_left_edge`)**.
  - Vạch kế tiếp là **vạch phân cách trung tâm (`uk_center_divider`)**.
  - Các vạch bên phải là **làn xe đi ngược chiều (`uk_oncoming_right_edge`)**.
  - Hệ thống tính đường tâm dẫn hướng (`centerline`) nằm chính giữa `uk_left_edge` và `uk_center_divider` (chế độ `uk_multi_lane_left_corridor`) để xe tuyệt đối không lấn sang làn ngược chiều bên phải.
- Khi phát hiện **2 vạch kẻ đường**:
  - Hệ thống xác định hành lang làn trái (`uk_standard_left_lane`) với trọng số ưu tiên bám sát làn trái (`0.55 * left_border + 0.45 * right_border`), đồng thời dịch chuyển điểm nhìn trước (`lookahead_point`) theo gợi ý rẽ `map_turn_hint` từ bản đồ A*.
- Khi chỉ thấy **1 vạch kẻ đường** (ở khúc cua gấp hoặc đường quê nước Anh):
  - Tự động suy luận vị trí tâm làn trái cách mép đường hoặc vạch tim đường một nửa bề rộng làn chuẩn (`uk_single_left_edge` / `uk_single_center_divider`).

### 2.4. Hợp nhất Điều khiển Vision + Map UK (`src/control/controller.py`)
Trong mỗi chu kỳ điều khiển của `FH4AutonomousController.step()`:
1. Đọc gói UDP Telemetry từ `fh4_telemetry.py` lấy vận tốc `SpeedKmh`, góc hướng `Yaw`, và tọa độ 3D `(PositionX, PositionY, PositionZ)`.
2. Nếu có `map_navigator`, tự động gọi `compute_map_guidance(PositionX, PositionZ, Yaw)` để cập nhật góc hướng bản đồ, tốc độ giới hạn của đoạn đường và khoảng cách tới đích (`distance_to_goal_m`).
3. Nếu `distance_to_goal_m <= arrival_tolerance_m`, bộ điều khiển chuyển sang trạng thái `ARRIVED` và phanh dừng xe tại điểm đến.
4. Khi đang di chuyển, bộ điều khiển hợp nhất góc lái từ **Camera nhận diện làn đường UK (80%)** và **Hướng dẫn lộ trình bản đồ A\* (20%)** (`source="vision+map_uk"`), giúp xe vừa bám chặt làn đường bên trái vừa rẽ đúng hướng tại các ngã ba/ngã tư/vòng xuyến để tới đích.

---

## 3. Danh Sách Các File CSV Số Liệu & Biểu Đồ Tại Thư Mục Gốc (`ROOT_DIR`)

Theo đúng yêu cầu, toàn bộ biểu đồ và file CSV số liệu đã được xuất trực tiếp tại thư mục gốc `h:\FPT\4\CPV\project\CPV301\`:

1. **[`fh4_map_road_graph.csv`](../fh4_map_road_graph.csv)**: Bảng dữ liệu toàn bộ các tuyến đường trên bản đồ FH4 (nút đầu, nút cuối, loại đường UK, làn chạy bên trái, tốc độ tối đa, chiều dài pixel & mét).
2. **[`fh4_route_waypoints.csv`](../fh4_route_waypoints.csv)**: Bảng tọa độ các mốc lộ trình A* từ điểm xuất phát tới điểm đến đã đánh dấu, kèm chỉ dẫn làn đường UK (`uk_instruction`) và khoảng cách tích lũy.
3. **[`fh4_map_route_chart.png`](../fh4_map_route_chart.png)**: Biểu đồ trực quan 3 khung: Bản đồ FH4 gốc, Bản đồ lộ trình A* đã đánh dấu điểm đến, và Đồ thị tốc độ khuyến nghị theo quãng đường.
4. **[`preprocessing_metrics.csv`](../preprocessing_metrics.csv)**: Bảng số liệu kết quả chạy tiền xử lý ảnh mặt đường (số làn phát hiện, phân loại làn UK, số pixel biên/màu trong ROI, góc lệch hướng, độ cong, thời gian xử lý ms).
5. **[`preprocessing_pipeline_chart.png`](../preprocessing_pipeline_chart.png)**: Biểu đồ minh họa trực quan 6 bước tiền xử lý nâng cao và phân loại làn đường bên trái chuẩn UK.
6. **[`training_results.csv`](../training_results.csv)**: Số liệu quá trình huấn luyện mô hình YOLO11-seg trên tập dữ liệu `data/processing/`.
7. **[`evaluation_metrics.csv`](../evaluation_metrics.csv)**: Bảng so sánh độ chính xác (Accuracy), tỷ lệ báo sai (FP), bỏ sót (FN) và tốc độ xử lý (FPS) giữa 3 phương pháp (`Hough_Transform`, `LaneNet`, `Proposed_Spline`).
8. **[`training_and_evaluation_chart.png`](../training_and_evaluation_chart.png)**: Biểu đồ tổng hợp kết quả huấn luyện YOLO11-seg và so sánh hiệu năng các phương pháp.
9. **[`fh4_log.csv`](../fh4_log.csv)**: Nhật ký quá trình điều khiển xe tự hành ghi lại vận tốc, góc lái, ga, phanh, loại làn UK, tọa độ `(position_x, position_z)` và khoảng cách tới điểm đến.
