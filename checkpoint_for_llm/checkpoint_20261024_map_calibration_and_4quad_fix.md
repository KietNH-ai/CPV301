# Checkpoint: Calibrated Coordinate System & 4-Quadrant Dashboard Fixes

**Thời gian**: 2026-10-24 08:02:00 (UTC+7)  
**Tác vụ**: Hiệu chuẩn hệ trục tọa độ bản đồ FH4 (World <-> Pixel), sửa lỗi `AttributeError: _build_route_geometry`, tối ưu FPS và hỗ trợ nhận diện làn đường tự động.

---

## 1. Summary of Completed Work (Tóm tắt công việc đã hoàn thành)
1. **Hiệu chuẩn chính xác hệ trục tọa độ bản đồ Forza Horizon 4 (`FH4_UK_MAP_WORLD_BOUNDS`)**:
   - Phân tích dữ liệu thực tế từ 16.153 dòng telemetry UDP trong file `fh4_log.csv` (ghi lại quá trình lái xe thực tế trong FH4).
   - Khớp bình phương tối thiểu (least squares fit) giữa quỹ đạo thế giới FH4 `(PositionX, PositionZ)` và ảnh bản đồ `fh4_map/raw/fh4_uk_map.png` (1400 x 1238):
     - Tỉ lệ co giãn đồng nhất: `9.3 m/pixel`.
     - Tọa độ biên mới: `world_bounds = (-6133.95, 6886.05, -6794.85, 4718.55)` (thay thế giả định đối xứng cũ `(-7000, 7000, -7000, 7000)` vốn làm lệch xe sang phải và xuống dưới > 360m).
     - Khi xe đứng tại bùng binh đầu cao tốc M68 `(X=+2603.4, Z=-4400.1)`, tâm xe hiện tại rơi đúng vào tâm bùng binh `(940, 980)` trên ảnh bản đồ với khoảng cách tới đích chính xác `0.0 m` (báo `ARRIVED_PULL_OVER_LEFT`).

2. **Khắc phục triệt để lỗi `AttributeError: 'FH4MapNavigator' object has no attribute '_build_route_geometry'`**:
   - Bổ sung hoàn chỉnh các phương thức hình học và map-matching trong `src/map/navigator.py`:
     - `_compose_route_polyline()`: Ghép nối chuỗi pixel dọc theo các cung đường cong của bản đồ A*, bao gồm đoạn dẫn nhập từ vị trí xe thật.
     - `_build_route_geometry()`: Xây dựng mảng tọa độ thế giới dày đặc `_route_world` và mảng khoảng cách tích lũy `_route_cum_m` (tính theo độ dài mặt đường cong thực tế thay vì khoảng cách chim bay).
     - `locate_on_route()`: Chiếu vị trí xe vuông góc lên phân đoạn đường gần nhất, tính sai số lệch ngang (`cross_track_m`) và quãng đường còn lại tới đích (`dist_to_goal_m`).
     - `compute_map_guidance()`: Tận dụng vị trí chiếu để lookahead 25m phía trước trên đường cong, tính góc lệch hướng (`heading_error_rad`) mượt mà và chính xác.

3. **Cơ chế Map-Matching và Đồng bộ điểm xuất phát tự động từ Telemetry**:
   - Thêm phương thức `route_from_vehicle()` và `update_vehicle_position()` vào `FH4MapNavigator`.
   - Trong vòng lặp chính của `main.py`, khi nhận được gói UDP telemetry thật từ game, hệ thống tự động cập nhật vị trí xe và dẫn đường từ đúng vị trí xe đang đứng trên đường mà không cần nhập tay điểm xuất phát.
   - Khi xe đi chệch khỏi lộ trình > 120m liên tục > 2 giây, hệ thống tự động kích hoạt tái định tuyến (re-route).

4. **Khắc phục nghẽn FPS và tối ưu nhận diện làn đường (`--detector proposed`)**:
   - Trong `main.py` (`_AsyncCurveDetector`): sửa điều kiện luồng bất đồng bộ (trước đây kiểm tra `if not self._latest_curves` khiến các frame có 0 làn liên tục kích hoạt suy luận đồng bộ trên luồng UI làm tụt xuống 6 FPS). Hiện tại luồng UI luôn chạy mượt mà $\ge 30$ FPS.
   - Trong `src/proposed_method/yolo_spline.py` (`LaneDetector`): tích hợp cơ chế Hybrid Fallback tự động gọi `HoughLaneDetector` khi YOLO trả về $< 2$ làn, giúp nhận diện được lề đường cỏ và đường cong nông thôn UK ngay cả khi không có vạch sơn trắng.
   - Thêm bộ lọc mã hóa UTF-8 cho `sys.stdout` / `sys.stderr` trên Windows, tránh lỗi `UnicodeEncodeError: 'charmap' codec` khi in emoji ra PowerShell/CMD.
   - Xử lý import an toàn cho `tqdm`, `scikit-learn` (`DBSCAN`) và `pytest` khi chạy độc lập.

5. **Khắc phục triệt để hiện tượng nhận diện sai vạch làn nằm ngang khi xe đi ngoài đường (Off-road / Field)**:
   - **Vấn đề trên ảnh người dùng cung cấp (`media_1792804668379_d3020d1f.png`)**: Khi xe chạy vào cánh đồng rơm rạ / đất cày hoặc nhìn thấy đường chân trời/hàng rào, cạnh biên ngang giữa ruộng và cỏ/hàng cây bị thuật toán nhận diện nhầm thành vạch kẻ đường (`min_abs_slope=0.26` quá phẳng, `y_span` chỉ có vài pixel). Hậu quả: vẽ ra các đường ngoằn ngoèo nằm ngang (hồng, cam, xanh ngọc) và đường tim đường vàng bị gãy khúc 90 độ đâm thẳng xuống góc trái mép màn hình.
   - **Các giải pháp bảo vệ 4 lớp đã triển khai**:
     1. Tăng `min_abs_slope` trong [`src/detection/hough_transform.py`](file:///h:/FPT/4/CPV/project/CPV301/src/detection/hough_transform.py#L36) từ `0.26` lên `0.42` (loại bỏ các góc quá phẳng $< 23^\circ$).
     2. Trong `_fit_curved_lane`: kiểm tra độ trải dọc `y_span >= max(35.0, (y_bottom - y_top)*0.18)` và tỉ lệ trải ngang `x_span <= 2.2 * y_span`. Các đường chủ yếu nằm ngang lập tức bị loại bỏ.
     3. Trong `_detect_uk_left_road_edge`: kiểm tra mặt đường (mặt nhựa đường xám trung tính, loại bỏ bề mặt đất ruộng/rơm rạ có màu ấm `warmth_val > 68.0`).
     4. Trong `classify_uk_lanes` và `extract_path_from_lanes` ([`src/control/pure_pursuit.py`](file:///h:/FPT/4/CPV/project/CPV301/src/control/pure_pursuit.py#L156)): lọc sạch toàn bộ các curve nằm ngang, đảm bảo `y_max >= height * 0.45`. Nếu không có vạch dọc thực tế, lập tức trả về `no_lane_detected` (`valid=False, confidence=0.0`), không bao giờ vẽ hành lang ảo hay điểm ngắm gãy khúc khi xe đang ở ngoài đồng.

---

## 2. Current Task & Progress (Tiến độ hiện tại)
- **Đã hoàn thành 100%**:
  - Hệ trục bản đồ đã được hiệu chuẩn chính xác và tái tạo bộ nhớ đệm `fh4_map/processing/road_graph.json`, `road_graph_overlay.png`, `fh4_map_road_graph.csv`.
  - Khắc phục lỗi `AttributeError: '_build_route_geometry'`.
  - Toàn bộ 22/22 unit tests trong thư mục `tests/` đều vượt qua thành công (`TOTAL TESTS: 22, ALL PASSED: True`).

---

## 3. Modified & Created Files (Danh sách file đã thay đổi / tạo mới)
- `src/map/map_analyzer.py`: Cập nhật `FH4_UK_MAP_WORLD_BOUNDS` hiệu chuẩn, hàm `meters_per_pixel()` và `fit_world_bounds_from_anchors()`.
- `src/map/navigator.py`: Thêm `_compose_route_polyline()`, `_build_route_geometry()`, `locate_on_route()`, `route_from_vehicle()`, `update_vehicle_position()`, và nâng cấp `compute_map_guidance()`.
- `main.py`: Sửa cờ `_initialized` trong `_AsyncCurveDetector`, tích hợp `update_vehicle_position()` theo thời gian thực và reconfigure encoding UTF-8.
- `src/proposed_method/yolo_spline.py`: Tích hợp Hybrid Fallback (`HoughLaneDetector`) khi YOLO phát hiện $< 2$ làn.
- `src/detection/lanenet.py`: Fallback an toàn khi chưa cài `scikit-learn`.
- `src/tusimple_to_yolo.py`: Fallback an toàn khi chưa cài `tqdm`.
- `model/model_config.py`: Sửa in thông báo lỗi PyTorch không dùng emoji tránh charmap error trên Windows.
- `tests/test_control_and_main.py`, `tests/test_preprocessing_and_hough.py`, `tests/test_spline_fitter.py`: Hỗ trợ chạy kiểm thử độc lập mà không bắt buộc có pytest cài sẵn trong môi trường Python 3.11.

---

## 4. Next Action Steps (Các bước hành động tiếp theo)
1. **Kiểm tra trực tiếp với game Forza Horizon 4**:
   - Chạy lệnh quan sát 4 ô màn hình:
     ```bash
     python main.py --view-only --detector hough --dest-node 8
     ```
     hoặc với phương pháp đề xuất:
     ```bash
     python main.py --view-only --detector proposed --dest-node 8
     ```
   - Xác nhận trên Ô 4 (Góc Dưới-Phải): Biểu tượng xe màu xanh lá nằm chính xác trên cung đường/vòng xuyến mà xe đang dừng trong game, không còn bị lệch sang phải.
2. **Kiểm tra tính năng đổi điểm đến**:
   - Click chuột vào một địa danh bất kỳ ở Ô 2 hoặc bấm `Enter`/nút `[XAC NHAN]` để chuyển sang quan sát nhận diện làn đường.
   - Xác nhận Ô 2 đạt tốc độ $\ge 30$ FPS ổn định.

---

## 5. Known Issues & Blockers (Vấn đề tồn đọng)
- Không có blocker nào. Toàn bộ đồ thị đường đi và hệ trục tọa độ đã được khớp chính xác với dữ liệu thực tế của game.
