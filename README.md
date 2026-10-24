# Lane Detection & UK Autonomous Navigation trong Forza Horizon 4

Dự án môn **CPV301 - Thị giác máy tính** xây dựng hệ thống phát hiện làn đường, phân tích bản đồ toàn cục và điều khiển xe tự hành theo **luật giao thông Vương Quốc Anh (UK Left-Hand Traffic - đi bên trái)** trong game **Forza Horizon 4**, sử dụng **camera bumper** kết hợp tọa độ thực thời gian thực từ **UDP Telemetry (`fh4_telemetry.py`)**.

> ⚠️ **Cảnh báo**: Chạy hoàn toàn ở chế độ **offline**. Forza Horizon 4 có anti-cheat; sử dụng phần mềm giả lập tay cầm khi online có thể dẫn đến khóa tài khoản.
>
> 📖 **Tài liệu giải thích chi tiết kiến trúc & các nâng cấp mới nhất**: Xem [`docs/explain.md`](docs/explain.md).

## Mục tiêu

1. **Phân tích bản đồ `fh4_map` & Định tuyến điểm đến (A\*)**:
   - Đọc bản đồ Forza Horizon 4 từ `fh4_map/raw/`, trích xuất mặt nạ đường giao thông, làm mảnh khung xương (Skeletonization) và lập đồ thị mạng lưới đường (`RoadNode`, `RoadEdge`) lưu vào `fh4_map/processing/`.
   - Cho phép người dùng **đánh dấu điểm đến trên bản đồ** (click chuột trực tiếp với `--pick-dest`, chọn nút giao `--dest-node`, tọa độ ảnh `--dest-pixel` hoặc tọa độ thế giới `--dest-world`).
   - Kết hợp lộ trình A* với tọa độ thực (`PositionX`, `PositionY`, `PositionZ`, `Yaw`) từ `fh4_telemetry.py` và **phân loại làn đường chuẩn nước Anh (đi làn bên trái, vào bùng binh theo chiều kim đồng hồ)** để tự lái đến đích.
2. **Chuẩn hóa luồng dữ liệu qua các Notebook**:
   - `notebooks/01_pre_processing.ipynb`: Tiền xử lý nâng cao 6 bước (CLAHE + Gaussian Blur + Canny + HLS White/Yellow Mask + UK ROI + B-Spline + Phân loại làn trái UK) và xuất toàn bộ ảnh & bộ dữ liệu YOLO-seg vào **`data/processing/`**.
   - `notebooks/02_train.ipynb`: Huấn luyện YOLO11-seg bằng ảnh trong **`data/processing/`** và xuất trọng số (`best.pt`), ảnh dự đoán cùng báo cáo đánh giá vào **`data/output/`**.
3. **Tổng hợp toàn bộ Biểu đồ (`.png`) và Bảng số liệu (`.csv`) ngay tại thư mục gốc (`ROOT_DIR`)**:
   - Tất cả các file số liệu CSV và biểu đồ phân tích của Bản đồ, Tiền xử lý, Huấn luyện, Đánh giá và Điều khiển đều được xuất trực tiếp ra thư mục gốc của dự án để dễ dàng kiểm tra và đưa vào báo cáo.

## Kiến trúc tổng thể

```
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│   fh4_map/raw    │ ──► │ fh4_map/processing │ ──► │  A* Route Navigator  │
│  (Bản đồ FH4 UK) │     │ (Road Graph + CSV) │     │ (Điểm đến + Hướng rẽ)│
└──────────────────┘     └────────────────────┘     └──────────┬───────────┘
                                                               │
┌────────────┐   ┌───────────────┐   ┌─────────────┐   ┌───────▼───────┐   ┌───────────────┐   ┌────────────┐
│  Capture   │─► │ Preprocessing │─► │  Detection  │─► │UK Left-Hand   │─► │Control Fusion │─► │   Input    │
│ (mss, ROI) │   │data/processing│   │Hough/YOLO/  │   │Spline & Lane  │   │PurePursuit+PID│   │ (vgamepad) │
└────────────┘   └───────────────┘   │   LaneNet   │   │Classification │   │+fh4_telemetry │   └────────────┘
                                     └─────────────┘   └───────────────┘   └───────────────┘
```

## Trạng thái hiện tại

| Module | Trạng thái | Ghi chú |
|---|---|---|
| `fh4_map/` (`raw/` & `processing/`) | ✅ Hoàn thành | Chứa bản đồ gốc `fh4_uk_map.png`, mặt nạ đường `road_mask.png`, khung xương `road_skeleton.png`, đồ thị `road_graph_overlay.png`, `planned_route_map.png` |
| `src/map/` (`map_analyzer.py`, `navigator.py`) | ✅ Hoàn thành | Trích xuất đồ thị đường đi, chọn điểm đến tương tác/CLI, định tuyến A* theo luật giao thông Anh (LHT) |
| `src/preprocessing/` (`batch_processor.py`, `preprocessor.py`, `enhance.py`, `transform.py`) | ✅ Hoàn thành | Pipeline 6 bước + xuất ảnh/nhãn sang `data/processing/` + xuất CSV & biểu đồ ra Root |
| `src/control/` (`pure_pursuit.py`, `pid.py`, `controller.py`, `telemetry.py`) | ✅ Hoàn thành | Phân loại làn đường UK (`classify_uk_lanes`), hợp nhất Vision + Map A* (`vision+map_uk`), tự động dừng khi tới đích |
| `test_fh4/fh4_telemetry.py` & `FH4_controler.py` | ✅ Hoàn thành | Giải mã gói UDP 324 bytes kèm tọa độ 3D (`PositionX`, `PositionY`, `PositionZ`, `Yaw`, `Pitch`, `Roll`) |
| `notebooks/01_pre_processing.ipynb` | ✅ Hoàn thành | Tiền xử lý nâng cao 6 bước & xuất dữ liệu vào `data/processing/` + CSV/biểu đồ ở Root |
| `notebooks/02_train.ipynb` | ✅ Hoàn thành | Huấn luyện YOLO11-seg từ `data/processing/` & xuất ra `data/output/` + CSV/biểu đồ ở Root |
| `tests/` (pytest) | ✅ Hoàn thành | 21/21 unit tests PASSED |

## Danh sách Biểu đồ (`.png`) và Bảng số liệu (`.csv`) tại Thư mục gốc (`ROOT_DIR`)

| Tên file tại Root | Loại | Mô tả nội dung |
|---|---|---|
| `fh4_map_road_graph.csv` | CSV | Danh sách các đoạn đường (`RoadEdge`), nút giao (`RoadNode`), giới hạn tốc độ và làn UK trên bản đồ FH4 |
| `fh4_route_waypoints.csv` | CSV | Lộ trình A* chi tiết từng mốc tọa độ `(world_x, world_z)`, góc hướng và chỉ dẫn làn trái UK tới điểm đến |
| `fh4_map_route_chart.png` | Biểu đồ | Biểu đồ 3-panel: Bản đồ gốc FH4 (`fh4_map/raw`), Lộ trình A* (`fh4_map/processing`), và Hồ sơ tốc độ theo quãng đường |
| `preprocessing_metrics.csv` | CSV | Số liệu thống kê tiền xử lý từng ảnh mẫu (số làn phát hiện, loại làn UK, pixel biên/màu trong ROI, góc lệch, độ cong, thời gian xử lý) |
| `preprocessing_pipeline_chart.png` | Biểu đồ | Biểu đồ 6 bước tiền xử lý ảnh mặt đường và phân loại làn trái UK |
| `training_results.csv` | CSV | Lịch sử huấn luyện YOLO11-seg qua các epoch (`box_loss`, `seg_loss`, `mAP50`, `mAP50-95`) trên `data/processing/` |
| `evaluation_metrics.csv` | CSV | Bảng so sánh Accuracy, FP, FN, FPS giữa các phương pháp (`Hough_Transform`, `LaneNet`, `Proposed_Spline`) |
| `training_and_evaluation_chart.png` | Biểu đồ | Biểu đồ so sánh độ chính xác/FPS và đường cong hội tụ hàm mất mát huấn luyện |
| `fh4_log.csv` | CSV | Nhật ký điều khiển tự hành thời gian thực kèm tọa độ xe (`position_x`, `position_z`), loại làn UK và khoảng cách tới đích |

## Cấu trúc thư mục

```
CPV301/
├── fh4_map/                       # ✅ Thư mục bản đồ Forza Horizon 4 (Vương Quốc Anh)
│   ├── raw/                       # Bản đồ gốc (fh4_uk_map.png)
│   └── processing/                # Mặt nạ đường, khung xương, đồ thị & bản đồ lộ trình A*
├── config/
│   ├── data.yaml                  # ✅ Cấu hình dataset YOLO trỏ vào data/processing
│   └── bev.yaml                   # ✅ Cấu hình Bird's Eye View (default, fh4_bumper, tusimple)
├── src/
│   ├── map/                       # ✅ map_analyzer.py, navigator.py (Đồ thị bản đồ & Định tuyến A* UK)
│   ├── capture/                   # ✅ screen_capture.py (mss screen capture)
│   ├── preprocessing/             # ✅ batch_processor.py, enhance.py, preprocessor.py, transform.py
│   ├── detection/                 # ✅ hough_transform.py, lanenet.py, yolo11_detector.py
│   ├── fitting/                   # ✅ spline_fitter.py (B-spline, 3 cấp fallback)
│   ├── proposed_method/           # ✅ yolo_spline.py (Enhance + YOLO11 + Spline)
│   ├── control/                   # ✅ pure_pursuit.py (UK LHT), pid.py, controller.py, telemetry.py
│   ├── input/                     # ✅ actuator.py (vgamepad / keyboard / dry_run)
│   ├── evaluation/                # ✅ compare_methods.py, train_and_export.py
│   └── path.py                    # ✅ Quản lý đường dẫn tập trung
├── data/
│   ├── raw/                       # Ảnh gốc + test_driving.mp4 + TUSimple/
│   ├── processing/                # ✅ Ảnh đã tiền xử lý + tập train/valid YOLO-seg (từ 01_pre_processing.ipynb)
│   └── output/                    # ✅ Trọng số best.pt, ảnh dự đoán & báo cáo (từ 02_train.ipynb)
├── notebooks/
│   ├── 01_pre_processing.ipynb    # ✅ Tiền xử lý nâng cao -> xuất ra data/processing/ & Root
│   └── 02_train.ipynb             # ✅ Train trên data/processing/ -> xuất ra data/output/ & Root
├── docs/
│   └── explain.md                 # ✅ Giải thích chi tiết toàn bộ hệ thống & nâng cấp
├── fh4_map_road_graph.csv         # ✅ [Root CSV] Đồ thị mạng lưới đường FH4
├── fh4_route_waypoints.csv        # ✅ [Root CSV] Lộ trình A* tới điểm đến
├── fh4_map_route_chart.png        # ✅ [Root Chart] Biểu đồ lộ trình bản đồ FH4 UK
├── preprocessing_metrics.csv      # ✅ [Root CSV] Số liệu tiền xử lý ảnh
├── preprocessing_pipeline_chart.png # ✅ [Root Chart] Biểu đồ 6 bước tiền xử lý
├── training_results.csv           # ✅ [Root CSV] Kết quả huấn luyện YOLO11-seg
├── evaluation_metrics.csv         # ✅ [Root CSV] Bảng đánh giá các phương pháp
├── training_and_evaluation_chart.png # ✅ [Root Chart] Biểu đồ huấn luyện & đánh giá
├── fh4_log.csv                    # ✅ [Root CSV] Nhật ký điều khiển tự hành
└── main.py                        # ✅ Vòng lặp tự hành trung tâm tích hợp Bản đồ + Làn trái UK
```

## Sử dụng nhanh

### 1. Chạy tiền xử lý (`data/raw` -> `data/processing` + Root CSV/Chart)
Mở và chạy `notebooks/01_pre_processing.ipynb` hoặc chạy lệnh:
```bash
python -m src.preprocessing.batch_processor
```

### 2. Chạy huấn luyện & đánh giá (`data/processing` -> `data/output` + Root CSV/Chart)
Mở và chạy `notebooks/02_train.ipynb` hoặc chạy lệnh:
```bash
python -m src.evaluation.train_and_export
```

### 3. Chạy hệ thống tự lái kết hợp Bản đồ `fh4_map`, Điểm đến & Làn đường Vương Quốc Anh (`main.py`)
```bash
# A. Mở bản đồ FH4 để click chuột chọn điểm đến trực tiếp, sau đó chạy tự lái:
python main.py --control --detector hough --pick-dest

# B. Đánh dấu điểm đến bằng mã nút giao (ví dụ Node 12: Edinburgh City Centre) ở chế độ mô phỏng video:
python main.py --view-only --detector hough --dest-node 12 --video data/raw/test_driving.mp4 --save data/output/fh4_sim_viewonly.mp4

# C. Đánh dấu điểm đến bằng tọa độ pixel (U, V) hoặc tọa độ thế giới (PositionX, PositionZ):
python main.py --view-only --detector hough --dest-pixel 850 130
python main.py --view-only --detector hough --dest-world 2900 4180
```

### 4. Chạy bộ kiểm thử tự động (21 Unit Tests)
```bash
pytest -v
```
