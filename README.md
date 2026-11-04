# Lane Detection & UK Autonomous Navigation trong Forza Horizon 4 (CPV301)

Dự án môn **CPV301 - Thị giác máy tính** xây dựng hệ thống phân tích nhãn dữ liệu làn đường **TuSimple**, phát hiện làn đường lai (**YOLO11n-seg + Cubic B-Spline**), phân tích bản đồ toàn cục (**A\***) và điều khiển xe tự hành theo **luật giao thông Vương Quốc Anh (UK Left-Hand Traffic - đi bên trái)** trong game **Forza Horizon 4**, sử dụng **camera bumper** kết hợp tọa độ thực thời gian thực từ **UDP Telemetry (`fh4_telemetry.py`)**.

> ⚠️ **Cảnh báo**: Khi chạy cùng game Forza Horizon 4, hãy chạy hoàn toàn ở chế độ **offline (Horizon Solo)**. Forza Horizon 4 có anti-cheat; sử dụng phần mềm giả lập tay cầm khi online có thể dẫn đến khóa tài khoản.
>
> 📖 **Tài liệu giải thích chi tiết kiến trúc & các nâng cấp mới nhất**: Xem [`docs/explain.md`](docs/explain.md).

---

## Hướng dẫn Cài đặt Môi trường (Environment Setup)

Dự án chia tách rõ **2 tệp cấu hình thư viện** tùy theo nhu cầu sử dụng của bạn:
- **`requirements.txt`**: Chỉ cài các thư viện cốt lõi phục vụ **Phân tích nhãn, Tiền xử lý CLAHE, Huấn luyện YOLO11-seg + B-Spline, Đánh giá thực nghiệm, Chạy Notebook (`00`, `01`, `02`) và Kiểm thử `pytest`** (không cài các thư viện bắt cửa sổ/điều khiển tay cầm chạy song song với game FH4).
- **`requirements(testFH4).txt`**: Bao gồm toàn bộ `requirements.txt` **cộng thêm** các thư viện chuyên dụng để chạy song song thời gian thực cùng game **Forza Horizon 4** (`mss`, `vgamepad`, `pydirectinput`, `keyboard`, `psutil`).

### Bước 1: Tạo và kích hoạt môi trường ảo Python (Python 3.10 – 3.12)

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Trên Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Bước 2: Chọn chế độ cài đặt thư viện

#### Cách A — Chỉ chạy Phân tích dữ liệu, Huấn luyện mô hình, Notebook & Kiểm thử (Không cần mở game FH4)
```powershell
pip install -r requirements.txt
```

#### Cách B — Chạy đầy đủ cả Hệ thống Tự lái song song với game Forza Horizon 4 (Windows)
```powershell
pip install -r "requirements(testFH4).txt"
```
> **Thiết lập bổ sung khi chạy thực tế trong Forza Horizon 4 (`--control`):**
> 1. **ViGEmBus Driver**: Khi cài `vgamepad`, trình điều khiển tay cầm Xbox 360 ảo **ViGEmBus** sẽ được cài đặt (hoặc tải trực tiếp từ [ViGEmBus Releases](https://github.com/nefarius/ViGEmBus/releases)).
> 2. **Bật UDP Telemetry trong game**: Vào *Forza Horizon 4 $\rightarrow$ Settings $\rightarrow$ HUD and Gameplay*:
>    - **Data Out**: `ON`
>    - **Data Out IP Address**: `127.0.0.1`
>    - **Data Out IP Port**: `5300`

---

## Mục tiêu & Luồng Dữ liệu Chuẩn

1. **Phân tích & Tái tạo Nhãn toàn bộ tập dữ liệu TuSimple (`notebooks/00_label_analysis.ipynb`)**:
   - Quét **100% tập dữ liệu TuSimple (`6,408` clip có nhãn**: `3,626` clip trong `train_set` + `2,782` clip trong `test_set`, chỉ đọc khung hình `20.jpg` có nhãn của mỗi clip).
   - Giải mã mặt nạ phân đoạn `seg_label/*.png` (giá trị pixel `{0, 1..6}` vốn hiển thị màu đen trên trình xem ảnh thường) sang ảnh màu trực quan, kiểm toán chất lượng nhãn, lọc nhiễu và xuất toàn bộ bảng thống kê, biểu đồ EDA cùng `6,408` file nhãn `.txt` sạch vào **`data/processing/label_analysis/`**.
2. **Tiền xử lý nâng cao & Chuyển đổi YOLO11-seg (`notebooks/01_pre_processing.ipynb`)**:
   - Áp dụng pipeline tiền xử lý 6 bước (**CLAHE** cân bằng sáng cục bộ trên kênh L + Gaussian Blur + Canny + HLS White/Yellow Mask + UK ROI + B-Spline + Phân loại làn trái UK).
   - Chia tập `train_set` (`3,626` clip) theo tỷ lệ chuẩn 80/20 không trùng lặp và xuất trực tiếp vào **`data/processing/`**:
     - **`data/processing/train/`**: **`2,901`** ảnh đã tăng cường CLAHE + **`2,901`** nhãn đa giác YOLO11-seg (`11,007` đa giác làn đường).
     - **`data/processing/valid/`**: **`725`** ảnh đã tăng cường CLAHE + **`725`** nhãn đa giác YOLO11-seg (`2,772` đa giác làn đường).
3. **Huấn luyện YOLO11n-seg + Nội suy B-Spline & Đánh giá trên Ground-Truth (`notebooks/02_train.ipynb`)**:
   - Fine-tune mô hình **`yolo11n-seg.pt`** trên dữ liệu `data/processing/` (khai báo trong `config/data.yaml`), lưu trọng số tốt nhất tại **`data/output/best.pt`**.
   - Kết hợp dự đoán của `YOLO11n-seg` với thuật toán bám đỉnh sơn đường và nội suy đường cong bậc 3 **`SplineFitter` (`Proposed_Spline`)**, xuất ảnh dự đoán kiểm thử vào **`data/output/predictions/`** và đối sánh thực nghiệm với `Hough_Transform`, `LaneNet` trên nhãn Ground-Truth thực tế của tập `data/processing/valid/`.
4. **Phân tích bản đồ `fh4_map` & Định tuyến điểm đến (A\*) theo luật Vương Quốc Anh**:
   - Trích xuất mặt nạ đường giao thông từ `fh4_map/raw/`, làm mảnh khung xương (Skeletonization) và lập đồ thị mạng lưới đường (`RoadNode`, `RoadEdge`) lưu vào `fh4_map/processing/`.
   - Kết hợp lộ trình A* với tọa độ thực (`PositionX`, `PositionY`, `PositionZ`, `Yaw`) từ `fh4_telemetry.py` và điều khiển xe chạy đúng **làn bên trái chuẩn Vương Quốc Anh (UK Left-Hand Traffic)**.

---

## Kiến trúc tổng thể

```
┌──────────────────────┐     ┌─────────────────────────┐     ┌────────────────────────┐
│   data/raw/TUSimple  │ ──► │ 00_label_analysis.ipynb │ ──► │ 01_pre_processing.ipynb│
│ (6,408 Labeled Clips)│     │ (Audit & Clean Polygons)│     │ (CLAHE: 2901 Tr/725 Va)│
└──────────────────────┘     └─────────────────────────┘     └───────────┬────────────┘
                                                                         │
┌──────────────────┐     ┌────────────────────┐     ┌────────────────────▼───┐
│   fh4_map/raw    │ ──► │ fh4_map/processing │ ──► │     02_train.ipynb     │
│  (Bản đồ FH4 UK) │     │ (Road Graph + A*)  │     │(YOLO11n-seg + B-Spline)│
└──────────────────┘     └─────────┬──────────┘     └────────────┬───────────┘
                                   │                             │
┌────────────┐   ┌───────────────┐ │ ┌─────────────┐   ┌─────────▼─────┐   ┌───────────────┐   ┌────────────┐
│  Capture   │─► │ Preprocessing │─┼►│  Detection  │─► │UK Left-Hand   │─► │Control Fusion │─► │   Input    │
│ (mss, ROI) │   │   (CLAHE)     │ │ │Hough/YOLO11/│   │Spline & Lane  │   │PurePursuit+PID│   │ (vgamepad) │
└────────────┘   └───────────────┘ │ │   LaneNet   │   │Classification │   │+fh4_telemetry │   └────────────┘
                                   │ └─────────────┘   └───────────────┘   └───────▲───────┘
                                   └───────────────────────────────────────────────┘
```

---

## Trạng thái hiện tại

| Module / Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| `notebooks/00_label_analysis.ipynb` | ✅ Hoàn thành | Quét 100% `6,408` nhãn TuSimple (`train_set` + `test_set`), giải mã `seg_label/*.png`, xuất báo cáo & nhãn sạch vào `data/processing/label_analysis/` |
| `notebooks/01_pre_processing.ipynb` | ✅ Hoàn thành | Tiền xử lý CLAHE 6 bước toàn bộ dataset, xuất `2,901` mẫu `train` + `725` mẫu `valid` vào `data/processing/` & xuất `preprocessing_metrics.csv` ra Root |
| `notebooks/02_train.ipynb` | ✅ Hoàn thành | Huấn luyện `YOLO11n-seg` từ `data/processing/`, nội suy `SplineFitter`, đánh giá trên Ground-Truth `valid` (`Accuracy = 85.83%`) & xuất ra `data/output/` + Root |
| `fh4_map/` (`raw/` & `processing/`) | ✅ Hoàn thành | Bản đồ gốc `fh4_uk_map.png`, mặt nạ đường `road_mask.png`, khung xương `road_skeleton.png`, đồ thị `road_graph_overlay.png`, `planned_route_map.png` |
| `src/map/` (`map_analyzer.py`, `navigator.py`, `dashboard.py`) | ✅ Hoàn thành | Đồ thị đường đi, giao diện 4 ô (4-Quadrant Dashboard >= 30 FPS), định tuyến A* theo luật giao thông Anh (LHT) |
| `src/control/` (`pure_pursuit.py`, `pid.py`, `controller.py`, `telemetry.py`) | ✅ Hoàn thành | Phân loại làn đường UK (`classify_uk_lanes`), hợp nhất Vision + Map A* (`vision+map_uk`), giải mã gói UDP 324 bytes |
| `tests/` (`pytest`) | ✅ Hoàn thành | **23/23 unit tests PASSED (100%)** |

---

## Danh sách Biểu đồ (`.png`) và Bảng số liệu (`.csv`) tại Thư mục gốc (`ROOT_DIR`)

| Tên file tại Root | Loại | Mô tả nội dung |
|---|---|---|
| `fh4_map_road_graph.csv` | CSV | Danh sách các đoạn đường (`RoadEdge`), nút giao (`RoadNode`), giới hạn tốc độ và làn UK trên bản đồ FH4 |
| `fh4_route_waypoints.csv` | CSV | Lộ trình A* chi tiết từng mốc tọa độ `(world_x, world_z)`, góc hướng và chỉ dẫn làn trái UK tới điểm đến |
| `fh4_map_route_chart.png` | Biểu đồ | Biểu đồ 3-panel: Bản đồ gốc FH4 (`fh4_map/raw`), Lộ trình A* (`fh4_map/processing`), và Hồ sơ tốc độ theo quãng đường |
| `preprocessing_metrics.csv` | CSV | Số liệu thống kê tiền xử lý trên toàn bộ ảnh (số làn phát hiện, loại làn UK, pixel biên/màu trong ROI, góc lệch, độ cong, thời gian xử lý) |
| `preprocessing_pipeline_chart.png` | Biểu đồ | Biểu đồ 6 bước tiền xử lý ảnh mặt đường và phân loại làn trái UK |
| `training_results.csv` | CSV | Lịch sử huấn luyện YOLO11-seg qua các epoch (`box_loss`, `seg_loss`, `mAP50`, `mAP50-95`) trên `data/processing/` |
| `evaluation_metrics.csv` | CSV | Bảng so sánh thực nghiệm `Accuracy`, `FP`, `FN`, `FPS` giữa `Hough_Transform`, `LaneNet`, và `Proposed_Spline` trên tập `data/processing/valid/` |
| `training_and_evaluation_chart.png` | Biểu đồ | Biểu đồ so sánh độ chính xác/FPS trên tập validation và đường cong hội tụ hàm mất mát huấn luyện |
| `fh4_log.csv` | CSV | Nhật ký điều khiển tự hành thời gian thực kèm tọa độ xe (`position_x`, `position_z`), loại làn UK và khoảng cách tới đích |

---

## Cấu trúc thư mục

```
CPV301/
├── requirements.txt               # ✅ Thư viện chuẩn (Phân tích, Tiền xử lý, Train YOLO, Notebook, Pytest)
├── requirements(testFH4).txt      # ✅ Thư viện đầy đủ để chạy song song thời gian thực với game Forza Horizon 4
├── fh4_map/                       # ✅ Thư mục bản đồ Forza Horizon 4 (Vương Quốc Anh)
│   ├── raw/                       # Bản đồ gốc (fh4_uk_map.png)
│   └── processing/                # Mặt nạ đường, khung xương, đồ thị & bản đồ lộ trình A*
├── config/
│   ├── data.yaml                  # ✅ Cấu hình dataset YOLO11-seg trỏ vào data/processing
│   └── bev.yaml                   # ✅ Cấu hình Bird's Eye View (default, fh4_bumper, tusimple)
├── src/
│   ├── map/                       # ✅ map_analyzer.py, navigator.py, dashboard.py (Đồ thị & Giao diện 4 ô FH4)
│   ├── capture/                   # ✅ screen_capture.py (mss screen capture)
│   ├── preprocessing/             # ✅ batch_processor.py, enhance.py, preprocessor.py, transform.py
│   ├── detection/                 # ✅ hough_transform.py, lanenet.py
│   ├── fitting/                   # ✅ spline_fitter.py (Nội suy đường tâm đa giác & Cubic B-Spline)
│   ├── proposed_method/           # ✅ yolo_spline.py (Enhance + YOLO11n-seg + Paint-Ridge Snap + B-Spline)
│   ├── control/                   # ✅ pure_pursuit.py (UK LHT), pid.py, controller.py, telemetry.py
│   ├── input/                     # ✅ actuator.py (vgamepad / pydirectinput / dry_run)
│   ├── evaluation/                # ✅ compare_methods.py, train_and_export.py
│   ├── pipeline_builder.py        # ✅ Bộ điều phối pipeline phát hiện & nội suy làn đường
│   ├── tusimple_to_yolo.py        # ✅ Chuyển đổi TuSimple JSON -> YOLO11-seg kèm CLAHE & kiểm chứng nhãn
│   └── path.py                    # ✅ Quản lý đường dẫn tập trung
├── data/
│   ├── raw/                       # Ảnh gốc + test_driving.mp4 + TUSimple/ (6,408 clips)
│   ├── processing/                # ✅ Dữ liệu đã xử lý:
│   │   ├── label_analysis/        # Báo cáo CSV, biểu đồ EDA, mask tô màu & 6,408 nhãn sạch (từ 00_label_analysis.ipynb)
│   │   ├── train/                 # 2,901 ảnh CLAHE + 2,901 nhãn YOLO-seg (từ 01_pre_processing.ipynb)
│   │   └── valid/                 # 725 ảnh CLAHE + 725 nhãn YOLO-seg (từ 01_pre_processing.ipynb)
│   └── output/                    # ✅ Trọng số best.pt, ảnh dự đoán predictions/ & báo cáo (từ 02_train.ipynb)
├── notebooks/
│   ├── 00_label_analysis.ipynb    # ✅ Bước 0: Phân tích, giải mã & hậu kiểm 100% nhãn TuSimple (6,408 mẫu)
│   ├── 01_pre_processing.ipynb    # ✅ Bước 1: Tiền xử lý nâng cao 6 bước -> xuất ra data/processing/ & Root
│   └── 02_train.ipynb             # ✅ Bước 2: Huấn luyện YOLO11-seg + B-Spline & đánh giá -> data/output/ & Root
├── docs/
│   └── explain.md                 # ✅ Giải thích chi tiết toàn bộ hệ thống & nâng cấp
├── tests/                         # ✅ Bộ kiểm thử tự động (23/23 bài test PASSED)
└── main.py                        # ✅ Vòng lặp tự hành trung tâm tích hợp Bản đồ + Làn trái UK (4-Quadrant UI)
```

---

## Sử dụng nhanh

### 1. Phân tích & Kiểm toán toàn bộ nhãn TuSimple (`notebooks/00_label_analysis.ipynb`)
Mở và chạy [`notebooks/00_label_analysis.ipynb`](notebooks/00_label_analysis.ipynb) để quét toàn bộ `6,408` mẫu có nhãn của TuSimple và xuất kết quả ra `data/processing/label_analysis/`.

### 2. Chạy tiền xử lý (`data/raw` $\rightarrow$ `data/processing` + Root CSV/Chart)
Mở và chạy [`notebooks/01_pre_processing.ipynb`](notebooks/01_pre_processing.ipynb) hoặc chạy lệnh:
```bash
python -m src.preprocessing.batch_processor
```

### 3. Chạy huấn luyện & đánh giá (`data/processing` $\rightarrow$ `data/output` + Root CSV/Chart)
Mở và chạy [`notebooks/02_train.ipynb`](notebooks/02_train.ipynb) hoặc chạy lệnh:
```bash
# Huấn luyện YOLO11n-seg + xuất dự đoán & báo cáo đánh giá:
python -m src.evaluation.train_and_export

# Hoặc chỉ chạy riêng bước suy luận & đánh giá nhanh với trọng số best.pt hiện có:
python -c "from src.evaluation.train_and_export import run_evaluation_and_export; run_evaluation_and_export(max_eval_images=30)"
```

### 4. Chạy hệ thống tự lái kết hợp Bản đồ `fh4_map`, Điểm đến & Làn đường Vương Quốc Anh (`main.py`)
*(Yêu cầu cài đặt `pip install -r "requirements(testFH4).txt"` khi chạy trực tiếp với game)*
```bash
# A. Mở bản đồ FH4 để click chuột chọn điểm đến trực tiếp, sau đó chạy tự lái:
python main.py --control --detector yolo --weights data/output/best.pt --pick-dest

# B. Đánh dấu điểm đến bằng mã nút giao (ví dụ Node 12: Edinburgh City Centre) ở chế độ mô phỏng video:
python main.py --view-only --detector hough --dest-node 12 --video data/raw/test_driving.mp4 --save data/output/fh4_sim_viewonly.mp4

# C. Đánh dấu điểm đến bằng tọa độ pixel (U, V) hoặc tọa độ thế giới (PositionX, PositionZ):
python main.py --view-only --detector hough --dest-pixel 850 130
python main.py --view-only --detector hough --dest-world 2900 4180
```

### 5. Chạy bộ kiểm thử tự động (23 Unit Tests)
```bash
pytest -v
```
