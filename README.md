# Lane Detection & Self-Driving trong Forza Horizon 4

Dự án môn **CPV301 - Thị giác máy tính** xây dựng pipeline phát hiện làn đường và điều khiển xe tự động trong game **Forza Horizon 4**, sử dụng **camera bumper** (góc nhìn cố định gắn trên xe).

> ⚠️ **Cảnh báo**: Chạy hoàn toàn ở chế độ **offline**. Forza Horizon 4 có anti-cheat; sử dụng phần mềm giả lập tay cầm khi online có thể dẫn đến khóa tài khoản.

## Mục tiêu

1. Xây dựng pipeline điều khiển xe tự lái trong Forza Horizon 4 theo hướng **modular**:
   `Chụp màn hình → Tiền xử lý (ROI + BEV + Canny) → Phát hiện làn đường → Khớp spline → Điều khiển (Pure Pursuit + PID) → Gửi lệnh vào game`.
2. So sánh các phương pháp phát hiện làn đường: **Classic CV (Hough + Spline)** làm phương pháp chính, **YOLO11-seg** làm phương pháp học sâu so sánh, **LaneNet** (phiên bản cổ điển) làm baseline thứ ba.
3. Đánh giá trên cả tập dữ liệu **TuSimple** (chuẩn học thuật) và môi trường thực tế **Forza Horizon 4**.

## Kiến trúc tổng thể

```
┌────────────┐   ┌───────────────┐   ┌─────────────┐   ┌───────────────┐   ┌───────────────┐   ┌────────────┐
│  Capture   │→  │ Preprocessing │→  │  Detection  │→  │Spline Fitting │→  │   Control     │→  │   Input    │
│ (mss, ROI) │   │CLAHE+Canny+BEV│   │Hough/YOLO/  │   │ (scipy spline)│   │PurePursuit+PID│   │ (vgamepad) │
└────────────┘   └───────────────┘   │   LaneNet   │   └───────────────┘   └───────────────┘   └────────────┘
                                     └─────────────┘
```

- **Camera**: Bumper cam (góc nhìn cố định, không rung) → ma trận biến đổi Bird's Eye View (BEV) được hiệu chỉnh cố định một lần (hiện đang hardcode mặc định trong `src/preprocessing/transform.py`).
- **Điều khiển**: Góc lái tính bằng Pure Pursuit từ spline curve; tốc độ điều khiển bằng PID. Logic điều khiển độc lập đã có trong `test_fh4/`, **chưa tích hợp vào `src/`**.
- **Chế độ an toàn**: `main.py` (vòng lặp điều khiển, hỗ trợ `--view-only`) **đang chờ triển khai**.

## Trạng thái hiện tại

| Module | Trạng thái | Ghi chú |
|---|---|---|
| `src/preprocessing/` (ROI, CLAHE, Canny, BEV) | ✅ Hoàn thành | `preprocessor.py` (grayscale+CLAHE+blur → Canny → ROI → morph closing), `enhance.py`, `transform.py` (BEV warp/unwarp) |
| `src/detection/hough_transform.py` | ✅ Hoàn thành | HoughLinesP + gom cụm trái/phải + fit đường thẳng, có CLI chạy ảnh |
| `src/detection/lanenet.py` | ✅ Hoàn thành (phiên bản cổ điển) | Binary segmentation + DBSCAN instance clustering + fit đa thức (không dùng NN đã huấn luyện), có CLI |
| `src/yolo_detector.py` + `src/detection/yolo11_detector.py` | ✅ Hoàn thành | Wrapper YOLO11-seg (ultralytics), làm phương pháp so sánh |
| `src/fitting/spline_fitter.py` | ✅ Hoàn thành | B-spline với 3 cấp fallback, có test |
| `src/proposed_method/yolo_spline.py` | ✅ Hoàn thành | Phương pháp đề xuất: Enhance (CLAHE+denoise) + YOLO11-seg + Spline; API `LaneDetector` / `DetectionResult.visualize()` |
| `src/evaluation/compare_methods.py` | ✅ Hoàn thành | Metric theo chuẩn TuSimple (Accuracy / FP / FN) + đo FPS, có CLI xuất JSON |
| `src/pipelinebuilder.py` | ✅ Hoàn thành | Xử lý video frame-by-frame (đã gỡ bản class trùng) |
| `src/__init__.py` | ✅ Hoàn thành | Public API thống nhất, `__all__` khớp 100% với import thật |
| `src/path.py` | ✅ Hoàn thành | Quản lý đường dẫn tập trung, tự tạo thư mục |
| `model/demo.py` | ✅ Hoàn thành | Chạy so sánh 3 phương pháp trên ảnh mẫu, xuất lưới so sánh + `comparison_report.json` |
| `test_fh4/` (FH4_controler.py + fh4_telemetry.py) | ✅ Hoàn thành (độc lập) | Đọc gói telemetry UDP của FH4 + tính lái/tăng ga/phanh + gửi lệnh qua virtual gamepad/keyboard + ghi CSV; **chưa nối với pipeline `src/`** |
| `tests/` (pytest) | ✅ Hoàn thành | `test_spline_fitter.py`, `test_preprocessing_and_hough.py` (đã sửa import) |
| `docs/` | ✅ Hoàn thành | 8 tài liệu `00`–`07` (hướng dẫn lý thuyết + project report + user guide) |
| `data/output/` (kết quả sơ bộ) | ✅ Có | Ảnh so sánh từng bước, video output, `comparison_report.json` |
| `src/capture/` (mss screen capture) | ⏳ Chưa triển khai | Cần để chạy thật trên Forza |
| Tích hợp `src/control/` + `src/input/` | ⏳ Chưa tích hợp | Logic đã có trong `test_fh4/`, cần chuyển vào `src/` và nối với pipeline thị giác |
| `main.py` (vòng lặp điều khiển, `--view-only` / `--control`) | ⏳ Chưa triển khai | Ghép toàn bộ pipeline + capture + control (`test_fh4/main.py` mới là placeholder rỗng) |
| `config/bev.yaml` | ⏳ Chưa có | Tham số BEV đang để mặc định hardcode trong `transform.py` |
| Fine-tune YOLO trên ảnh Forza | ⏳ Chưa thực hiện | Cần thu thập & label ~500–1000 ảnh game |
| Đánh giá đầy đủ trên TuSimple + Forza | ⏳ Chưa thực hiện | Mới có kết quả sơ bộ trên ảnh mẫu (xem mục Kết quả đánh giá) |

> **Lưu ý về LaneNet**: Đã triển khai dưới dạng **phiên bản cổ điển** (binary segmentation theo gradient + phân cụm DBSCAN, không huấn luyện mạng nơ-ron) để làm baseline so sánh trong `model/demo.py`. Phiên bản end-to-end dựa trên học sâu chưa triển khai do chi phí huấn luyện cao và không cần thiết với môi trường game (đường rõ nét, classic CV đã chạy tốt).

### 📌 Các điểm cần dọn dẹp (còn lại)

- `test_fh4/main.py` hiện đang là file rỗng (placeholder), chưa có code vòng lặp điều khiển.
- `data/` và `fh4_log.csv` **được giữ cố ý** làm dữ liệu mẫu và log chạy thử (không phải vấn đề cần gỡ).

## Cấu trúc thư mục

```
CPV301/
├── config/
│   └── data.yaml                  # Cấu hình dataset YOLO (1 class: lane)
│   # bev.yaml                     # (chưa có) tham số BEV đang hardcode trong transform.py
├── src/
│   ├── preprocessing/             # ✅ enhance.py, preprocessor.py, transform.py (BEV)
│   ├── detection/                 # ✅ hough_transform.py, lanenet.py, yolo11_detector.py (re-export)
│   ├── fitting/                   # ✅ spline_fitter.py (B-spline, 3 cấp fallback)
│   ├── proposed_method/           # ✅ yolo_spline.py (Enhance + YOLO11 + Spline)
│   ├── evaluation/                # ✅ compare_methods.py (metric TuSimple + FPS)
│   ├── yolo_detector.py           # ✅ Wrapper YOLO11-seg
│   ├── spline_fitter.py           # ✅ shortcut re-export
│   ├── pipelinebuilder.py         # ✅ Xử lý video (tên thống nhất)
│   ├── __init__.py                # ✅ Public API thống nhất
│   └── path.py                    # ✅ Quản lý đường dẫn tập trung
│   # capture/                     # (chưa có) mss screen capture
│   # control/ , input/            # (chưa tích hợp vào src) — logic nằm trong test_fh4/
├── test_fh4/                      # ✅ Độc lập: fh4_telemetry.py (UDP) + FH4_controler.py (điều khiển + gamepad/keyboard)
│   └── main.py                    # (placeholder rỗng) vòng lặp điều khiển, đang chờ viết
├── model/
│   └── demo.py                    # ✅ Chạy so sánh 3 phương pháp, xuất grid + comparison_report.json
├── tests/                         # ✅ pytest: test_spline_fitter.py, test_preprocessing_and_hough.py
├── notebooks/                     # 4 notebook tham khảo (preprocessing, train, demo)
├── docs/                          # ✅ 8 tài liệu 00–07 (hướng dẫn + project report + user guide)
├── data/
│   ├── raw/                       # ảnh mẫu + test_driving.mp4
│   ├── processed/
│   └── output/                    # ✅ kết quả sơ bộ: ảnh so sánh, video, comparison_report.json
├── checkpoint_for_llm/            # Checkpoint cho agent (theo AGENTS.md)
├── fh4_log.csv                    # Log chạy thử controller (giữ lại để test)
├── main.py                        # (chưa có ở root) vòng lặp điều khiển --view-only / --control
├── requirements.txt
├── pytest.ini
├── AGENTS.md / GEMINI.md          # Quy tắc làm việc với LLM/agent (token budget + checkpoint)
├── LICENSE
└── README.md
```

## Cài đặt

```bash
# Clone
git clone https://github.com/KietNH-ai/CPV301.git
cd CPV301

# Tạo môi trường ảo (Python 3.10+)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Cài phụ thuộc
pip install -r requirements.txt
```

> **PyTorch & CUDA**: Nên cài PyTorch theo phiên bản CUDA của máy từ
> https://pytorch.org/get-started/locally/ trước khi chạy `pip install -r requirements.txt`.
>
> **vgamepad** chỉ hỗ trợ Windows (Forza Horizon 4 cũng chỉ chạy Windows).

## Sử dụng

### 1. Chạy pipeline trên video (không cần game)

```python
from src.detection.hough_transform import HoughLaneDetector
from src.fitting.spline_fitter import SplineFitter
from src.pipelinebuilder import PipelineBuilder

detector = HoughLaneDetector()
fitter = SplineFitter(spline_order=3, smoothing=3.0)
pipeline = PipelineBuilder(detector=detector, spline_fitter=fitter)
pipeline.process_video('test_driving.mp4', 'output_result.mp4')
```

### 2. Phương pháp đề xuất (Enhance + YOLO11-seg + Spline)

```python
from src.proposed_method.yolo_spline import LaneDetector

detector = LaneDetector(yolo_weights='yolo11n-seg.pt', spline_order=3, confidence_threshold=0.5)
result = detector.detect('data/raw/20.jpg')
result.visualize(save_path='data/output/result.jpg')
```

### 3. Chạy demo so sánh 3 phương pháp (Hough / LaneNet / Proposed)

```bash
python model/demo.py
# Hoặc xuất báo cáo đánh giá dạng JSON:
python -m src.evaluation.compare_methods --methods hough lanenet proposed --output data/output/comparison_report.json
```

Kết quả được lưu vào `data/output/` (ảnh so sánh từng bước, video, `comparison_grid.jpg`, `comparison_report.json`).

### 4. Chạy tự lái trong Forza Horizon 4 (chưa có)

`main.py` với chế độ `--view-only` / `--control` **đang chờ triển khai**. Hiện tại phần điều khiển xe chạy độc lập trong `test_fh4/`:

```bash
# Đọc telemetry + điều khiển (chỉ Windows, chỉ chạy offline!)
python test_fh4/FH4_controler.py
```

### 5. Train/fine-tune YOLO trên ảnh Forza (chưa thực hiện)

```bash
yolo detect train data=config/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640
```

## Kết quả đánh giá

> ✅ **Đã có kết quả sơ bộ** trên ảnh mẫu trong `data/raw/` (sample nhỏ, **chưa phải** benchmark đầy đủ trên TuSimple hay Forza). File gốc: `data/output/comparison_report.json`.

| Phương pháp | Accuracy | FP | FN | FPS |
|---|---|---|---|---|
| Hough Transform | 0.7931 | 0.0 | 0.0 | 89.25 |
| LaneNet (phiên bản cổ điển) | 1.0000 | 0.0 | 0.0 | 9.45 |
| Proposed (YOLO11 + Spline) | 1.0000 | 0.0 | 0.0 | 9.29 |

> ⚠️ Accuracy = 1.0 của LaneNet và Proposed chỉ phản ánh ảnh mẫu đơn giản; cần đánh giá lại trên tập lớn hơn. Các chỉ số đầy đủ sẽ cập nhật sau khi chạy thí nghiệm trên TuSimple và Forza (Accuracy, FP, FN theo chuẩn TuSimple, FPS, độ trễ end-to-end, tỷ lệ xe đi đúng làn trong Forza).

## Tài liệu tham khảo

1. **Tusimple Benchmark**: https://github.com/TuSimple/tusimple-benchmark
2. **YOLO11 / Ultralytics**: https://docs.ultralytics.com/
3. **Spline Fitting**: De Boor, C. (2001). *A Practical Guide to Splines*.
4. **Pure Pursuit**: Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm*.
5. **LaneNet** (nghiên cứu lý thuyết + triển khai phiên bản cổ điển): Neven, D. et al. (2018). *Towards End-to-End Lane Detection: An Instance Segmentation Approach*.

## Thành viên nhóm

- **Nhóm trưởng**: Nguyễn Hữu Kiệt
- **Thành viên**: Nguyễn Hoàng Phương, Lê Huy Hùng, Nguyễn Thiện Nhân

## Giấy phép

Dự án phân phối theo giấy phép MIT. Xem file `LICENSE`.
