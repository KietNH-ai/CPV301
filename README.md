# Lane Detection &amp; Self-Driving trong Forza Horizon 4

Dự án môn **CPV301 - Thị giác máy tính** xây dựng pipeline phát hiện làn đường và điều khiển xe tự động trong game **Forza Horizon 4**, sử dụng **camera bumper** (góc nhìn cố định gắn trên xe).

> ⚠️ **Cảnh báo**: Chạy hoàn toàn ở chế độ **offline**. Forza Horizon 4 có anti-cheat; sử dụng phần mềm giả lập tay cầm khi online có thể dẫn đến khóa tài khoản.

## Mục tiêu

1. Xây dựng pipeline điều khiển xe tự lái trong Forza Horizon 4 theo hướng **modular**:
   `Chụp màn hình → Tiền xử lý (ROI + BEV + Canny) → Phát hiện làn đường → Khớp spline → Điều khiển (Pure Pursuit + PID) → Gửi lệnh vào game`.
2. So sánh các phương pháp phát hiện làn đường: **Classic CV (Hough + Spline)** làm phương pháp chính, **YOLO11-seg** làm phương pháp học sâu so sánh.
3. Đánh giá trên cả tập dữ liệu **TuSimple** (chuẩn学术界) và môi trường thực tế **Forza Horizon 4**.

## Kiến trúc tổng thể

```
┌────────────┐   ┌───────────────┐   ┌─────────────┐   ┌───────────────┐   ┌───────────────┐   ┌────────────┐
│  Capture   │→  │ Preprocessing │→  │  Detection  │→  │Spline Fitting │→  │   Control     │→  │   Input    │
│ (mss, ROI) │   │CLAHE+Canny+BEV│   │Hough / YOLO │   │ (scipy spline)│   │PurePursuit+PID│   │ (vgamepad) │
└────────────┘   └───────────────┘   └─────────────┘   └───────────────┘   └───────────────┘   └────────────┘
```

- **Camera**: Bumper cam (góc nhìn cố định, không rung) → ma trận biến đổi Bird's Eye View (BEV) được hiệu chỉnh cố định một lần.
- **Điều khiển**: Góc lái tính bằng Pure Pursuit từ spline curve; tốc độ điều khiển bằng PID.
- **Chế độ an toàn**: `main.py` hỗ trợ chế độ `--view-only` (chỉ vẽ làn đường, không gửi lệnh vào game) để test pipeline trước khi điều khiển thật.

## Trạng thái hiện tại

| Module | Trạng thái | Ghi chú |
|---|---|---|
| `src/fitting/spline_fitter.py` | ✅ Hoàn thành | Có fallback 3 tầng, có test |
| `src/yolo_detector.py` | ✅ Hoàn thành | Wrapper YOLO11-seg, làm phương pháp so sánh |
| `src/pipeline_builder.py` | ✅ Hoàn thành | (đang đổi tên từ `piplinebuilder.py`) |
| `src/path.py` | ✅ Hoàn thành | Quản lý đường dẫn tập trung |
| `tests/test_spline_fitter.py` | ✅ Hoàn thành | 7 test case, cover edge case |
| `docs/` (6 hướng dẫn) | ✅ Hoàn thành | Tài liệu lý thuyết các phương pháp |
| `src/preprocessing/` (ROI, CLAHE, Canny, BEV) | 🚧 Đang triển khai | Chưa có code, đang refactor từ notebook |
| `src/detection/hough_transform.py` | 🚧 Đang triển khai | Baseline classic CV |
| `src/capture/` (mss screen capture) | ⏳ Chưa triển khai | Cần cho chạy trên Forza |
| `src/control/` (Pure Pursuit + PID) | ⏳ Chưa triển khai | Phần then chốt để điều khiển xe |
| `src/input/` (vgamepad) | ⏳ Chưa triển khai | Gửi lệnh vào game |
| `main.py` (vòng lặp điều khiển) | ⏳ Chưa triển khai | Ghép toàn bộ pipeline |
| Fine-tune YOLO trên ảnh Forza | ⏳ Chưa triển khai | Cần thu thập &amp; label ~500–1000 ảnh game |
| Kết quả thí nghiệm so sánh | ⏳ Chưa thực hiện | Sẽ cập nhật sau khi chạy |

> **Lưu ý**: LaneNet chỉ được nghiên cứu lý thuyết (trong `docs/05_how_to_use_lanenet.md`) nhưng **không triển khai** do chi phí huấn luyện cao và không cần thiết với môi trường game (đường rõ nét, classic CV đã chạy tốt).

## Cấu trúc thư mục

```
CPV301/
├── config/
│   ├── data.yaml              # Cấu hình dataset YOLO (TuSimple / ảnh Forza)
│   └── bev.yaml               # ROI + ma trận homography cho bumper cam (đang thêm)
├── src/
│   ├── capture/               # (đang thêm) Chụp màn hình bằng mss, crop ROI
│   ├── preprocessing/         # (đang thêm) CLAHE, blur, Canny, biến đổi BEV
│   ├── detection/
│   │   ├── yolo_detector.py   # YOLO11-seg (phương pháp so sánh)
│   │   └── hough_transform.py # (đang thêm) Baseline Hough Transform
│   ├── fitting/
│   │   └── spline_fitter.py   # Khớp B-spline cho làn đường cong
│   ├── control/               # (đang thêm) Pure Pursuit + PID tốc độ
│   ├── input/                 # (đang thêm) vgamepad gửi lệnh vào Forza
│   ├── pipeline_builder.py    # Pipeline xử lý video/frame
│   └── path.py                # Quản lý đường dẫn
├── tests/                     # Unit test (pytest)
├── docs/                      # Tài liệu lý thuyết các phương pháp
├── data/                      # raw / processed / output (dùng .gitkeep)
├── notebooks/                 # Notebook phân tích (tham khảo)
├── main.py                    # (đang thêm) Vòng lặp điều khiển, có --view-only
├── requirements.txt
├── pytest.ini
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

> **PyTorch &amp; CUDA**: Nên cài PyTorch theo phiên bản CUDA của máy từ
> https://pytorch.org/get-started/locally/ trước khi chạy `pip install -r requirements.txt`.
>
> **vgamepad** chỉ hỗ trợ Windows (Forza Horizon 4 cũng chỉ chạy Windows).

## Sử dụng

### 1. Chạy pipeline trên video (không cần game)

```python
from src.detection.hough_transform import HoughLaneDetector  # sẵn sàng sau khi thêm
from src.fitting.spline_fitter import SplineFitter
from src.pipeline_builder import PipelineBuilder

detector = HoughLaneDetector()
fitter = SplineFitter(spline_order=3, smoothing=3.0)
pipeline = PipelineBuilder(detector=detector, spline_fitter=fitter)
pipeline.process_video('input_video.mp4', 'output_result.mp4')
```

### 2. Chạy tự lái trong Forza Horizon 4 (sắp có)

```bash
# Chế độ chỉ xem (vẽ làn đường, không gửi lệnh) - an toàn để test
python main.py --view-only

# Chế độ điều khiển thật (chỉ khi offline!)
python main.py --control
```

### 3. Train/fine-tune YOLO trên ảnh Forza (sắp có)

```bash
yolo detect train data=config/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640
```

## Kết quả đánh giá

> 🚧 **Đang thực hiện** — sẽ cập nhật sau khi chạy thí nghiệm trên TuSimple và Forza.

Các chỉ số đo lường: Accuracy, FP, FN (theo chuẩn TuSimple), FPS, độ trễ end-to-end, và tỷ lệ xe đi đúng làn trong Forza.

## Tài liệu tham khảo

1. **Tusimple Benchmark**: https://github.com/TuSimple/tusimple-benchmark
2. **YOLO11 / Ultralytics**: https://docs.ultralytics.com/
3. **Spline Fitting**: De Boor, C. (2001). *A Practical Guide to Splines*.
4. **Pure Pursuit**: Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm*.
5. **LaneNet** (nghiên cứu lý thuyết): Neven, D. et al. (2018). *Towards End-to-End Lane Detection: an Instance Segmentation Approach*.

## Thành viên nhóm

- **Nhóm trưởng**: Nguyễn Hữu Kiệt
- **Thành viên**: Nguyễn Hoàng Phương, Lê Huy Hùng, Nguyễn Thiện Nhân

## Giấy phép

Dự án phân phối theo giấy phép MIT. Xem file `LICENSE`.
