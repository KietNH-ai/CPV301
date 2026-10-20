# 🛠️ Hướng Dẫn Cài Đặt, Chạy Thực Nghiệm & Kiểm Thử Dự Án (CPV301)

Tài liệu này hướng dẫn từng bước cách chạy toàn bộ các module, xuất ảnh/video kết quả, chạy notebook trình diễn và kiểm thử tự động cho dự án **Phát Hiện Làn Đường (CPV301)**.

---

## 1. Cấu Trúc Thư Mục Hiện Tại Của Dự Án

```text
CPV301/
├── config/
│   └── data.yaml                        # Cấu hình huấn luyện YOLOv11-seg (1 class: lane)
├── data/
│   ├── raw/                             # Dữ liệu đầu vào (20.jpg, curved_road.jpg, test_driving.mp4)
│   ├── processed/                       # Dữ liệu sau tiền xử lý / chuẩn bị train YOLO
│   └── output/                          # Kết quả xuất ra (ảnh so sánh, video .mp4, file .json)
├── docs/
│   ├── 00_lane_detection_tutorial.md    # Tổng quan lý thuyết hệ thống
│   ├── 01_how_to_preprocessing.md       # Lý thuyết Tiền xử lý ảnh
│   ├── 02_how_to_use_yolo_11.md         # Hướng dẫn YOLOv11-seg
│   ├── 03_how_to_use_spline_fitting.md  # Lý thuyết làm mượt B-Spline
│   ├── 04_how_to_use_hough_transform.md # Lý thuyết Hough Transform
│   ├── 05_how_to_use_lanenet.md         # Lý thuyết mạng đa nhánh LaneNet
│   ├── 06_project_report.md             # 📄 BÁO CÁO TỔNG KẾT DỰ ÁN
│   └── 07_user_guide.md                 # 🛠️ FILE HƯỚNG DẪN NÀY
├── model/
│   └── demo.py                          # Script chạy tự động toàn bộ thực nghiệm & xuất báo cáo
├── notebooks/
│   ├── pre_processing.ipynb             # Notebook từng bước Tiền xử lý + Hough + Spline (Cell 0 -> 7)
│   └── demo.ipynb                       # Notebook tổng hợp so sánh các phương pháp & chạy video
├── src/
│   ├── __init__.py                      # Khai báo các class xuất ra của toàn dự án
│   ├── path.py                          # Quản lý đường dẫn tự động
│   ├── yolo_detector.py                 # Class YOLOLaneDetector
│   ├── piplinebuilder.py                # Class PipelineBuilder xử lý video
│   ├── pipelinebuilder.py               # Alias chống lỗi chính tả khi import
│   ├── preprocessing/                   # Gói tiền xử lý (ImagePreprocessor, CLAHE, Bird's Eye View)
│   ├── detection/                       # Gói phát hiện làn (HoughLaneDetector, LaneNetDetector, YOLO)
│   ├── fitting/                         # Gói khớp đường cong (SplineFitter)
│   ├── proposed_method/                 # Gói giải pháp đề xuất (LaneDetector, DetectionResult)
│   └── evaluation/                      # Gói chấm điểm TuSimple (Accuracy, FP, FN, FPS)
└── tests/
    ├── test_spline_fitter.py            # 7 bài test cho SplineFitter & PipelineBuilder
    └── test_preprocessing_and_hough.py  # 4 bài test cho Preprocessing, Hough, LaneNet & Evaluation
```

---

## 2. Cài Đặt Môi Trường Nhanh

Mở **PowerShell** hoặc **Terminal** tại thư mục gốc `CPV301` và chạy:

```powershell
# Cài đặt các thư viện cốt lõi để chạy toàn bộ module và kiểm thử
pip install numpy scipy scikit-learn opencv-python matplotlib pytest pyyaml

# (Tùy chọn) Nếu muốn chạy hoặc huấn luyện trực tiếp mô hình YOLOv11 bằng PyTorch:
pip install torch torchvision ultralytics
```

---

## 3. Cách Chạy Từng Phần Của Dự Án

### Cách 1: Chạy 1 lệnh duy nhất để xuất toàn bộ Ảnh so sánh & Báo cáo JSON (Khuyên dùng)
Chạy file `model/demo.py`:
```powershell
python model/demo.py
```
**Kết quả thu được ngay trong `data/output/`**:
- `data/output/01_preprocessing_mask.jpg`: Ảnh mặt nạ biên trong vùng ROI sau khi nối nét đứt.
- `data/output/02_hough_transform.jpg`: Kết quả phát hiện của Hough Transform.
- `data/output/03_lanenet_detection.jpg`: Kết quả phân cụm làn đường của LaneNet.
- `data/output/04_proposed_spline_fitting.jpg`: Kết quả làm mượt B-Spline + tô vùng làn đường di chuyển.
- `data/output/comparison_grid.jpg`: Ảnh ghép 2×2 đối chiếu cả 4 bước trên cùng 1 khung hình.
- `data/output/comparison_report.json`: Bảng chỉ số `accuracy`, `fp`, `fn`, `fps`.

---

### Cách 2: Chạy từng Module độc lập bằng dòng lệnh (CLI)

Bạn có thể chạy riêng lẻ từng thuật toán trên bất kỳ bức ảnh nào:

```powershell
# 1. Chỉ chạy Hough Transform trên ảnh bất kỳ
python src/detection/hough_transform.py --input data/raw/20.jpg --output data/output/02_hough_transform.jpg

# 2. Chỉ chạy LaneNet trên ảnh bất kỳ
python src/detection/lanenet.py --input data/raw/20.jpg --output data/output/03_lanenet_detection.jpg

# 3. Chạy công cụ đánh giá so sánh các phương pháp
python src/evaluation/compare_methods.py --dataset data/raw --methods hough lanenet proposed --output data/output/comparison_report.json
```

---

### Cách 3: Chạy qua Jupyter Notebook (Trực quan hóa khi thuyết trình)

Mở 2 file notebook trong VS Code hoặc Jupyter Lab:
1. **`notebooks/pre_processing.ipynb`**:
   - Trình bày chi tiết từng bước tiền xử lý ảnh từ **Cell 0 $\rightarrow$ Cell 6** (Đọc ảnh $\rightarrow$ Ảnh xám $\rightarrow$ Canny $\rightarrow$ Cắt khung hình thang ROI $\rightarrow$ Đóng hình thái học Morphological Closing).
   - **Cell 7** trình bày bước chuyển tiếp từ mặt nạ `connected_mask` sang phát hiện `HoughLaneDetector` và làm mượt `SplineFitter`.
2. **`notebooks/demo.ipynb`**:
   - Trình bày kết quả đối sánh tổng thể trên ảnh thực tế TuSimple (`20.jpg`), đối sánh trên **đường cong gắt (`curved_road.jpg`)** để chứng minh ưu thế của B-Spline so với Hough Transform, in bảng điểm TuSimple và xuất 2 video `.mp4` (`output_hough.mp4` và `output_proposed.mp4`).

---

### Cách 4: Sử dụng trong Code Python của riêng bạn

#### A. Chạy Phương pháp Đề xuất trên Ảnh lẻ:
```python
from src.proposed_method.yolo_spline import LaneDetector

# Khởi tạo bộ phát hiện (nếu đã có file trọng số YOLO riêng, truyền vào yolo_weights)
detector = LaneDetector(
    yolo_weights="yolo11n-seg.pt",
    spline_order=3,
    smoothing=2.5,
    confidence_threshold=0.5,
)

# Nếu muốn dùng nhánh LaneNet + Spline mà không cần tải trọng số YOLO:
from src.detection.lanenet import LaneNetDetector
detector = LaneDetector(detector=LaneNetDetector(), spline_order=3, smoothing=2.5)

result = detector.detect("data/raw/20.jpg")
result.visualize(save_path="data/output/my_result.jpg")
```

#### B. Chạy Xử lý Video bằng `PipelineBuilder`:
```python
from src import HoughLaneDetector, LaneNetDetector, SplineFitter, PipelineBuilder

hough = HoughLaneDetector()
lanenet = LaneNetDetector()
spline = SplineFitter(spline_order=3, smoothing=2.5)

# 1. Xuất video chỉ dùng Hough Transform cổ điển
pipe_hough = PipelineBuilder(detector=hough, spline_fitter=None)
pipe_hough.process_video("test_driving.mp4", "output_hough.mp4")

# 2. Xuất video dùng Phát hiện + Làm mượt B-Spline
pipe_proposed = PipelineBuilder(detector=lanenet, spline_fitter=spline)
pipe_proposed.process_video("test_driving.mp4", "output_proposed.mp4")
```

---

## 4. Chạy Kiểm Thử Tự Động (Unit Tests)

Trước khi commit hoặc nộp bài, hãy chạy lệnh sau để kiểm tra toàn bộ 11 bài test của hệ thống:

```powershell
python -m pytest -v
```
Nếu màn hình hiện `11 passed`, toàn bộ các hàm tiền xử lý, phát hiện làn đường, làm mượt đường cong B-Spline, ghép nối pipeline và chấm điểm đánh giá đều đang hoạt động chính xác 100%.

---

## 5. Hướng Dẫn Đẩy Code Lên GitHub (Khi Bạn Đã Kiểm Tra Xong)

Khi bạn đã xem qua các file kết quả và muốn đẩy phần việc của mình lên GitHub:

```powershell
# 1. Kiểm tra danh sách các file đã thêm/sửa
git status

# 2. Đưa các file vào vùng chuẩn bị commit
git add src/ tests/ model/ notebooks/ docs/ data/

# 3. Tạo commit ghi nhận đóng góp của bạn
git commit -m "feat: complete preprocessing, Hough, LaneNet, proposed pipeline, evaluation, demo and report"

# 4. Đẩy lên nhánh GitHub của nhóm
git push origin main
```
