# Checkpoint: Khôi phục Tiền xử lý & Huấn luyện lại YOLO11-seg với Early Stopping

**Thời gian**: 2026-10-24 08:52:00 (UTC+7)  
**Tác vụ**: Giữ nguyên pipeline tiền xử lý (pre_processing) chuẩn, cấu hình và kích hoạt huấn luyện lại YOLO11-seg trên tập `data/processing/` với Early Stopping (`patience=5`).

---

## 1. Summary of Completed Work (Tóm tắt công việc đã hoàn thành)
1. **Làm rõ yêu cầu người dùng**:
   - Người dùng xác nhận: *"giữ nguyên tiền xử lý, nhưng train lại, vì output nó vẽ ra quá tệ, à nhớ thêm earlystoping"*.
   - Nguyên nhân output trước đây vẽ xấu/gãy khúc: Model trước đó chỉ được chạy thử nghiệm `epochs=2` trên CPU dẫn đến `mAP50(M)` của segmentation mask chỉ đạt `0.017` (1.7%), các mask dự đoán bị đứt đoạn, méo mó và rời rạc.

2. **Giữ nguyên & Chuẩn hóa module Tiền xử lý (`src/preprocessing/preprocessor.py`)**:
   - Khôi phục nguyên vẹn các tham số ROI tiêu chuẩn khớp với tài liệu và bài lab của nhóm (`notebooks/01_pre_processing.ipynb`):
     - `roi_top_y_ratio = 0.42`
     - `roi_top_left_x_ratio = 0.35`
     - `roi_top_right_x_ratio = 0.65`
     - `roi_bottom_left_x_ratio = 0.02`
     - `roi_bottom_right_x_ratio = 0.98`
     - `roi_bottom_margin_px = 5`
   - Khôi phục hàm `detect_edges` về thuật toán Canny nguyên bản (`cv2.Canny`), loại bỏ bộ lọc ridge tương phản ngang thực nghiệm (tránh kích hoạt nhiễu khi nhìn vào cánh đồng ruộng rơm rạ / cỏ khô).
   - Vẫn duy trì bộ đệm mặt nạ ROI (`_get_cached_roi_mask`) để đảm bảo hiệu năng $\ge 30$ FPS khi chạy thời gian thực.

3. **Tích hợp tính năng Early Stopping (`patience=5`) vào quá trình Huấn luyện**:
   - Cập nhật [`src/evaluation/train_and_export.py`](file:///h:/FPT/4/CPV/project/CPV301/src/evaluation/train_and_export.py):
     - Bổ sung tham số `patience: int = 5` vào hàm `run_training_and_export()`.
     - Truyền `patience=patience` trực tiếp vào `model.train()` của Ultralytics YOLO. Khi độ chính xác validation (mAP hoặc loss) không cải thiện sau 5 epochs liên tiếp, quá trình huấn luyện sẽ tự động kích hoạt Early Stopping và lưu lại checkpoint có kết quả tốt nhất (`best.pt`).
     - Bật tính năng `cache=True` (RAM cache) giúp nạp trước 256 ảnh huấn luyện vào bộ nhớ, tăng tốc độ xử lý trên CPU từ 2-3 lần (loại bỏ nghẽn I/O đọc ổ đĩa liên tục mỗi epoch).
     - Đặt `workers=2` và `batch=8` tối ưu hóa đa luồng trên CPU 8 nhân.
   - Cập nhật [`notebooks/02_train.ipynb`](file:///h:/FPT/4/CPV/project/CPV301/notebooks/02_train.ipynb):
     - Đồng bộ Cell 3 với `PATIENCE = 5` và hiển thị chú thích rõ ràng về cơ chế Early Stopping.

4. **Kích hoạt tiến trình Huấn luyện (Background Task)**:
   - Tiến trình huấn luyện đã khởi chạy nền qua môi trường Python ảo `.\.venv\Scripts\python.exe -m src.evaluation.train_and_export`.
   - File cấu hình thực tế `data/output/yolo_train/tusimple_lane/args.yaml` đã xác nhận:
     - `task: segment`
     - `epochs: 15`
     - `patience: 5` (Early Stopping kích hoạt)
     - `cache: true`
     - `device: cpu` (chạy trên CPU an toàn, không chiếm VRAM GPU của game Forza Horizon 4 đang mở).

---

## 2. Current Task & Progress (Tiến độ hiện tại)
- Tiến trình huấn luyện YOLO11-seg đang chạy trong nền (Background Task).
- Sau khi hoàn thành tất cả các epoch (hoặc dừng sớm khi Early Stopping kích hoạt):
  1. Trọng số tốt nhất được tự động sao chép sang `data/output/best.pt`.
  2. Bảng chỉ số huấn luyện lưu tại `training_results.csv`.
  3. Bảng đánh giá so sánh lưu tại `evaluation_metrics.csv`.
  4. Biểu đồ tổng hợp quá trình huấn luyện & đánh giá xuất ra `training_and_evaluation_chart.png`.
  5. Ảnh dự đoán kèm phân loại làn đường UK xuất ra `data/output/predictions/`.

---

## 3. Modified & Created Files (Danh sách file đã thay đổi / tạo mới)
- [`src/preprocessing/preprocessor.py`](file:///h:/FPT/4/CPV/project/CPV301/src/preprocessing/preprocessor.py): Khôi phục nguyên bản tiền xử lý (Canny thuần + ROI chuẩn 0.42/0.35/0.65/0.02/0.98/5).
- [`src/evaluation/train_and_export.py`](file:///h:/FPT/4/CPV/project/CPV301/src/evaluation/train_and_export.py): Thêm tham số `patience=5` (Early Stopping), `cache=True`, `workers=2`, `batch=8`.
- [`notebooks/02_train.ipynb`](file:///h:/FPT/4/CPV/project/CPV301/notebooks/02_train.ipynb): Đồng bộ cấu hình Early Stopping trong Cell 3.
- `checkpoint_for_llm/checkpoint_20261024_retrain_yolo_earlystopping.md`: File checkpoint tiến độ hiện tại.

---

## 4. Next Action Steps (Các bước hành động tiếp theo)
1. **Theo dõi tiến trình huấn luyện**:
   - Kiểm tra log và file `training_results.csv` khi các epoch tiếp theo hoàn thành.
2. **Kiểm tra trọng số mới `data/output/best.pt`**:
   - Đánh giá chất lượng phân đoạn trên tập validation và xác nhận giá trị `mAP50(M)` cao vượt trội so với phiên bản 2 epochs cũ.
3. **Thử nghiệm chạy thực tế với game**:
   - Chạy lệnh nhận diện làn đường bằng model mới:
     ```bash
     python main.py --view-only --detector proposed --weights data/output/best.pt
     ```
   - Xác nhận đường cong và hành lang lái được vẽ mịn màng, không còn hiện tượng gãy khúc hay vẽ đường ảo.

---

## 5. Known Issues & Blockers
- Không có blocker. Quá trình huấn luyện đang chạy ổn định trên CPU nền mà không làm gián đoạn game hay các tác vụ khác.
