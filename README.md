# 🚗 Phát Hiện Làn Đường Bằng Thị Giác Máy Tính

## 📋 Giới Thiệu Dự Án

Dự án này tập trung vào bài toán **phát hiện làn đường** (Lane Detection) - một thành phần cốt lõi trong hệ thống lái xe tự động và hỗ trợ người lái. Chúng tôi triển khai và so sánh nhiều phương pháp xử lý ảnh khác nhau trên tập dữ liệu **Tusimple**, đồng thời đề xuất một giải pháp tối ưu kết hợp **YOLO11** và **Spline Fitting** để xử lý hiệu quả các đoạn đường cong phức tạp.

## 🎯 Mục Tiêu

- Triển khai và đánh giá nhiều phương pháp phát hiện làn đường: YOLO11, Spline Fitting, Hough Transform, Lanenet
- So sánh hiệu năng với nghiên cứu tham khảo **"Lane Detection Using Machine Learning"**
- Đề xuất giải pháp tối ưu hóa giai đoạn xử lý ảnh
- Cải thiện độ chính xác trong việc phát hiện làn đường cong, đặc biệt ở các góc cua

## 📊 Tập Dữ Liệu: Tusimple

**Tusimple** là một trong những bộ dữ liệu chuẩn cho bài toán phát hiện làn đường, bao gồm:
- **3,626 video clip** (1 giây mỗi clip, 20 frame/clip)
- **36,268 frame** tổng cộng
- **7,000 frame** được chú thích cho huấn luyện
- **2,000 frame** cho kiểm thử
- Đặc điểm: Đa dạng điều kiện giao thông, đường cong, đường thẳng, các tình huống che khuất

## 🛠️ Các Phương Pháp Triển Khai

### 1. **YOLO11** (You Only Look Once)
- Mô hình phát hiện đối tượng hiện đại, tốc độ cao
- Ưu điểm: Xử lý thời gian thực, độ chính xác tốt
- Ứng dụng: Phát hiện vùng chứa làn đường như các đối tượng

### 2. **Hough Transform**
- Phương pháp cổ điển dựa trên biến đổi hình học
- Ưu điểm: Hiệu quả với đường thẳng, dễ triển khai
- Hạn chế: Kém hiệu quả với đường cong, nhạy cảm với nhiễu

### 3. **LaneNet**
- Mạng nơ-ron tích chập đa nhiệm (multi-task)
- Kiến trúc: Nhánh phân đoạn nhị phân + nhánh nhúng phiên bản
- Ưu điểm: Xử lý tốt làn đường thay đổi số lượng

### 4. **Spline Fitting**
- Phương pháp khớp đường cong sử dụng hàm spline
- Ưu điểm: Xử lý mượt mà các đoạn đường cong
- Ứng dụng: Tái tạo hình dạng làn đường từ các điểm phát hiện

## 💡 Đề Xuất Tối Ưu: YOLO11 + Spline Fitting

### Vấn Đề Cần Giải Quyết
Các phương pháp truyền thống như Hough Transform gặp khó khăn lớn với:
- Đường cong ở góc cua
- Làn đường có độ cong lớn
- Tình huống che khuất một phần

### Giải Pháp Đề Xuất

**Giai đoạn 1: Xử Lý Ảnh Tối Ưu**
- Cân bằng histogram thích ứng (CLAHE) để cải thiện độ tương phản
- Lọc nhiễu bằng bộ lọc trung vị và Gaussian
- Phát hiện cạnh bằng Canny với ngưỡng thích ứng
- Biến đổi góc nhìn (Bird's Eye View) để giảm méo dạng

**Giai đoạn 2: Phát Hiện Với YOLO11**
- Sử dụng YOLO11 để phát hiện nhanh chóng vùng chứa làn đường
- Trích xuất các điểm đặc trưng trên làn đường
- Lọc bỏ các phát hiện sai dựa trên vị trí và hình dạng

**Giai đoạn 3: Khớp Đường Cong Bằng Spline Fitting**
- Áp dụng **Cubic Spline Interpolation** lên các điểm đã phát hiện
- Tái tạo hình dạng chính xác của làn đường cong
- Đảm bảo tính liên tục và mượt mà của đường cong kết quả

### Ưu Điểm Của Giải Pháp
✅ **Xử lý tốt đường cong**: Spline fitting mô hình hóa chính xác các đoạn cong phức tạp  
✅ **Tốc độ cao**: YOLO11 đảm bảo khả năng xử lý thời gian thực  
✅ **Độ bền với nhiễu**: Giai đoạn xử lý ảnh tối ưu giảm tác động của nhiễu  
✅ **Chính xác cao**: Kết hợp điểm mạnh của cả hai phương pháp  

## 📁 Cấu Trúc Dự Án

```
lane-detection-project/
├── data/
│   ├── raw/              # Tập dữ liệu Tusimple
│   ├── processed/             # Dữ liệu sau xử lý
│   └── output/                # Kết quả đầu ra
├── models/
│   ├── yolo11/                # Mô hình YOLO11
│   ├── lanenet/               # Mô hình LaneNet
│   └── weights/               # Trọng số đã huấn luyện
├── src/
│   ├── preprocessing/         # Tiền xử lý ảnh
│   │   ├── enhance.py         # Cải thiện chất lượng ảnh
│   │   └── transform.py       # Biến đổi góc nhìn
│   ├── detection/
│   │   ├── yolo11_detector.py # YOLO11 implementation
│   │   ├── hough_transform.py # Hough Transform
│   │   └── lanenet.py         # LaneNet implementation
│   ├── fitting/
│   │   └── spline_fitter.py   # Spline Fitting
│   ├── proposed_method/       # Giải pháp đề xuất
│   │   └── yolo_spline.py     # YOLO11 + Spline Fitting
│   └── evaluation/            # Đánh giá hiệu năng
├── notebooks/                 # Jupyter notebooks phân tích
├── results/                   # Báo cáo kết quả
├── requirements.txt           # Thư viện phụ thuộc
└── README.md                  # File này
```

## 🚀 Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- CUDA 11.0+ (để tăng tốc GPU)
- OpenCV 4.5+
- PyTorch 1.9+

### Cài Đặt Thư Viện

```bash
# Clone repository
git clone https://github.com/your-team/lane-detection-project.git
cd lane-detection-project

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt phụ thuộc
pip install -r requirements.txt
```

### Chuẩn Bị Dữ Liệu

```bash
# Tải tập dữ liệu Tusimple
# Đặt dữ liệu vào thư mục data/tusimple/
# Cấu trúc:
# data/tusimple/
#   ├── clips/
#   ├── label_data_*.json
#   └── test_label.json
```

## 🎮 Sử Dụng

### Chạy phương pháp đề xuất (YOLO11 + Spline Fitting)

```python
from src.proposed_method.yolo_spline import LaneDetector

# Khởi tạo bộ phát hiện
detector = LaneDetector(
    yolo_weights='models/weights/yolo11_lane.pt',
    spline_order=3,
    confidence_threshold=0.5
)

# Phát hiện làn đường trên ảnh
result = detector.detect('data/test_image.jpg')
result.visualize(save_path='output/result.jpg')
```

### Chạy đánh giá so sánh

```bash
# Đánh giá tất cả các phương pháp
python src/evaluation/compare_methods.py \
    --dataset data/tusimple \
    --methods yolo11 hough lanenet proposed \
    --output results/comparison_report.json
```

### Sử dụng các module riêng lẻ

```bash
# Chỉ chạy Hough Transform
python src/detection/hough_transform.py --input data/sample.jpg

# Chỉ chạy LaneNet
python src/detection/lanenet.py --input data/sample.jpg

# Chạy Spline Fitting trên các điểm đã có
python src/fitting/spline_fitter.py --points data/points.json
```

## 📈 Kết Quả Đánh Giá

### Đo Lường Hiệu Năng
Chúng tôi sử dụng các chỉ số chuẩn của Tusimple benchmark:

| Chỉ số | Mô tả |
|--------|-------|
| **Accuracy** | Tỷ lệ điểm phát hiện chính xác |
| **FP** | False Positive - Số điểm phát hiện sai |
| **FN** | False Negative - Số điểm bị bỏ sót |

### Kết Quả So Sánh Dự Kiến

| Phương pháp | Accuracy | FP | FN | Tốc độ (FPS) | Xử lý đường cong |
|-------------|----------|----|----|--------------|-------------------|
| Hough Transform | Trung bình | Cao | Trung bình | 30+ | Kém |
| LaneNet | Tốt | Trung bình | Thấp | ~15 | Tốt |
| YOLO11 | Tốt | Thấp | Trung bình | ~45 | Trung bình |
| **YOLO11 + Spline Fitting** | **Rất tốt** | **Thấp** | **Thấp** | **~35** | **Xuất sắc** |

## 📚 Tài Liệu Tham Khảo

1. **Nghiên cứu so sánh**: *Lane Detection Using Machine Learning*
2. **Tusimple Dataset**: https://github.com/TuSimple/tusimple-benchmark
3. **YOLO11**: Phát triển từ YOLO series, tối ưu cho phát hiện đối tượng
4. **LaneNet**: Neven, D. et al. (2018). "Towards End-to-End Lane Detection: an Instance Segmentation Approach"
5. **Spline Fitting**: De Boor, C. (2001). "A Practical Guide to Splines"

## 👥 Thành Viên Nhóm

- **Nhóm trưởng**: Nguyễn Hữu Kiệt
- **Thành viên 1**: Nguyễn Hoàng Phương
- **Thành viên 2**: Lê Huy Hùng
- **Thành viên 3**: Nguyễn Thiện Nhân


## 📝 Công Việc Đang Thực Hiện

- [x] Nghiên cứu lý thuyết các phương pháp
- [x] Thu thập và phân tích tập dữ liệu Tusimple
- [x] Triển khai Hough Transform
- [x] Triển khai LaneNet
- [x] Triển khai YOLO11
- [x] Triển khai Spline Fitting
- [x] Phát triển giải pháp đề xuất YOLO11 + Spline Fitting
- [ ] Thực hiện thí nghiệm so sánh
- [ ] Viết báo cáo chi tiết
- [ ] Chuẩn bị thuyết trình

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork repository
2. Tạo branch tính năng (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 Giấy Phép

Dự án này được phân phối dưới giấy phép MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên Hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ qua email: [your-team-email@example.com]

---

⭐ **Nếu dự án hữu ích, hãy cho chúng tôi một sao trên GitHub!** ⭐
