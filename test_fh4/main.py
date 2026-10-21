from model.model_config import build_pipeline, GENERAL

# Khởi tạo 1 lần dùng cho toàn bộ vòng lặp điều khiển
pipeline = build_pipeline("hough")
print(f"Pipeline sẵn sàng — chạy trên {GENERAL.default_device}")