import sys
from pathlib import Path

# Đảm bảo project root nằm trong sys.path để import được `model` & `src`
# khi chạy trực tiếp: python test_fh4/main.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.model_config import build_pipeline, GENERAL  # noqa: E402

# Khởi tạo 1 lần dùng cho toàn bộ vòng lặp điều khiển
pipeline = build_pipeline("hough")
print(f"Pipeline sẵn sàng — chạy trên {GENERAL.default_device}")
