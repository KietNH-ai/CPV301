"""
test_fh4/main.py — Điểm vào chạy vòng lặp thị giác + điều khiển FH4.
Kết nối `model.model_config.build_pipeline` với vòng lặp điều khiển chính tại `main.py`.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.model_config import build_pipeline, GENERAL  # noqa: E402
from main import main as run_main  # noqa: E402

# Khởi tạo 1 lần khi import trực tiếp để kiểm tra nhanh
pipeline = build_pipeline("hough")

if __name__ == "__main__":
    print(f"Pipeline sẵn sàng — chạy trên {GENERAL.default_device}")
    run_main()
