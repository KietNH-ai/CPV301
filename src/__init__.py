"""
Lane Detection Project (CPV301)
Public API của package `src`. Import các thành phần chính ở đây để
`from src import <Ten>` hoạt động thống nhất cho toàn bộ pipeline.
Mọi tên liệt kê trong `__all__` đều phải được import thật ở trên.
"""

import sys
from pathlib import Path

# Đảm bảo ROOT_DIR luôn nằm trong sys.path khi import package `src`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---- Đường dẫn & thư mục dự án ----
from src.path import (
    ROOT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DATA_DIR,
    MODEL_DIR,
    WEIGHTS_DIR,
    CONFIG_DIR,
    DOCS_DIR,
    CHECKPOINT_FOR_LLM_DIR,
)

# ---- Tiền xử lý ảnh ----
from src.preprocessing.preprocessor import ImagePreprocessor
from src.preprocessing.enhance import apply_clahe, denoise_image, adaptive_canny
from src.preprocessing.transform import PerspectiveTransformer

# ---- Phát hiện làn đường ----
from src.yolo_detector import YOLOLaneDetector
from src.detection.hough_transform import HoughLaneDetector
from src.detection.lanenet import LaneNetDetector

# ---- Khớp curve & phương pháp đề xuất ----
from src.fitting.spline_fitter import SplineFitter
from src.proposed_method.yolo_spline import LaneDetector, DetectionResult

# ---- Điều phối pipeline & đánh giá ----
from src.pipeline_builder import PipelineBuilder
from src.evaluation.compare_methods import (
    evaluate_lane_predictions,
    benchmark_detector_on_frames,
)

__all__ = [
    # Đường dẫn & thư mục
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUTPUT_DATA_DIR",
    "MODEL_DIR",
    "WEIGHTS_DIR",
    "CONFIG_DIR",
    "DOCS_DIR",
    "CHECKPOINT_FOR_LLM_DIR",
    # Tiền xử lý ảnh
    "ImagePreprocessor",
    "apply_clahe",
    "denoise_image",
    "adaptive_canny",
    "PerspectiveTransformer",
    # Phát hiện làn đường
    "YOLOLaneDetector",
    "HoughLaneDetector",
    "LaneNetDetector",
    # Khớp curve & phương pháp đề xuất
    "SplineFitter",
    "LaneDetector",
    "DetectionResult",
    # Điều phối & đánh giá
    "PipelineBuilder",
    "evaluate_lane_predictions",
    "benchmark_detector_on_frames",
]
