"""
Lane Detection Project (CPV301)
"""

import sys
from pathlib import Path

# Đảm bảo ROOT_DIR luôn nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from src.yolo_detector import YOLOLaneDetector
from src.fitting.spline_fitter import SplineFitter
from src.preprocessing.preprocessor import ImagePreprocessor
from src.preprocessing.enhance import apply_clahe, denoise_image, adaptive_canny
from src.preprocessing.transform import PerspectiveTransformer
from src.detection.hough_transform import HoughLaneDetector
from src.detection.lanenet import LaneNetDetector
from src.proposed_method.yolo_spline import LaneDetector, DetectionResult
from src.evaluation.compare_methods import (
    evaluate_lane_predictions,
    benchmark_detector_on_frames,
)

__all__ = [
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
    "YOLOLaneDetector",
    "SplineFitter",
    "PipelineBuilder",
    "ImagePreprocessor",
    "apply_clahe",
    "denoise_image",
    "adaptive_canny",
    "PerspectiveTransformer",
    "HoughLaneDetector",
    "LaneNetDetector",
    "LaneDetector",
    "DetectionResult",
    "evaluate_lane_predictions",
    "benchmark_detector_on_frames",
]
