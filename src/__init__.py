"""
Lane Detection & Autonomous Driving Project (CPV301)
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
    PROCESSING_DATA_DIR,
    OUTPUT_DATA_DIR,
    FH4_MAP_DIR,
    FH4_MAP_RAW_DIR,
    FH4_MAP_PROCESSING_DIR,
    MODEL_DIR,
    WEIGHTS_DIR,
    CONFIG_DIR,
    DOCS_DIR,
    CHECKPOINT_FOR_LLM_DIR,
)

# ---- Chụp màn hình (mss) ----
from src.capture.screen_capture import ScreenCapture, get_screen_quadrant_geometry

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

# ---- Bản đồ FH4 & Định tuyến A* (UK Left-Hand Traffic) ----
from src.map import (
    RoadNode,
    RoadEdge,
    FH4MapAnalyzer,
    FH4MapNavigator,
    FH4QuadDashboard,
    ensure_sample_fh4_map,
    DEFAULT_FH4_LANDMARKS,
    DEFAULT_FH4_ROAD_SEGMENTS,
)

# ---- Điều khiển (Pure Pursuit + PID + Telemetry) & Đầu vào (Actuator) ----
from src.input.actuator import VehicleActuator
from src.control import (
    PathData,
    PurePursuitController,
    classify_uk_lanes,
    extract_path_from_lanes,
    PIDSpeedController,
    SpeedGovernor,
    SteeringController,
    MapData,
    VehicleState,
    ControlCommand,
    InputBuffer,
    FH4CSVLogger,
    FH4AutonomousController,
    TelemetryReceiver,
    decode_fh4_packet,
    PACKET_FIELDS,
    TYPE_INFO,
    EXPECTED_PARSED_BYTES,
)

# ---- Điều phối pipeline & đánh giá ----
from src.pipeline_builder import PipelineBuilder
from src.evaluation.compare_methods import (
    evaluate_lane_predictions,
    benchmark_detector_on_frames,
    load_tusimple_samples,
)

__all__ = [
    # Đường dẫn & thư mục
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "PROCESSING_DATA_DIR",
    "OUTPUT_DATA_DIR",
    "FH4_MAP_DIR",
    "FH4_MAP_RAW_DIR",
    "FH4_MAP_PROCESSING_DIR",
    "MODEL_DIR",
    "WEIGHTS_DIR",
    "CONFIG_DIR",
    "DOCS_DIR",
    "CHECKPOINT_FOR_LLM_DIR",
    # Chụp màn hình
    "ScreenCapture",
    "get_screen_quadrant_geometry",
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
    # Bản đồ FH4 & Định tuyến A*
    "RoadNode",
    "RoadEdge",
    "FH4MapAnalyzer",
    "FH4MapNavigator",
    "FH4QuadDashboard",
    "ensure_sample_fh4_map",
    "DEFAULT_FH4_LANDMARKS",
    "DEFAULT_FH4_ROAD_SEGMENTS",
    # Điều khiển & đầu vào
    "VehicleActuator",
    "PathData",
    "PurePursuitController",
    "classify_uk_lanes",
    "extract_path_from_lanes",
    "PIDSpeedController",
    "SpeedGovernor",
    "SteeringController",
    "MapData",
    "VehicleState",
    "ControlCommand",
    "InputBuffer",
    "FH4CSVLogger",
    "FH4AutonomousController",
    "TelemetryReceiver",
    "decode_fh4_packet",
    "PACKET_FIELDS",
    "TYPE_INFO",
    "EXPECTED_PARSED_BYTES",
    # Điều phối & đánh giá
    "PipelineBuilder",
    "evaluate_lane_predictions",
    "benchmark_detector_on_frames",
    "load_tusimple_samples",
]
