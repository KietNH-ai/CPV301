"""
model_config.py - Centralized model configuration & loader for CPV301 Lane Detection
--------------------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Literal

# ============ BƯỚC 1: Xử lý đường dẫn TRƯỚC hết mọi thứ ============
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ====================================================================

# Bây giờ mới import từ src
from src.path import ROOT_DIR
_PROJECT_ROOT = ROOT_DIR

# ============ BƯỚC 2: Kiểm tra thiết bị (an toàn nếu torch lỗi) ============
DEVICE = "cpu"  # mặc định
try:
    import torch
    DEVICE = str(torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    ))
except Exception as e:
    print(f"⚠️ Không thể import PyTorch, dùng CPU: {e}")
# ===========================================================================


# ============ Cấu hình chung ============
@dataclass
class GeneralConfig:
    project_root: str = str(_PROJECT_ROOT)
    checkpoint_dir: str = os.path.join(str(_PROJECT_ROOT), "checkpoint_for_llm")
    default_device: str = DEVICE
    img_size: tuple[int, int] = (640, 360)
    fps_target: int = 30
    batch_size: int = 1


GENERAL = GeneralConfig()


# ============ Hough Transform ============
@dataclass
class HoughConfig:
    rho: int = 1
    theta: float = 0.01745
    threshold: int = 15
    min_line_length: int = 30
    max_line_gap: int = 10
    angle_tolerance: float = 0.3


HOUGH = HoughConfig()


# ============ Spline Fitting ============
@dataclass
class SplineConfig:
    spline_order: int = 3
    smoothing: float = 3.0
    min_points_per_curve: int = 5
    num_sample_points: int = 100


SPLINE = SplineConfig()


# ============ LaneNet ============
@dataclass
class LaneNetConfig:
    use_clustering: bool = True
    eps: float = 1.5
    min_samples: int = 15
    binary_threshold: float = 0.5
    weights_path: Optional[str] = None


LANENET = LaneNetConfig()


# ============ YOLO11-Seg + Spline ============
@dataclass
class YOLOConfig:
    model_name: str = "yolo11n-seg.pt"
    model_dir: str = os.path.join(GENERAL.checkpoint_dir, "yolo")
    conf_threshold: float = 0.5
    iou_threshold: float = 0.7
    imgsz: int = 640
    device: str = GENERAL.default_device
    auto_download: bool = True

    @property
    def full_weights_path(self) -> str:
        return os.path.join(self.model_dir, self.model_name)


YOLO = YOLOConfig()


# ============ Pipeline Factory ============
DetectorType = Literal["hough", "lanenet", "yolo"]


def build_detector(detector_type: DetectorType, **override_kwargs):
    if detector_type == "hough":
        from src.detection.hough_transform import HoughLaneDetector
        cfg = {**HOUGH.__dict__, **override_kwargs}
        return HoughLaneDetector(**cfg)

    elif detector_type == "lanenet":
        from src.detection.lanenet import LaneNetDetector
        cfg = {**LANENET.__dict__, **override_kwargs}
        return LaneNetDetector(**cfg)

    elif detector_type == "yolo":
        from src.proposed_method.yolo_spline import YoloSplineDetector
        cfg = {**YOLO.__dict__, **override_kwargs}
        return YoloSplineDetector(**cfg)

    raise ValueError(
        f"detector_type không hợp lệ: {detector_type}. "
        "Chọn: 'hough', 'lanenet', 'yolo'"
    )


def build_spline_fitter(**override_kwargs):
    from src.fitting.spline_fitter import SplineFitter
    cfg = {**SPLINE.__dict__, **override_kwargs}
    return SplineFitter(**cfg)


def build_pipeline(
    detector_type: DetectorType = "hough",
    detector_cfg: Optional[Dict[str, Any]] = None,
    spline_cfg: Optional[Dict[str, Any]] = None,
):
    from src.pipelinebuilder import PipelineBuilder

    detector = build_detector(detector_type, **(detector_cfg or {}))
    fitter = build_spline_fitter(**(spline_cfg or {}))

    return PipelineBuilder(
        detector=detector,
        spline_fitter=fitter,
        device=GENERAL.default_device,
    )


if __name__ == "__main__":
    print(f"✅ CPV301 Model Config — thiết bị: {GENERAL.default_device}")
    print(f"   Project root: {GENERAL.project_root}")
    print(f"   YOLO weights: {YOLO.full_weights_path}")
    print(f"   Hough threshold: {HOUGH.threshold} | Spline order: {SPLINE.spline_order}")