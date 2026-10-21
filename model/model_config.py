"""
model_config.py - Cấu hình & factory tập trung cho CPV301 Lane Detection
------------------------------------------------------------------------
Mọi trường trong các dataclass config dưới đây ĐỀU khớp 1:1 với tham số
của constructor thật trong `src/` (không còn truyền kwargs thừa / thiếu).

Hợp đồng chung với `PipelineBuilder`:
    detector.detect(frame) -> (results, lane_masks)
    detector.draw_lanes(frame, lane_masks) -> annotated_frame  # fallback
    spline_fitter.fit_and_draw(frame, lane_masks) -> annotated_frame

Phương pháp "proposed" (Enhance + YOLO11-seg + Spline) được bọc bằng
`ProposedDetectorAdapter` để về đúng hợp đồng trên; nó tự vẽ spline +
vùng đi được nội bộ, nên khi dựng pipeline cho "proposed" thì
`spline_fitter=None` (PipelineBuilder sẽ gọi detector.draw_lanes).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# ============ BƯỚC 1: Xử lý đường dẫn TRƯỚC hết mọi thứ ============
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.path import ROOT_DIR  # noqa: E402

_PROJECT_ROOT = ROOT_DIR

# ============ BƯỚC 2: Phát hiện thiết bị (an toàn nếu torch lỗi) ============
DEVICE = "cpu"
try:
    import torch  # noqa: F401

    DEVICE = str(
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
except Exception as e:  # pragma: no cover - phụ thuộc môi trường
    print(f"⚠️ Không thể import PyTorch, dùng CPU: {e}")


# ===========================================================================
# ============ Cấu hình chung ============
@dataclass
class GeneralConfig:
    project_root: str = str(_PROJECT_ROOT)
    checkpoint_dir: str = os.path.join(str(_PROJECT_ROOT), "checkpoint_for_llm")
    default_device: str = DEVICE
    img_size: Tuple[int, int] = (640, 360)
    fps_target: int = 30
    batch_size: int = 1


GENERAL = GeneralConfig()


# ============ Tiền xử lý ảnh (ImagePreprocessor) ============
@dataclass
class PreprocessorConfig:
    canny_low: int = 50
    canny_high: int = 150
    use_clahe: bool = False
    blur_ksize: int = 5
    morph_kernel_size: Tuple[int, int] = (3, 3)
    morph_iterations: int = 2
    roi_top_y_ratio: float = 0.42
    roi_top_left_x_ratio: float = 0.35
    roi_top_right_x_ratio: float = 0.65
    roi_bottom_left_x_ratio: float = 0.02
    roi_bottom_right_x_ratio: float = 0.98
    roi_bottom_margin_px: int = 5


PREPROCESSOR = PreprocessorConfig()


def build_preprocessor(**override_kwargs):
    """Trả về `ImagePreprocessor`; các tham số khớp 1:1 với constructor."""
    from src.preprocessing.preprocessor import ImagePreprocessor

    cfg = {**PREPROCESSOR.__dict__, **override_kwargs}
    return ImagePreprocessor(**cfg)


# ============ Hough Transform ============
@dataclass
class HoughConfig:
    # preprocessor: để None => detector tự tạo ImagePreprocessor() mặc định.
    # Truyền `preprocessor=...` qua detector_cfg nếu muốn dùng riêng.
    rho: float = 1.0
    theta: float = 0.01745  # ~np.pi/180
    threshold: int = 40
    min_line_length: int = 40
    max_line_gap: int = 50
    min_abs_slope: float = 0.3
    group_left_right: bool = True


HOUGH = HoughConfig()


# ============ Spline Fitting ============
@dataclass
class SplineConfig:
    spline_order: int = 3
    smoothing: float = 3.0
    num_points: int = 100
    default_color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 4


SPLINE = SplineConfig()


# ============ LaneNet (phiên bản cổ điển) ============
@dataclass
class LaneNetConfig:
    # preprocessor: None => detector tự tạo mặc định.
    eps: float = 0.06
    min_samples: int = 15
    max_lanes: int = 4
    poly_degree: int = 2


LANENET = LaneNetConfig()


# ============ YOLO11-Seg (thuần) & Proposed (YOLO + Spline) ============
@dataclass
class YOLOConfig:
    # --- tham số thật của YOLOLaneDetector / LaneDetector ---
    model_name: str = "yolo11n-seg.pt"
    conf_threshold: float = 0.5
    # (proposed only)
    spline_order: int = 3
    smoothing: float = 3.0
    enhance_input: bool = True
    # --- metadata (không truyền vào constructor) ---
    model_dir: str = os.path.join(GENERAL.checkpoint_dir, "yolo")
    imgsz: int = 640
    iou_threshold: float = 0.7
    device: str = GENERAL.default_device
    auto_download: bool = True

    @property
    def full_weights_path(self) -> str:
        return os.path.join(self.model_dir, self.model_name)


YOLO = YOLOConfig()


# ===========================================================================
# ============ Adapter cho phương pháp proposed ============
class ProposedDetectorAdapter:
    """
    Bọc `src.proposed_method.yolo_spline.LaneDetector` (Enhance + YOLO11-seg
    + Spline) về đúng hợp đồng của `PipelineBuilder`:
        detect(frame) -> (DetectionResult, raw_masks)
        draw_lanes(frame, masks) -> annotated_frame (đã vẽ spline + vùng đi được)

    Phương pháp này tự fit spline nội bộ, nên khi dùng trong PipelineBuilder
    thì `spline_fitter=None`; nhánh fallback `detector.draw_lanes(...)` sẽ
    trả về khung hình đã chú thích đầy đủ.
    """

    def __init__(
        self,
        yolo_weights: str = "yolo11n-seg.pt",
        spline_order: int = 3,
        smoothing: float = 3.0,
        confidence_threshold: float = 0.5,
        enhance_input: bool = True,
    ):
        from src.proposed_method.yolo_spline import LaneDetector

        self._detector = LaneDetector(
            yolo_weights=yolo_weights,
            spline_order=spline_order,
            smoothing=smoothing,
            confidence_threshold=confidence_threshold,
            enhance_input=enhance_input,
        )
        self._last_annotated: Optional[Any] = None

    def detect(self, frame):
        result = self._detector.detect_frame(frame)
        self._last_annotated = result.annotated_frame
        return result, result.raw_masks

    def draw_lanes(self, frame, lane_masks):
        if self._last_annotated is not None:
            return self._last_annotated
        return self._detector.draw_lanes(frame, lane_masks)


# ===========================================================================
# ============ Factory ============
DetectorType = Literal["hough", "lanenet", "yolo", "proposed"]

# Tên các tham số thật của từng constructor — chỉ truyền đúng những trường này.
_HOUGH_PARAMS = {
    "preprocessor",
    "rho",
    "theta",
    "threshold",
    "min_line_length",
    "max_line_gap",
    "min_abs_slope",
    "group_left_right",
}
_LANENET_PARAMS = {
    "preprocessor",
    "eps",
    "min_samples",
    "max_lanes",
    "poly_degree",
}
_YOLO_PARAMS = {"model_path", "confidence_threshold"}
_PROPOSED_PARAMS = {
    "yolo_weights",
    "spline_order",
    "smoothing",
    "confidence_threshold",
    "enhance_input",
}
_SPLINE_PARAMS = {
    "spline_order",
    "smoothing",
    "num_points",
    "default_color",
    "thickness",
}


def _filter_kwargs(cfg: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    """Chỉ giữ những khóa khớp với tham số constructor thật."""
    return {k: v for k, v in cfg.items() if k in allowed}


def build_detector(detector_type: DetectorType, **override_kwargs):
    """
    Dựng detector theo tên. Các tham số không khớp constructor sẽ bị loại bỏ
    (an toàn khi truyền thẳng cả dataclass config).
    """
    if detector_type == "hough":
        from src.detection.hough_transform import HoughLaneDetector

        cfg = _filter_kwargs({**HOUGH.__dict__, **override_kwargs}, _HOUGH_PARAMS)
        return HoughLaneDetector(**cfg)

    if detector_type == "lanenet":
        from src.detection.lanenet import LaneNetDetector

        cfg = _filter_kwargs({**LANENET.__dict__, **override_kwargs}, _LANENET_PARAMS)
        return LaneNetDetector(**cfg)

    if detector_type == "yolo":
        from src.yolo_detector import YOLOLaneDetector

        cfg = {
            "model_path": override_kwargs.pop("model_path", YOLO.full_weights_path),
            "confidence_threshold": override_kwargs.pop(
                "confidence_threshold", YOLO.conf_threshold
            ),
        }
        # Cho phép ghi đè trực tiếp model_path/confidence_threshold.
        cfg.update(_filter_kwargs(override_kwargs, _YOLO_PARAMS))
        return YOLOLaneDetector(**cfg)

    if detector_type == "proposed":
        cfg = {
            "yolo_weights": override_kwargs.pop(
                "yolo_weights", YOLO.full_weights_path
            ),
            "confidence_threshold": override_kwargs.pop(
                "confidence_threshold", YOLO.conf_threshold
            ),
            "spline_order": override_kwargs.pop("spline_order", YOLO.spline_order),
            "smoothing": override_kwargs.pop("smoothing", YOLO.smoothing),
            "enhance_input": override_kwargs.pop(
                "enhance_input", YOLO.enhance_input
            ),
        }
        cfg.update(_filter_kwargs(override_kwargs, _PROPOSED_PARAMS))
        return ProposedDetectorAdapter(**cfg)

    raise ValueError(
        f"detector_type không hợp lệ: {detector_type!r}. "
        "Chọn: 'hough', 'lanenet', 'yolo', 'proposed'"
    )


def build_spline_fitter(**override_kwargs):
    from src.fitting.spline_fitter import SplineFitter

    cfg = _filter_kwargs({**SPLINE.__dict__, **override_kwargs}, _SPLINE_PARAMS)
    return SplineFitter(**cfg)


def build_pipeline(
    detector_type: DetectorType = "hough",
    detector_cfg: Optional[Dict[str, Any]] = None,
    spline_cfg: Optional[Dict[str, Any]] = None,
):
    """
    Dựng `PipelineBuilder` với detector + spline fitter tương ứng.
    Với detector_type='proposed', spline fitter để None vì adapter đã tự
    vẽ spline + vùng đi được nội bộ (PipelineBuilder dùng draw_lanes fallback).
    """
    from src.pipeline_builder import PipelineBuilder

    detector = build_detector(detector_type, **(detector_cfg or {}))
    if detector_type == "proposed":
        fitter = None
    else:
        fitter = build_spline_fitter(**(spline_cfg or {}))
    return PipelineBuilder(detector=detector, spline_fitter=fitter)


if __name__ == "__main__":
    print(f"✅ CPV301 Model Config — thiết bị: {GENERAL.default_device}")
    print(f"   Project root: {GENERAL.project_root}")
    print(f"   YOLO weights: {YOLO.full_weights_path}")
    print(
        f"   Hough threshold={HOUGH.threshold} | "
        f"LaneNet eps={LANENET.eps} | "
        f"Spline order={SPLINE.spline_order}"
    )
    # Smoke-test: dựng được các factory không cần torch/ultralytics.
    _ = build_preprocessor()
    _ = build_spline_fitter()
    _ = build_detector("hough")
    _ = build_detector("lanenet")
    _ = build_pipeline("hough")
    _ = build_pipeline("lanenet")
    print("✅ Smoke test: preprocessor / spline / hough / lanenet / pipeline OK.")
    print("ℹ️  'yolo' & 'proposed' cần `ultralytics` (và weights) để khởi tạo.")
