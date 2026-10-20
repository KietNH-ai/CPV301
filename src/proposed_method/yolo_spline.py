import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from src.yolo_detector import YOLOLaneDetector
from src.fitting.spline_fitter import SplineFitter
from src.preprocessing.enhance import apply_clahe, denoise_image


class DetectionResult:
    """
    Container for a single frame/image detection result from `LaneDetector`.
    Matches the `result.visualize(save_path=...)` interface documented in `README.md`.
    """

    def __init__(
        self,
        original_frame: np.ndarray,
        annotated_frame: np.ndarray,
        raw_masks: List[np.ndarray],
        fitted_curves: List[np.ndarray],
    ):
        self.original_frame = original_frame
        self.annotated_frame = annotated_frame
        self.raw_masks = raw_masks
        self.fitted_curves = fitted_curves

    def visualize(self, save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Save (if `save_path` is provided) and return the annotated image."""
        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), self.annotated_frame)
        return self.annotated_frame


class LaneDetector:
    """
    Proposed hybrid method: Optimized Image Enhancement + YOLOv11-seg + Cubic B-Spline Fitting.
    Matches the API in `README.md`:
    ```python
    from src.proposed_method.yolo_spline import LaneDetector
    detector = LaneDetector(yolo_weights='models/weights/yolo11_lane.pt', spline_order=3, confidence_threshold=0.5)
    result = detector.detect('data/test_image.jpg')
    result.visualize(save_path='output/result.jpg')
    ```
    """

    def __init__(
        self,
        yolo_weights: str = "yolo11n-seg.pt",
        spline_order: int = 3,
        smoothing: float = 3.0,
        confidence_threshold: float = 0.5,
        enhance_input: bool = True,
        detector: Optional[object] = None,
    ):
        self.yolo_weights = yolo_weights
        self.spline_order = spline_order
        self.smoothing = smoothing
        self.confidence_threshold = confidence_threshold
        self.enhance_input = enhance_input

        self.spline_fitter = SplineFitter(
            spline_order=spline_order,
            smoothing=smoothing,
        )
        self._detector = detector

    @property
    def detector(self):
        if self._detector is None:
            self._detector = YOLOLaneDetector(
                model_path=self.yolo_weights,
                confidence_threshold=self.confidence_threshold,
            )
        return self._detector

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Stage 1: CLAHE contrast enhancement + Gaussian/Median denoising."""
        if not self.enhance_input:
            return frame
        enhanced = apply_clahe(frame)
        return denoise_image(enhanced, gaussian_ksize=3, median_ksize=3)

    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run Stage 1 (Enhance) -> Stage 2 (YOLO11) -> Stage 3 (Spline Fitting) on a BGR frame."""
        proc_frame = self.preprocess_frame(frame)
        _, raw_masks = self.detector.detect(proc_frame)
        fitted_curves = self.spline_fitter.fit_multiple_lanes(raw_masks)
        annotated = self.spline_fitter.fit_and_draw(frame, raw_masks)

        if len(fitted_curves) >= 2:
            # Sort left-to-right and shade drivable corridor between the two primary lanes
            sorted_curves = sorted(fitted_curves, key=lambda c: float(c[0, 0]))
            annotated = self.spline_fitter.draw_lane_area(
                annotated, sorted_curves[0], sorted_curves[1], alpha=0.20
            )

        return DetectionResult(
            original_frame=frame,
            annotated_frame=annotated,
            raw_masks=raw_masks,
            fitted_curves=fitted_curves,
        )

    def detect(self, image_or_path: Union[str, Path, np.ndarray]):
        """
        Detect lanes on either a file path (`str`/`Path`) or an already loaded `np.ndarray` frame.
        If `image_or_path` is an `np.ndarray`, returns `(results, fitted_curves)` for `PipelineBuilder`
        compatibility, or if a path is passed, returns a `DetectionResult`.
        """
        if isinstance(image_or_path, (str, Path)):
            img_path = Path(image_or_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise ValueError(f"Unable to read image: {img_path}")
            return self.detect_frame(frame)

        res = self.detect_frame(image_or_path)
        return res, res.fitted_curves

    def draw_lanes(self, frame: np.ndarray, lane_masks: List[np.ndarray]) -> np.ndarray:
        """Compatible with `PipelineBuilder`."""
        return self.spline_fitter.fit_and_draw(frame, lane_masks)
