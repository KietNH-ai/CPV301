import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from src.yolo_detector import YOLOLaneDetector
from src.detection.hough_transform import HoughLaneDetector
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
        smoothing: float = 2.5,
        confidence_threshold: float = 0.25,
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
        self._hough_fallback: Optional[HoughLaneDetector] = None

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

    @staticmethod
    def _extract_yolo_guided_ego_masks(yolo_res, frame: np.ndarray) -> List[np.ndarray]:
        """
        Combine YOLO11 instance detections (`boxes.xyxy` + `masks.xy`) with resolution-adaptive
        horizontal paint-ridge snapping inside each detected ego-lane corridor so dashed and curved
        highway markings are tracked accurately across any frame resolution.
        """
        h, w = frame.shape[:2]
        mid_x = w / 2.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        kw = max(9, int(round(w * 0.024)) | 1)
        kh = max(3, int(round(h * 0.010)) | 1)
        ridge = cv2.subtract(gray, cv2.boxFilter(gray, cv2.CV_8U, (kw, kh)))

        boxes = (
            yolo_res.boxes.xyxy.cpu().numpy()
            if (getattr(yolo_res, "boxes", None) is not None and len(yolo_res.boxes))
            else np.empty((0, 4))
        )
        confs = (
            yolo_res.boxes.conf.cpu().numpy()
            if (getattr(yolo_res, "boxes", None) is not None and len(yolo_res.boxes))
            else np.empty((0,))
        )

        min_box_h = 0.20 * h
        min_box_w = 0.085 * w
        min_x1 = 0.042 * w
        max_x2 = 0.992 * w

        left_ego, right_ego = None, None
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = [float(v) for v in box]
            if y2 < 0.66 * h or (y2 - y1) < min_box_h or (x2 - x1) < min_box_w:
                continue
            xc = 0.5 * (x1 + x2)
            if xc < mid_x and x1 >= min_x1:
                if left_ego is None or conf > left_ego[0]:
                    left_ego = (float(conf), (x1, y1, x2, y2))
            elif xc >= mid_x and x2 <= max_x2:
                if right_ego is None or conf > right_ego[0]:
                    right_ego = (float(conf), (x1, y1, x2, y2))

        y_top_i = int(h * 0.42)
        y_bot_i = max(y_top_i + 10, h - max(3, int(round(h * 0.014))))
        step_y = max(2, int(round(h / 180.0)))
        half_sw_diag = max(10, int(round(w * 0.030)))
        half_sw_nom = max(16, int(round(w * 0.047)))
        ys = np.linspace(float(y_bot_i), float(y_top_i), 50, dtype=np.float32)
        guided_masks: List[np.ndarray] = []

        for side, ego_box in (("left", left_ego), ("right", right_ego)):
            if ego_box is None:
                continue
            pts = []
            _, (bx1, by1, bx2, by2) = ego_box
            for y in range(max(y_top_i, int(by1)), min(y_bot_i + 2, int(by2)), step_y):
                t = (y - by1) / max(1.0, by2 - by1)
                x_diag = (bx2 - t * (bx2 - bx1)) if side == "left" else (bx1 + t * (bx2 - bx1))
                xa, xb = max(0, int(x_diag - half_sw_diag)), min(w, int(x_diag + half_sw_diag))
                row_r = ridge[y, xa:xb]
                if row_r.size > 0 and int(row_r.max()) >= 10:
                    pts.append([float(xa + int(np.argmax(row_r))), float(y)])
                else:
                    pts.append([float(x_diag), float(y)])

            if len(pts) >= 4:
                p_arr = np.array(pts, dtype=np.float32)
                deg = 2 if len(p_arr) >= 20 else 1
                poly = np.polyfit(p_arr[:, 1], p_arr[:, 0], deg=deg)
                if deg == 2:
                    max_curv = 0.0018 * (720.0 / max(120.0, float(h)))
                    poly[0] = float(np.clip(poly[0], -max_curv, max_curv))
                xs = np.polyval(poly, ys)
                guided_masks.append(np.column_stack((np.clip(xs, 0, w - 1), ys)).astype(np.int32))

        return guided_masks

    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run Stage 1 (Enhance) -> Stage 2 (YOLO11) -> Stage 3 (Spline Fitting) on a BGR frame."""
        proc_frame = self.preprocess_frame(frame)
        raw_masks = []
        yolo_results = None
        try:
            yolo_results, raw_masks = self.detector.detect(proc_frame)
        except Exception:
            raw_masks = []

        if (
            yolo_results is not None
            and isinstance(yolo_results, (list, tuple))
            and len(yolo_results) > 0
            and hasattr(yolo_results[0], "boxes")
        ):
            guided = self._extract_yolo_guided_ego_masks(yolo_results[0], frame)
            if len(guided) >= 2:
                raw_masks = guided
            elif len(guided) == 1:
                raw_masks = guided

        if len(raw_masks) < 2:
            if self._hough_fallback is None:
                self._hough_fallback = HoughLaneDetector()
            _, hough_masks = self._hough_fallback.detect(frame)
            if hough_masks:
                if not raw_masks:
                    raw_masks = hough_masks
                else:
                    mid_x = frame.shape[1] / 2.0
                    has_left = any(float(np.mean(m[:, 0])) < mid_x for m in raw_masks)
                    for hm in hough_masks:
                        hm_left = float(np.mean(hm[:, 0])) < mid_x
                        if (hm_left and not has_left) or (not hm_left and has_left):
                            raw_masks = list(raw_masks) + [hm]
                            break

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
