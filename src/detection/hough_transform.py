import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.preprocessor import ImagePreprocessor


class HoughLaneDetector:
    """
    Classical Lane Detector using Preprocessing + Probabilistic Hough Transform (cv2.HoughLinesP).
    
    Implements the interface required by `PipelineBuilder`:
    - `detect(frame) -> (raw_lines, lane_masks)`
    - `draw_lanes(frame, lane_masks) -> annotated_frame`
    """

    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 40,
        min_line_length: int = 40,
        max_line_gap: int = 50,
        min_abs_slope: float = 0.3,
        group_left_right: bool = True,
    ):
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_abs_slope = min_abs_slope
        self.group_left_right = group_left_right

    def detect_from_mask(
        self, binary_mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Run Probabilistic Hough Transform directly on a 2D binary edge/connected mask
        (such as `connected_mask` from Cell 6 of `notebooks/pre_processing.ipynb`).
        """
        height, width = binary_mask.shape[:2]
        raw_lines = cv2.HoughLinesP(
            binary_mask,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if raw_lines is None or len(raw_lines) == 0:
            return raw_lines, []

        if not self.group_left_right:
            masks = []
            for line in raw_lines:
                x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
                masks.append(np.array([[x1, y1], [x2, y2]], dtype=np.int32))
            return raw_lines, masks

        left_pts = []
        right_pts = []
        mid_x = width / 2.0

        for line in raw_lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 1e-3:
                continue
            slope = dy / dx
            if abs(slope) < self.min_abs_slope:
                # Ignore near-horizontal lines (hood, horizon, bumper shadows)
                continue

            # Sample intermediate points along the segment weighted by length
            seg_len = int(max(2, np.hypot(dx, dy) // 8))
            xs = np.linspace(x1, x2, seg_len)
            ys = np.linspace(y1, y2, seg_len)
            seg_pts = np.column_stack((xs, ys))

            if slope < 0 and (x1 < mid_x * 1.15 or x2 < mid_x * 1.15):
                left_pts.append(seg_pts)
            elif slope > 0 and (x1 > mid_x * 0.85 or x2 > mid_x * 0.85):
                right_pts.append(seg_pts)

        lane_masks: List[np.ndarray] = []
        y_bottom = height - 5
        y_top = int(height * self.preprocessor.roi_top_y_ratio)

        for pts_group in (left_pts, right_pts):
            if not pts_group:
                continue
            all_pts = np.vstack(pts_group)
            # Fit straight line x = m*y + b representing classic Hough lane line
            try:
                poly = np.polyfit(all_pts[:, 1], all_pts[:, 0], deg=1)
                y_line = np.linspace(
                    min(y_bottom, int(all_pts[:, 1].max())),
                    max(y_top, int(all_pts[:, 1].min())),
                    20,
                )
                x_line = np.polyval(poly, y_line)
                x_line = np.clip(x_line, 0, width - 1)
                lane_masks.append(np.column_stack((x_line, y_line)).astype(np.int32))
            except Exception:
                lane_masks.append(all_pts.astype(np.int32))

        return raw_lines, lane_masks

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Preprocess a BGR frame (or accept a 2D binary mask) and detect lanes with Hough Transform.
        """
        if frame.ndim == 2:
            binary_mask = frame
        else:
            binary_mask = self.preprocessor.process(frame)

        return self.detect_from_mask(binary_mask)

    def draw_raw_segments(
        self,
        frame: np.ndarray,
        raw_lines: Optional[np.ndarray],
        color: Tuple[int, int, int] = (255, 100, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw individual raw line segments returned by cv2.HoughLinesP."""
        annotated = frame.copy()
        if raw_lines is None:
            return annotated
        for line in raw_lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
        return annotated

    def draw_lanes(
        self,
        frame: np.ndarray,
        lane_masks: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 4,
    ) -> np.ndarray:
        """
        Draw the detected Hough lane lines onto the frame.
        Compatible with `PipelineBuilder`.
        """
        annotated = frame.copy()
        if not lane_masks:
            return annotated
        cv2.polylines(
            annotated,
            lane_masks,
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        return annotated


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from src.path import RAW_DATA_DIR, OUTPUT_DATA_DIR

    parser = argparse.ArgumentParser(description="Run Hough Transform Lane Detection on an image.")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_DIR / "20.jpg"), help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DATA_DIR / "02_hough_transform.jpg"),
        help="Path to save annotated output image",
    )
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {args.input}")

    det = HoughLaneDetector()
    _, masks = det.detect(img)
    out_img = det.draw_lanes(img, masks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out_img)
    print(f"[HoughLaneDetector] Detected {len(masks)} lane(s). Saved result to {args.output}")

