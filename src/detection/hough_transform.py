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
    Classical & Hybrid Road-Boundary Lane Detector for Forza Horizon 4 (UK Left-Hand Traffic):
      1. Preprocessing + Horizontal Local-Contrast Paint Ridge + Probabilistic Hough Transform (`cv2.HoughLinesP`).
      2. Curved polynomial / ridge-slice tracking (replaces rigid deg=1 straight lines so curves over
         hill crests and bends are followed accurately all the way to the horizon).
      3. Adaptive UK Left Asphalt-vs-Grass Boundary detection (`_detect_uk_left_road_edge`) when driving on
         unmarked UK rural/country roads where only the center divider has white paint.

    Implements the interface required by `PipelineBuilder`:
      - `detect(frame) -> (raw_lines, lane_masks)`
      - `draw_lanes(frame, lane_masks) -> annotated_frame`
    """

    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 32,
        min_line_length: int = 28,
        max_line_gap: int = 55,
        min_abs_slope: float = 0.42,
        group_left_right: bool = True,
        detect_unmarked_uk_edge: bool = True,
    ):
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_abs_slope = min_abs_slope
        self.group_left_right = group_left_right
        self.detect_unmarked_uk_edge = detect_unmarked_uk_edge

    @staticmethod
    def _fit_curved_lane(
        pts: np.ndarray,
        y_top: int,
        y_bottom: int,
        width: int,
        num_samples: int = 28,
        y_step: int = 5,
    ) -> Optional[np.ndarray]:
        """
        Fit a smooth curved trajectory x(y) to candidate lane points using slice-median binning,
        outlier rejection, and blended cubic/quadratic curve interpolation so both straight lower
        sections and curved hill-crest sections are preserved accurately.
        """
        if pts is None or len(pts) < 2:
            return None

        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)

        y_bins = (np.round(ys / float(y_step)) * y_step).astype(np.int32)
        unique_y = np.unique(y_bins)
        if len(unique_y) < 2:
            return None

        med_x = np.empty(len(unique_y), dtype=np.float32)
        slice_w = np.empty(len(unique_y), dtype=np.float32)
        for i, yb in enumerate(unique_y):
            mask = y_bins == yb
            x_slice = xs[mask]
            med_x[i] = float(np.median(x_slice))
            slice_w[i] = float(min(15.0, np.sqrt(len(x_slice))))

        y_span = float(unique_y.max() - unique_y.min())
        x_span = float(med_x.max() - med_x.min())
        # Loại bỏ các đường chủ yếu nằm ngang (chân trời, luống cày, hàng rào)
        if y_span < max(35.0, float(y_bottom - y_top) * 0.18):
            return None
        if x_span > 2.2 * y_span and y_span < float(y_bottom - y_top) * 0.45:
            return None

        deg = 3 if (len(unique_y) >= 10 and y_span >= 75.0) else (2 if (len(unique_y) >= 5 and y_span >= 40.0) else 1)
        try:
            uy_f = unique_y.astype(np.float32)
            poly2 = np.polyfit(uy_f, med_x, deg=min(2, deg), w=slice_w)
            if len(unique_y) >= 6:
                pred = np.polyval(poly2, uy_f)
                res = np.abs(med_x - pred)
                mad = float(np.median(res))
                inliers = res <= max(12.0, 2.2 * mad)
                if int(np.sum(inliers)) >= max(3, deg + 1):
                    uy_f = uy_f[inliers]
                    med_x = med_x[inliers]
                    slice_w = slice_w[inliers]

            poly = np.polyfit(uy_f, med_x, deg=min(deg, len(uy_f) - 1), w=slice_w)
            y_max_eval = float(min(y_bottom, int(uy_f.max()) + 8))
            y_min_eval = float(max(y_top, int(uy_f.min()) - 4))
            y_line = np.linspace(y_max_eval, y_min_eval, num_samples, dtype=np.float32)
            x_poly = np.polyval(poly, y_line)

            # Blend with smoothed empirical slice medians inside the observed y range
            if len(uy_f) >= 6:
                x_emp = np.interp(y_line, uy_f, med_x)
                x_emp_smooth = cv2.GaussianBlur(x_emp.reshape(-1, 1), (1, 5), 0).reshape(-1)
                in_obs = (y_line >= uy_f.min()) & (y_line <= uy_f.max())
                x_line = np.where(in_obs, 0.55 * x_poly + 0.45 * x_emp_smooth, x_poly)
            else:
                x_line = x_poly

            x_line = np.clip(x_line, 0.0, float(width - 1))
            return np.column_stack((x_line, y_line)).astype(np.int32)
        except Exception:
            return np.column_stack((med_x, unique_y)).astype(np.int32)

    @staticmethod
    def _extract_paint_ridge_mask(frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Extract bright white and yellow painted road markings inside the ROI using horizontal
        local-contrast Top-Hat filtering (`gray - boxFilter(gray, (31, 7))`) and chroma gating.
        Excludes artificial high-saturation HUD overlays (cyan/green/red/blue).
        """
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        local_bg = cv2.boxFilter(gray, cv2.CV_8U, (31, 7))
        ridge = cv2.subtract(gray, local_bg)

        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        chroma = cv2.subtract(max_c, min_c)

        # White / light-grey road paint: strong horizontal brightness ridge + low color saturation
        white_paint = (ridge >= 14) & (gray >= 135) & (chroma <= 38)

        # Yellow road paint: positive ridge + warm red/green dominance over blue
        r_i16 = r.astype(np.int16)
        g_i16 = g.astype(np.int16)
        b_i16 = b.astype(np.int16)
        yellow_paint = (
            (ridge >= 11)
            & (r >= 135)
            & (g >= 115)
            & ((r_i16 - b_i16) >= 32)
            & ((g_i16 - b_i16) >= 24)
            & (b >= 15)
        )

        paint_mask = np.where(white_paint | yellow_paint, np.uint8(255), np.uint8(0))
        return cv2.bitwise_and(paint_mask, roi_mask)

    @staticmethod
    def _refine_curve_with_paint_ridge(
        initial_pts: np.ndarray,
        paint_mask: np.ndarray,
        y_top: int,
        y_bottom: int,
        width: int,
    ) -> np.ndarray:
        """
        Given initial Hough points for a lane line, snap and trace along `paint_mask` from bottom
        to top (`y_bottom` -> `y_top`) so the detected line locks onto the center of the painted
        road marking and follows curved bends over hill crests.
        """
        if initial_pts is None or len(initial_pts) < 2:
            return initial_pts

        ys_init = initial_pts[:, 1].astype(np.float32)
        xs_init = initial_pts[:, 0].astype(np.float32)

        try:
            deg_seed = 2 if len(ys_init) >= 8 else 1
            seed_poly = np.polyfit(ys_init, xs_init, deg=deg_seed)
            if deg_seed == 2:
                seed_poly[0] = float(np.clip(seed_poly[0], -0.0035, 0.0035))
        except Exception:
            return initial_pts

        step_y = 4
        search_half_w = max(24, int(width * 0.065))
        ridge_pts = []

        y_start = int(min(y_bottom - 4, max(ys_init.max(), height_or_bottom := y_bottom - 12)))
        curr_x = float(np.polyval(seed_poly, float(y_start)))
        curr_dx_dy = float((np.polyval(seed_poly, float(y_start - 10)) - curr_x) / -10.0)
        curr_dx_dy = float(np.clip(curr_dx_dy, -2.4, 2.4))

        for y in range(y_start, y_top - 1, -step_y):
            y0 = max(0, y - 2)
            y1 = min(paint_mask.shape[0], y + 3)
            if y >= ys_init.min():
                pred_x = 0.5 * float(np.polyval(seed_poly, float(y))) + 0.5 * (curr_x - curr_dx_dy * step_y)
            else:
                pred_x = curr_x - curr_dx_dy * step_y
            pred_x = float(np.clip(pred_x, 0.0, float(width - 1)))

            x_lo = max(0, int(round(pred_x - search_half_w)))
            x_hi = min(width, int(round(pred_x + search_half_w)))
            if x_hi <= x_lo:
                continue

            strip = paint_mask[y0:y1, x_lo:x_hi]
            nz_y, nz_x = np.nonzero(strip)
            if len(nz_x) >= 2:
                found_x = float(x_lo + np.median(nz_x))
                dx_step = (found_x - curr_x) / float(-step_y)
                curr_dx_dy = 0.60 * curr_dx_dy + 0.40 * float(np.clip(dx_step, -2.5, 2.5))
                curr_x = found_x
                # Strong weight on true paint ridge pixels so paint overrides noisy edges
                ridge_pts.append(np.array([[found_x, float(y)]] * 5, dtype=np.float32))
            else:
                curr_x = pred_x

        if len(ridge_pts) >= 6:
            # When strong paint ridge points are found, prioritize them over raw Canny Hough segments
            return np.vstack(ridge_pts + [initial_pts[::2]])
        if ridge_pts:
            return np.vstack([initial_pts] + ridge_pts)
        return initial_pts

    @staticmethod
    def _detect_uk_left_road_edge(
        frame: np.ndarray,
        right_curve: np.ndarray,
        y_top: int,
        y_bottom: int,
    ) -> Optional[np.ndarray]:
        """
        Detect the unmarked UK Left Road Edge (asphalt-to-grass / curb transition) when driving
        on UK country roads where only the right-side center divider (`uk_center_divider`) is painted.
        """
        height, width = frame.shape[:2]
        if right_curve is None or len(right_curve) < 2:
            return None

        rc = np.asarray(right_curve, dtype=np.float32)
        sort_idx = np.argsort(rc[:, 1])
        rc_y = rc[sort_idx, 1]
        rc_x = rc[sort_idx, 0]

        y_min_rc = int(max(y_top, rc_y.min()))
        y_max_rc = int(min(y_bottom, rc_y.max()))
        rc_yspan = float(y_max_rc - y_min_rc)
        rc_xspan = float(rc_x.max() - rc_x.min())
        if rc_yspan < max(35.0, float(y_bottom - y_top) * 0.20):
            return None
        if rc_xspan > 2.2 * rc_yspan and rc_yspan < float(y_bottom - y_top) * 0.40:
            return None

        # Verify this is a real road scene (not a pure-black synthetic test image or off-road field)
        mid_y = int(0.5 * (y_min_rc + y_max_rc))
        mid_xr = int(np.interp(mid_y, rc_y, rc_x))
        sample_x0 = max(0, mid_xr - int(0.18 * width))
        sample_x1 = max(sample_x0 + 4, mid_xr - int(0.04 * width))
        check_patch = frame[max(0, mid_y - 10) : min(height, mid_y + 10), sample_x0:sample_x1]
        if check_patch.size == 0 or float(np.mean(check_patch)) < 22.0:
            return None
        # Kiểm tra mặt đường: đất ruộng / rơm rạ có màu ấm rất cao (R - B >> 0), mặt đường nhựa xám trung tính
        patch_b = float(np.mean(check_patch[:, :, 0]))
        patch_g = float(np.mean(check_patch[:, :, 1]))
        patch_r = float(np.mean(check_patch[:, :, 2]))
        warmth_val = max(0.0, patch_r - patch_b) + max(0.0, patch_g - patch_b)
        if warmth_val > 68.0:
            return None

        band = frame[y_min_rc:y_max_rc, :].astype(np.float32)
        b_band = band[:, :, 0]
        g_band = band[:, :, 1]
        r_band = band[:, :, 2]
        gray_band = 0.114 * b_band + 0.587 * g_band + 0.299 * r_band
        warmth_band = np.maximum(0.0, r_band - b_band) + 0.65 * np.maximum(0.0, g_band - b_band)

        grad_gray = np.abs(cv2.Sobel(gray_band, cv2.CV_32F, 1, 0, ksize=3))
        grad_warmth = np.abs(cv2.Sobel(warmth_band, cv2.CV_32F, 1, 0, ksize=3))

        # Exclude any dark non-game UI sidebar on the far left (e.g. VS Code bar at x < 0.07*W)
        min_valid_x = max(8, int(width * 0.055))
        left_strip_mean = np.mean(gray_band[:, : int(width * 0.08)], axis=0)
        dark_cols = np.where(left_strip_mean < 48.0)[0]
        if len(dark_cols) > 0:
            min_valid_x = max(min_valid_x, int(dark_cols.max()) + 6)

        # Vanishing horizon height for FH4 bumper/hood camera perspective
        y_vanish = float(height) * 0.355
        y_slices = np.linspace(y_max_rc - 4, y_min_rc + 4, 24, dtype=np.int32)
        edge_pts = []

        for y_abs in y_slices:
            yb = int(np.clip(y_abs - y_min_rc, 2, band.shape[0] - 3))
            x_r = float(np.interp(float(y_abs), rc_y, rc_x))

            # Perspective scale p(y) in [0.06, 1.0]: narrow near hill crest (~0.09*W), wide at bumper (~0.56*W)
            p_y = float(np.clip((float(y_abs) - y_vanish) / max(1.0, float(y_bottom) - y_vanish), 0.06, 1.0))
            w_nom = float(width) * (0.065 + 0.505 * (p_y ** 0.95))

            ref_x0 = int(np.clip(round(x_r - 0.36 * w_nom), min_valid_x, width - 6))
            ref_x1 = int(np.clip(round(x_r - 0.08 * w_nom), ref_x0 + 3, width - 2))
            ref_b = float(np.median(b_band[yb - 2 : yb + 3, ref_x0:ref_x1]))
            ref_g = float(np.median(g_band[yb - 2 : yb + 3, ref_x0:ref_x1]))
            ref_r = float(np.median(r_band[yb - 2 : yb + 3, ref_x0:ref_x1]))
            ref_w = float(np.median(warmth_band[yb - 2 : yb + 3, ref_x0:ref_x1]))

            search_x_min = max(min_valid_x, int(round(x_r - 1.25 * w_nom)))
            search_x_max = min(int(round(x_r - 0.65 * w_nom)), width - 16)
            if search_x_max <= search_x_min + 6:
                x_est = max(float(min_valid_x), x_r - w_nom)
                edge_pts.append([x_est, float(y_abs)])
                continue

            row_b = np.mean(b_band[yb - 2 : yb + 3, :], axis=0)
            row_g = np.mean(g_band[yb - 2 : yb + 3, :], axis=0)
            row_r = np.mean(r_band[yb - 2 : yb + 3, :], axis=0)
            row_w = np.mean(warmth_band[yb - 2 : yb + 3, :], axis=0)
            row_grad = np.mean(grad_gray[yb - 2 : yb + 3, :] + 1.3 * grad_warmth[yb - 2 : yb + 3, :], axis=0)

            dissim = (
                0.7 * np.abs(row_b - ref_b)
                + 0.8 * np.abs(row_g - ref_g)
                + 1.3 * np.abs(row_r - ref_r)
                + 1.8 * np.maximum(0.0, row_w - ref_w)
            )
            dissim_smooth = cv2.blur(dissim.reshape(1, -1), (1, 9)).reshape(-1)

            xs_cand = np.arange(search_x_min, search_x_max, dtype=np.int32)
            win = max(8, int(round(0.025 * width)))
            left_idx = np.clip(xs_cand - win, min_valid_x, width - 1)
            right_idx = np.clip(xs_cand + win, min_valid_x, width - 1)

            step_contrast = dissim_smooth[left_idx] - dissim_smooth[right_idx]
            local_edge = 0.28 * row_grad[xs_cand]
            x_nominal = x_r - w_nom
            prior_penalty = 14.0 * np.abs(xs_cand.astype(np.float32) - x_nominal) / max(15.0, w_nom)

            score = step_contrast + local_edge - prior_penalty
            best_x = float(xs_cand[int(np.argmax(score))])
            edge_pts.append([best_x, float(y_abs)])

        if len(edge_pts) < 4:
            return None

        pts_arr = np.asarray(edge_pts, dtype=np.float32)
        try:
            poly_left = np.polyfit(pts_arr[:, 1], pts_arr[:, 0], deg=2)
            poly_left[0] = float(np.clip(poly_left[0], -0.0040, 0.0040))
            y_out = np.linspace(float(y_max_rc), float(y_min_rc), 28, dtype=np.float32)
            x_out = np.polyval(poly_left, y_out)
            x_r_out = np.interp(y_out, rc_y, rc_x)
            p_out = np.clip((y_out - y_vanish) / max(1.0, float(y_bottom) - y_vanish), 0.06, 1.0)
            min_sep = float(width) * (0.055 + 0.32 * (p_out ** 0.95))
            x_out = np.minimum(x_out, x_r_out - min_sep)
            x_out = np.clip(x_out, float(min_valid_x), float(width - 1))
            return np.column_stack((x_out, y_out)).astype(np.int32)
        except Exception:
            return pts_arr.astype(np.int32)

    def detect_from_mask(
        self,
        binary_mask: np.ndarray,
        bgr_frame: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Run Probabilistic Hough Transform on a 2D binary edge/connected mask and fit smooth
        curved lane boundaries. When `bgr_frame` is provided, also refines curves along painted
        road ridges up to the horizon and detects the unmarked UK left asphalt-to-grass boundary.
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
        min_valid_x = int(width * 0.045) if bgr_frame is not None else 0

        for line in raw_lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
            if x1 < min_valid_x and x2 < min_valid_x:
                continue
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 1e-3:
                continue
            slope = dy / dx
            if abs(slope) < self.min_abs_slope:
                continue

            seg_len = int(max(2, np.hypot(dx, dy) // 6))
            xs = np.linspace(x1, x2, seg_len)
            ys = np.linspace(y1, y2, seg_len)
            seg_pts = np.column_stack((xs, ys))

            x_mean = 0.5 * (x1 + x2)
            y_mean = 0.5 * (y1 + y2)
            if slope < 0 and (x_mean < mid_x * 1.12 or (y_mean < height * 0.56 and x_mean < mid_x * 1.25)):
                left_pts.append(seg_pts)
            elif slope > 0 and (x_mean > mid_x * 0.78 or (y_mean < height * 0.58 and x_mean > mid_x * 0.60)):
                right_pts.append(seg_pts)

        y_bottom = height - self.preprocessor.roi_bottom_margin_px
        y_top = int(height * self.preprocessor.roi_top_y_ratio)

        paint_mask: Optional[np.ndarray] = None
        if bgr_frame is not None and bgr_frame.ndim == 3:
            roi_mask = self.preprocessor._get_cached_roi_mask(height, width)
            paint_mask = self._extract_paint_ridge_mask(bgr_frame, roi_mask)

        left_curve: Optional[np.ndarray] = None
        right_curve: Optional[np.ndarray] = None

        if left_pts:
            all_left = np.vstack(left_pts)
            if paint_mask is not None:
                all_left = self._refine_curve_with_paint_ridge(all_left, paint_mask, y_top, y_bottom, width)
            left_curve = self._fit_curved_lane(all_left, y_top, y_bottom, width)

        if right_pts:
            all_right = np.vstack(right_pts)
            if paint_mask is not None:
                all_right = self._refine_curve_with_paint_ridge(all_right, paint_mask, y_top, y_bottom, width)
            right_curve = self._fit_curved_lane(all_right, y_top, y_bottom, width)

        # If driving on an unmarked UK country road where only the center divider (`right_curve`)
        # has white paint, automatically detect the left asphalt-vs-grass road edge!
        if (
            self.detect_unmarked_uk_edge
            and bgr_frame is not None
            and bgr_frame.ndim == 3
            and right_curve is not None
            and left_curve is None
        ):
            left_curve = self._detect_uk_left_road_edge(bgr_frame, right_curve, y_top, y_bottom)

        lane_masks: List[np.ndarray] = []
        if left_curve is not None:
            lane_masks.append(left_curve)
        if right_curve is not None:
            lane_masks.append(right_curve)

        return raw_lines, lane_masks

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Preprocess a BGR frame (or accept a 2D binary mask) and detect lanes with Hough Transform
        + Curved Ridge Tracking + UK Left Road Edge Segmentation.
        """
        if frame.ndim == 2:
            return self.detect_from_mask(frame, bgr_frame=None)

        binary_mask = self.preprocessor.process(frame)
        return self.detect_from_mask(binary_mask, bgr_frame=frame)

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
