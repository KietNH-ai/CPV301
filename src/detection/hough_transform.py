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

        roi_h = max(40.0, float(y_bottom - y_top))
        eff_y_step = max(3, int(round(roi_h / 50.0))) if y_step == 5 else max(2, int(y_step))
        y_bins = (np.round(ys / float(eff_y_step)) * eff_y_step).astype(np.int32)

        # Vectorized slice-median via lexsort (replaces O(N * K) Python loop)
        order = np.lexsort((xs, y_bins))
        y_sorted = y_bins[order]
        x_sorted = xs[order]
        unique_y, first_idx, counts = np.unique(y_sorted, return_index=True, return_counts=True)
        if len(unique_y) < 2:
            return None

        mid_hi = first_idx + (counts // 2)
        mid_lo = first_idx + ((counts - 1) // 2)
        med_x = 0.5 * (x_sorted[mid_lo] + x_sorted[mid_hi])
        slice_w = np.minimum(15.0, np.sqrt(counts.astype(np.float32)))

        y_span = float(unique_y[-1] - unique_y[0])
        x_span = float(med_x.max() - med_x.min())
        # Loại bỏ các đường chủ yếu nằm ngang (chân trời, luống cày, hàng rào)
        if y_span < max(16.0, roi_h * 0.16):
            return None
        if x_span > 2.2 * y_span and y_span < roi_h * 0.45:
            return None

        deg = (
            3
            if (len(unique_y) >= 10 and y_span >= roi_h * 0.25)
            else (2 if (len(unique_y) >= 5 and y_span >= roi_h * 0.14) else 1)
        )
        try:
            uy_f = unique_y.astype(np.float32)
            poly2 = np.polyfit(uy_f, med_x, deg=min(2, deg), w=slice_w)
            if len(unique_y) >= 6:
                pred = np.polyval(poly2, uy_f)
                res = np.abs(med_x - pred)
                mad = float(np.median(res))
                inlier_tol = max(5.0, width * 0.0095, 2.2 * mad)
                inliers = res <= inlier_tol
                if int(np.sum(inliers)) >= max(3, deg + 1):
                    uy_f = uy_f[inliers]
                    med_x = med_x[inliers]
                    slice_w = slice_w[inliers]

            poly = np.polyfit(uy_f, med_x, deg=min(deg, len(uy_f) - 1), w=slice_w)
            pad_bot = max(3, int(round(roi_h * 0.025)))
            pad_top = max(2, int(round(roi_h * 0.015)))
            y_max_eval = float(min(y_bottom, int(uy_f.max()) + pad_bot))
            y_min_eval = float(max(y_top, int(uy_f.min()) - pad_top))
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
        Extract bright white and yellow painted road markings inside the ROI using resolution-adaptive
        horizontal local-contrast Top-Hat filtering and chroma gating.
        Excludes artificial high-saturation HUD overlays (cyan/green/red/blue).
        """
        height, width = frame.shape[:2]
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kw = max(9, int(round(width * 0.024)) | 1)
        kh = max(3, int(round(height * 0.010)) | 1)
        local_bg = cv2.boxFilter(gray, cv2.CV_8U, (kw, kh))
        ridge = cv2.subtract(gray, local_bg)

        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        chroma = cv2.subtract(max_c, min_c)

        # White / light-grey road paint: positive ridge + low color saturation
        white_paint = (ridge >= 8) & (gray >= 80) & (chroma <= 42)

        # Yellow road paint: positive ridge + warm red/green dominance over blue
        r_i16 = r.astype(np.int16)
        g_i16 = g.astype(np.int16)
        b_i16 = b.astype(np.int16)
        yellow_paint = (
            (ridge >= 7)
            & (r >= 95)
            & (g >= 85)
            & ((r_i16 - b_i16) >= 25)
            & ((g_i16 - b_i16) >= 18)
            & (b >= 10)
        )

        paint_mask = np.where(white_paint | yellow_paint, np.uint8(255), np.uint8(0))
        paint_mask = cv2.bitwise_and(paint_mask, roi_mask)
        roi_area = max(1, int(np.count_nonzero(roi_mask)))
        paint_nz = int(np.count_nonzero(paint_mask))
        # Quá nhiều vạch sáng phủ kín màn hình (>8.5% ROI) => nhiễu màn hình, lưới gạch, tường hoa, văn phòng
        if paint_nz > 0.085 * roi_area:
            return np.zeros_like(paint_mask)
        return paint_mask

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

        roi_h = max(40.0, float(y_bottom - y_top))
        step_y = max(4, int(round(roi_h / 36.0)))
        search_half_w = max(10, int(width * 0.055))
        ridge_pts = []

        y_min_init = float(ys_init.min())
        y_start = int(min(y_bottom - 3, max(ys_init.max(), y_bottom - max(6, int(roi_h * 0.03)))))
        if y_start <= y_top:
            return initial_pts

        y_steps_arr = np.arange(y_start, y_top - 1, -step_y, dtype=np.int32)
        poly_preds = np.polyval(seed_poly, y_steps_arr.astype(np.float32))
        curr_x = float(poly_preds[0])
        dy_probe = max(4.0, roi_h * 0.025)
        curr_dx_dy = float((np.polyval(seed_poly, float(y_start - dy_probe)) - curr_x) / -dy_probe)
        curr_dx_dy = float(np.clip(curr_dx_dy, -2.4, 2.4))

        for idx, y in enumerate(y_steps_arr):
            if y >= y_min_init:
                pred_x = 0.5 * float(poly_preds[idx]) + 0.5 * (curr_x - curr_dx_dy * step_y)
            else:
                pred_x = curr_x - curr_dx_dy * step_y
            pred_x = float(np.clip(pred_x, 0.0, float(width - 1)))

            x_lo = max(0, int(round(pred_x - search_half_w)))
            x_hi = min(width, int(round(pred_x + search_half_w)))
            if x_hi <= x_lo:
                continue

            row_strip = paint_mask[y, x_lo:x_hi]
            nz_x = np.flatnonzero(row_strip)
            if len(nz_x) == 0 and 0 < y < paint_mask.shape[0] - 1:
                nz_x = np.flatnonzero(paint_mask[y - 1, x_lo:x_hi] | paint_mask[y + 1, x_lo:x_hi])
            if len(nz_x) >= 1:
                found_x = float(x_lo) + 0.5 * float(int(nz_x[0]) + int(nz_x[-1]))
                dx_step = (found_x - curr_x) / float(-step_y)
                curr_dx_dy = 0.60 * curr_dx_dy + 0.40 * float(np.clip(dx_step, -2.5, 2.5))
                curr_x = found_x
                ridge_pts.append([found_x, float(y)])
            else:
                curr_x = pred_x

        if len(ridge_pts) >= 4:
            r_arr = np.repeat(np.asarray(ridge_pts, dtype=np.float32), 4, axis=0)
            return np.vstack([r_arr, initial_pts[::2]])
        if len(ridge_pts) >= 2:
            r_arr = np.repeat(np.asarray(ridge_pts, dtype=np.float32), 3, axis=0)
            return np.vstack([initial_pts, r_arr])
        # If no genuine road paint was found along this line, it is a non-road edge (shadow/fence/wall/UI)
        return None

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

        # Vanishing horizon height for FH4 bumper/hood camera perspective
        y_vanish = float(height) * 0.355
        y_slices = np.linspace(max(y_min_rc + 2, y_max_rc - 4), min(y_max_rc - 2, y_min_rc + 4), 16, dtype=np.int32)
        rows_bgr = frame[y_slices, :].astype(np.float32)
        b_rows = rows_bgr[:, :, 0]
        g_rows = rows_bgr[:, :, 1]
        r_rows = rows_bgr[:, :, 2]
        gray_rows = 0.114 * b_rows + 0.587 * g_rows + 0.299 * r_rows
        warmth_rows = np.maximum(0.0, r_rows - b_rows) + 0.65 * np.maximum(0.0, g_rows - b_rows)

        grad_gray = np.abs(np.gradient(gray_rows, axis=1))
        grad_warmth = np.abs(np.gradient(warmth_rows, axis=1))
        row_grad_all = grad_gray + 1.3 * grad_warmth

        # Exclude any dark non-game UI sidebar on the far left (e.g. VS Code bar at x < 0.07*W)
        min_valid_x = max(8, int(width * 0.055))
        left_strip_mean = np.mean(gray_rows[:, : max(1, int(width * 0.08))], axis=0)
        dark_cols = np.where(left_strip_mean < 48.0)[0]
        if len(dark_cols) > 0:
            min_valid_x = max(min_valid_x, int(dark_cols.max()) + 6)

        edge_pts = []
        win = max(6, int(round(0.025 * width)))
        x_rs = np.interp(y_slices.astype(np.float32), rc_y, rc_x)

        for idx, y_abs in enumerate(y_slices):
            x_r = float(x_rs[idx])

            # Perspective scale p(y) in [0.06, 1.0]: narrow near hill crest (~0.09*W), wide at bumper (~0.56*W)
            p_y = float(np.clip((float(y_abs) - y_vanish) / max(1.0, float(y_bottom) - y_vanish), 0.06, 1.0))
            w_nom = float(width) * (0.065 + 0.505 * (p_y ** 0.95))

            ref_x0 = int(np.clip(round(x_r - 0.36 * w_nom), min_valid_x, width - 6))
            ref_x1 = int(np.clip(round(x_r - 0.08 * w_nom), ref_x0 + 3, width - 2))
            ref_b = float(np.mean(b_rows[idx, ref_x0:ref_x1]))
            ref_g = float(np.mean(g_rows[idx, ref_x0:ref_x1]))
            ref_r = float(np.mean(r_rows[idx, ref_x0:ref_x1]))
            ref_w = float(np.mean(warmth_rows[idx, ref_x0:ref_x1]))

            search_x_min = max(min_valid_x, int(round(x_r - 1.25 * w_nom)))
            search_x_max = min(int(round(x_r - 0.65 * w_nom)), width - 16)
            if search_x_max <= search_x_min + 6:
                x_est = max(float(min_valid_x), x_r - w_nom)
                edge_pts.append([x_est, float(y_abs)])
                continue

            dissim = (
                0.7 * np.abs(b_rows[idx] - ref_b)
                + 0.8 * np.abs(g_rows[idx] - ref_g)
                + 1.3 * np.abs(r_rows[idx] - ref_r)
                + 1.8 * np.maximum(0.0, warmth_rows[idx] - ref_w)
            )
            dissim_smooth = cv2.blur(dissim.reshape(1, -1), (1, 9)).reshape(-1)

            xs_cand = np.arange(search_x_min, search_x_max, dtype=np.int32)
            left_idx = np.clip(xs_cand - win, min_valid_x, width - 1)
            right_idx = np.clip(xs_cand + win, min_valid_x, width - 1)

            step_contrast = dissim_smooth[left_idx] - dissim_smooth[right_idx]
            local_edge = 0.28 * row_grad_all[idx, xs_cand]
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
        scale_w = min(1.0, max(0.42, float(width) / 1280.0))
        eff_min_len = max(12, int(round(self.min_line_length * scale_w)))
        eff_max_gap = max(20, int(round(self.max_line_gap * scale_w)))
        eff_thresh = max(18, int(round(self.threshold * (0.65 + 0.35 * scale_w))))
        raw_lines = cv2.HoughLinesP(
            binary_mask,
            rho=self.rho,
            theta=self.theta,
            threshold=eff_thresh,
            minLineLength=eff_min_len,
            maxLineGap=eff_max_gap,
        )

        if raw_lines is None or len(raw_lines) == 0:
            return raw_lines, []

        if not self.group_left_right:
            masks = []
            for line in raw_lines:
                x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
                masks.append(np.array([[x1, y1], [x2, y2]], dtype=np.int32))
            return raw_lines, masks

        # Vectorized segment filtering and point expansion (replaces per-segment Python loop)
        mid_x = width / 2.0
        min_valid_x = float(int(width * 0.045)) if bgr_frame is not None else 0.0
        ego_min_slope = max(0.35, float(self.min_abs_slope))

        segs = raw_lines.reshape(-1, 4).astype(np.float32)
        valid_x = (segs[:, 0] >= min_valid_x) | (segs[:, 2] >= min_valid_x)
        segs = segs[valid_x]
        if len(segs) == 0:
            return raw_lines, []

        dx = segs[:, 2] - segs[:, 0]
        dy = segs[:, 3] - segs[:, 1]
        valid_dx = np.abs(dx) >= 1e-3
        segs = segs[valid_dx]
        dx = dx[valid_dx]
        dy = dy[valid_dx]
        if len(segs) == 0:
            return raw_lines, []

        slopes = dy / dx
        valid_slope = np.abs(slopes) >= self.min_abs_slope
        segs = segs[valid_slope]
        dx = dx[valid_slope]
        dy = dy[valid_slope]
        slopes = slopes[valid_slope]
        if len(segs) == 0:
            return raw_lines, []

        x_mean = 0.5 * (segs[:, 0] + segs[:, 2])
        y_mean = 0.5 * (segs[:, 1] + segs[:, 3])
        x_proj = x_mean + (float(height) - y_mean) / slopes

        is_left = ((slopes < 0.0) & (x_mean < mid_x * 1.15)) | ((slopes > 0.0) & (x_proj < mid_x * 0.98) & (x_mean < mid_x * 0.85))
        is_left = is_left & (x_proj >= -0.30 * width)
        is_ego_left = is_left & (x_proj >= -0.05 * width) & (x_proj <= 0.51 * width) & (np.abs(slopes) >= ego_min_slope)

        is_right = ((slopes > 0.0) & (x_mean > mid_x * 0.75)) | ((slopes < 0.0) & (x_proj > mid_x * 1.02) & (x_mean > mid_x * 0.85))
        is_right = is_right & (x_proj <= 1.30 * width)
        is_ego_right = is_right & (x_proj >= 0.49 * width) & (x_proj <= 1.20 * width) & (np.abs(slopes) >= ego_min_slope)

        t_vec = np.linspace(0.0, 1.0, 6, dtype=np.float32)[None, :]

        def _expand_segs(sel_mask: np.ndarray) -> Optional[np.ndarray]:
            if not np.any(sel_mask):
                return None
            s = segs[sel_mask]
            s_dx = dx[sel_mask][:, None]
            s_dy = dy[sel_mask][:, None]
            xs = (s[:, 0:1] + t_vec * s_dx).reshape(-1)
            ys = (s[:, 1:2] + t_vec * s_dy).reshape(-1)
            return np.column_stack((xs, ys))

        y_bottom = height - self.preprocessor.roi_bottom_margin_px
        y_top = int(height * self.preprocessor.roi_top_y_ratio)
        roi_h = max(40.0, float(y_bottom - y_top))

        all_left = _expand_segs(is_left)
        ego_l_arr = _expand_segs(is_ego_left)
        if ego_l_arr is not None:
            if int(np.sum(is_ego_left)) >= 2 and float(ego_l_arr[:, 1].max() - ego_l_arr[:, 1].min()) >= 0.18 * roi_h:
                all_left = ego_l_arr
            elif all_left is not None:
                all_left = np.vstack([ego_l_arr, ego_l_arr, all_left])

        all_right = _expand_segs(is_right)
        ego_r_arr = _expand_segs(is_ego_right)
        if ego_r_arr is not None:
            if int(np.sum(is_ego_right)) >= 2 and float(ego_r_arr[:, 1].max() - ego_r_arr[:, 1].min()) >= 0.18 * roi_h:
                all_right = ego_r_arr
            elif all_right is not None:
                all_right = np.vstack([ego_r_arr, ego_r_arr, all_right])

        paint_mask: Optional[np.ndarray] = None
        if bgr_frame is not None and bgr_frame.ndim == 3:
            roi_mask = self.preprocessor._get_cached_roi_mask(height, width)
            paint_mask = self._extract_paint_ridge_mask(bgr_frame, roi_mask)

        left_curve: Optional[np.ndarray] = None
        right_curve: Optional[np.ndarray] = None

        if all_left is not None:
            if paint_mask is not None:
                all_left = self._refine_curve_with_paint_ridge(all_left, paint_mask, y_top, y_bottom, width)
            if all_left is not None:
                left_curve = self._fit_curved_lane(all_left, y_top, y_bottom, width)

        if all_right is not None:
            if paint_mask is not None:
                all_right = self._refine_curve_with_paint_ridge(all_right, paint_mask, y_top, y_bottom, width)
            if all_right is not None:
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

        # Geometric & Physical Plausibility Gating (chống nhận diện vạch ảo ngoài đời thực)
        min_reach_y = height * 0.62
        if left_curve is not None and float(np.max(left_curve[:, 1])) < min_reach_y:
            left_curve = None
        if right_curve is not None and float(np.max(right_curve[:, 1])) < min_reach_y:
            right_curve = None

        if left_curve is not None and right_curve is not None:
            idx_lb = int(np.argmax(left_curve[:, 1]))
            idx_rb = int(np.argmax(right_curve[:, 1]))
            x_l_bot = float(left_curve[idx_lb, 0])
            x_r_bot = float(right_curve[idx_rb, 0])
            sep_bot = x_r_bot - x_l_bot
            if (
                sep_bot < 0.14 * float(width)
                or sep_bot > 0.92 * float(width)
                or x_l_bot > mid_x * 1.15
                or x_r_bot < mid_x * 0.85
            ):
                # Hình học bất khả thi (2 vạch đè lên nhau, quá rộng, hoặc đều nằm lệch cùng 1 bên lề)
                left_curve = None
                right_curve = None
        elif left_curve is not None:
            idx_lb = int(np.argmax(left_curve[:, 1]))
            if float(left_curve[idx_lb, 0]) > mid_x * 0.98:
                left_curve = None
        elif right_curve is not None:
            idx_rb = int(np.argmax(right_curve[:, 1]))
            if float(right_curve[idx_rb, 0]) < mid_x * 1.02:
                right_curve = None

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
