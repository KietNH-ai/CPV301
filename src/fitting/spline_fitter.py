import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional, Union


class SplineFitter:
    """
    Fits smooth B-spline curves to detected lane points or segmentation masks.
    
    This replaces jagged, discontinuous, or noisy lane masks with mathematically
    smooth curves, specifically optimized for sharp turns and complex curves.
    """

    def __init__(
        self,
        spline_order: int = 3,
        smoothing: float = 3.0,
        num_points: int = 100,
        default_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 4,
    ):
        self.spline_order = spline_order
        self.smoothing = smoothing
        self.num_points = num_points
        self.default_color = default_color
        self.thickness = thickness

    def _extract_centerline(self, points: np.ndarray, y_step: int = 4) -> np.ndarray:
        """
        Extracts a clean centerline from polygon or scattered mask coordinates.
        Groups points by horizontal slices (y-coordinate) and averages their x-positions
        using vectorized NumPy bincount for high-FPS real-time performance.
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)

        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)

        y_binned = (np.round(y / y_step) * y_step).astype(np.int32)
        unique_y, inv_idx = np.unique(y_binned, return_inverse=True)
        counts = np.bincount(inv_idx).astype(np.float32)
        sum_x = np.bincount(inv_idx, weights=x).astype(np.float32)
        avg_x = sum_x / np.maximum(1.0, counts)

        centerline = np.column_stack((avg_x, unique_y.astype(np.float32)))

        # Sort from bottom of the image (largest y) to top (smallest y)
        if len(centerline) > 0:
            sort_idx = np.argsort(-centerline[:, 1])
            centerline = centerline[sort_idx]

        return centerline

    def fit_lane(self, mask_points: Union[np.ndarray, List]) -> Optional[np.ndarray]:
        """
        Fit a smooth B-spline curve to a single lane mask/contour.
        """
        pts = np.asarray(mask_points)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return None

        centerline = self._extract_centerline(pts)
        if len(centerline) < 2:
            return None

        diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        valid_mask = np.insert(diffs > 1e-4, 0, True)
        centerline = centerline[valid_mask]

        n_pts = len(centerline)
        if n_pts < 2:
            return None

        x_pts = centerline[:, 0]
        y_pts = centerline[:, 1]
        y_span = float(y_pts.max() - y_pts.min())
        x_span = float(x_pts.max() - x_pts.min())
        if y_span < 25.0 or (x_span > 2.6 * y_span and y_span < 75.0):
            return None

        k = min(self.spline_order, n_pts - 1)

        try:
            tck, u = splprep([x_pts, y_pts], s=self.smoothing, k=k)
            u_new = np.linspace(u.min(), u.max(), self.num_points)
            x_smooth, y_smooth = splev(u_new, tck)

            curve = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
            return curve

        except Exception:
            try:
                deg = min(2, n_pts - 1)
                poly = np.polyfit(y_pts, x_pts, deg=deg)
                y_new = np.linspace(y_pts.min(), y_pts.max(), self.num_points)
                x_new = np.polyval(poly, y_new)
                curve = np.column_stack((x_new, y_new)).astype(np.int32)
                return curve
            except Exception:
                t_old = np.linspace(0, 1, n_pts)
                t_new = np.linspace(0, 1, self.num_points)
                x_interp = np.interp(t_new, t_old, x_pts)
                y_interp = np.interp(t_new, t_old, y_pts)
                return np.column_stack((x_interp, y_interp)).astype(np.int32)

    def fit_multiple_lanes(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Fit smooth curves to a list of lane masks."""
        fitted_curves = []
        for mask in masks:
            curve = self.fit_lane(mask)
            if curve is not None and len(curve) > 0:
                fitted_curves.append(curve)
        return fitted_curves

    def draw_fitted_curves(
        self,
        frame: np.ndarray,
        fitted_curves: List[np.ndarray],
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Draw already-fitted curves onto a copy of `frame` without re-fitting."""
        annotated = frame.copy()
        if not fitted_curves:
            return annotated

        for idx, curve in enumerate(fitted_curves):
            color = self.default_color
            if colors and idx < len(colors):
                color = colors[idx]

            cv2.polylines(
                annotated,
                [curve],
                isClosed=False,
                color=color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA,
            )
        return annotated

    def fit_and_draw(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Fits splines to all lane masks and draws smooth curves onto the frame.
        """
        if not masks:
            return frame.copy()

        fitted_curves = self.fit_multiple_lanes(masks)
        return self.draw_fitted_curves(frame, fitted_curves, colors=colors)

    def draw_lane_area(
        self,
        frame: np.ndarray,
        left_curve: np.ndarray,
        right_curve: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.25,
    ) -> np.ndarray:
        """Draws a semi-transparent colored drivable corridor between two lanes."""
        if len(left_curve) < 2 or len(right_curve) < 2:
            return frame

        overlay = frame.copy()
        poly_pts = np.vstack([left_curve, np.flipud(right_curve)])
        cv2.fillPoly(overlay, [poly_pts], color=color)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
