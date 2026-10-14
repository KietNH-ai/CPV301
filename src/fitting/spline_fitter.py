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
        thickness: int = 4
    ):
        """
        Initialize the SplineFitter.
        
        Args:
            spline_order (int): Order of the B-spline (k=3 for cubic spline).
            smoothing (float): Smoothing factor 's'. Higher values reject more noise.
            num_points (int): Number of interpolated points along the fitted curve.
            default_color (tuple): BGR color tuple for drawing lanes.
            thickness (int): Thickness of the drawn lane lines.
        """
        self.spline_order = spline_order
        self.smoothing = smoothing
        self.num_points = num_points
        self.default_color = default_color
        self.thickness = thickness

    def _extract_centerline(self, points: np.ndarray, y_step: int = 4) -> np.ndarray:
        """
        Extracts a clean centerline from polygon or scattered mask coordinates.
        Groups points by horizontal slices (y-coordinate) and averages their x-positions.
        
        Args:
            points: (N, 2) array of (x, y) coordinates.
            y_step: Bin size along y-axis to group and average points.
            
        Returns:
            (M, 2) array of sorted, deduplicated centerline points.
        """
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
            
        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        
        # Bin y coordinates by step size to prevent clustering
        y_binned = (np.round(y / y_step) * y_step).astype(np.int32)
        unique_y = np.unique(y_binned)
        
        centerline = []
        for uy in unique_y:
            matching_x = x[y_binned == uy]
            avg_x = np.mean(matching_x)
            centerline.append([avg_x, float(uy)])
            
        centerline = np.array(centerline, dtype=np.float32)
        
        # Sort from bottom of the image (largest y) to top (smallest y)
        if len(centerline) > 0:
            sort_idx = np.argsort(-centerline[:, 1])
            centerline = centerline[sort_idx]
            
        return centerline

    def fit_lane(self, mask_points: Union[np.ndarray, List]) -> Optional[np.ndarray]:
        """
        Fit a smooth B-spline curve to a single lane mask/contour.
        
        Args:
            mask_points: Array-like of shape (N, 2) containing (x, y) points.
            
        Returns:
            np.ndarray of shape (num_points, 2) with smoothed integer coordinates,
            or None if fitting is not possible.
        """
        pts = np.asarray(mask_points)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return None
            
        # Extract centerline and sort along y-axis
        centerline = self._extract_centerline(pts)
        if len(centerline) < 2:
            return None

        # Filter out identical consecutive points which cause splprep to fail
        diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        valid_mask = np.insert(diffs > 1e-4, 0, True)
        centerline = centerline[valid_mask]
        
        n_pts = len(centerline)
        if n_pts < 2:
            return None

        x_pts = centerline[:, 0]
        y_pts = centerline[:, 1]
        
        # Determine spline degree (cannot exceed n_pts - 1)
        k = min(self.spline_order, n_pts - 1)
        
        try:
            # Fit B-spline parameterized by arc length
            # Note: s=0 interpolates through all points, s>0 smooths
            tck, u = splprep([x_pts, y_pts], s=self.smoothing, k=k)
            u_new = np.linspace(u.min(), u.max(), self.num_points)
            x_smooth, y_smooth = splev(u_new, tck)
            
            curve = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
            return curve
            
        except Exception:
            # Fallback 1: Polynomial fitting (quadratic or linear x = f(y))
            try:
                deg = min(2, n_pts - 1)
                poly = np.polyfit(y_pts, x_pts, deg=deg)
                y_new = np.linspace(y_pts.min(), y_pts.max(), self.num_points)
                x_new = np.polyval(poly, y_new)
                curve = np.column_stack((x_new, y_new)).astype(np.int32)
                return curve
            except Exception:
                # Fallback 2: Direct linear interpolation along points
                t_old = np.linspace(0, 1, n_pts)
                t_new = np.linspace(0, 1, self.num_points)
                x_interp = np.interp(t_new, t_old, x_pts)
                y_interp = np.interp(t_new, t_old, y_pts)
                return np.column_stack((x_interp, y_interp)).astype(np.int32)

    def fit_multiple_lanes(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Fit smooth curves to a list of lane masks.
        
        Args:
            masks: List of (N, 2) point arrays.
            
        Returns:
            List of (num_points, 2) smooth curve arrays.
        """
        fitted_curves = []
        for mask in masks:
            curve = self.fit_lane(mask)
            if curve is not None and len(curve) > 0:
                fitted_curves.append(curve)
        return fitted_curves

    def fit_and_draw(
        self,
        frame: np.ndarray,
        masks: List[np.ndarray],
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Fits splines to all lane masks and draws smooth curves onto the frame.
        
        Args:
            frame (np.ndarray): The input video frame (BGR format).
            masks (list): List of detected lane masks/contours from detector.
            colors (list, optional): List of BGR colors for each lane.
            
        Returns:
            np.ndarray: Annotated video frame.
        """
        annotated = frame.copy()
        if not masks:
            return annotated
            
        fitted_curves = self.fit_multiple_lanes(masks)
        
        for idx, curve in enumerate(fitted_curves):
            color = self.default_color
            if colors and idx < len(colors):
                color = colors[idx]
                
            # Draw antialiased smooth polyline
            cv2.polylines(
                annotated,
                [curve],
                isClosed=False,
                color=color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA
            )
            
        return annotated

    def draw_lane_area(
        self,
        frame: np.ndarray,
        left_curve: np.ndarray,
        right_curve: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.25
    ) -> np.ndarray:
        """
        Draws a semi-transparent colored drivable corridor between two lanes.
        
        Args:
            frame: Input image/frame.
            left_curve: (M, 2) coordinates of left lane curve.
            right_curve: (N, 2) coordinates of right lane curve.
            color: BGR fill color (default green).
            alpha: Transparency factor (0.0 to 1.0).
            
        Returns:
            Annotated frame with highlighted lane area.
        """
        if len(left_curve) < 2 or len(right_curve) < 2:
            return frame
            
        overlay = frame.copy()
        # Form closed polygon: left curve down + right curve up
        poly_pts = np.vstack([left_curve, np.flipud(right_curve)])
        cv2.fillPoly(overlay, [poly_pts], color=color)
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
