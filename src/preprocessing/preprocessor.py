import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class ImagePreprocessor:
    """
    Classical image preprocessing pipeline for lane detection.
    
    Encapsulates the steps prototyped in `notebooks/pre_processing.ipynb`:
    1. Grayscale conversion (and optional CLAHE + Gaussian blur enhancement)
    2. Canny edge detection
    3. Trapezoidal Region of Interest (ROI) masking
    4. Morphological closing to connect broken edge segments
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        use_clahe: bool = False,
        blur_ksize: int = 5,
        morph_kernel_size: Tuple[int, int] = (3, 3),
        morph_iterations: int = 2,
        roi_top_y_ratio: float = 0.42,
        roi_top_left_x_ratio: float = 0.35,
        roi_top_right_x_ratio: float = 0.65,
        roi_bottom_left_x_ratio: float = 0.02,
        roi_bottom_right_x_ratio: float = 0.98,
        roi_bottom_margin_px: int = 5,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.use_clahe = use_clahe
        self.blur_ksize = blur_ksize
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.roi_top_y_ratio = roi_top_y_ratio
        self.roi_top_left_x_ratio = roi_top_left_x_ratio
        self.roi_top_right_x_ratio = roi_top_right_x_ratio
        self.roi_bottom_left_x_ratio = roi_bottom_left_x_ratio
        self.roi_bottom_right_x_ratio = roi_bottom_right_x_ratio
        self.roi_bottom_margin_px = roi_bottom_margin_px

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Step 1: Convert BGR image to single-channel grayscale (with optional CLAHE & blur)."""
        if frame.ndim == 2:
            gray = frame.copy()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if self.blur_ksize and self.blur_ksize > 1:
            k = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        return gray

    def detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Step 2: Apply Canny edge detection on grayscale image."""
        return cv2.Canny(gray, threshold1=self.canny_low, threshold2=self.canny_high)

    def get_roi_vertices(self, height: int, width: int) -> np.ndarray:
        """Compute trapezoid ROI vertices matching `notebooks/pre_processing.ipynb`."""
        bottom_y = max(0, height - self.roi_bottom_margin_px)
        top_y = int(height * self.roi_top_y_ratio)
        return np.array(
            [
                [
                    (int(width * self.roi_bottom_left_x_ratio), bottom_y),
                    (int(width * self.roi_top_left_x_ratio), top_y),
                    (int(width * self.roi_top_right_x_ratio), top_y),
                    (int(width * self.roi_bottom_right_x_ratio), bottom_y),
                ]
            ],
            dtype=np.int32,
        )

    def apply_roi_mask(self, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Step 3: Mask out regions outside the road trapezoid ROI."""
        height, width = edges.shape[:2]
        roi_vertices = self.get_roi_vertices(height, width)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        roi_edges = cv2.bitwise_and(edges, mask)
        return roi_edges, mask

    def connect_edges(self, roi_edges: np.ndarray) -> np.ndarray:
        """Step 4: Connect nearby edge pixels using morphological closing."""
        lane_mask = (roi_edges > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        connected_mask = cv2.morphologyEx(
            lane_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.morph_iterations,
        )
        return connected_mask

    def process(self, frame: np.ndarray, return_steps: bool = False):
        """
        Run the complete preprocessing pipeline on a single BGR frame.
        
        Args:
            frame: Input BGR image (H, W, 3).
            return_steps: If True, returns a dictionary of intermediate outputs
                          ('gray', 'edges', 'roi_mask', 'roi_edges', 'connected_mask').
                          
        Returns:
            connected_mask (np.ndarray) or Tuple[np.ndarray, Dict[str, np.ndarray]].
        """
        gray = self.to_grayscale(frame)
        edges = self.detect_edges(gray)
        roi_edges, roi_mask = self.apply_roi_mask(edges)
        connected_mask = self.connect_edges(roi_edges)

        if return_steps:
            steps = {
                "gray": gray,
                "edges": edges,
                "roi_mask": roi_mask,
                "roi_edges": roi_edges,
                "connected_mask": connected_mask,
            }
            return connected_mask, steps

        return connected_mask

    @staticmethod
    def highlight_mask(
        frame: np.ndarray,
        binary_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        base_weight: float = 0.65,
        overlay_weight: float = 0.9,
    ) -> np.ndarray:
        """Overlay a binary mask onto the original BGR image (matching Cell 5 & 6)."""
        highlighted = frame.copy()
        highlighted[binary_mask > 0] = color
        return cv2.addWeighted(frame, base_weight, highlighted, overlay_weight, 0)
