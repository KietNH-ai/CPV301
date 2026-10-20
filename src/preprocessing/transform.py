import cv2
import numpy as np
from typing import Tuple, Optional


class PerspectiveTransformer:
    """
    Inverse Perspective Mapping (Bird's Eye View) transformer for road lane analysis.
    """

    def __init__(
        self,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
    ):
        self.src_points = src_points
        self.dst_points = dst_points

    def _default_points(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        src = np.float32(
            [
                [width * 0.35, height * 0.42],  # Top-left
                [width * 0.65, height * 0.42],  # Top-right
                [width * 0.98, height - 5],     # Bottom-right
                [width * 0.02, height - 5],     # Bottom-left
            ]
        )
        margin_x = width * 0.20
        dst = np.float32(
            [
                [margin_x, 0],
                [width - margin_x, 0],
                [width - margin_x, height],
                [margin_x, height],
            ]
        )
        return src, dst

    def get_matrices(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (M, M_inv) perspective transformation matrices."""
        default_src, default_dst = self._default_points(height, width)
        src = np.float32(self.src_points) if self.src_points is not None else default_src
        dst = np.float32(self.dst_points) if self.dst_points is not None else default_dst
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return M, M_inv

    def warp_to_birds_eye(self, frame: np.ndarray) -> np.ndarray:
        """Transform camera perspective view to top-down Bird's Eye View."""
        height, width = frame.shape[:2]
        M, _ = self.get_matrices(height, width)
        return cv2.warpPerspective(frame, M, (width, height), flags=cv2.INTER_LINEAR)

    def unwarp_to_camera_view(self, bev_frame: np.ndarray) -> np.ndarray:
        """Project top-down Bird's Eye View back to original camera perspective."""
        height, width = bev_frame.shape[:2]
        _, M_inv = self.get_matrices(height, width)
        return cv2.warpPerspective(bev_frame, M_inv, (width, height), flags=cv2.INTER_LINEAR)
