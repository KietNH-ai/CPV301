import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_DEFAULT_BEV_YAML = Path(__file__).resolve().parent.parent.parent / "config" / "bev.yaml"


class PerspectiveTransformer:
    """
    Inverse Perspective Mapping (Bird's Eye View) transformer for road lane analysis.
    Supports loading calibration ratios from `config/bev.yaml` while preserving
    exact default fallback behavior when custom points are not supplied.
    """

    def __init__(
        self,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        config_path: Optional[Union[str, Path]] = None,
        preset: Optional[str] = None,
    ):
        self.src_points = src_points
        self.dst_points = dst_points
        self.config_path = Path(config_path) if config_path is not None else _DEFAULT_BEV_YAML
        self.preset = preset
        self._bev_cfg: Optional[Dict[str, Any]] = self._load_yaml_config()

    def _load_yaml_config(self) -> Optional[Dict[str, Any]]:
        """Load BEV preset configuration from YAML if available."""
        if yaml is None or self.config_path is None or not self.config_path.exists():
            return None
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            presets = data.get("presets", {})
            active = self.preset or data.get("active_preset", "default")
            if active in presets:
                return presets[active]
            return data if "src_ratios" in data else None
        except Exception:
            return None

    @classmethod
    def from_yaml(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        preset: str = "default",
    ) -> "PerspectiveTransformer":
        """Create a PerspectiveTransformer configured from `config/bev.yaml`."""
        return cls(config_path=config_path, preset=preset)

    def _default_points(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._bev_cfg and "src_ratios" in self._bev_cfg and "dst_ratios" in self._bev_cfg:
            sr = self._bev_cfg["src_ratios"]
            dr = self._bev_cfg["dst_ratios"]
            tl_x, tl_y = sr.get("top_left", [0.35, 0.42])
            tr_x, tr_y = sr.get("top_right", [0.65, 0.42])
            br_x = float(sr.get("bottom_right_x", 0.98))
            bl_x = float(sr.get("bottom_left_x", 0.02))
            bottom_margin = float(sr.get("bottom_margin_px", 5))
            bottom_y = max(0.0, float(height) - bottom_margin)

            src = np.float32(
                [
                    [width * float(tl_x), height * float(tl_y)],  # Top-left
                    [width * float(tr_x), height * float(tr_y)],  # Top-right
                    [width * br_x, bottom_y],                     # Bottom-right
                    [width * bl_x, bottom_y],                     # Bottom-left
                ]
            )
            margin_x = width * float(dr.get("margin_x", 0.20))
            top_y = height * float(dr.get("top_y", 0.0))
            bot_y = height * float(dr.get("bottom_y", 1.0))
            dst = np.float32(
                [
                    [margin_x, top_y],
                    [width - margin_x, top_y],
                    [width - margin_x, bot_y],
                    [margin_x, bot_y],
                ]
            )
            return src, dst

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

    def warp_points(
        self,
        points: np.ndarray,
        height: int,
        width: int,
        inverse: bool = False,
    ) -> np.ndarray:
        """Transform (N, 2) coordinate array between camera view and Bird's Eye View."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts_reshaped = pts.reshape(-1, 1, 2)
        M, M_inv = self.get_matrices(height, width)
        mat = M_inv if inverse else M
        transformed = cv2.perspectiveTransform(pts_reshaped, mat)
        return transformed.reshape(-1, 2)
