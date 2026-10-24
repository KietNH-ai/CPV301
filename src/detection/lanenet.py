import sys
from pathlib import Path
import cv2
import numpy as np
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None
from typing import List, Tuple, Optional, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.preprocessor import ImagePreprocessor
from src.preprocessing.transform import PerspectiveTransformer


class LaneNetDetector:
    """
    LaneNet-style two-branch Lane Detection module (following `docs/05_how_to_use_lanenet.md`):
    - Branch 1 (Binary Segmentation): Separates lane pixels from background using ROI + color/edge cues
      and filters out horizontal vehicle bumper edges via directional Sobel gradients.
    - Branch 2 (Instance Embedding & Clustering): Computes spatial-orientation embeddings for candidate
      lane pixels and clusters them into distinct lane instances via DBSCAN.
    - H-Net / Perspective Curve Fitting: Fits smooth polynomial curves per clustered lane instance.
    """

    INSTANCE_COLORS: List[Tuple[int, int, int]] = [
        (255, 180, 0),
        (0, 220, 255),
        (255, 0, 180),
        (0, 255, 128),
    ]

    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        eps: float = 0.06,
        min_samples: int = 15,
        max_lanes: int = 4,
        poly_degree: int = 2,
    ):
        self.preprocessor = preprocessor or ImagePreprocessor(use_clahe=False, blur_ksize=5)
        self.perspective = PerspectiveTransformer()
        self.eps = eps
        self.min_samples = min_samples
        self.max_lanes = max_lanes
        self.poly_degree = poly_degree

    def _binary_segmentation_branch(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Branch 1: Classify every pixel inside the road ROI as 'Lane' (255) or 'Background' (0),
        suppressing horizontal vehicle edges using directional gradients.
        """
        gray = self.preprocessor.to_grayscale(frame)
        edge_mask = self.preprocessor.process(frame)

        # Compute directional Sobel gradients to suppress horizontal car bumpers
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        non_horizontal = (np.abs(gx) > np.abs(gy) * 0.45).astype(np.uint8) * 255

        filtered_edges = cv2.bitwise_and(edge_mask, non_horizontal)
        connected = self.preprocessor.connect_edges(filtered_edges)
        return connected, gx

    def _instance_embedding_and_clustering(
        self, binary_mask: np.ndarray, gx: np.ndarray
    ) -> List[np.ndarray]:
        """
        Branch 2: Compute normalized feature embeddings for candidate lane pixels
        and cluster them into individual lane lines using DBSCAN.
        """
        height, width = binary_mask.shape[:2]
        ys, xs = np.nonzero(binary_mask)
        if len(xs) < self.min_samples:
            return []

        if len(xs) > 2500:
            step = max(1, len(xs) // 2500)
            xs = xs[::step]
            ys = ys[::step]

        # Project (x, y) along perspective lines toward horizon vanishing point
        vp_x = width * 0.50
        vp_y = height * 0.36
        dy = np.maximum(ys - vp_y, 25.0)
        projected_base_x = vp_x + (xs - vp_x) * ((height - vp_y) / dy)

        # Include horizontal side (left vs right of ego-lane center) in embedding
        side_feature = np.where(xs < vp_x, -0.15, 0.15)
        feat_x = (projected_base_x / float(width)) + side_feature
        feat_y = (ys / float(height)) * 0.18
        embeddings = np.column_stack((feat_x, feat_y))

        if DBSCAN is None:
            # Thuật toán gom cụm dự phòng khi chưa cài scikit-learn (phân nhóm theo trục chiếu đáy đường)
            left_mask = xs < vp_x
            right_mask = xs >= vp_x
            clusters_fb = []
            for m in (left_mask, right_mask):
                if np.sum(m) >= max(3, self.min_samples):
                    pts = np.column_stack((xs[m], ys[m]))
                    clusters_fb.append((len(pts), pts))
            return [pts for _, pts in clusters_fb[: self.max_lanes]]

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(embeddings)
        labels = clustering.labels_

        clusters: List[Tuple[int, np.ndarray]] = []
        for lbl in set(labels):
            if lbl == -1:
                continue
            pts = np.column_stack((xs[labels == lbl], ys[labels == lbl]))
            y_span = pts[:, 1].max() - pts[:, 1].min()
            x_span = pts[:, 0].max() - pts[:, 0].min()
            if y_span >= height * 0.10 and x_span >= 15:
                clusters.append((len(pts), pts))

        clusters.sort(key=lambda item: item[0], reverse=True)
        return [pts for _, pts in clusters[: self.max_lanes]]

    def _fit_instance_curve(
        self, pts: np.ndarray, height: int, width: int
    ) -> Optional[np.ndarray]:
        """Fit a smooth polynomial curve x = f(y) for a clustered lane instance."""
        if len(pts) < 3:
            return None
        try:
            deg = min(self.poly_degree, len(pts) - 1)
            poly = np.polyfit(pts[:, 1], pts[:, 0], deg=deg)
            y_max = min(height - 8, int(pts[:, 1].max() + 30))
            y_min = max(int(height * 0.45), int(pts[:, 1].min()))
            y_vals = np.linspace(y_max, y_min, 40)
            x_vals = np.clip(np.polyval(poly, y_vals), 0, width - 1)
            return np.column_stack((x_vals, y_vals)).astype(np.int32)
        except Exception:
            return pts.astype(np.int32)

    def detect(self, frame: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Run LaneNet binary segmentation + instance clustering on a frame.
        
        Returns:
            results: Dict containing 'binary_mask'.
            lane_masks: List of (N, 2) coordinate arrays for each detected lane instance.
        """
        height, width = frame.shape[:2]
        binary_mask, gx = self._binary_segmentation_branch(frame)
        clusters = self._instance_embedding_and_clustering(binary_mask, gx)

        lane_masks: List[np.ndarray] = []
        for pts in clusters:
            curve = self._fit_instance_curve(pts, height, width)
            if curve is not None and len(curve) >= 2:
                lane_masks.append(curve)

        # Sort lanes from left to right by bottom-most x coordinate
        lane_masks.sort(key=lambda m: float(m[0, 0]))
        return {"binary_mask": binary_mask}, lane_masks

    def draw_lanes(
        self,
        frame: np.ndarray,
        lane_masks: List[np.ndarray],
        thickness: int = 4,
    ) -> np.ndarray:
        """Draw instance-segmented lane curves with distinct colors per lane."""
        annotated = frame.copy()
        for idx, curve in enumerate(lane_masks):
            color = self.INSTANCE_COLORS[idx % len(self.INSTANCE_COLORS)]
            cv2.polylines(
                annotated,
                [curve],
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

    parser = argparse.ArgumentParser(description="Run LaneNet Lane Detection on an image.")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_DIR / "20.jpg"), help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DATA_DIR / "03_lanenet_detection.jpg"),
        help="Path to save annotated output image",
    )
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {args.input}")

    det = LaneNetDetector()
    _, masks = det.detect(img)
    out_img = det.draw_lanes(img, masks)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, out_img)
    print(f"[LaneNetDetector] Detected {len(masks)} lane instance(s). Saved result to {args.output}")

