import sys
import json
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.fitting.spline_fitter import SplineFitter


def evaluate_lane_predictions(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    y_samples: np.ndarray,
    dist_threshold_px: float = 20.0,
) -> Dict[str, float]:
    """
    Compute standard TuSimple benchmark metrics between predicted lane curves and ground-truth lanes:
    - Accuracy: Fraction of ground-truth y-sample points matched within `dist_threshold_px`.
    - FP (False Positive rate): Fraction of predicted lanes that do not match any ground-truth lane.
    - FN (False Negative rate): Fraction of ground-truth lanes missed by predictions.
    """
    if len(gt_lanes) == 0:
        return {
            "accuracy": 1.0 if len(pred_lanes) == 0 else 0.0,
            "fp": 1.0 if len(pred_lanes) > 0 else 0.0,
            "fn": 0.0,
        }
    if len(pred_lanes) == 0:
        return {"accuracy": 0.0, "fp": 0.0, "fn": 1.0}

    def _interp_x_at_y(curve: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        pts = np.asarray(curve, dtype=np.float32)
        sort_idx = np.argsort(pts[:, 1])
        ys = pts[sort_idx, 1]
        xs = pts[sort_idx, 0]
        unique_ys, u_idx = np.unique(ys, return_index=True)
        unique_xs = xs[u_idx]
        valid = (y_grid >= ys.min() - 25) & (y_grid <= ys.max() + 25)
        out = np.full_like(y_grid, np.nan, dtype=np.float32)
        if np.any(valid):
            out[valid] = np.interp(y_grid[valid], unique_ys, unique_xs)
        # Extrapolate within ROI y_grid if curve covers at least 25% of ROI height
        if len(unique_ys) >= 2 and (ys.max() - ys.min()) >= 60.0 and not np.all(valid):
            try:
                deg = min(2, len(unique_ys) - 1)
                poly = np.polyfit(unique_ys, unique_xs, deg=deg)
                out[~valid] = np.polyval(poly, y_grid[~valid])
            except Exception:
                pass
        return out

    gt_sampled = [_interp_x_at_y(g, y_samples) for g in gt_lanes]
    pred_sampled = [_interp_x_at_y(p, y_samples) for p in pred_lanes]

    matched_gt = set()
    matched_pred = set()
    lane_accuracies: List[float] = []

    for g_idx, g_xs in enumerate(gt_sampled):
        valid_g = ~np.isnan(g_xs)
        n_valid_g = int(np.sum(valid_g))
        if n_valid_g == 0:
            continue

        best_acc = 0.0
        best_p_idx = None
        for p_idx, p_xs in enumerate(pred_sampled):
            both_valid = valid_g & (~np.isnan(p_xs))
            if not np.any(both_valid):
                continue
            diffs = np.abs(g_xs[both_valid] - p_xs[both_valid])
            hits = int(np.sum(diffs <= dist_threshold_px))
            acc = hits / float(n_valid_g)
            if acc > best_acc:
                best_acc = acc
                best_p_idx = p_idx

        lane_accuracies.append(best_acc)
        if best_acc >= 0.5 and best_p_idx is not None:
            matched_gt.add(g_idx)
            matched_pred.add(best_p_idx)

    accuracy = float(np.mean(lane_accuracies)) if lane_accuracies else 0.0
    fp = float(len(pred_lanes) - len(matched_pred)) / float(max(1, len(pred_lanes)))
    fn = float(len(gt_lanes) - len(matched_gt)) / float(max(1, len(gt_lanes)))

    return {
        "accuracy": round(accuracy, 4),
        "fp": round(max(0.0, fp), 4),
        "fn": round(max(0.0, fn), 4),
    }


def benchmark_detector_on_frames(
    detector: Any,
    frames_and_gt: List[Tuple[np.ndarray, List[np.ndarray]]],
    spline_fitter: SplineFitter = None,
    y_samples: np.ndarray = None,
) -> Dict[str, float]:
    """
    Run a detector (and optional `SplineFitter`) across a list of `(frame, gt_lanes)` pairs
    and return average `accuracy`, `fp`, `fn`, and `fps`.
    """
    if not frames_and_gt:
        return {"accuracy": 0.0, "fp": 0.0, "fn": 0.0, "fps": 0.0}

    h = frames_and_gt[0][0].shape[0]
    if y_samples is None:
        y_samples = np.linspace(int(h * 0.45), h - 10, 30)

    accs, fps_list, fns = [], [], []
    start_time = time.perf_counter()

    for frame, gt_lanes in frames_and_gt:
        _, masks = detector.detect(frame)
        if spline_fitter is not None and masks:
            pred_lanes = spline_fitter.fit_multiple_lanes(masks)
        else:
            pred_lanes = masks

        metrics = evaluate_lane_predictions(pred_lanes, gt_lanes, y_samples)
        accs.append(metrics["accuracy"])
        fps_list.append(metrics["fp"])
        fns.append(metrics["fn"])

    elapsed = max(1e-6, time.perf_counter() - start_time)
    fps = len(frames_and_gt) / elapsed

    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "fp": round(float(np.mean(fps_list)), 4),
        "fn": round(float(np.mean(fns)), 4),
        "fps": round(float(fps), 2),
    }


def load_tusimple_samples(
    tusimple_root: Path,
    max_samples: int = 20,
    ego_lanes_only: bool = True,
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Load `(frame, gt_lanes)` pairs from a TuSimple directory (`data/raw/TUSimple`).
    Automatically searches `train_set/label_data_*.json` and `test_label.json`.
    When `ego_lanes_only=True`, filters ground-truth lanes to the primary left & right ego lanes
    within the camera ROI so evaluation matches the ego-lane detection ROI.
    """
    root = Path(tusimple_root)
    candidates = [
        (root / "train_set", root / "train_set" / "label_data_0313.json"),
        (root / "train_set", root / "train_set" / "label_data_0531.json"),
        (root / "train_set", root / "train_set" / "label_data_0601.json"),
        (root / "test_set", root / "test_label.json"),
        (root, root / "label_data_0313.json"),
        (root, root / "test_label.json"),
    ]

    samples: List[Tuple[np.ndarray, List[np.ndarray]]] = []
    for base_dir, label_file in candidates:
        if not label_file.exists():
            continue
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= max_samples:
                    return samples
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = base_dir / rec["raw_file"]
                if not img_path.exists():
                    continue
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                mid_x = w / 2.0
                h_samples = rec.get("h_samples", [])
                all_lanes: List[np.ndarray] = []
                for lane_xs in rec.get("lanes", []):
                    pts = [
                        [int(x), int(y)]
                        for x, y in zip(lane_xs, h_samples)
                        if x is not None and x >= 0 and y >= int(h * 0.42)
                    ]
                    if len(pts) >= 2:
                        all_lanes.append(np.array(pts, dtype=np.int32))
                if ego_lanes_only and len(all_lanes) > 2:
                    left_cands = [ln for ln in all_lanes if ln[-1, 0] < mid_x]
                    right_cands = [ln for ln in all_lanes if ln[-1, 0] >= mid_x]
                    gt_lanes = []
                    if left_cands:
                        gt_lanes.append(max(left_cands, key=lambda ln: ln[-1, 0]))
                    if right_cands:
                        gt_lanes.append(min(right_cands, key=lambda ln: ln[-1, 0]))
                else:
                    gt_lanes = all_lanes
                if gt_lanes:
                    samples.append((frame, gt_lanes))
    return samples


if __name__ == "__main__":
    from model.demo import run_demo_comparison
    from src.preprocessing.preprocessor import ImagePreprocessor
    from src.detection.hough_transform import HoughLaneDetector
    from src.detection.lanenet import LaneNetDetector

    parser = argparse.ArgumentParser(description="Compare Lane Detection Methods.")
    parser.add_argument("--dataset", type=str, default="data/raw", help="Path to dataset folder")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hough", "lanenet", "proposed"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=15,
        help="Max TuSimple frames to benchmark when --dataset points to TuSimple",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/comparison_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if (dataset_path / "train_set").exists() or (dataset_path / "test_label.json").exists():
        tusimple_frames = load_tusimple_samples(dataset_path, max_samples=args.max_samples)
        if tusimple_frames:
            pre = ImagePreprocessor(use_clahe=True, blur_ksize=5)
            hough = HoughLaneDetector(preprocessor=pre, min_line_length=30, threshold=25)
            lanenet = LaneNetDetector(preprocessor=pre, min_samples=12)
            spline = SplineFitter(spline_order=3, smoothing=2.5, num_points=100)
            report = {}
            if "hough" in args.methods:
                report["Hough_Transform"] = benchmark_detector_on_frames(hough, tusimple_frames)
            if "lanenet" in args.methods:
                report["LaneNet"] = benchmark_detector_on_frames(lanenet, tusimple_frames)
            if "proposed" in args.methods:
                report["Proposed_Spline"] = benchmark_detector_on_frames(
                    lanenet, tusimple_frames, spline_fitter=spline
                )
        else:
            report = run_demo_comparison()
    else:
        report = run_demo_comparison()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation report to {out_path}")
