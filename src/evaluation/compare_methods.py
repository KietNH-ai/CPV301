import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
        valid = (y_grid >= ys.min() - 10) & (y_grid <= ys.max() + 10)
        out = np.full_like(y_grid, np.nan, dtype=np.float32)
        if np.any(valid):
            out[valid] = np.interp(y_grid[valid], ys, xs)
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


if __name__ == "__main__":
    import sys

    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from model.demo import run_demo_comparison

    parser = argparse.ArgumentParser(description="Compare Lane Detection Methods.")
    parser.add_argument("--dataset", type=str, default="data/raw", help="Path to dataset folder")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["hough", "lanenet", "proposed"],
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/comparison_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    report = run_demo_comparison()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation report to {out_path}")

