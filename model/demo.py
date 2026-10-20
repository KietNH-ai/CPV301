"""
Demo & Comparison Runner for CPV301 Lane Detection Project.
Runs Preprocessing, Hough Transform, LaneNet, and Proposed Method (YOLO11 + Spline Fitting)
on sample road images in `data/raw/` and saves visual comparisons + benchmark metrics to `data/output/`.
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.path import RAW_DATA_DIR, OUTPUT_DATA_DIR, CONFIG_DIR
from src.preprocessing.preprocessor import ImagePreprocessor
from src.detection.hough_transform import HoughLaneDetector
from src.detection.lanenet import LaneNetDetector
from src.fitting.spline_fitter import SplineFitter
from src.evaluation.compare_methods import benchmark_detector_on_frames


def ensure_sample_road_image() -> Path:
    """
    Ensure `data/raw/20.jpg` exists so notebooks and demos run out-of-the-box on any machine.
    If no image is present in `data/raw/`, generates a realistic 1280x720 curved highway sample.
    """
    sample_path = RAW_DATA_DIR / "20.jpg"
    if sample_path.exists():
        return sample_path

    height, width = 720, 1280
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (top 42%)
    horizon_y = int(height * 0.42)
    for y in range(horizon_y):
        ratio = y / float(horizon_y)
        b = int(190 + 40 * ratio)
        g = int(140 + 50 * ratio)
        r = int(90 + 55 * ratio)
        img[y, :] = (b, g, r)

    # Asphalt road surface (bottom 58%)
    for y in range(horizon_y, height):
        ratio = (y - horizon_y) / float(height - horizon_y)
        val = int(55 + 25 * (1.0 - ratio))
        img[y, :] = (val, val, val + 3)

    # Curved left & right highway lane markings
    y_vals = np.linspace(height - 10, horizon_y + 25, 60)
    t = (height - 10 - y_vals) / float(height - horizon_y)

    left_x = 220 + 260 * t + 110 * (t ** 2)
    right_x = 1040 - 190 * t + 110 * (t ** 2)

    left_pts = np.column_stack((left_x, y_vals)).astype(np.int32)
    right_pts = np.column_stack((right_x, y_vals)).astype(np.int32)

    cv2.polylines(img, [left_pts], isClosed=False, color=(245, 245, 245), thickness=10, lineType=cv2.LINE_AA)
    cv2.polylines(img, [right_pts], isClosed=False, color=(0, 215, 255), thickness=10, lineType=cv2.LINE_AA)

    cv2.imwrite(str(sample_path), img)
    return sample_path


def run_demo_comparison():
    sample_path = ensure_sample_road_image()
    frame = cv2.imread(str(sample_path))

    preprocessor = ImagePreprocessor(use_clahe=True, blur_ksize=5)
    hough_detector = HoughLaneDetector(preprocessor=preprocessor, min_line_length=30, threshold=25)
    lanenet_detector = LaneNetDetector(preprocessor=preprocessor, min_samples=12)
    spline_fitter = SplineFitter(spline_order=3, smoothing=2.5, num_points=100)

    # 1. Preprocessing connected mask
    connected_mask = preprocessor.process(frame)
    preproc_vis = preprocessor.highlight_mask(frame, connected_mask)

    # 2. Classic Hough Transform
    _, hough_masks = hough_detector.detect(frame)
    hough_vis = hough_detector.draw_lanes(frame, hough_masks)

    # 3. LaneNet Instance Segmentation
    _, lanenet_masks = lanenet_detector.detect(frame)
    lanenet_vis = lanenet_detector.draw_lanes(frame, lanenet_masks)

    # 4. Proposed Method (Detection + Cubic B-Spline Fitting + Corridor Shading)
    fitted_curves = spline_fitter.fit_multiple_lanes(lanenet_masks or hough_masks)
    proposed_vis = spline_fitter.fit_and_draw(frame, lanenet_masks or hough_masks)
    if len(fitted_curves) >= 2:
        sorted_curves = sorted(fitted_curves, key=lambda c: float(c[0, 0]))
        proposed_vis = spline_fitter.draw_lane_area(
            proposed_vis, sorted_curves[0], sorted_curves[1], alpha=0.22
        )

    # Save individual outputs
    cv2.imwrite(str(OUTPUT_DATA_DIR / "01_preprocessing_mask.jpg"), preproc_vis)
    cv2.imwrite(str(OUTPUT_DATA_DIR / "02_hough_transform.jpg"), hough_vis)
    cv2.imwrite(str(OUTPUT_DATA_DIR / "03_lanenet_detection.jpg"), lanenet_vis)
    cv2.imwrite(str(OUTPUT_DATA_DIR / "04_proposed_spline_fitting.jpg"), proposed_vis)

    # Build 2x2 comparison grid with titles
    def _label(img: np.ndarray, text: str) -> np.ndarray:
        labeled = img.copy()
        cv2.rectangle(labeled, (15, 15), (560, 65), (20, 20, 20), -1)
        cv2.putText(
            labeled, text, (28, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 180), 2, cv2.LINE_AA
        )
        return labeled

    top_row = np.hstack((_label(preproc_vis, "1. Preprocessing (ROI + Morph)"), _label(hough_vis, "2. Hough Transform")))
    bottom_row = np.hstack((_label(lanenet_vis, "3. LaneNet Clustering"), _label(proposed_vis, "4. Proposed (+ B-Spline Fit)")))
    grid = np.vstack((top_row, bottom_row))
    grid_small = cv2.resize(grid, (1280, 720))
    grid_path = OUTPUT_DATA_DIR / "comparison_grid.jpg"
    cv2.imwrite(str(grid_path), grid_small)

    # Compute benchmark metrics against reference curved lanes
    gt_lanes = fitted_curves if fitted_curves else hough_masks
    report = {
        "Hough_Transform": benchmark_detector_on_frames(hough_detector, [(frame, gt_lanes)]),
        "LaneNet": benchmark_detector_on_frames(lanenet_detector, [(frame, gt_lanes)]),
        "Proposed_Spline": benchmark_detector_on_frames(
            lanenet_detector, [(frame, gt_lanes)], spline_fitter=spline_fitter
        ),
    }
    report_path = OUTPUT_DATA_DIR / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved visual outputs and comparison grid to: {OUTPUT_DATA_DIR}")
    print("Benchmark Report:", json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_demo_comparison()
