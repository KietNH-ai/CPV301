import cv2
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from src import (
    ImagePreprocessor,
    PerspectiveTransformer,
    apply_clahe,
    denoise_image,
    adaptive_canny,
    HoughLaneDetector,
    LaneNetDetector,
    LaneDetector,
    SplineFitter,
    PipelineBuilder,
    evaluate_lane_predictions,
    benchmark_detector_on_frames,
)


def _create_synthetic_road_frame(
    height: int = 720, width: int = 1280, curved: bool = False
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Create a synthetic BGR road image with two white lane lines inside ROI and return (frame, gt_lanes)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y_vals = np.linspace(height - 20, int(height * 0.45), 40)

    if not curved:
        left_x = np.linspace(250, 540, 40)
        right_x = np.linspace(1030, 740, 40)
    else:
        # Sharp right-hand curve
        t = (height - 20 - y_vals) / float(height * 0.55)
        left_x = 250 + 220 * t + 140 * (t ** 2)
        right_x = 980 + 120 * t + 140 * (t ** 2)

    left_gt = np.column_stack((left_x, y_vals)).astype(np.int32)
    right_gt = np.column_stack((right_x, y_vals)).astype(np.int32)

    cv2.polylines(frame, [left_gt], isClosed=False, color=(255, 255, 255), thickness=10)
    cv2.polylines(frame, [right_gt], isClosed=False, color=(255, 255, 255), thickness=10)
    # Sky distraction outside ROI (y=100)
    cv2.line(frame, (100, 100), (1100, 100), (255, 255, 255), thickness=6)
    return frame, [left_gt, right_gt]


def test_image_preprocessor_and_enhancements():
    frame, _ = _create_synthetic_road_frame()
    pre = ImagePreprocessor(use_clahe=True, blur_ksize=5)

    connected_mask, steps = pre.process(frame, return_steps=True)

    assert connected_mask.shape == (720, 1280)
    assert set(steps.keys()) == {"gray", "edges", "roi_mask", "roi_edges", "connected_mask"}
    assert np.count_nonzero(steps["roi_edges"][:200, :]) == 0
    assert np.count_nonzero(connected_mask[350:, :]) > 0

    enhanced = apply_clahe(frame)
    denoised = denoise_image(enhanced)
    edges_auto = adaptive_canny(denoised)
    assert edges_auto.shape == (720, 1280)

    transformer = PerspectiveTransformer()
    bev = transformer.warp_to_birds_eye(frame)
    unwarped = transformer.unwarp_to_camera_view(bev)
    assert bev.shape == frame.shape
    assert unwarped.shape == frame.shape


def test_hough_lane_detector_and_pipeline():
    frame, _ = _create_synthetic_road_frame(curved=False)
    detector = HoughLaneDetector(min_line_length=30, threshold=25)

    raw_lines, lane_masks = detector.detect(frame)

    assert raw_lines is not None
    assert len(lane_masks) == 2

    annotated_hough = detector.draw_lanes(frame, lane_masks)
    assert annotated_hough.shape == frame.shape
    assert not np.array_equal(annotated_hough, frame)

    spline_fitter = SplineFitter(spline_order=3, smoothing=2.0)
    pipeline = PipelineBuilder(detector=detector, spline_fitter=spline_fitter)
    _, masks = pipeline.detector.detect(frame)
    annotated_spline = pipeline.spline_fitter.fit_and_draw(frame, masks)
    assert annotated_spline.shape == frame.shape


def test_lanenet_detector():
    frame, _ = _create_synthetic_road_frame(curved=True)
    detector = LaneNetDetector(min_samples=10)

    results, lane_masks = detector.detect(frame)
    assert "binary_mask" in results
    assert len(lane_masks) >= 2

    annotated = detector.draw_lanes(frame, lane_masks)
    assert annotated.shape == frame.shape


def test_proposed_method_and_evaluation(tmp_path):
    frame, gt_lanes = _create_synthetic_road_frame(curved=True)

    class MockYOLODetector:
        def detect(self, img):
            return None, gt_lanes

    proposed = LaneDetector(detector=MockYOLODetector(), spline_order=3, smoothing=1.5)
    result = proposed.detect_frame(frame)
    out_img_path = tmp_path / "proposed_result.jpg"
    vis = result.visualize(save_path=out_img_path)

    assert out_img_path.exists()
    assert vis.shape == frame.shape
    assert len(result.fitted_curves) == 2

    # Benchmark comparison on curved road: Proposed (YOLO+Spline) vs Straight Hough
    hough = HoughLaneDetector(min_line_length=25, threshold=20)
    metrics_hough = benchmark_detector_on_frames(hough, [(frame, gt_lanes)])
    metrics_proposed = benchmark_detector_on_frames(
        MockYOLODetector(), [(frame, gt_lanes)], spline_fitter=proposed.spline_fitter
    )

    assert metrics_proposed["accuracy"] >= 0.90
    assert metrics_proposed["accuracy"] > metrics_hough["accuracy"]
    assert metrics_proposed["fps"] > 0
