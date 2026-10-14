import numpy as np
import pytest
import cv2

from src.fitting.spline_fitter import SplineFitter
from src.piplinebuilder import PipelineBuilder

class DummyDetector:
    """Mock detector returning synthetic lane polygon masks for testing."""
    def __init__(self, masks=None):
        self.masks = masks or []
        
    def detect(self, frame):
        return None, self.masks
        
    def draw_lanes(self, frame, masks):
        annotated = frame.copy()
        for mask in masks:
            cv2.polylines(annotated, [mask], isClosed=False, color=(0, 255, 0), thickness=2)
        return annotated


def test_spline_fitter_initialization():
    fitter = SplineFitter(spline_order=3, smoothing=2.5, num_points=80, thickness=3)
    assert fitter.spline_order == 3
    assert fitter.smoothing == 2.5
    assert fitter.num_points == 80
    assert fitter.thickness == 3


def test_spline_fitter_curved_lane():
    fitter = SplineFitter(spline_order=3, smoothing=1.0, num_points=50)
    
    # Generate synthetic curved lane coordinates: y in [100, 500], x = 300 + 0.001 * (y - 300)^2
    y_vals = np.linspace(500, 100, 25)
    x_vals = 300 + 0.001 * (y_vals - 300)**2
    points = np.column_stack((x_vals, y_vals)).astype(np.int32)
    
    fitted = fitter.fit_lane(points)
    
    assert fitted is not None
    assert fitted.shape == (50, 2)
    # y coordinates should cover the original range
    assert fitted[:, 1].min() <= 120
    assert fitted[:, 1].max() >= 480


def test_spline_fitter_yolo_polygon_mask():
    fitter = SplineFitter(spline_order=3, smoothing=2.0, num_points=60)
    
    # A YOLO mask is typically a closed polygon around the lane boundaries
    # Left edge: x = 200, y going down from 100 to 500
    # Right edge: x = 220, y going up from 500 to 100
    left_edge = [[200, y] for y in range(100, 500, 20)]
    right_edge = [[220, y] for y in range(500, 100, -20)]
    polygon = np.array(left_edge + right_edge, dtype=np.int32)
    
    fitted = fitter.fit_lane(polygon)
    
    assert fitted is not None
    assert fitted.shape == (60, 2)
    # The centerline x should be approximately centered around 210
    mean_x = np.mean(fitted[:, 0])
    assert 200 <= mean_x <= 220


def test_spline_fitter_edge_cases():
    fitter = SplineFitter()
    
    # Empty points
    assert fitter.fit_lane([]) is None
    assert fitter.fit_lane(np.empty((0, 2))) is None
    
    # Single point
    assert fitter.fit_lane([[100, 200]]) is None
    
    # 2 points (cannot do cubic spline, should gracefully fall back to linear)
    two_pts = np.array([[100, 400], [150, 100]])
    fitted_two = fitter.fit_lane(two_pts)
    assert fitted_two is not None
    assert fitted_two.shape == (fitter.num_points, 2)
    
    # 3 points (k=2 quadratic spline or fallback)
    three_pts = np.array([[100, 400], [120, 250], [160, 100]])
    fitted_three = fitter.fit_lane(three_pts)
    assert fitted_three is not None
    assert fitted_three.shape == (fitter.num_points, 2)
    
    # Duplicate points
    dup_pts = np.array([[100, 100], [100, 100], [100, 100]])
    fitted_dup = fitter.fit_lane(dup_pts)
    # All identical points should return None since length < 2 after deduplication
    assert fitted_dup is None


def test_fit_and_draw():
    fitter = SplineFitter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    lane1 = np.array([[200, 400], [210, 300], [230, 200], [260, 100]])
    lane2 = np.array([[440, 400], [430, 300], [410, 200], [380, 100]])
    
    annotated = fitter.fit_and_draw(frame, [lane1, lane2])
    assert annotated.shape == (480, 640, 3)
    # Image should have drawn pixels (non-zero)
    assert np.count_nonzero(annotated) > 0


def test_draw_lane_area():
    fitter = SplineFitter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    curve1 = np.array([[200, 400], [220, 300], [250, 200]])
    curve2 = np.array([[400, 400], [380, 300], [350, 200]])
    
    shaded = fitter.draw_lane_area(frame, curve1, curve2, color=(0, 255, 0), alpha=0.3)
    assert shaded.shape == (480, 640, 3)
    assert np.count_nonzero(shaded) > 0


def test_pipeline_builder_integration():
    # Verify that PipelineBuilder works properly with SplineFitter and DummyDetector
    lane = np.array([[200, 400], [210, 300], [220, 200], [230, 100]])
    detector = DummyDetector(masks=[lane])
    fitter = SplineFitter()
    
    pipeline = PipelineBuilder(detector=detector, spline_fitter=fitter)
    assert pipeline.detector is detector
    assert pipeline.spline_fitter is fitter
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, masks = pipeline.detector.detect(frame)
    annotated = pipeline.spline_fitter.fit_and_draw(frame, masks)
    
    assert annotated.shape == (480, 640, 3)
    assert np.count_nonzero(annotated) > 0
