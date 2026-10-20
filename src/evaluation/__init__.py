"""
Evaluation and Method Comparison package (TuSimple Accuracy, FP, FN, FPS).
"""

from src.evaluation.compare_methods import (
    evaluate_lane_predictions,
    benchmark_detector_on_frames,
)

__all__ = ["evaluate_lane_predictions", "benchmark_detector_on_frames"]
