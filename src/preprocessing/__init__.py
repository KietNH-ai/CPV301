"""
Image Preprocessing package for CPV301 Lane Detection.
"""

from src.preprocessing.preprocessor import ImagePreprocessor
from src.preprocessing.enhance import apply_clahe, denoise_image, adaptive_canny
from src.preprocessing.transform import PerspectiveTransformer
from src.preprocessing.batch_processor import (
    remove_small_blobs,
    build_uk_roi_mask,
    extract_adaptive_lane_color_mask,
    preprocess_single_road_image,
    load_or_build_sample_label_verification,
    collect_labeled_road_images,
    run_batch_preprocessing,
)

__all__ = [
    "ImagePreprocessor",
    "apply_clahe",
    "denoise_image",
    "adaptive_canny",
    "PerspectiveTransformer",
    "remove_small_blobs",
    "build_uk_roi_mask",
    "extract_adaptive_lane_color_mask",
    "preprocess_single_road_image",
    "load_or_build_sample_label_verification",
    "collect_labeled_road_images",
    "run_batch_preprocessing",
]
