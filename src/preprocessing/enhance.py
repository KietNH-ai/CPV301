import cv2
import numpy as np
from typing import Tuple


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Supports both grayscale (2D) and BGR color (3D) images.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_image(
    image: np.ndarray,
    gaussian_ksize: int = 5,
    median_ksize: int = 3,
) -> np.ndarray:
    """
    Reduce high-frequency road/sensor noise using Median and Gaussian filters.
    """
    out = image.copy()
    if median_ksize and median_ksize > 1:
        mk = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        out = cv2.medianBlur(out, mk)
    if gaussian_ksize and gaussian_ksize > 1:
        gk = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        out = cv2.GaussianBlur(out, (gk, gk), 0)
    return out


def adaptive_canny(
    gray: np.ndarray,
    sigma: float = 0.33,
) -> np.ndarray:
    """
    Compute Canny edges using automatic thresholds derived from median pixel intensity.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower == upper:
        lower, upper = 50, 150
    return cv2.Canny(gray, lower, upper)
