"""
src/preprocessing/batch_processor.py — Quy trình Tiền xử lý nâng cao & Xuất dữ liệu
cho `notebooks/01_pre_processing.ipynb`:
  1. Tăng cường tương phản cục bộ CLAHE + Khử nhiễu Gaussian
  2. Dò biên Canny kết hợp Lọc màu vạch kẻ đường HLS (Trắng & Vàng) trong vùng ROI
  3. Khử đốm nhiễu nhỏ (Connected Components) + Đóng hình thái học (Morphological Closing)
  4. Phát hiện Hough có trọng số + Tinh chỉnh theo dải trắng + Làm mượt B-Spline
     + Phân loại làn đường chuẩn Vương quốc Anh (UK Left-Hand Traffic)
  5. Xuất ảnh đã xử lý + tập train/valid chuẩn YOLO-seg vào `data/processing/`
  6. Xuất bảng số liệu `preprocessing_metrics.csv` và biểu đồ `preprocessing_pipeline_chart.png` ở `ROOT_DIR`.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.path import ROOT_DIR, RAW_DATA_DIR, PROCESSING_DATA_DIR
from src.preprocessing.enhance import apply_clahe, denoise_image
from src.fitting.spline_fitter import SplineFitter
from src.control.pure_pursuit import classify_uk_lanes, extract_path_from_lanes
from src.tusimple_to_yolo import convert_tusimple_to_yolo


def remove_small_blobs(binary_edges: np.ndarray, min_area: int = 15) -> np.ndarray:
    """Xóa các đốm nhiễu nhỏ (vân nhựa đường, lá cây, vết bẩn) bằng Connected Components."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_edges, connectivity=8)
    cleaned = np.zeros_like(binary_edges)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def build_uk_roi_mask(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo mặt nạ ROI hình đa giác 5 đỉnh nhìn xa cho góc cua và làn đường UK."""
    roi_vertices = np.array(
        [
            [
                (0, height - 1),
                (int(width * 0.36), int(height * 0.41)),
                (int(width * 0.64), int(height * 0.41)),
                (width - 1, int(height * 0.86)),
                (width - 1, height - 1),
            ]
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, roi_vertices, 255)
    return mask, roi_vertices


def preprocess_single_road_image(
    image_bgr: np.ndarray,
    spline_fitter: Optional[SplineFitter] = None,
) -> Dict[str, Any]:
    """
    Chạy đầy đủ các bước tiền xử lý nâng cao trên 1 khung hình BGR và trả về
    tất cả ảnh trung gian + kết quả phân loại làn UK + số liệu thống kê.
    """
    t0 = time.perf_counter()
    fitter = spline_fitter or SplineFitter(spline_order=3, smoothing=2.5, num_points=80)
    height, width = image_bgr.shape[:2]

    # Bước 1: Tăng cường tương phản CLAHE + ảnh xám + làm mờ Gaussian
    enhanced_bgr = apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 2: Dò biên Canny + Mặt nạ màu vạch đường HLS (Trắng + Vàng)
    canny_edges = cv2.Canny(gray_blur, threshold1=50, threshold2=150)
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, (0, 185, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(hls, (15, 90, 90), (38, 255, 255))
    color_lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Bước 3: Áp mặt nạ ROI
    roi_mask, roi_vertices = build_uk_roi_mask(height, width)
    roi_edges = cv2.bitwise_and(canny_edges, roi_mask)
    roi_color = cv2.bitwise_and(color_lane_mask, roi_mask)

    # Bước 4: Lọc nhiễu đốm nhỏ + Đóng hình thái học (Morphological Closing)
    clean_edges = remove_small_blobs(roi_edges, min_area=12)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected_edges = cv2.morphologyEx(clean_edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Bước 5: Hough Transform có trọng số + Tinh chỉnh trên dải màu vạch đường
    y_bottom = height - 5
    y_top = int(roi_vertices[:, :, 1].min())
    lines = cv2.HoughLinesP(
        connected_edges,
        rho=1.0,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=18,
        maxLineGap=140,
    )

    left_pts, left_w = [], []
    right_pts, right_w = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.35:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            if slope < 0 and max(x1, x2) < width * 0.62:
                left_pts.extend([(x1, y1), (x2, y2)])
                left_w.extend([length, length])
            elif slope > 0 and min(x1, x2) > width * 0.38:
                right_pts.extend([(x1, y1), (x2, y2)])
                right_w.extend([length, length])

    ys_color, xs_color = np.nonzero(roi_color)

    def _fit_and_refine_curve(pts_list, w_list) -> Optional[np.ndarray]:
        if len(pts_list) < 2:
            return None
        pts_arr = np.array(pts_list, dtype=np.float32)
        a, b = np.polyfit(pts_arr[:, 1], pts_arr[:, 0], deg=1, w=np.array(w_list, dtype=np.float32))
        if len(ys_color) > 30:
            x_pred = a * ys_color + b
            keep = (np.abs(xs_color - x_pred) < 32) & (ys_color >= y_top)
            if int(keep.sum()) >= 40:
                ref_pts = np.column_stack((xs_color[keep], ys_color[keep])).astype(np.int32)
                curve = fitter.fit_lane(ref_pts)
                if curve is not None:
                    return curve
        y_vals = np.linspace(y_bottom, y_top, 50)
        x_vals = np.clip(a * y_vals + b, 0, width - 1)
        raw_line = np.column_stack((x_vals, y_vals)).astype(np.int32)
        fitted_raw = fitter.fit_lane(raw_line)
        return fitted_raw if fitted_raw is not None else raw_line

    detected_curves: List[np.ndarray] = []
    l_curve = _fit_and_refine_curve(left_pts, left_w)
    r_curve = _fit_and_refine_curve(right_pts, right_w)
    if l_curve is not None:
        detected_curves.append(l_curve)
    if r_curve is not None:
        detected_curves.append(r_curve)

    # Bước 6: Phân loại làn đường chuẩn UK (Left-Hand Traffic) & Vẽ kết quả trực quan
    uk_info = classify_uk_lanes(detected_curves, image_bgr.shape)
    path_data = extract_path_from_lanes(detected_curves, image_bgr.shape, driving_side="uk_left")

    highlighted = image_bgr.copy()
    if l_curve is not None and r_curve is not None:
        highlighted = fitter.draw_lane_area(highlighted, l_curve, r_curve, color=(0, 220, 90), alpha=0.22)

    for label_name, curve in uk_info["labeled_curves"].items():
        color = (255, 200, 0) if label_name == "uk_left_edge" else (0, 165, 255)
        curve_pts = np.asarray(curve, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(highlighted, [curve_pts], isClosed=False, color=color, thickness=6, lineType=cv2.LINE_AA)

    if path_data.valid and path_data.centerline:
        cl_pts = np.array(path_data.centerline, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(highlighted, [cl_pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    total_px = float(max(1, height * width))
    edge_px = int(np.count_nonzero(connected_edges))
    color_px = int(np.count_nonzero(roi_color))
    return {
        "original": image_bgr,
        "gray": gray_blur,
        "enhanced_gray": gray_blur,
        "canny_edges": canny_edges,
        "roi_edges": roi_edges,
        "clean_edges": connected_edges,
        "roi_color": roi_color,
        "color_mask": roi_color,
        "highlighted": highlighted,
        "detected_curves": detected_curves,
        "uk_lane_type": uk_info["uk_lane_type"],
        "heading_error_rad": path_data.heading_error_rad or 0.0,
        "curvature": path_data.curvature or 0.0,
        "edge_pixels": edge_px,
        "color_pixels": color_px,
        "num_hough_segments": 0 if lines is None else int(len(lines)),
        "num_lanes": len(detected_curves),
        "num_detected_lanes": len(detected_curves),
        "edge_Validate_ratio": edge_px / total_px,
        "white_mask_ratio": color_px / total_px,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def run_batch_preprocessing(
    max_tusimple_clips: int = 12,
    max_highlight_samples: int = 12,
    raw_dir: Optional[Path] = None,
    processing_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    max_highlight_images: Optional[int] = None,
    max_yolo_clips: Optional[int] = None,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    """
    Chạy toàn bộ pipeline tiền xử lý cho `notebooks/01_pre_processing.ipynb`:
      1. Xuất các ảnh đã tiền xử lý & tô làn đường (`highlighted_*.jpg`) vào `data/processing/`
         và `data/processing/highlighted/`.
      2. Chuyển đổi tập dữ liệu TuSimple sang định dạng YOLO11-seg (`train/` & `valid/`) trong `data/processing/`.
      3. Xuất bảng số liệu `preprocessing_metrics.csv` và biểu đồ `preprocessing_pipeline_chart.png` ở `ROOT_DIR`.
    """
    raw_base = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    proc_base = Path(processing_dir) if processing_dir is not None else PROCESSING_DATA_DIR
    root_base = Path(root_dir) if root_dir is not None else ROOT_DIR
    if max_highlight_images is not None:
        max_highlight_samples = max_highlight_images
    if max_yolo_clips is not None:
        max_tusimple_clips = max_yolo_clips

    proc_base.mkdir(parents=True, exist_ok=True)
    highlight_dir = proc_base / "highlighted"
    highlight_dir.mkdir(parents=True, exist_ok=True)

    # Thu thập danh sách ảnh mẫu
    sample_images: List[Path] = []
    for p in sorted(raw_base.glob("*.jpg")):
        sample_images.append(p)
    tusimple_dir = raw_base / "TUSimple"
    if tusimple_dir.exists():
        for p in sorted(tusimple_dir.rglob("20.jpg"))[:max_highlight_samples]:
            sample_images.append(p)
            if len(sample_images) >= max_highlight_samples:
                break

    metrics_rows: List[Dict[str, Any]] = []
    first_result: Optional[Dict[str, Any]] = None
    first_bgr: Optional[np.ndarray] = None

    fitter = SplineFitter(spline_order=3, smoothing=2.5, num_points=80)
    for idx, img_path in enumerate(sample_images):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        res = preprocess_single_road_image(bgr, spline_fitter=fitter)
        if first_result is None:
            first_result = res
            first_bgr = bgr

        out_name = f"highlighted_{idx:02d}_{img_path.name}"
        cv2.imwrite(str(highlight_dir / out_name), res["highlighted"])

        metrics_rows.append(
            {
                "sample_id": idx,
                "image_name": img_path.name,
                "output_file": f"data/processing/highlighted/{out_name}",
                "num_lanes_detected": res["num_lanes"],
                "uk_lane_type": res["uk_lane_type"],
                "edge_pixels_in_roi": res["edge_pixels"],
                "color_pixels_in_roi": res["color_pixels"],
                "heading_error_rad": round(float(res["heading_error_rad"]), 5),
                "curvature": round(float(res["curvature"]), 5),
                "processing_time_ms": res["elapsed_ms"],
            }
        )

    # Xuất bộ dữ liệu train/valid chuẩn YOLO-seg vào data/processing/
    yolo_stats = {"train": {"imgs": 0, "polys": 0}, "valid": {"imgs": 0, "polys": 0}}
    if tusimple_dir.exists():
        yolo_stats = convert_tusimple_to_yolo(
            tusimple_root=tusimple_dir,
            output_dir=proc_base,
            val_ratio=val_ratio,
            lane_thickness=12,
            max_clips=max_tusimple_clips,
        )

    # Xuất file CSV số liệu tiền xử lý ở ngay ROOT_DIR
    csv_root_path = root_base / "preprocessing_metrics.csv"
    if metrics_rows:
        with open(csv_root_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    # Xuất biểu đồ tổng hợp quy trình tiền xử lý ở ngay ROOT_DIR
    chart_root_path = root_base / "preprocessing_pipeline_chart.png"
    if first_result is not None and first_bgr is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        panels = [
            (cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB), "1. Original Road Image"),
            (first_result["gray"], "2. CLAHE + Gaussian Denoised Gray"),
            (first_result["roi_edges"], "3. Canny Edges inside UK ROI"),
            (first_result["clean_edges"], "4. Blob Filtered + Morph Closed Edges"),
            (first_result["roi_color"], "5. HLS White/Yellow Lane Mask"),
            (
                cv2.cvtColor(first_result["highlighted"], cv2.COLOR_BGR2RGB),
                f"6. Final UK Lane Fit ({first_result['uk_lane_type']})",
            ),
        ]
        for ax, (panel_img, title) in zip(axes.flat, panels):
            ax.imshow(panel_img, cmap="gray" if panel_img.ndim == 2 else None)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

        plt.suptitle(
            "CPV301 — Multi-Stage Lane Preprocessing & UK Left-Hand Lane Extraction Pipeline",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(str(chart_root_path), dpi=140)
        plt.close(fig)

    return {
        "processed_samples": len(metrics_rows),
        "yolo_dataset_stats": yolo_stats,
        "csv_path": str(csv_root_path),
        "chart_path": str(chart_root_path),
    }


if __name__ == "__main__":
    summary = run_batch_preprocessing()
    print("Batch preprocessing summary:", summary)
