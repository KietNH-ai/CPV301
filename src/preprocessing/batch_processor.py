"""
src/preprocessing/batch_processor.py — Quy trình Tiền xử lý nâng cao, Kiểm chứng Label
& Xuất dữ liệu toàn diện cho `notebooks/01_pre_processing.ipynb`:
  0. Trực quan hóa & Kiểm chứng song song 2 loại File Label:
     - Nhãn gốc TuSimple JSON (`lanes` + `h_samples` tại `20.jpg` của mỗi clip)
     - Nhãn đa giác chuẩn hóa YOLO11-seg (`data/processing/{train,valid}/labels/*.txt`)
  1. Tăng cường tương phản cục bộ CLAHE + Khử nhiễu Gaussian
  2. Dò biên Canny kết hợp Lọc màu vạch kẻ đường HLS thích ứng (Trắng & Vàng) trong vùng ROI
  3. Khử đốm nhiễu nhỏ (Connected Components) + Đóng hình thái học (Morphological Closing)
  4. Phát hiện Hough có trọng số + Tinh chỉnh theo dải màu + Làm mượt B-Spline
     + Phân loại làn đường chuẩn Vương quốc Anh (UK Left-Hand Traffic)
  5. Chuyển đổi toàn bộ Dataset TuSimple có nhãn (quét theo `raw_file` trong JSON, không quét mù
     134.571 file `.jpg`) kèm tiền xử lý CLAHE (`apply_preprocessing=True`) sang `data/processing/`
  6. Xuất bảng số liệu `preprocessing_metrics.csv` và biểu đồ `preprocessing_pipeline_chart.png` ở `ROOT_DIR`.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
>>>>>>> 500ca77ce0c2ce178cb43db4bfe404b777a15902
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.path import ROOT_DIR, RAW_DATA_DIR, PROCESSING_DATA_DIR
from src.preprocessing.enhance import apply_clahe
from src.fitting.spline_fitter import SplineFitter
from src.control.pure_pursuit import classify_uk_lanes, extract_path_from_lanes
from src.tusimple_to_yolo import (
    convert_tusimple_to_yolo,
    iter_labeled_tusimple_samples,
    lane_to_polygons,
    polygon_to_yolo_line,
    render_label_verification_pair,
)


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


def extract_adaptive_lane_color_mask(
    image_bgr: np.ndarray,
    enhanced_bgr: np.ndarray,
    gray_blur: np.ndarray,
) -> np.ndarray:
    """
    Trích xuất mặt nạ màu vạch kẻ đường (Trắng & Vàng) trên không gian HLS kết hợp
    Top-Hat hình thái học trên ảnh đã qua CLAHE để bắt tốt vạch trắng dưới trời âm u / bóng râm.
    """
    hls_raw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    hls_enh = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HLS)

    # Vạch trắng sáng rõ trên ảnh gốc hoặc ảnh đã cân bằng sáng CLAHE
    white_raw = cv2.inRange(hls_raw, (0, 180, 0), (255, 255, 255))
    white_enh = cv2.inRange(hls_enh, (0, 165, 0), (255, 255, 85))

    # Bộ lọc Top-Hat nổi bật các dải vạch sáng hẹp so với nền mặt đường xám tối (như 20.jpg)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    tophat = cv2.morphologyEx(gray_blur, cv2.MORPH_TOPHAT, kernel_tophat)
    _, tophat_mask = cv2.threshold(tophat, 22, 255, cv2.THRESH_BINARY)
    bright_road_band = cv2.inRange(gray_blur, 115, 255)
    white_tophat = cv2.bitwise_and(tophat_mask, bright_road_band)

    # Vạch vàng (tiêu chuẩn vạch phân làn / lề đường)
    yellow_raw = cv2.inRange(hls_raw, (15, 85, 80), (38, 255, 255))
    yellow_enh = cv2.inRange(hls_enh, (15, 85, 80), (38, 255, 255))

    white_mask = cv2.bitwise_or(cv2.bitwise_or(white_raw, white_enh), white_tophat)
    yellow_mask = cv2.bitwise_or(yellow_raw, yellow_enh)
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return remove_small_blobs(combined, min_area=10)


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

    # Bước 2: Dò biên Canny + Mặt nạ màu vạch đường HLS thích ứng (Trắng + Vàng)
    canny_edges = cv2.Canny(gray_blur, threshold1=50, threshold2=150)
    color_lane_mask = extract_adaptive_lane_color_mask(image_bgr, enhanced_bgr, gray_blur)

    # Bước 3: Áp mặt nạ ROI đa giác
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
    edge_ratio = edge_px / total_px
    return {
        "original": image_bgr,
        "enhanced_bgr": enhanced_bgr,
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
        "edge_validate_ratio": edge_ratio,
        "edge_Validate_ratio": edge_ratio,
        "white_mask_ratio": color_px / total_px,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def load_or_build_sample_label_verification(
    raw_dir: Optional[Path] = None,
    processing_dir: Optional[Path] = None,
    lane_thickness: int = 12,
) -> Dict[str, Any]:
    """
    Tải 1 mẫu nhãn thực tế từ `data/raw/TUSimple` (nếu có) hoặc khởi tạo bản ghi chuẩn
    cấu trúc TuSimple JSON (`raw_file`, `h_samples`, `lanes`) trên ảnh mẫu `data/raw/20.jpg`
    để phục vụ Bước 1: Trực quan hóa & Kiểm chứng song song 2 định dạng Label:
      1. Nhãn gốc JSON của TuSimple (`lanes` + `h_samples` với `-2` và `x >= 0`).
      2. Nhãn đa giác YOLO11-seg (`<class_id> <x1> <y1> ... <xn> <yn>`).
    """
    raw_base = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    proc_base = Path(processing_dir) if processing_dir is not None else PROCESSING_DATA_DIR
    tusimple_dir = raw_base / "TUSimple"

    labeled_samples = iter_labeled_tusimple_samples(tusimple_dir, include_test=True, max_samples=1)
    if labeled_samples:
        img_path, rec, subset_tag = labeled_samples[0]
        image_bgr = cv2.imread(str(img_path))
    else:
        # Fallback khi chạy trên máy chỉ có ảnh mẫu `data/raw/20.jpg`
        img_path = raw_base / "20.jpg"
        if not img_path.exists():
            jpgs = sorted(raw_base.glob("*.jpg"))
            img_path = jpgs[0] if jpgs else img_path
        image_bgr = cv2.imread(str(img_path)) if img_path.exists() else np.zeros((720, 1280, 3), dtype=np.uint8)
        res = preprocess_single_road_image(image_bgr)
        h_samples = list(range(240, 720, 10))  # 48 mốc Y chuẩn của TuSimple (240..710)
        lanes: List[List[int]] = []
        for curve in res["detected_curves"]:
            curve_arr = np.asarray(curve, dtype=np.float32)
            order = np.argsort(curve_arr[:, 1])
            ys_c = curve_arr[order, 1]
            xs_c = curve_arr[order, 0]
            y_min, y_max = float(ys_c.min()), float(ys_c.max())
            lane_row: List[int] = []
            for y_s in h_samples:
                if y_min <= y_s <= y_max:
                    x_interp = int(round(float(np.interp(y_s, ys_c, xs_c))))
                    lane_row.append(max(0, min(image_bgr.shape[1] - 1, x_interp)))
                else:
                    lane_row.append(-2)
            lanes.append(lane_row)
        rec = {
            "raw_file": f"clips/sample/{img_path.name}",
            "h_samples": h_samples,
            "lanes": lanes,
        }
        subset_tag = "sample_raw"

    # Kiểm tra nếu đã có file label .txt trong data/processing/train/labels
    yolo_lines: Optional[List[str]] = None
    label_txt_path: Optional[Path] = None
    train_lbl_dir = proc_base / "train" / "labels"
    if train_lbl_dir.exists():
        stem_candidate = str(Path(rec["raw_file"]).parent).replace("/", "_").replace("\\", "_")
        cand_path = train_lbl_dir / f"{stem_candidate}_{Path(rec['raw_file']).stem}.txt"
        if cand_path.exists():
            label_txt_path = cand_path
            yolo_lines = cand_path.read_text(encoding="utf-8").strip().splitlines()

    verif = render_label_verification_pair(
        image_bgr=image_bgr,
        tusimple_record=rec,
        yolo_lines=yolo_lines,
        lane_thickness=lane_thickness,
        apply_clahe_on_yolo_view=True,
    )
    verif["image_path"] = str(img_path)
    verif["subset"] = subset_tag
    verif["tusimple_record"] = rec
    verif["label_txt_path"] = str(label_txt_path) if label_txt_path else "generated_in_memory"
    return verif


def collect_labeled_road_images(
    raw_dir: Path,
    include_test: bool = True,
    max_samples: Optional[int] = None,
) -> List[Tuple[Path, str]]:
    """
    Thu thập danh sách các bức ảnh CÓ NHÃN thực sự để chạy đánh giá tiền xử lý:
      1. Các file ảnh mẫu trực tiếp trong `data/raw/*.jpg`
      2. Toàn bộ các ảnh `20.jpg` được liệt kê trong các file JSON Ground-Truth của
         `data/raw/TUSimple` (`label_data_0313.json`, `label_data_0531.json`,
         `label_data_0601.json`, và `test_label.json` nếu `include_test=True`).
      Tuyệt đối KHÔNG quét mù 134.571 file `.jpg` không có nhãn trong `clips/`.
    """
    collected: List[Tuple[Path, str]] = []
    seen_paths = set()

    for p in sorted(raw_dir.glob("*.jpg")):
        rp = p.resolve()
        if rp not in seen_paths:
            seen_paths.add(rp)
            collected.append((p, "raw_root"))
            if max_samples is not None and max_samples > 0 and len(collected) >= max_samples:
                return collected

    tusimple_dir = raw_dir / "TUSimple"
    if tusimple_dir.exists():
        remaining = None if max_samples is None else max(0, max_samples - len(collected))
        for img_path, _rec, subset_tag in iter_labeled_tusimple_samples(
            tusimple_dir,
            include_test=include_test,
            max_samples=remaining,
        ):
            rp = img_path.resolve()
            if rp not in seen_paths:
                seen_paths.add(rp)
                collected.append((img_path, subset_tag))
                if max_samples is not None and max_samples > 0 and len(collected) >= max_samples:
                    return collected

    return collected


def _save_pipeline_chart_agg(
    first_bgr: np.ndarray,
    first_result: Dict[str, Any],
    chart_path: Path,
) -> None:
    """
    Lưu biểu đồ 6 bước tiền xử lý bằng `Figure` + `FigureCanvasAgg` độc lập,
    không gọi `matplotlib.use('Agg')` toàn cục để tránh lỗi `plt.show()` trong Jupyter Notebook.
    """
    fig = Figure(figsize=(18, 10), dpi=140)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
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

    fig.suptitle(
        "CPV301 — Multi-Stage Lane Preprocessing & UK Left-Hand Lane Extraction Pipeline",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(chart_path), dpi=140)


def run_batch_preprocessing(
    max_tusimple_clips: Optional[int] = None,
    max_highlight_samples: Optional[int] = None,
    raw_dir: Optional[Path] = None,
    processing_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    max_highlight_images: Optional[int] = None,
    max_yolo_clips: Optional[int] = None,
    val_ratio: float = 0.2,
    apply_preprocessing: bool = True,
    include_test_set: bool = False,
    include_test_in_metrics: bool = True,
    max_saved_highlight_files: Optional[int] = 100,
) -> Dict[str, Any]:
    """
    Chạy toàn bộ pipeline tiền xử lý nâng cao cho `notebooks/01_pre_processing.ipynb`:
      1. Quét toàn bộ các ảnh có nhãn Ground-Truth của TuSimple (theo danh sách `raw_file`
         trong các file JSON, mặc định `max_highlight_samples=None` = không giới hạn 60 ảnh)
         để đánh giá tiền xử lý và ghi thống kê đầy đủ ra `preprocessing_metrics.csv`.
      2. Chuyển đổi toàn bộ tập dữ liệu TuSimple sang định dạng YOLO11-seg (`train/` & `valid/`)
         trong `data/processing/` (mặc định `max_tusimple_clips=None` = toàn bộ 3.626 clip,
         kèm `apply_preprocessing=True` áp dụng CLAHE lên ảnh huấn luyện).
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

    # Thu thập danh sách ảnh có nhãn (không quét mù 134.571 ảnh không nhãn)
    sample_entries = collect_labeled_road_images(
        raw_base,
        include_test=include_test_in_metrics,
        max_samples=max_highlight_samples,
    )

    metrics_rows: List[Dict[str, Any]] = []
    first_result: Optional[Dict[str, Any]] = None
    first_bgr: Optional[np.ndarray] = None

    fitter = SplineFitter(spline_order=3, smoothing=2.5, num_points=80)
    for idx, (img_path, subset_tag) in enumerate(
        tqdm(sample_entries, desc="Tiền xử lý & Phân tích làn UK", unit="img")
    ):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        res = preprocess_single_road_image(bgr, spline_fitter=fitter)
        if first_result is None:
            first_result = res
            first_bgr = bgr

        out_name = f"highlighted_{idx:04d}_{img_path.name}"
        save_highlight = (
            max_saved_highlight_files is None or idx < max_saved_highlight_files
        )
        if save_highlight:
            cv2.imwrite(str(highlight_dir / out_name), res["highlighted"])
            out_rel_file = f"data/processing/highlighted/{out_name}"
        else:
            out_rel_file = ""

        try:
            rel_source = str(img_path.relative_to(raw_base)).replace("\\", "/")
        except ValueError:
            rel_source = img_path.name

        metrics_rows.append(
            {
                "sample_id": idx,
                "subset": subset_tag,
                "image_name": rel_source,
                "output_file": out_rel_file,
                "num_lanes_detected": res["num_lanes"],
                "uk_lane_type": res["uk_lane_type"],
                "edge_pixels_in_roi": res["edge_pixels"],
                "color_pixels_in_roi": res["color_pixels"],
                "heading_error_rad": round(float(res["heading_error_rad"]), 5),
                "curvature": round(float(res["curvature"]), 5),
                "processing_time_ms": res["elapsed_ms"],
            }
        )

    # Xuất bộ dữ liệu train/valid chuẩn YOLO-seg (kèm tiền xử lý CLAHE) vào data/processing/
    yolo_stats = {
        "train": {"imgs": 0, "polys": 0, "skipped": 0},
        "valid": {"imgs": 0, "polys": 0, "skipped": 0},
        "apply_preprocessing": apply_preprocessing,
    }
    tusimple_dir = raw_base / "TUSimple"
    if tusimple_dir.exists():
        yolo_stats = convert_tusimple_to_yolo(
            tusimple_root=tusimple_dir,
            output_dir=proc_base,
            val_ratio=val_ratio,
            lane_thickness=12,
            max_clips=max_tusimple_clips,
            apply_preprocessing=apply_preprocessing,
            include_test_set=include_test_set,
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
        _save_pipeline_chart_agg(first_bgr, first_result, chart_root_path)

    return {
        "processed_samples": len(metrics_rows),
        "yolo_dataset_stats": yolo_stats,
        "apply_preprocessing": apply_preprocessing,
        "csv_path": str(csv_root_path),
        "chart_path": str(chart_root_path),
    }


if __name__ == "__main__":
    summary = run_batch_preprocessing()
    print("Batch preprocessing summary:", summary)
