"""
src/evaluation/train_and_export.py — Huấn luyện YOLO11-seg từ `data/processing/`,
xuất model + ảnh dự đoán vào `data/output/`, và xuất bảng CSV + Biểu đồ ra `ROOT_DIR`.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.path import ROOT_DIR, CONFIG_DIR, PROCESSING_DATA_DIR, OUTPUT_DATA_DIR
from src.preprocessing.preprocessor import ImagePreprocessor
from src.detection.hough_transform import HoughLaneDetector
from src.detection.lanenet import LaneNetDetector
from src.fitting.spline_fitter import SplineFitter
from src.control.pure_pursuit import extract_path_from_lanes
from src.evaluation.compare_methods import benchmark_detector_on_frames


def _load_processing_val_pairs(max_samples: int = 10):
    """Đọc các cặp (frame, gt_lanes) từ `data/processing/valid/{images,labels}`."""
    img_dir = PROCESSING_DATA_DIR / "valid" / "images"
    lbl_dir = PROCESSING_DATA_DIR / "valid" / "labels"
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg"))[:max_samples]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_lanes = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                vals = [float(x) for x in line.split()]
                if len(vals) < 7:
                    continue
                pts = vals[1:]
                poly = np.array(
                    [[int(pts[i] * w), int(pts[i + 1] * h)] for i in range(0, len(pts) - 1, 2)],
                    dtype=np.int32,
                )
                if len(poly) >= 2:
                    gt_lanes.append(poly)
        if gt_lanes:
            pairs.append((frame, gt_lanes))
    return pairs


def run_training_and_export(
    epochs: int = 15,
    patience: int = 5,
    imgsz: int = 320,
    batch: int = 8,
    processing_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    max_eval_images: int = 12,
    cache: Union[bool, str] = True,
    workers: int = 2,
) -> Dict[str, Any]:
    """
    Thực hiện luồng của `notebooks/02_train.ipynb`:
      1. Kiểm tra dữ liệu trong `data/processing/` (nếu chưa có thì tự động gọi `run_batch_preprocessing()`).
      2. Huấn luyện `yolo11n-seg.pt` trên dữ liệu `data/processing/` (qua `config/data.yaml`) kèm
         Early Stopping (patience) và lưu kết quả vào `data/output/yolo_train/tusimple_lane/`.
      3. Chạy dự đoán trên tập `data/processing/valid/images`, kết hợp B-Spline & phân loại làn UK,
         lưu ảnh kết quả vào `data/output/predictions/`.
      4. Xuất bảng CSV (`training_results.csv`, `evaluation_metrics.csv`) và biểu đồ
         (`training_and_evaluation_chart.png`) ở ngay thư mục gốc `ROOT_DIR` và `data/output/`.
    """
    import torch
    from ultralytics import YOLO

    proc_base = Path(processing_dir) if processing_dir is not None else PROCESSING_DATA_DIR
    out_base = Path(output_dir) if output_dir is not None else OUTPUT_DATA_DIR
    root_base = Path(root_dir) if root_dir is not None else ROOT_DIR

    train_img_dir = proc_base / "train" / "images"
    if not train_img_dir.exists() or not list(train_img_dir.glob("*.jpg")):
        from src.preprocessing.batch_processor import run_batch_preprocessing
        run_batch_preprocessing(max_tusimple_clips=12, max_highlight_samples=10)

    out_base.mkdir(parents=True, exist_ok=True)
    pred_dir = out_base / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    train_device = 0 if torch.cuda.is_available() else "cpu"
    weights_src = root_base / "yolo11n-seg.pt"
    if not weights_src.exists():
        weights_src = root_base / "checkpoint_for_llm" / "yolo" / "yolo11n-seg.pt"

    # Đảm bảo config/data.yaml trỏ tuyệt đối tới thư mục data/processing
    data_yaml_path = CONFIG_DIR / "data.yaml"
    data_yaml_content = (
        "# YOLOv11 Lane Detection Dataset Configuration\n\n"
        "# Set the base path to your processing data folder (populated by 01_pre_processing.ipynb / tusimple_to_yolo.py)\n"
        f"path: {proc_base.resolve().as_posix()}\n\n"
        "# Define where the training and validation images are (relative to 'path')\n"
        "train: train/images\n"
        "val: valid/images\n\n"
        "# Classes (Since we are only detecting lanes, there is 1 class)\n"
        "nc: 1\n"
        "names:\n"
        "  0: lane\n"
    )
    data_yaml_path.write_text(data_yaml_content, encoding="utf-8")

    model = YOLO(str(weights_src) if weights_src.exists() else "yolo11n-seg.pt")
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        patience=patience,  # Early stopping sau 'patience' epochs nếu val metric không cải thiện
        imgsz=imgsz,
        batch=batch,
        device=train_device,
        project=str(out_base / "yolo_train"),
        name="tusimple_lane",
        exist_ok=True,
        verbose=True,
        plots=True,
        cache=cache,
        workers=workers,
    )

    run_dir = out_base / "yolo_train" / "tusimple_lane"
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, out_base / "best.pt")

    # 1. Sao chép results.csv của quá trình train ra ROOT_DIR / training_results.csv
    ultralytics_csv = run_dir / "results.csv"
    root_train_csv = root_base / "training_results.csv"
    if ultralytics_csv.exists():
        shutil.copy2(ultralytics_csv, root_train_csv)

    # Xóa bản sao weights trung gian trong yolo_train/ để tiết kiệm dung lượng (đã có data/output/best.pt)
    if (run_dir / "weights").exists():
        shutil.rmtree(run_dir / "weights", ignore_errors=True)

    # 2. Chạy inference + B-Spline + UK Lane Classification trên ảnh valid trong data/processing/valid/images
    trained_model = YOLO(str(out_base / "best.pt") if (out_base / "best.pt").exists() else str(weights_src))
    spline_fitter = SplineFitter(spline_order=3, smoothing=2.5, num_points=80)
    pre = ImagePreprocessor(use_clahe=True, blur_ksize=5)
    lanenet_fallback = LaneNetDetector(preprocessor=pre, min_samples=12)

    val_imgs = sorted((proc_base / "valid" / "images").glob("*.jpg"))[:max_eval_images]
    for idx, img_path in enumerate(val_imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        preds = trained_model.predict(frame, conf=0.25, verbose=False)
        masks = []
        if preds and preds[0].masks is not None:
            for m in preds[0].masks.xy:
                if len(m) >= 2:
                    masks.append(np.array(m, dtype=np.int32))
        if not masks:
            _, masks = lanenet_fallback.detect(frame)

        curves = spline_fitter.fit_multiple_lanes(masks)
        annotated = spline_fitter.fit_and_draw(frame, masks)
        if len(curves) >= 2:
            sorted_c = sorted(curves, key=lambda c: float(c[0, 0]))
            annotated = spline_fitter.draw_lane_area(annotated, sorted_c[0], sorted_c[1], alpha=0.22)

        path_data = extract_path_from_lanes(curves, frame.shape, driving_side="uk_left")
        if path_data.valid and path_data.centerline:
            cl_pts = np.array(path_data.centerline, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [cl_pts], False, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            annotated,
            f"UK Lane: {path_data.uk_lane_type} | Heading: {np.degrees(path_data.heading_error_rad or 0.0):+.1f} deg",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 160),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(pred_dir / f"pred_{idx:02d}_{img_path.name}"), annotated)
        if idx == 0:
            cv2.imwrite(str(OUTPUT_DATA_DIR / "val_prediction_sample.jpg"), annotated)

    # 3. Đánh giá so sánh các phương pháp và xuất evaluation_metrics.csv ở ROOT_DIR & data/output/
    from model.demo import run_demo_comparison
    demo_report = run_demo_comparison()

    eval_rows = []
    for method_name, metrics in demo_report.items():
        eval_rows.append(
            {
                "method": method_name,
                "dataset_source": "data/processing",
                "uk_traffic_mode": "left_hand_traffic",
                "accuracy": metrics["accuracy"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "fps": metrics["fps"],
            }
        )

    root_eval_csv = root_base / "evaluation_metrics.csv"
    with open(root_eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)

    # 4. Vẽ biểu đồ tổng hợp Huấn luyện & Đánh giá xuất ra ROOT_DIR
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel trái: So sánh Accuracy & FPS giữa các phương pháp
    methods = [r["method"] for r in eval_rows]
    accs = [r["accuracy"] * 100.0 for r in eval_rows]
    fps_vals = [r["fps"] for r in eval_rows]
    x_pos = np.arange(len(methods))
    bars = axes[0].bar(x_pos, accs, width=0.48, color=["#4c78a8", "#f58518", "#54a24b"])
    for bar, acc, fps_v in zip(bars, accs, fps_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            f"{acc:.1f}%\n({fps_v:.1f} FPS)",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim(0, 118)
    axes[0].set_ylabel("TuSimple Accuracy (%)")
    axes[0].set_title("1. Method Accuracy & FPS Comparison (data/output)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel phải: Biểu đồ Loss quá trình huấn luyện từ training_results.csv
    if root_train_csv.exists():
        with open(root_train_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        cleaned_rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
        ep_list = [int(float(r.get("epoch", idx + 1))) for idx, r in enumerate(cleaned_rows)]
        box_loss = [float(r.get("train/box_loss", 0.0)) for r in cleaned_rows]
        seg_loss = [float(r.get("train/seg_loss", 0.0)) for r in cleaned_rows]
        axes[1].plot(ep_list, box_loss, marker="o", linewidth=2.2, label="train/box_loss", color="#e45756")
        axes[1].plot(ep_list, seg_loss, marker="s", linewidth=2.2, label="train/seg_loss", color="#72b7b2")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("2. YOLO11-seg Training Loss on data/processing")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

    plt.suptitle(
        "CPV301 — Training on data/processing & Evaluation in data/output",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    chart_root_path = root_base / "training_and_evaluation_chart.png"
    fig.savefig(str(chart_root_path), dpi=140)
    plt.close(fig)

    return {
        "best_weights": str(OUTPUT_DATA_DIR / "best.pt"),
        "predictions_dir": str(pred_dir),
        "training_csv": str(root_train_csv),
        "evaluation_csv": str(root_eval_csv),
        "chart_path": str(chart_root_path),
    }


if __name__ == "__main__":
    out = run_training_and_export(epochs=15, patience=5, imgsz=320, batch=8)
    print("Training & Export complete:", out)
