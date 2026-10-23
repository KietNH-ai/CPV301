"""
tusimple_to_yolo.py — Chuyển dataset TuSimple sang định dạng YOLOv8/11 segmentation.

Đầu vào (TuSimple train set):
    <tusimple_root>/
        clips/0313-1/20/20.jpg   (và các clip 0531*, 0601*)
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json

Đầu ra (khớp config/data.yaml):
    <output_dir>/
        train/images/  train/labels/
        valid/images/  valid/labels/

Mỗi làn đường (lane) được vẽ thành dải line dày trên mask nhị phân, sau đó lấy
contour làm polygon segmentation. Chia train/val theo CLIP (không theo ảnh) để
tránh leak: cùng một clip không nằm đồng thời ở hai tập.

Usage:
    python scripts/tusimple_to_yolo.py \
        --tusimple_root data/external/tusimple \
        --output_dir data/processed \
        --val_ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Đảm bảo import được src khi chạy script trực tiếp
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LABEL_FILES = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]


def lane_to_polygons(
    lanes: List[List[float]],
    h_samples: List[float],
    img_h: int,
    img_w: int,
    thickness: int = 12,
    epsilon_factor: float = 0.002,
) -> List[np.ndarray]:
    """
    Chuyển list các làn (mỗi làn là list tọa độ x ứng với từng y trong h_samples,
    x = -2 nghĩa là thiếu) thành list polygon (N,2) dạng int, tọa độ pixel.
    """
    polygons: List[np.ndarray] = []
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for lane_xs in lanes:
        pts = [
            (int(x), int(y))
            for x, y in zip(lane_xs, h_samples)
            if x is not None and x >= 0
        ]
        if len(pts) < 2:
            continue
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        # Vẽ làn thành dải dày để mask có diện tích (segmentation polygon cần diện tích)
        cv2.polylines(mask, [pts_arr], isClosed=False, color=255, thickness=thickness)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask.fill(0)  # reset cho làn tiếp theo
        for cnt in contours:
            if len(cnt) < 3:
                continue
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                polygons.append(approx.reshape(-1, 2).astype(np.int32))
    return polygons


def polygon_to_yolo_line(polygon: np.ndarray, img_w: int, img_h: int, class_id: int = 0) -> str:
    """Chuẩn hóa polygon về [0,1] và xuất dòng label YOLO-seg."""
    coords = []
    for x, y in polygon:
        coords.append(f"{min(max(x / img_w, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(y / img_h, 0.0), 1.0):.6f}")
    return f"{class_id} " + " ".join(coords)


def clip_id_of(raw_file: str) -> str:
    """raw_file = 'clips/0313-1/20/20.jpg' -> 'clips/0313-1/20'"""
    return str(Path(raw_file).parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tusimple_root", required=True, type=Path,
                    help="Thư mục gốc TuSimple chứa clips/ và label_data_*.json")
    ap.add_argument("--output_dir", type=Path, default=_ROOT / "data" / "processed",
                    help="Thư mục đầu ra (mặc định: data/processed)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Tỷ lệ validation (theo clip)")
    ap.add_argument("--lane_thickness", type=int, default=12,
                    help="Độ dày (px) khi vẽ làn thành mask")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root: Path = args.tusimple_root.resolve()
    out: Path = args.output_dir.resolve()

    if not (root / "test_set" /"clips").is_dir():
        sys.exit(f"❌ Không tìm thấy {root / "test_set" /"clips"}. Kiểm tra lại --tusimple_root.")

    # 1. Đọc tất cả label, gom theo clip
    samples_by_clip: Dict[str, List[Tuple[Path, dict]]] = {}
    for lf in LABEL_FILES:
        lf_path = root / lf
        if not lf_path.exists():
            print(f"⚠️  Bỏ qua (không tồn tại): {lf_path}")
            continue
        with open(lf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = root / rec["raw_file"]
                cid = clip_id_of(rec["raw_file"])
                samples_by_clip.setdefault(cid, []).append((img_path, rec))

    clip_ids = sorted(samples_by_clip.keys())
    random.shuffle(clip_ids)
    n_val = max(1, int(len(clip_ids) * args.val_ratio))
    val_clips = set(clip_ids[:n_val])
    print(f"📦 Tổng clip: {len(clip_ids)} | val: {len(val_clips)} | train: {len(clip_ids) - len(val_clips)}")

    # 2. Tạo thư mục đầu ra
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"train": {"imgs": 0, "polys": 0, "skipped": 0},
             "valid": {"imgs": 0, "polys": 0, "skipped": 0}}

    for cid in tqdm(clip_ids, desc="Chuyển đổi", unit="clip"):
        split = "valid" if cid in val_clips else "train"
        for img_path, rec in samples_by_clip[cid]:
            if not img_path.exists():
                stats[split]["skipped"] += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                stats[split]["skipped"] += 1
                continue
            img_h, img_w = img.shape[:2]
            polygons = lane_to_polygons(
                rec["lanes"], rec["h_samples"], img_h, img_w,
                thickness=args.lane_thickness,
            )
            if not polygons:
                stats[split]["skipped"] += 1
                continue

            name = img_path.name  # 20.jpg — đổi tên tránh đụng độ giữa các clip
            stem = cid.replace("/", "_").replace("\\", "_")  # clips_0313-1_20
            out_name = f"{stem}_{img_path.stem}"

            shutil.copy2(img_path, out / split / "images" / f"{out_name}.jpg")
            label_lines = [polygon_to_yolo_line(p, img_w, img_h, class_id=0) for p in polygons]
            with open(out / split / "labels" / f"{out_name}.txt", "w", encoding="utf-8") as lf:
                lf.write("\n".join(label_lines) + "\n")

            stats[split]["imgs"] += 1
            stats[split]["polys"] += len(polygons)

    print("\n✅ Hoàn tất. Thống kê:")
    for split, s in stats.items():
        print(f"   {split:5s}: {s['imgs']:4d} ảnh | {s['polys']:5d} polygon | bỏ qua {s['skipped']}")
    print(f"\n📂 Đầu ra: {out}")
    print("👉 Tiếp theo: chỉnh config/data.yaml (path=../data/processed) rồi chạy train.")


if __name__ == "__main__":
    main()
