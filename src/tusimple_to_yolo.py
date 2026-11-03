"""
tusimple_to_yolo.py — Chuyển dataset TuSimple sang định dạng YOLOv8/11 segmentation
kèm tiền xử lý tăng cường tương phản CLAHE và công cụ kiểm chứng nhãn (Label Verification).

Đầu vào (hỗ trợ tự động nhận diện cấu trúc thư mục TuSimple):
  Cách 1 (trỏ vào thư mục gốc `data/raw/TUSimple`):
    <tusimple_root>/
        train_set/
            clips/0313-1/20/20.jpg   (và các clip 0531*, 0601*)
            label_data_0313.json     (2.858 dòng)
            label_data_0531.json     (358 dòng)
            label_data_0601.json     (410 dòng) -> Tổng train_set: 3.626 ảnh có nhãn
        test_set/
            clips/...
        test_label.json              (2.782 dòng) -> Tổng toàn bộ: 6.408 ảnh có nhãn

  Cách 2 (trỏ trực tiếp vào `train_set/` hoặc `test_set/`):
    <tusimple_root>/
        clips/0313-1/20/20.jpg
        label_data_0313.json ...

Lưu ý quan trọng:
  - Mỗi thư mục clip của TuSimple có 20 khung hình liên tiếp (`1.jpg` .. `20.jpg`), nhưng chỉ
    khung hình cuối cùng (`20.jpg`) được gán nhãn Ground-Truth trong các file `.json`.
  - Module này chỉ quét theo danh sách `raw_file` trong các file JSON (tối đa 6.408 ảnh có nhãn),
    TUYỆT ĐỐI KHÔNG quét mù 134.571 file `.jpg` không có nhãn trong `clips/`.

Đầu ra (khớp `config/data.yaml`):
    <output_dir>/
        train/images/  train/labels/
        valid/images/  valid/labels/

Usage:
    python -m src.tusimple_to_yolo \
        --tusimple_root data/raw/TUSimple \
        --output_dir data/processing \
        --val_ratio 0.2 \
        --apply_preprocessing
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Đảm bảo import được src khi chạy script trực tiếp
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocessing.enhance import apply_clahe

TRAIN_LABEL_FILES = [
    "label_data_0313.json",
    "label_data_0531.json",
    "label_data_0601.json",
]
TEST_LABEL_FILES = ["test_label.json"]
LABEL_FILES = TRAIN_LABEL_FILES


def extract_gt_points_from_record(rec: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
    """
    Trích xuất danh sách các làn đường từ 1 bản ghi JSON của TuSimple.
    Mỗi làn là danh sách các điểm pixel (x, y) hợp lệ (bỏ qua các mốc x == -2).
    """
    lanes = rec.get("lanes", [])
    h_samples = rec.get("h_samples", [])
    extracted_lanes: List[List[Tuple[int, int]]] = []
    for lane_xs in lanes:
        pts = [
            (int(x), int(y))
            for x, y in zip(lane_xs, h_samples)
            if x is not None and x >= 0
        ]
        if len(pts) >= 2:
            extracted_lanes.append(pts)
    return extracted_lanes


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
    x = -2 nghĩa là thiếu) thành list polygon (N, 2) dạng int, tọa độ pixel.
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
    """Chuẩn hóa polygon về [0, 1] và xuất dòng label YOLO-seg."""
    coords = []
    for x, y in polygon:
        coords.append(f"{min(max(x / img_w, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(y / img_h, 0.0), 1.0):.6f}")
    return f"{class_id} " + " ".join(coords)


def parse_yolo_seg_lines(
    lines: List[str],
    img_w: int,
    img_h: int,
) -> List[np.ndarray]:
    """
    Giải mã các dòng trong file nhãn YOLO11-seg `.txt` (`<class_id> <x1> <y1> ... <xn> <yn>`)
    về danh sách các đa giác tọa độ pixel `np.ndarray` kích thước (N, 2).
    """
    polygons: List[np.ndarray] = []
    for raw_line in lines:
        parts = raw_line.strip().split()
        if len(parts) < 7:  # Tối thiểu 1 class_id + 3 đỉnh (6 số thực)
            continue
        coords = [float(v) for v in parts[1:]]
        pts = [
            [int(round(coords[i] * img_w)), int(round(coords[i + 1] * img_h))]
            for i in range(0, len(coords) - 1, 2)
        ]
        if len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.int32))
    return polygons


def parse_yolo_seg_label_file(label_path: Path, img_w: int, img_h: int) -> List[np.ndarray]:
    """Đọc file `.txt` chuẩn YOLO11-seg và chuyển về danh sách polygon pixel (N, 2)."""
    if not Path(label_path).exists():
        return []
    text = Path(label_path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return parse_yolo_seg_lines(text.splitlines(), img_w=img_w, img_h=img_h)


def render_label_verification_pair(
    image_bgr: np.ndarray,
    tusimple_record: Dict[str, Any],
    yolo_lines: Optional[List[str]] = None,
    lane_thickness: int = 12,
    apply_clahe_on_yolo_view: bool = True,
) -> Dict[str, Any]:
    """
    Tạo cặp ảnh trực quan hóa song song để kiểm chứng nhãn (Bước 1 trong kế hoạch):
      1. `json_gt_vis`: Ảnh gốc BGR + các điểm rời rạc (x, y) đọc từ `lanes` & `h_samples` của JSON TuSimple.
      2. `yolo_poly_vis`: Ảnh đã qua tiền xử lý CLAHE + đa giác (polygon) đọc từ dòng lệnh YOLO11-seg `.txt`.
    """
    img_h, img_w = image_bgr.shape[:2]
    lane_colors = [
        (0, 255, 0),    # Xanh lá
        (0, 200, 255),  # Vàng cam
        (255, 180, 0),  # Xanh dương sáng
        (255, 0, 255),  # Tím hồng
        (0, 255, 255),  # Vàng
    ]

    # 1. Vẽ các điểm (x, y) từ JSON gốc của TuSimple (h_samples + lanes)
    json_gt_vis = image_bgr.copy()
    gt_point_lanes = extract_gt_points_from_record(tusimple_record)
    total_gt_points = 0
    for lane_idx, pts in enumerate(gt_point_lanes):
        color = lane_colors[lane_idx % len(lane_colors)]
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(json_gt_vis, [pts_arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        for x_px, y_px in pts:
            cv2.circle(json_gt_vis, (x_px, y_px), radius=4, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(json_gt_vis, (x_px, y_px), radius=5, color=color, thickness=1, lineType=cv2.LINE_AA)
            total_gt_points += 1

    # 2. Tạo hoặc giải mã các dòng YOLO11-seg (.txt)
    if yolo_lines is None:
        raw_polys = lane_to_polygons(
            tusimple_record.get("lanes", []),
            tusimple_record.get("h_samples", []),
            img_h=img_h,
            img_w=img_w,
            thickness=lane_thickness,
        )
        yolo_lines = [polygon_to_yolo_line(p, img_w=img_w, img_h=img_h, class_id=0) for p in raw_polys]

    decoded_polys = parse_yolo_seg_lines(yolo_lines, img_w=img_w, img_h=img_h)
    base_yolo_img = (
        apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
        if apply_clahe_on_yolo_view
        else image_bgr.copy()
    )
    yolo_poly_vis = base_yolo_img.copy()
    overlay = base_yolo_img.copy()

    for poly_idx, poly in enumerate(decoded_polys):
        color = lane_colors[poly_idx % len(lane_colors)]
        poly_reshaped = poly.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [poly_reshaped], color)
        cv2.polylines(yolo_poly_vis, [poly_reshaped], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        for vx, vy in poly:
            cv2.circle(yolo_poly_vis, (int(vx), int(vy)), radius=3, color=(255, 255, 255), thickness=-1)

    yolo_poly_vis = cv2.addWeighted(overlay, 0.38, yolo_poly_vis, 0.62, 0)

    return {
        "json_gt_vis": json_gt_vis,
        "yolo_poly_vis": yolo_poly_vis,
        "gt_point_lanes": gt_point_lanes,
        "total_gt_points": total_gt_points,
        "yolo_lines": yolo_lines,
        "decoded_polygons": decoded_polys,
    }


def clip_id_of(raw_file: str) -> str:
    """raw_file = 'clips/0313-1/20/20.jpg' -> 'clips/0313-1/20'"""
    return str(Path(raw_file).parent)


def find_tusimple_label_sources(
    root: Path,
    include_test: bool = False,
) -> List[Tuple[Path, Path, str]]:
    """
    Tìm danh sách `(data_root, label_json_path, subset_tag)` cho tập `train_set`
    và tùy chọn `test_set` mà không làm dừng chương trình nếu không tìm thấy.
    """
    root = Path(root).resolve()
    sources: List[Tuple[Path, Path, str]] = []

    # 1. Tìm tập train_set
    train_candidates = [root, root / "train_set"]
    for cand in train_candidates:
        if (cand / "clips").is_dir():
            for lf in TRAIN_LABEL_FILES:
                lf_path = cand / lf
                if lf_path.exists():
                    sources.append((cand, lf_path, "train_set"))
            if sources:
                break

    # 2. Tìm tập test_set (nếu được yêu cầu hoặc nếu root trỏ thẳng vào test_set)
    if include_test or not sources:
        test_candidates = [root / "test_set", root]
        for cand in test_candidates:
            if (cand / "clips").is_dir():
                for tf in TEST_LABEL_FILES:
                    tf_path = cand / tf
                    if not tf_path.exists() and (root / tf).exists():
                        tf_path = root / tf
                    if tf_path.exists():
                        pair = (cand, tf_path, "test_set")
                        if pair not in sources:
                            sources.append(pair)
                if any(s[2] == "test_set" for s in sources):
                    break

    return sources


def iter_labeled_tusimple_samples(
    tusimple_root: Path,
    include_test: bool = True,
    max_samples: Optional[int] = None,
) -> List[Tuple[Path, Dict[str, Any], str]]:
    """
    Đọc danh sách các bức ảnh CÓ NHÃN Ground-Truth (`20.jpg` của mỗi clip) trực tiếp
    từ các file JSON (`label_data_0313.json`, `label_data_0531.json`, `label_data_0601.json`,
    và `test_label.json` khi `include_test=True`).

    Tuyệt đối KHÔNG quét mù 134.571 file `.jpg` không có nhãn trong `clips/`.
    """
    sources = find_tusimple_label_samples_sources(tusimple_root, include_test=include_test)
    labeled_samples: List[Tuple[Path, Dict[str, Any], str]] = []

    for data_root, lf_path, subset_tag in sources:
        with open(lf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = data_root / rec["raw_file"]
                if img_path.exists():
                    labeled_samples.append((img_path, rec, subset_tag))
                    if max_samples is not None and max_samples > 0 and len(labeled_samples) >= max_samples:
                        return labeled_samples

    return labeled_samples


def find_tusimple_label_samples_sources(
    root: Path,
    include_test: bool = True,
) -> List[Tuple[Path, Path, str]]:
    """Alias nội bộ trả về các nguồn JSON của TuSimple."""
    return find_tusimple_label_sources(root, include_test=include_test)


def resolve_tusimple_data_dir(root: Path) -> Tuple[Path, List[Path]]:
    """
    Tự động tìm thư mục chứa `clips/` và danh sách các file JSON label
    tương ứng dù người dùng truyền `TUSimple`, `TUSimple/train_set`, hay `TUSimple/test_set`.
    """
    sources = find_tusimple_label_sources(root, include_test=False)
    if sources:
        data_root = sources[0][0]
        label_paths = [s[1] for s in sources if s[0] == data_root]
        return data_root, label_paths

    clips_check = root / "train_set" / "clips"
    sys.exit(
        f"❌ Không tìm thấy thư mục clips và label JSON trong '{root}' "
        f"(hoặc '{clips_check}'). Kiểm tra lại --tusimple_root."
    )


def convert_tusimple_to_yolo(
    tusimple_root: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    lane_thickness: int = 12,
    seed: int = 42,
    max_clips: Optional[int] = None,
    apply_preprocessing: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    include_test_set: bool = False,
) -> dict:
    """
    Chuyển đổi bộ dữ liệu TuSimple sang chuẩn YOLO11-seg (`train/` và `valid/`):
      - `max_clips=None`: Xử lý TOÀN BỘ 3.626 clip có nhãn của `train_set`
        (và thêm 2.782 clip của `test_set` nếu `include_test_set=True`).
      - `apply_preprocessing=True`: Áp dụng cân bằng sáng cục bộ CLAHE lên ảnh trước khi
        lưu vào `data/processing/{train,valid}/images/` thay vì chỉ copy ảnh thô.
    """
    random.seed(seed)
    root = Path(tusimple_root).resolve()
    out = Path(output_dir).resolve()

    sources = find_tusimple_label_sources(root, include_test=include_test_set)
    if not sources:
        data_root, label_paths = resolve_tusimple_data_dir(root)
        sources = [(data_root, lp, "train_set") for lp in label_paths]

    print(
        f"📂 Đọc dữ liệu TuSimple từ {len(sources)} file JSON nhãn "
        f"(apply_preprocessing={apply_preprocessing}, max_clips={max_clips or 'ALL'})"
    )

    # 1. Đọc tất cả label, gom theo clip
    samples_by_clip: Dict[str, List[Tuple[Path, dict]]] = {}
    for data_root, lf_path, subset_tag in sources:
        with open(lf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = data_root / rec["raw_file"]
                cid = f"{subset_tag}/{clip_id_of(rec['raw_file'])}" if include_test_set else clip_id_of(rec["raw_file"])
                samples_by_clip.setdefault(cid, []).append((img_path, rec))

    clip_ids = sorted(samples_by_clip.keys())
    random.shuffle(clip_ids)
    if max_clips is not None and max_clips > 0:
        clip_ids = clip_ids[:max_clips]

    n_val = max(1, int(len(clip_ids) * val_ratio)) if len(clip_ids) > 1 else 0
    val_clips = set(clip_ids[:n_val])
    print(f"📦 Tổng clip xử lý: {len(clip_ids)} | val: {len(val_clips)} | train: {len(clip_ids) - len(val_clips)}")

    # 2. Tạo thư mục đầu ra
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "train": {"imgs": 0, "polys": 0, "skipped": 0},
        "valid": {"imgs": 0, "polys": 0, "skipped": 0},
        "apply_preprocessing": apply_preprocessing,
    }

    for cid in tqdm(clip_ids, desc="Chuyển đổi TuSimple -> YOLO11-seg", unit="clip"):
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
                rec["lanes"],
                rec["h_samples"],
                img_h,
                img_w,
                thickness=lane_thickness,
            )
            if not polygons:
                stats[split]["skipped"] += 1
                continue

            stem = cid.replace("/", "_").replace("\\", "_")
            out_name = f"{stem}_{img_path.stem}"
            dst_img_path = out / split / "images" / f"{out_name}.jpg"

            if apply_preprocessing:
                processed_img = apply_clahe(
                    img,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
                cv2.imwrite(str(dst_img_path), processed_img)
            else:
                shutil.copy2(img_path, dst_img_path)

            label_lines = [polygon_to_yolo_line(p, img_w, img_h, class_id=0) for p in polygons]
            with open(out / split / "labels" / f"{out_name}.txt", "w", encoding="utf-8") as lf:
                lf.write("\n".join(label_lines) + "\n")

            stats[split]["imgs"] += 1
            stats[split]["polys"] += len(polygons)

    print("\n✅ Hoàn tất. Thống kê:")
    for split in ("train", "valid"):
        s = stats[split]
        print(f"   {split:5s}: {s['imgs']:4d} ảnh | {s['polys']:5d} polygon | bỏ qua {s['skipped']}")
    print(f"\n📂 Đầu ra: {out}")
    print("👉 Tiếp theo: kiểm tra config/data.yaml (path: ../data/processing) rồi chạy train.")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tusimple_root",
        required=True,
        type=Path,
        help="Thư mục gốc TuSimple (vd: data/raw/TUSimple hoặc data/raw/TUSimple/train_set)",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=_ROOT / "data" / "processing",
        help="Thư mục đầu ra (mặc định: data/processing)",
    )
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Tỷ lệ validation (theo clip, mặc định: 0.2)",
    )
    ap.add_argument(
        "--lane_thickness",
        type=int,
        default=12,
        help="Độ dày (px) khi vẽ làn thành mask",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Giới hạn số lượng clip chuyển đổi (mặc định: None = xử lý toàn bộ 3.626 clip)",
    )
    ap.add_argument(
        "--apply_preprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Áp dụng tiền xử lý CLAHE lên ảnh khi xuất sang data/processing (mặc định: True)",
    )
    ap.add_argument(
        "--include_test_set",
        action="store_true",
        default=False,
        help="Bao gồm cả tập test_set (2.782 ảnh có nhãn) khi chuyển đổi",
    )
    args = ap.parse_args(argv)
    convert_tusimple_to_yolo(
        tusimple_root=args.tusimple_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        lane_thickness=args.lane_thickness,
        seed=args.seed,
        max_clips=args.max_clips,
        apply_preprocessing=args.apply_preprocessing,
        include_test_set=args.include_test_set,
    )


if __name__ == "__main__":
    main()
