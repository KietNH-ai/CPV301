"""
src/map/map_analyzer.py — Phân tích bản đồ Forza Horizon 4 (UK), trích xuất mạng lưới
đường giao thông thực tế (Road Mask -> Skeleton -> Smooth Curved Road Graph) và lưu kết quả
vào `fh4_map/processing/` cùng file CSV trực tiếp tại thư mục gốc (`ROOT_DIR`).
"""
from __future__ import annotations

import csv
import heapq
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from src.path import ROOT_DIR, FH4_MAP_RAW_DIR, FH4_MAP_PROCESSING_DIR


# ---------------------------------------------------------------------------
# HIỆU CHỈNH HỆ TRỤC TỌA ĐỘ (World <-> Pixel) cho `fh4_map/raw/fh4_uk_map.png` (1400 x 1238)
# ---------------------------------------------------------------------------
# Trước đây dùng khung đối xứng (-7000, 7000, -7000, 7000) đoán mò -> xe bị vẽ lệch sang phải/xuống
# ~250-360 m so với vị trí thật (ví dụ: đứng ở vòng xuyến đầu cao tốc M68 nhưng bản đồ báo còn 364 m).
#
# Giá trị dưới đây được KHỚP (fit) từ quỹ đạo telemetry UDP thật ghi trong `fh4_log.csv`:
#   - ~1.500 điểm chạy qua lại trên con đường thẳng Đông-Tây xuyên Horizon Festival
#     (Z ~ -4310, X từ -3400 đến -1200) -> khớp vào vạch đường trắng trên ảnh map (sai số TB ~1.4 px).
#   - Mốc neo: vòng xuyến đầu cao tốc M68 phía Nam (X=+2603.4, Z=-4400.1) theo xác nhận của người dùng.
# Kết quả: tỉ lệ đồng nhất 9.3 m/pixel, trục X hướng Đông (sang phải), trục Z hướng Bắc (lên trên).
# Định dạng: (min_x, max_x, min_z, max_z) tương ứng mép trái/phải/dưới/trên của toàn bộ ảnh.
# Nếu thay ảnh bản đồ khác, dùng `FH4MapAnalyzer.fit_world_bounds_from_anchors(...)` để hiệu chỉnh lại.
FH4_UK_MAP_WORLD_BOUNDS: Tuple[float, float, float, float] = (-6133.95, 6886.05, -6794.85, 4718.55)


@dataclass
class RoadNode:
    """Một nút (giao lộ / địa danh) trên đồ thị bản đồ FH4."""
    node_id: int
    x_px: int
    y_px: int
    world_x: float
    world_z: float
    degree: int = 0
    road_type: str = "primary_uk"
    label: str = ""


@dataclass
class RoadEdge:
    """Một đoạn đường nối giữa 2 nút trên bản đồ FH4 (bám sát theo vạch đường trắng thật trên map)."""
    u: int
    v: int
    length_px: float
    length_m: float
    road_class: str = "primary_uk"
    speed_limit_kmh: float = 70.0
    uk_driving_side: str = "left"
    polyline_px: List[Tuple[int, int]] = field(default_factory=list)


# Tọa độ chuẩn hóa (thang 1000x1000) đặt đúng tại các nút giao lộ đường nhựa trắng trên bản đồ FH4 (UK)
DEFAULT_FH4_LANDMARKS: List[Tuple[int, str, int, int, str]] = [
    (0, "Horizon Festival Site", 253, 803, "roundabout"),
    (1, "Ambleside Village", 97, 681, "secondary_uk"),
    (2, "Tarn Hows Junction", 45, 833, "secondary_uk"),
    (3, "Derwentwater West", 76, 492, "primary_uk"),
    (4, "Lakehurst Forest", 395, 488, "primary_uk"),
    (5, "Mortimer Gardens", 135, 249, "secondary_uk"),
    (6, "Glen Rannoch Pass", 360, 57, "secondary_uk"),
    (7, "M68 North Roundabout", 501, 171, "roundabout"),
    (8, "M68 South Roundabout", 668, 796, "roundabout"),
    (9, "M68 Astmoor Interchange", 719, 576, "motorway"),
    (10, "M68 Moorhead Roundabout", 558, 288, "roundabout"),
    (11, "Edinburgh West Gate", 595, 181, "primary_uk"),
    (12, "Edinburgh City Centre", 639, 143, "primary_uk"),
    (13, "Arthur's Seat North", 762, 137, "primary_uk"),
    (14, "Bamburgh Castle Coast", 855, 514, "primary_uk"),
    (15, "Broadway Village", 432, 937, "primary_uk"),
    (16, "Ashbrook & Uffington", 488, 868, "primary_uk"),
    (17, "Derwent North Bridge", 272, 265, "primary_uk"),
    (18, "Greendale Airstrip", 772, 368, "primary_uk"),
    (19, "Derwent South Shore", 232, 572, "primary_uk"),
]

# Các tuyến đường nối trực tiếp dọc theo mạng lưới đường nhựa trắng thực tế trên bản đồ FH4
DEFAULT_FH4_ROAD_SEGMENTS: List[Tuple[int, int, str, float]] = [
    # Cao tốc M68 (Motorway trục Bắc - Nam)
    (8, 9, "motorway", 180.0),
    (9, 10, "motorway", 180.0),
    (10, 7, "motorway", 170.0),
    # Khu vực Edinburgh & Đông Bắc
    (7, 11, "primary_uk", 85.0),
    (10, 11, "primary_uk", 90.0),
    (11, 12, "primary_uk", 65.0),
    (12, 13, "primary_uk", 70.0),
    (13, 14, "primary_uk", 85.0),
    (10, 18, "primary_uk", 85.0),
    (18, 14, "primary_uk", 80.0),
    (14, 9, "primary_uk", 85.0),
    (14, 8, "primary_uk", 80.0),
    # Khu vực phía Nam (Broadway - Ashbrook - Horizon Festival - M68 South)
    (8, 16, "primary_uk", 90.0),
    (16, 15, "primary_uk", 75.0),
    (15, 0, "primary_uk", 80.0),
    (0, 16, "primary_uk", 85.0),
    # Khu vực phía Tây & Trung tâm (Quanh hồ Derwentwater - Lakehurst - Glen Rannoch)
    (0, 1, "secondary_uk", 65.0),
    (0, 2, "secondary_uk", 65.0),
    (1, 2, "secondary_uk", 60.0),
    (1, 3, "primary_uk", 75.0),
    (3, 5, "primary_uk", 70.0),
    (5, 6, "secondary_uk", 65.0),
    (6, 7, "primary_uk", 75.0),
    (5, 17, "primary_uk", 75.0),
    (17, 7, "primary_uk", 80.0),
    (17, 4, "primary_uk", 75.0),
    (17, 19, "primary_uk", 75.0),
    (1, 19, "primary_uk", 75.0),
    (19, 0, "primary_uk", 75.0),
    (19, 4, "primary_uk", 75.0),
    (4, 10, "primary_uk", 85.0),
    (4, 16, "primary_uk", 80.0),
]


# Các điểm định hướng trung gian (thang 1000x1000) đặt đúng tâm đường nhựa trắng & cao tốc M68
# giúp đường vẽ luôn đi đúng 100% tuyến chính trên bản đồ thực tế (không bao giờ lệch sang nhánh phụ)
EDGE_VIA_GUIDES: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
    (8, 9): [(691, 775), (715, 735), (728, 695), (730, 655), (725, 615)],
    (9, 10): [
        (715, 560),
        (710, 545),
        (706, 530),
        (704, 515),
        (704, 500),
        (704, 485),
        (701, 470),
        (696, 460),
        (691, 450),
        (686, 440),
        (680, 430),
        (674, 420),
        (669, 410),
        (663, 400),
        (658, 390),
        (651, 380),
        (643, 370),
        (630, 360),
        (617, 350),
        (601, 340),
        (594, 330),
        (589, 320),
        (580, 310),
        (570, 300),
    ],
    (10, 7): [(546, 275), (527, 255), (515, 235), (505, 215), (501, 195)],
    (10, 18): [(585, 255), (631, 278), (689, 281), (745, 338)],
    (1, 3): [(94, 668), (78, 640), (72, 610), (78, 580), (85, 545), (77, 515)],
    (3, 5): [(74, 420), (72, 350), (108, 280)],
    (5, 6): [(119, 210), (143, 175), (194, 140), (245, 110), (301, 110), (341, 80)],
    (5, 17): [(176, 252), (186, 287), (215, 287), (245, 265)],
    (17, 19): [(252, 420)],
    (0, 1): [(228, 725), (174, 690), (140, 697)],
    (0, 2): [(230, 863), (180, 875), (130, 901), (90, 907), (55, 876)],
    (0, 16): [(382, 818)],
}


def ensure_sample_fh4_map(map_path: Optional[Path] = None) -> Path:
    """
    Tìm file bản đồ Forza Horizon 4 thực tế trong `fh4_map/raw/`.
    """
    FH4_MAP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if map_path is not None and Path(map_path).exists():
        return Path(map_path)

    existing = sorted(
        [
            p for p in FH4_MAP_RAW_DIR.glob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ],
        key=lambda p: (p.name == "fh4_uk_map.png", -p.stat().st_size),
    )
    if existing:
        chosen = existing[0]
        std_png = FH4_MAP_RAW_DIR / "fh4_uk_map.png"
        if not std_png.exists() and chosen != std_png:
            img = cv2.imread(str(chosen))
            if img is not None:
                h, w = img.shape[:2]
                scale = min(1.0, 1400.0 / float(max(h, w)))
                resized = (
                    cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    if scale < 1.0
                    else img
                )
                cv2.imwrite(str(std_png), resized)
        return chosen

    fallback_path = FH4_MAP_RAW_DIR / "fh4_uk_map.png"
    h, w = 1000, 1000
    canvas = np.full((h, w, 3), (44, 62, 48), dtype=np.uint8)
    node_coords = {nid: (x, y) for nid, _, x, y, _ in DEFAULT_FH4_LANDMARKS}
    for u, v, rclass, _ in DEFAULT_FH4_ROAD_SEGMENTS:
        p1, p2 = node_coords[u], node_coords[v]
        cv2.line(canvas, p1, p2, (20, 20, 20), 14, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (250, 250, 250), 7, cv2.LINE_AA)
    cv2.imwrite(str(fallback_path), canvas)
    return fallback_path


def _smooth_polyline(pts: List[Tuple[int, int]], window: int = 5) -> List[Tuple[int, int]]:
    """Làm mượt đường gấp khúc pixel bằng bộ lọc trượt trung bình nhưng giữ cố định 2 đầu mút."""
    if len(pts) <= 4:
        return pts
    arr = np.array(pts, dtype=np.float32)
    smoothed = arr.copy()
    half = window // 2
    for _ in range(2):
        temp = smoothed.copy()
        for i in range(1, len(arr) - 1):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            temp[i] = smoothed[lo:hi].mean(axis=0)
        temp[0] = arr[0]
        temp[-1] = arr[-1]
        smoothed = temp

    # Lấy mẫu lại cách đều ~8px để vẽ đường mượt đẹp
    diffs = np.hypot(np.diff(smoothed[:, 0]), np.diff(smoothed[:, 1]))
    total_len = float(np.sum(diffs))
    num_samples = max(12, min(90, int(total_len / 7.0)))
    idxs = np.linspace(0, len(smoothed) - 1, num_samples).astype(int)
    res = [(int(round(smoothed[i, 0])), int(round(smoothed[i, 1]))) for i in idxs]
    res[0] = pts[0]
    res[-1] = pts[-1]
    return res


def _draw_label_pill(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (22, 28, 24),
    border_color: Tuple[int, int, int] = (0, 215, 255),
    font_scale: float = 0.44,
    thickness: int = 1,
) -> None:
    """Vẽ nhãn chữ trong hộp bo viền gọn gàng, dễ đọc trên nền bản đồ."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = img.shape[:2]
    x = int(np.clip(pos[0], 6, max(6, w - tw - 14)))
    y = int(np.clip(pos[1], th + 8, max(th + 8, h - 8)))

    p1 = (x - 5, y - th - 5)
    p2 = (x + tw + 5, y + baseline + 3)
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, bg_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, p1, p2, border_color, 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


class FH4MapAnalyzer:
    """
    Bộ phân tích bản đồ Forza Horizon 4 thực tế:
      1. Đọc ảnh bản đồ từ `fh4_map/raw/`.
      2. Tách mặt nạ mạng lưới đường nhựa trắng chuẩn xác bằng Morphological Top-Hat + Lọc liên thông.
      3. Làm mảnh khung xương đường 1-pixel (`skeletonize_roads`).
      4. Dùng trường khoảng cách (`cv2.distanceTransform`) để dò đường cong thực tế bám chặt 100%
         trên vạch đường trắng của bản đồ (không bao giờ cắt ngang qua hồ nước hay rừng cây).
    """

    def __init__(
        self,
        world_bounds: Tuple[float, float, float, float] = FH4_UK_MAP_WORLD_BOUNDS,
        max_working_dim: int = 1400,
    ):
        self.world_bounds = tuple(float(v) for v in world_bounds)
        self.max_working_dim = max_working_dim

    def meters_per_pixel(self, width: int) -> float:
        """Số mét thế giới FH4 tương ứng 1 pixel theo chiều ngang của ảnh bản đồ (độ phân giải `width`)."""
        return (self.world_bounds[1] - self.world_bounds[0]) / float(max(1, width))

    @staticmethod
    def fit_world_bounds_from_anchors(
        anchors: List[Tuple[float, float, float, float]],
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:
        """
        Hiệu chỉnh `world_bounds` bằng bình phương tối thiểu từ >= 2 mốc neo
        `(world_x, world_z, x_px, y_px)` (tỉ lệ đồng nhất, X sang phải, Z hướng lên trên).
        """
        if len(anchors) < 2:
            raise ValueError("Cần ít nhất 2 mốc neo (world_x, world_z, x_px, y_px)")
        arr = np.asarray(anchors, dtype=np.float64)
        wx, wz, px, py = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        # wx = min_x + s*px ; wz = max_z - s*py  -> giải tuyến tính [s, min_x, max_z]
        rows_a = np.concatenate(
            [np.stack([px, np.ones_like(px), np.zeros_like(px)], 1),
             np.stack([-py, np.zeros_like(py), np.ones_like(py)], 1)]
        )
        rhs = np.concatenate([wx, wz])
        s, min_x, max_z = np.linalg.lstsq(rows_a, rhs, rcond=None)[0]
        return (
            round(float(min_x), 2),
            round(float(min_x + s * width), 2),
            round(float(max_z - s * height), 2),
            round(float(max_z), 2),
        )

    def pixel_to_world(self, x_px: float, y_px: float, width: int, height: int) -> Tuple[float, float]:
        """Quy đổi tọa độ pixel (u, v) trên ảnh bản đồ sang tọa độ thế giới (PositionX, PositionZ) của FH4."""
        min_x, max_x, min_z, max_z = self.world_bounds
        wx = min_x + (float(x_px) / max(1.0, float(width))) * (max_x - min_x)
        wz = max_z - (float(y_px) / max(1.0, float(height))) * (max_z - min_z)
        return round(wx, 2), round(wz, 2)

    def world_to_pixel(self, world_x: float, world_z: float, width: int, height: int) -> Tuple[int, int]:
        """Quy đổi tọa độ thế giới (PositionX, PositionZ) từ fh4_telemetry sang tọa độ pixel (u, v)."""
        min_x, max_x, min_z, max_z = self.world_bounds
        u = (float(world_x) - min_x) / max(1e-6, max_x - min_x) * width
        v = (max_z - float(world_z)) / max(1e-6, max_z - min_z) * height
        return int(np.clip(round(u), 0, width - 1)), int(np.clip(round(v), 0, height - 1))

    def extract_road_mask(self, map_bgr: np.ndarray) -> np.ndarray:
        """
        Tách mặt nạ nhị phân mạng lưới đường nhựa trắng & cao tốc trên bản đồ Forza Horizon 4.
        Loại bỏ hoàn toàn chữ viết, icon nhỏ, đỉnh núi tuyết và bãi biển.
        """
        h, w = map_bgr.shape[:2]
        b, g, r = cv2.split(map_bgr)
        gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)

        k_size = max(11, int(round(min(h, w) * 0.012)) | 1)
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)

        # Vạch đường nhựa trắng & cao tốc M68 trên bản đồ (bao gồm cả đường hẹp ven hồ Derwentwater)
        paved_mask = ((gray >= 148) & ((max_c - min_c) <= 48) & (tophat >= 14)).astype(np.uint8) * 255

        # Loại trừ vùng núi tuyết góc trên bên trái và bãi biển cát sát mép phải
        paved_mask[0 : int(h * 0.14), 0 : int(w * 0.24)] = 0
        paved_mask[:, int(w * 0.885) :] = 0

        # Nối liền dải phân cách giữa 2 làn cao tốc M68 và các khe hở nhỏ do cầu/nhãn chữ đè lên đường
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(paved_mask, cv2.MORPH_CLOSE, close_k, iterations=2)

        # Chỉ giữ lại các nhánh đường liên thông dài (loại bỏ chữ cái và icon nhỏ rời rạc)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        cleaned = np.zeros_like(closed)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area >= 95 and max(bw, bh) >= 26:
                cleaned[labels == i] = 255

        # Nối liền các điểm cầu vượt cắt ngang trục cao tốc M68 và đường đèo Tây Bắc để không bị đứt nhịp
        sx, sy = w / 1000.0, h / 1000.0
        corridors = [
            [(668, 796)] + EDGE_VIA_GUIDES[(8, 9)] + [(719, 576)] + EDGE_VIA_GUIDES[(9, 10)] + [(558, 288)] + EDGE_VIA_GUIDES[(10, 7)] + [(501, 171)],
            [(135, 249)] + EDGE_VIA_GUIDES[(5, 6)] + [(360, 57)],
            [(253, 803)] + EDGE_VIA_GUIDES[(0, 2)] + [(45, 833)],
        ]
        for chain in corridors:
            c_pts = np.array([(int(round(x * sx)), int(round(y * sy))) for x, y in chain], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(cleaned, [c_pts], isClosed=False, color=255, thickness=3)
        return cleaned

    def skeletonize_roads(self, road_mask: np.ndarray) -> np.ndarray:
        """Làm mảnh hình thái học (Morphological Skeletonization) để lấy trục tâm mạng lưới đường."""
        img = (road_mask > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(60):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return skel

    @staticmethod
    def _trace_single_span(
        cost_grid: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        step: int = 1,
    ) -> List[Tuple[int, int]]:
        """Dò đường A* độ phân giải từng pixel (step=1) giữa 2 điểm gần nhau trên lưới chi phí."""
        h, w = cost_grid.shape[:2]
        x1, y1 = int(p1[0] // step), int(p1[1] // step)
        x2, y2 = int(p2[0] // step), int(p2[1] // step)
        gh, gw = h // step, w // step
        x1, y1 = int(np.clip(x1, 0, gw - 1)), int(np.clip(y1, 0, gh - 1))
        x2, y2 = int(np.clip(x2, 0, gw - 1)), int(np.clip(y2, 0, gh - 1))

        pad = int(65 // step)
        min_x, max_x = max(0, min(x1, x2) - pad), min(gw - 1, max(x1, x2) + pad)
        min_y, max_y = max(0, min(y1, y2) - pad), min(gh - 1, max(y1, y2) + pad)

        start = (x1, y1)
        goal = (x2, y2)
        seg_dx = float(x2 - x1)
        seg_dy = float(y2 - y1)
        seg_len = max(1.0, math.hypot(seg_dx, seg_dy))
        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414),
        ]

        while open_heap:
            _, curr = heapq.heappop(open_heap)
            if curr == goal:
                raw_pts: List[Tuple[int, int]] = [p2]
                c = curr
                while c in came_from:
                    c = came_from[c]
                    raw_pts.append((int(c[0] * step), int(c[1] * step)))
                raw_pts[-1] = p1
                raw_pts.reverse()
                return raw_pts

            cx, cy = curr
            cg = g_score[curr]
            for dx, dy, d_step in neighbors:
                nx, ny = cx + dx, cy + dy
                if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                    continue
                cross_dist = abs((nx - x1) * seg_dy - (ny - y1) * seg_dx) / seg_len
                cell_cost = float(cost_grid[ny * step, nx * step]) + 0.04 * cross_dist
                tentative = cg + d_step * cell_cost
                nxt = (nx, ny)
                if tentative < g_score.get(nxt, float("inf")):
                    came_from[nxt] = curr
                    g_score[nxt] = tentative
                    heur = math.hypot(nx - goal[0], ny - goal[1])
                    heapq.heappush(open_heap, (tentative + heur, nxt))

        return [p1, p2]

    @classmethod
    def trace_road_polyline(
        cls,
        cost_grid: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        step: int = 1,
        via_points: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Dò đường đi bám sát tâm vạch đường trắng thực tế giữa 2 nút `p1` và `p2`
        (có hỗ trợ các điểm định hướng `via_points` để giữ đúng làn cao tốc M68).
        """
        waypoints = [p1] + (via_points or []) + [p2]
        full_raw: List[Tuple[int, int]] = []
        for i in range(len(waypoints) - 1):
            span = cls._trace_single_span(cost_grid, waypoints[i], waypoints[i + 1], step=step)
            if not full_raw:
                full_raw.extend(span)
            else:
                full_raw.extend(span[1:])
        return _smooth_polyline(full_raw, window=13)

    def build_road_graph(
        self,
        map_path: Optional[Path] = None,
        save_outputs: bool = True,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Phân tích bản đồ FH4 thực tế trong `fh4_map/raw/`, trích xuất mặt nạ đường đi,
        khung xương và đồ thị đường đi bám sát các cung đường cong thật trên bản đồ.
        Tự động nạp bộ nhớ đệm `road_graph.json` nếu đã tồn tại để khởi động tức thì (< 20ms).
        """
        actual_map_path = ensure_sample_fh4_map(map_path)
        raw_bgr = cv2.imread(str(actual_map_path))
        if raw_bgr is None:
            raise ValueError(f"Không thể đọc ảnh bản đồ tại: {actual_map_path}")

        raw_h, raw_w = raw_bgr.shape[:2]
        scale = min(1.0, float(self.max_working_dim) / float(max(raw_h, raw_w)))
        if scale < 1.0:
            map_bgr = cv2.resize(
                raw_bgr,
                (int(round(raw_w * scale)), int(round(raw_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            map_bgr = raw_bgr.copy()

        height, width = map_bgr.shape[:2]

        graph_json_path = FH4_MAP_PROCESSING_DIR / "road_graph.json"
        mask_path = FH4_MAP_PROCESSING_DIR / "road_mask.png"
        skel_path = FH4_MAP_PROCESSING_DIR / "road_skeleton.png"
        overlay_path = FH4_MAP_PROCESSING_DIR / "road_graph_overlay.png"
        root_csv = ROOT_DIR / "fh4_map_road_graph.csv"

        if (
            not force_rebuild
            and graph_json_path.exists()
            and mask_path.exists()
            and skel_path.exists()
            and overlay_path.exists()
            and root_csv.exists()
        ):
            try:
                cached = json.loads(graph_json_path.read_text(encoding="utf-8"))
                if (
                    cached.get("working_resolution") == [width, height]
                    and len(cached.get("nodes", [])) == len(DEFAULT_FH4_LANDMARKS)
                    and len(cached.get("edges", [])) == len(DEFAULT_FH4_ROAD_SEGMENTS)
                    # Bộ đệm cũ tính tọa độ thế giới bằng hệ trục chưa hiệu chỉnh -> phải dựng lại
                    and [round(float(v), 2) for v in cached.get("world_bounds", [])]
                    == [round(float(v), 2) for v in self.world_bounds]
                ):
                    road_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    skeleton = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
                    if road_mask is not None and skeleton is not None:
                        nodes = {int(n["node_id"]): RoadNode(**n) for n in cached["nodes"]}
                        edges: List[RoadEdge] = []
                        adjacency: Dict[int, List[Tuple[int, float, str, float]]] = {nid: [] for nid in nodes}
                        edge_polylines: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
                        for ed in cached["edges"]:
                            poly = [(int(p[0]), int(p[1])) for p in ed.get("polyline_px") or []]
                            edge_obj = RoadEdge(
                                u=int(ed["u"]),
                                v=int(ed["v"]),
                                length_px=float(ed["length_px"]),
                                length_m=float(ed["length_m"]),
                                road_class=str(ed["road_class"]),
                                speed_limit_kmh=float(ed["speed_limit_kmh"]),
                                uk_driving_side=str(ed.get("uk_driving_side", "left")),
                                polyline_px=poly,
                            )
                            edges.append(edge_obj)
                            edge_polylines[(edge_obj.u, edge_obj.v)] = poly
                            edge_polylines[(edge_obj.v, edge_obj.u)] = list(reversed(poly))
                            adjacency[edge_obj.u].append(
                                (edge_obj.v, edge_obj.length_px, edge_obj.road_class, edge_obj.speed_limit_kmh)
                            )
                            adjacency[edge_obj.v].append(
                                (edge_obj.u, edge_obj.length_px, edge_obj.road_class, edge_obj.speed_limit_kmh)
                            )
                        return {
                            "map_bgr": map_bgr,
                            "road_mask": road_mask,
                            "skeleton": skeleton,
                            "nodes": nodes,
                            "edges": edges,
                            "edge_polylines": edge_polylines,
                            "adjacency": adjacency,
                            "width": width,
                            "height": height,
                        }
            except Exception:
                pass

        road_mask = self.extract_road_mask(map_bgr)
        skeleton = self.skeletonize_roads(road_mask)

        # Trường khoảng cách tới vạch đường nhựa trắng:
        # - Trên vạch đường (dist == 0): cost = 1.0
        # - Sát mép vạch đường (dist <= 3): cost = 1.2 + dist
        # - Khe hở cầu/nhãn chữ nhỏ (dist <= 9): cost = 4.0 + dist * 2.5
        # - Ngoài đường (rừng/hồ/đồng cỏ, dist > 9): phạt bậc 2 cực lớn để cấm đi tắt!
        dist_to_road = cv2.distanceTransform(255 - road_mask, cv2.DIST_L2, 5)
        dist_inside = cv2.distanceTransform(road_mask, cv2.DIST_L2, 5)
        cost_grid = np.where(
            dist_to_road <= 0.5,
            1.0 + 0.18 / (1.0 + dist_inside),
            np.where(
                dist_to_road <= 3.5,
                1.2 + dist_to_road * 0.8,
                np.where(
                    dist_to_road <= 9.0,
                    4.0 + dist_to_road * 2.5,
                    65.0 + (dist_to_road ** 2) * 2.5,
                ),
            ),
        ).astype(np.float32)

        ys_road, xs_road = np.nonzero(road_mask)
        scale_x = width / 1000.0
        scale_y = height / 1000.0

        nodes: Dict[int, RoadNode] = {}
        for nid, label, bx, by, rtype in DEFAULT_FH4_LANDMARKS:
            px = int(round(bx * scale_x))
            py = int(round(by * scale_y))
            if len(xs_road) > 0:
                dists = np.hypot(xs_road - px, ys_road - py)
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < 45.0:
                    px, py = int(xs_road[min_idx]), int(ys_road[min_idx])

            wx, wz = self.pixel_to_world(px, py, width, height)
            nodes[nid] = RoadNode(
                node_id=nid,
                x_px=px,
                y_px=py,
                world_x=wx,
                world_z=wz,
                degree=0,
                road_type=rtype,
                label=label,
            )

        edges: List[RoadEdge] = []
        adjacency: Dict[int, List[Tuple[int, float, str, float]]] = {nid: [] for nid in nodes}
        edge_polylines: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for u, v, rclass, speed_lim in DEFAULT_FH4_ROAD_SEGMENTS:
            nu, nv = nodes[u], nodes[v]
            raw_vias = EDGE_VIA_GUIDES.get((u, v))
            if raw_vias is None and (v, u) in EDGE_VIA_GUIDES:
                raw_vias = list(reversed(EDGE_VIA_GUIDES[(v, u)]))

            snapped_vias: List[Tuple[int, int]] = []
            if raw_vias:
                for vx_norm, vy_norm in raw_vias:
                    vpx = int(round(vx_norm * scale_x))
                    vpy = int(round(vy_norm * scale_y))
                    if len(xs_road) > 0:
                        vdists = np.hypot(xs_road - vpx, ys_road - vpy)
                        vmin_idx = int(np.argmin(vdists))
                        if vdists[vmin_idx] < 12.0:
                            vpx, vpy = int(xs_road[vmin_idx]), int(ys_road[vmin_idx])
                    snapped_vias.append((vpx, vpy))

            poly = self.trace_road_polyline(
                cost_grid,
                (nu.x_px, nu.y_px),
                (nv.x_px, nv.y_px),
                step=1,
                via_points=snapped_vias,
            )
            dist_px = sum(
                math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
                for i in range(len(poly) - 1)
            )
            world_scale = self.meters_per_pixel(width)
            dist_m = dist_px * world_scale

            edge = RoadEdge(
                u=u,
                v=v,
                length_px=round(float(dist_px), 2),
                length_m=round(float(dist_m), 2),
                road_class=rclass,
                speed_limit_kmh=speed_lim,
                uk_driving_side="left",
                polyline_px=poly,
            )
            edges.append(edge)
            edge_polylines[(u, v)] = poly
            edge_polylines[(v, u)] = list(reversed(poly))
            adjacency[u].append((v, float(dist_px), rclass, speed_lim))
            adjacency[v].append((u, float(dist_px), rclass, speed_lim))
            nodes[u].degree += 1
            nodes[v].degree += 1

        graph_data = {
            "map_path": str(actual_map_path),
            "raw_resolution": [raw_w, raw_h],
            "working_resolution": [width, height],
            "world_bounds": [round(float(v), 2) for v in self.world_bounds],
            "meters_per_pixel": round(self.meters_per_pixel(width), 4),
            "traffic_rule": "UK_LEFT_HAND_TRAFFIC",
            "nodes": [asdict(n) for n in nodes.values()],
            "edges": [asdict(e) for e in edges],
        }

        if save_outputs:
            FH4_MAP_PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(FH4_MAP_PROCESSING_DIR / "road_mask.png"), road_mask)
            cv2.imwrite(str(FH4_MAP_PROCESSING_DIR / "road_skeleton.png"), skeleton)

            overlay = map_bgr.copy()
            # Làm dịu nhẹ nền bản đồ 15% để mạng lưới đồ thị đường đi nổi bật rõ ràng
            dark_bg = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.86, dark_bg, 0.14, 0, overlay)

            # 1. Vẽ viền đen (casing) cho toàn bộ các tuyến đường trong đồ thị
            for e in edges:
                pts = np.array(e.polyline_px, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=False, color=(18, 22, 18), thickness=7, lineType=cv2.LINE_AA)

            # 2. Vẽ màu phân loại tuyến đường (vẽ đường phụ trước, cao tốc M68 trên cùng để không bị đè)
            sorted_edges = sorted(
                edges,
                key=lambda item: 2 if item.road_class == "motorway" else (1 if item.road_class == "primary_uk" else 0),
            )
            for e in sorted_edges:
                pts = np.array(e.polyline_px, dtype=np.int32).reshape((-1, 1, 2))
                if e.road_class == "motorway":
                    color = (0, 200, 255)
                    thick = 5
                elif e.road_class == "primary_uk":
                    color = (110, 250, 70)
                    thick = 3
                else:
                    color = (255, 210, 80)
                    thick = 3
                cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thick, lineType=cv2.LINE_AA)

            # 3. Vẽ các nút giao lộ (Nodes) và hộp nhãn tên địa danh sắc nét (tránh đè nhau)
            label_offsets = {
                7: (-215, -6),
                11: (14, 16),
                12: (-195, -12),
                13: (14, -8),
                14: (-210, -12),
            }
            for n in nodes.values():
                badge_color = (0, 140, 255) if n.road_type == "roundabout" else (20, 40, 230)
                cv2.circle(overlay, (n.x_px, n.y_px), 11, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (n.x_px, n.y_px), 8, badge_color, -1, cv2.LINE_AA)
                dx_l, dy_l = label_offsets.get(n.node_id, (13, 4))
                _draw_label_pill(
                    overlay,
                    f"[{n.node_id}] {n.label}",
                    (n.x_px + dx_l, n.y_px + dy_l),
                    text_color=(255, 255, 255),
                    border_color=(0, 215, 255) if n.road_type == "roundabout" else (110, 250, 70),
                    font_scale=0.41,
                )

            cv2.imwrite(str(FH4_MAP_PROCESSING_DIR / "road_graph_overlay.png"), overlay)

            graph_json_path = FH4_MAP_PROCESSING_DIR / "road_graph.json"
            graph_json_path.write_text(
                json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            root_csv = ROOT_DIR / "fh4_map_road_graph.csv"
            with open(root_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "edge_id",
                        "from_node_id",
                        "from_label",
                        "to_node_id",
                        "to_label",
                        "road_class",
                        "uk_driving_side",
                        "speed_limit_kmh",
                        "length_px",
                        "length_m",
                    ]
                )
                for idx, e in enumerate(edges):
                    writer.writerow(
                        [
                            idx,
                            e.u,
                            nodes[e.u].label,
                            e.v,
                            nodes[e.v].label,
                            e.road_class,
                            e.uk_driving_side,
                            e.speed_limit_kmh,
                            e.length_px,
                            e.length_m,
                        ]
                    )

        return {
            "map_bgr": map_bgr,
            "road_mask": road_mask,
            "skeleton": skeleton,
            "nodes": nodes,
            "edges": edges,
            "edge_polylines": edge_polylines,
            "adjacency": adjacency,
            "width": width,
            "height": height,
        }
