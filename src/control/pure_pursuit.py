"""
src/control/pure_pursuit.py — Thuật toán Pure Pursuit & cầu nối Thị giác -> Quỹ đạo lái (Vision-to-Path)
tích hợp Phân loại làn đường chuẩn Vương quốc Anh (UK Left-Hand Traffic) cho Forza Horizon 4.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PathData:
    """Interface for output from the vision/path detection module."""

    timestamp: float = 0.0
    valid: bool = False
    confidence: float = 0.0

    # Negative = road direction to the left.
    # Positive = road direction to the right.
    heading_error_rad: Optional[float] = None

    curvature: Optional[float] = None
    centerline: Optional[list] = None
    target_speed_kmh: Optional[float] = None
    lookahead_point: Optional[Tuple[int, int]] = None
    lateral_offset_px: Optional[float] = None
    driving_side: str = "uk_left"
    uk_lane_type: str = "unknown"
    classified_lanes: Dict[str, np.ndarray] = field(default_factory=dict)


class PurePursuitController:
    """
    Pure Pursuit lateral steering controller combined with EMA smoothing and deadband.
    Compatible with `SteeringController` interface (`calculate(heading_error_rad)` and `.reset()`).
    """

    def __init__(
        self,
        wheelbase_m: float = 2.65,
        lookahead_base_m: float = 7.5,
        lookahead_speed_gain: float = 0.16,
        min_lookahead_m: float = 5.0,
        max_lookahead_m: float = 22.0,
        max_steer_angle_deg: float = 32.0,
        max_steering: float = 120.0,
        smoothing: float = 0.35,
        deadband_deg: float = 1.2,
    ):
        self.wheelbase_m = wheelbase_m
        self.lookahead_base_m = lookahead_base_m
        self.lookahead_speed_gain = lookahead_speed_gain
        self.min_lookahead_m = min_lookahead_m
        self.max_lookahead_m = max_lookahead_m
        self.max_steer_angle_deg = max_steer_angle_deg
        self.max_steering = max_steering
        self.smoothing = smoothing
        self.deadband_deg = deadband_deg

        self._smoothed_steer_deg: Optional[float] = None

    def reset(self) -> None:
        self._smoothed_steer_deg = None

    def compute_lookahead_distance(self, speed_kmh: float = 60.0) -> float:
        """Compute velocity-dependent lookahead distance L_d (in meters)."""
        speed_mps = max(0.0, float(speed_kmh)) / 3.6
        ld = self.lookahead_base_m + self.lookahead_speed_gain * speed_mps
        return max(self.min_lookahead_m, min(self.max_lookahead_m, ld))

    def calculate(
        self,
        heading_error_rad: float,
        speed_kmh: float = 60.0,
    ) -> float:
        """
        Compute steering command in [-max_steering, +max_steering] using Pure Pursuit law:
            delta = arctan(2 * L * sin(alpha) / L_d)
        """
        ld = self.compute_lookahead_distance(speed_kmh)
        delta_rad = math.atan2(2.0 * self.wheelbase_m * math.sin(heading_error_rad), ld)
        delta_deg = math.degrees(delta_rad)

        if self._smoothed_steer_deg is None:
            self._smoothed_steer_deg = delta_deg
        else:
            self._smoothed_steer_deg = (
                self.smoothing * delta_deg
                + (1.0 - self.smoothing) * self._smoothed_steer_deg
            )

        steer_deg = self._smoothed_steer_deg
        if abs(steer_deg) < self.deadband_deg:
            return 0.0

        normalized = steer_deg / max(1.0, self.max_steer_angle_deg)
        steering_cmd = normalized * self.max_steering
        return max(-self.max_steering, min(self.max_steering, steering_cmd))


def _interp_lane_x(curve: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Interpolate x coordinates along a sorted vertical y_grid with smooth tangent extrapolation."""
    pts = np.asarray(curve, dtype=np.float32)
    if pts.ndim != 2 or len(pts) < 2:
        return np.full_like(y_grid, np.nan, dtype=np.float32)
    sort_idx = np.argsort(pts[:, 1])
    ys = pts[sort_idx, 1]
    xs = pts[sort_idx, 0]
    unique_ys, unique_idx = np.unique(ys, return_index=True)
    if len(unique_ys) < 2:
        return np.full_like(y_grid, float(xs[0]), dtype=np.float32)
    unique_xs = xs[unique_idx]
    interp_x = np.interp(y_grid, unique_ys, unique_xs).astype(np.float32)

    # Smooth tangent extrapolation beyond [unique_ys[0], unique_ys[-1]] to avoid vertical kinks
    y_min, y_max = float(unique_ys[0]), float(unique_ys[-1])
    if len(unique_ys) >= 3:
        k_top = min(4, len(unique_ys) - 1)
        dy_top = max(1.0, float(unique_ys[k_top] - unique_ys[0]))
        slope_top = float(unique_xs[k_top] - unique_xs[0]) / dy_top
        slope_top = float(np.clip(slope_top, -2.2, 2.2))
        above_mask = y_grid < y_min
        if np.any(above_mask):
            interp_x[above_mask] = float(unique_xs[0]) + slope_top * (y_grid[above_mask] - y_min)

        k_bot = max(0, len(unique_ys) - 1 - min(4, len(unique_ys) - 1))
        dy_bot = max(1.0, float(unique_ys[-1] - unique_ys[k_bot]))
        slope_bot = float(unique_xs[-1] - unique_xs[k_bot]) / dy_bot
        slope_bot = float(np.clip(slope_bot, -2.2, 2.2))
        below_mask = y_grid > y_max
        if np.any(below_mask):
            interp_x[below_mask] = float(unique_xs[-1]) + slope_bot * (y_grid[below_mask] - y_max)

    return interp_x


def classify_uk_lanes(
    lane_masks: List[np.ndarray],
    frame_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    """
    Phân loại các vạch kẻ đường phát hiện được theo quy tắc giao thông Vương quốc Anh
    (UK Left-Hand Traffic — xe chạy ở làn bên trái):
      - `uk_left_edge`: Vạch giới hạn lề đường bên trái (kerb / grass edge).
      - `uk_center_divider`: Vạch tim đường (nằm bên phải xe của chúng ta, ngăn cách với làn ngược chiều).
      - `uk_oncoming_right_edge`: Vạch lề phải của làn xe ngược chiều (nếu camera nhìn thấy toàn bộ mặt đường).
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mid_x = width / 2.0

    valid_curves = []
    for m in (lane_masks or []):
        if m is None:
            continue
        c = np.asarray(m, dtype=np.float32)
        if c.ndim != 2 or len(c) < 3:
            continue
        y_min = float(np.min(c[:, 1]))
        y_max = float(np.max(c[:, 1]))
        y_span = y_max - y_min
        x_span = float(np.max(c[:, 0]) - np.min(c[:, 0]))

        # Vạch đường thật phải có chiều sâu nhìn về phía trước xe (không phải luống cày / chân trời nằm ngang)
        if y_span < max(30.0, height * 0.15):
            continue
        overall_slope = y_span / max(1.0, x_span)
        if overall_slope < 0.36 and y_span < height * 0.40:
            continue
        if y_max < height * 0.45:
            continue
        valid_curves.append(c)
    if not valid_curves:
        return {
            "left_boundary": None,
            "right_boundary": None,
            "oncoming_edge": None,
            "uk_lane_type": "no_lane_detected",
            "labeled_curves": {},
        }

    # Sắp xếp các đường cong từ trái sang phải dựa theo hoành độ điểm thấp nhất (gần mũi xe nhất)
    curves_with_xbot = []
    for c in valid_curves:
        bot_idx = int(np.argmax(c[:, 1]))
        curves_with_xbot.append((float(c[bot_idx, 0]), c))
    curves_with_xbot.sort(key=lambda item: item[0])

    labeled: Dict[str, np.ndarray] = {}
    if len(curves_with_xbot) >= 3:
        # Có từ 3 vạch trở lên: chọn cặp vạch tạo thành hành lang làn bên trái (UK Left Lane)
        left_b = curves_with_xbot[0][1]
        center_div = curves_with_xbot[1][1]
        oncoming_r = curves_with_xbot[2][1]
        labeled["uk_left_edge"] = left_b
        labeled["uk_center_divider"] = center_div
        labeled["uk_oncoming_right_edge"] = oncoming_r
        return {
            "left_boundary": left_b,
            "right_boundary": center_div,
            "oncoming_edge": oncoming_r,
            "uk_lane_type": "uk_multi_lane_left_corridor",
            "labeled_curves": labeled,
        }

    if len(curves_with_xbot) == 2:
        x_left_bot, left_b = curves_with_xbot[0]
        x_right_bot, right_b = curves_with_xbot[1]
        span_ratio = (x_right_bot - x_left_bot) / max(1.0, float(width))

        # Nếu 2 vạch đều nằm cùng 1 phía (quá gần nhau), gộp thành 1 phía
        if abs(x_right_bot - x_left_bot) < width * 0.12:
            if x_left_bot < mid_x * 0.9:
                labeled["uk_left_edge"] = left_b
                return {
                    "left_boundary": left_b,
                    "right_boundary": None,
                    "oncoming_edge": None,
                    "uk_lane_type": "uk_single_left_edge",
                    "labeled_curves": labeled,
                }
            else:
                labeled["uk_center_divider"] = right_b
                return {
                    "left_boundary": None,
                    "right_boundary": right_b,
                    "oncoming_edge": None,
                    "uk_lane_type": "uk_single_center_divider",
                    "labeled_curves": labeled,
                }

        # Nếu khoảng cách 2 vạch quá rộng (>82% chiều ngang ảnh) và không có vạch tim đường ở giữa,
        # nghĩa là camera đang thấy 2 mép đường lớn 2 chiều -> đánh dấu wide_two_way_uk để bám nửa trái!
        if span_ratio > 0.82 and x_left_bot < width * 0.12 and x_right_bot > width * 0.88:
            labeled["uk_left_edge"] = left_b
            labeled["uk_oncoming_right_edge"] = right_b
            return {
                "left_boundary": left_b,
                "right_boundary": right_b,
                "oncoming_edge": right_b,
                "uk_lane_type": "uk_wide_unmarked_keep_left",
                "labeled_curves": labeled,
            }

        labeled["uk_left_edge"] = left_b
        labeled["uk_center_divider"] = right_b
        return {
            "left_boundary": left_b,
            "right_boundary": right_b,
            "oncoming_edge": None,
            "uk_lane_type": "uk_standard_left_lane",
            "labeled_curves": labeled,
        }

    # Chỉ phát hiện được 1 vạch kẻ đường
    x_bot, single_curve = curves_with_xbot[0]
    if x_bot < mid_x * 0.92:
        labeled["uk_left_edge"] = single_curve
        return {
            "left_boundary": single_curve,
            "right_boundary": None,
            "oncoming_edge": None,
            "uk_lane_type": "uk_single_left_edge",
            "labeled_curves": labeled,
        }
    else:
        labeled["uk_center_divider"] = single_curve
        return {
            "left_boundary": None,
            "right_boundary": single_curve,
            "oncoming_edge": None,
            "uk_lane_type": "uk_single_center_divider",
            "labeled_curves": labeled,
        }


def extract_path_from_lanes(
    lane_masks: List[np.ndarray],
    frame_shape: Tuple[int, ...],
    lookahead_ratio: float = 0.62,
    default_lane_width_ratio: float = 0.55,
    num_samples: int = 25,
    driving_side: str = "uk_left",
    map_turn_hint: Optional[str] = None,
) -> PathData:
    """
    Convert detected lane curves/masks into `PathData` for `FH4AutonomousController` / `PurePursuitController`,
    incorporating UK Left-Hand Traffic lane classification and optional map turn hints.
    """
    now = time.monotonic()
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mid_x = width / 2.0

    uk_info = classify_uk_lanes(lane_masks, frame_shape)
    left_curve = uk_info["left_boundary"]
    right_curve = uk_info["right_boundary"]
    uk_lane_type = uk_info["uk_lane_type"]

    if left_curve is None and right_curve is None:
        return PathData(
            timestamp=now,
            valid=False,
            confidence=0.0,
            driving_side=driving_side,
            uk_lane_type=uk_lane_type,
        )

    active_curves = [c for c in (left_curve, right_curve) if c is not None and len(c) >= 2]
    if not active_curves:
        return PathData(
            timestamp=now,
            valid=False,
            confidence=0.0,
            driving_side=driving_side,
            uk_lane_type="no_lane_detected",
        )
    min_detected_y = min(float(np.min(c[:, 1])) for c in active_curves)
    max_detected_y = max(float(np.max(c[:, 1])) for c in active_curves)
    if max_detected_y - min_detected_y < max(30.0, float(height) * 0.15):
        return PathData(
            timestamp=now,
            valid=False,
            confidence=0.0,
            driving_side=driving_side,
            uk_lane_type="no_lane_detected",
        )

    y_bottom = float(min(height - 8, max(height * 0.72, max_detected_y)))
    y_top = float(max(int(height * 0.40), min(int(height * 0.52), int(min_detected_y))))
    y_grid = np.linspace(y_bottom, y_top, num_samples, dtype=np.float32)

    y_vanish = float(height) * 0.31
    perspective_scale = np.clip((y_grid - y_vanish) / max(1.0, y_bottom - y_vanish), 0.14, 1.0) ** 0.90
    base_half_width = (width * min(default_lane_width_ratio, 0.48)) * 0.5
    half_width_profile = base_half_width * perspective_scale

    if left_curve is not None and right_curve is not None:
        x_left = _interp_lane_x(left_curve, y_grid)
        x_right = _interp_lane_x(right_curve, y_grid)
        if driving_side == "uk_left" and uk_lane_type == "uk_wide_unmarked_keep_left":
            # Đường 2 chiều không có vạch giữa: bám phần làn bên trái (30% tính từ lề trái)
            x_center = x_left + 0.30 * (x_right - x_left)
        else:
            x_center = 0.5 * (x_left + x_right)
        confidence = 0.95
    elif left_curve is not None:
        x_left = _interp_lane_x(left_curve, y_grid)
        x_center = x_left + half_width_profile
        confidence = 0.65
    else:
        x_right = _interp_lane_x(right_curve, y_grid)
        x_center = x_right - half_width_profile
        confidence = 0.65

    # Hỗ trợ dịch điểm nhìn trước khi có chỉ dẫn rẽ tại nút giao từ bản đồ (MapData)
    if map_turn_hint:
        hint = map_turn_hint.upper()
        if "TURN_LEFT" in hint:
            x_center = x_center - (1.0 - perspective_scale) * (width * 0.05)
        elif "TURN_RIGHT" in hint:
            x_center = x_center + (1.0 - perspective_scale) * (width * 0.05)

    # Đảm bảo đường tâm không bao giờ văng ra khỏi hành lang làn trái khi có cả 2 biên
    if left_curve is not None and right_curve is not None:
        margin = 0.08 * np.maximum(10.0, x_right - x_left)
        x_center = np.clip(x_center, x_left + margin, x_right - margin)

    x_center = np.clip(x_center, float(width * 0.05), float(width - 1))
    centerline_pts = [
        (int(round(float(x))), int(round(float(y))))
        for x, y in zip(x_center, y_grid)
    ]

    y_lookahead = float(np.clip(height * lookahead_ratio, y_top, y_bottom - 10.0))
    idx_la = int(np.argmin(np.abs(y_grid - y_lookahead)))
    la_x = float(x_center[idx_la])
    la_y = float(y_grid[idx_la])

    dx = la_x - mid_x
    dy = max(10.0, y_bottom - la_y)
    heading_error_rad = float(math.atan2(dx, dy))
    lateral_offset_px = float(x_center[0] - mid_x)

    y_norm = (y_bottom - y_grid) / max(1.0, y_bottom - y_top)
    x_norm = (x_center - mid_x) / max(1.0, float(width))
    try:
        poly = np.polyfit(y_norm, x_norm, deg=2)
        a, b = float(poly[0]), float(poly[1])
        curvature = abs(2.0 * a) / math.pow(1.0 + b * b, 1.5)
    except Exception:
        curvature = 0.0

    return PathData(
        timestamp=now,
        valid=True,
        confidence=round(confidence, 3),
        heading_error_rad=round(heading_error_rad, 5),
        curvature=round(float(curvature), 5),
        centerline=centerline_pts,
        lookahead_point=(int(round(la_x)), int(round(la_y))),
        lateral_offset_px=round(lateral_offset_px, 2),
        driving_side=driving_side,
        uk_lane_type=uk_lane_type,
        classified_lanes=uk_info["labeled_curves"],
    )
