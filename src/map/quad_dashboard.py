"""
src/map/quad_dashboard.py — Giao diện chia màn hình máy tính làm 4 phần (4-Quadrant Dashboard)
tốc độ cao (>= 30 FPS) cho hệ thống tự hành Forza Horizon 4 (UK Left-Hand Traffic):

  - Ô 1 (Góc Trên-Trái  - Top-Left)    : Cửa sổ Game (Raw Game Window / Clean Feed)
  - Ô 2 (Góc Trên-Phải  - Top-Right)   :
      + Trước khi xác nhận: Ô Chọn Điểm Đến (Hỗ trợ Cuộn bản đồ, Thanh cuộn, Kéo Pan, Lăn chuột Zoom, Click chọn Đích + Nút [XÁC NHẬN])
      + Sau khi xác nhận  : Tự động chuyển thành Ô Quan Sát Lane Detect (Hiển thị B-Spline, Hành lang làn trái UK, Lookahead, ROI & Binary Edge Mask PiP; có nút [Đổi Điểm Đến (M)] để mở lại bản đồ bất kỳ lúc nào)
  - Ô 3 (Góc Dưới-Trái  - Bottom-Left) : Quan sát Telemetry & Bộ điều khiển (Speedometer, RPM, Gear, Steer, Throttle/Brake, 3D Pos, FPS)
  - Ô 4 (Góc Dưới-Phải  - Bottom-Right): Quan sát Bản đồ & Lộ trình A* Động (Radar bám theo xe thời gian thực + Overview Minimap)
"""
from __future__ import annotations

from collections import deque
import ctypes
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.capture.screen_capture import get_screen_quadrant_geometry
from src.map.navigator import FH4MapNavigator


def _try_exclude_window_from_capture(window_title: str) -> bool:
    """
    Trên Windows 10/11 (2004+), bật cờ WDA_EXCLUDEFROMCAPTURE (0x11) cho cửa sổ OpenCV
    để `mss` chụp xuyên qua cửa sổ Dashboard xuống thẳng cửa sổ Game Forza Horizon 4
    mà không bao giờ bị hiện tượng lặp hình (hall-of-mirrors).
    """
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            res = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            return bool(res)
    except Exception:
        pass
    return False


def _try_snap_fh4_game_window(x: int, y: int, w: int, h: int) -> bool:
    """Nếu tìm thấy cửa sổ game Forza Horizon 4 trên Windows, tự động đưa về Ô 1 (Top-Left)."""
    try:
        user32 = ctypes.windll.user32
        for title in ("Forza Horizon 4",):
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                SWP_NOZORDER = 0x0004
                SWP_NOACTIVATE = 0x0010
                user32.SetWindowPos(hwnd, 0, int(x), int(y), int(w), int(h), SWP_NOZORDER | SWP_NOACTIVATE)
                return True
    except Exception:
        pass
    return False


def _try_focus_fh4_game_window() -> bool:
    """Đưa tiêu điểm (Focus) về lại cửa sổ Forza Horizon 4 ở Ô 1 sau khi người dùng xác nhận Điểm Đến ở Ô 2."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "Forza Horizon 4")
        if hwnd:
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return True
    except Exception:
        pass
    return False


class FH4QuadDashboard:
    """
    Bộ quản lý hiển thị & tương tác 4 Ô trên màn hình máy tính (2x2 Screen Quadrants),
    được tối ưu hóa hoàn toàn bằng OpenCV C++ / NumPy để chạy mượt >= 30..60+ FPS.

    Đặc biệt: Ô 2 (Góc Trên-Phải) khởi đầu là Ô Chọn Điểm Đến (có cuộn/zoom).
    Ngay sau khi người dùng bấm nút [XÁC NHẬN] (hoặc nhấn phím Enter / Space / Double-Click),
    Ô 2 lập tức chuyển thành Ô Quan Sát Lane Detect (hiển thị chuyên sâu nhận diện làn đường UK,
    B-Spline, hành lang làn trái, điểm nhìn trước và mặt nạ biên Canny PiP).
    Người dùng có thể bấm nút [Đổi Điểm Đến (M)] hoặc phím M bất kỳ lúc nào để chọn lại điểm đến.
    """

    MAIN_WINDOW = "CPV301 - FH4 4-Quadrant Autonomous System (Enter: Xac nhan Dich | M: Doi Dich | q/ESC: Thoat)"
    WIN_Q1 = "[O 1] Cua So Game (Forza Horizon 4)"
    WIN_Q2 = "[O 2] Chon Diem Den -> Quan Sat Lane Detect"
    WIN_Q3 = "[O 3] Quan Sat Telemetry & Dieu Khien"
    WIN_Q4 = "[O 4] Quan Sat Ban Do & Lo Trinh A*"

    def __init__(
        self,
        navigator: FH4MapNavigator,
        monitor: int = 1,
        quad_width: Optional[int] = None,
        quad_height: Optional[int] = None,
        layout: str = "quad",
        dest_confirmed: bool = False,
        on_route_changed: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.navigator = navigator
        self.layout = layout if layout in ("quad", "4windows", "game") else "quad"
        self.dest_confirmed: bool = bool(dest_confirmed)
        self.on_route_changed = on_route_changed

        geom = get_screen_quadrant_geometry(monitor=monitor)
        self.screen_geom = geom
        self.quad_w = int(quad_width or min(960, geom["quad_width"]))
        self.quad_h = int(quad_height or min(510, geom["quad_height"]))

        self.map_w = int(self.navigator.width)
        self.map_h = int(self.navigator.height)

        # ---- Trạng thái Cuộn / Phóng to của Ô 2 (Chọn Điểm Đến) ----
        self.picker_zoom: float = 1.0
        self.picker_min_zoom: float = max(
            float(self.quad_w) / float(max(1, self.map_w)),
            float(self.quad_h) / float(max(1, self.map_h)),
            0.45,
        )
        self.picker_max_zoom: float = 2.6

        dest_id = self.navigator.dest_node_id if self.navigator.dest_node_id is not None else 12
        dest_node = self.navigator.nodes.get(dest_id, self.navigator.nodes[0])
        self.picker_scroll_x: float = max(0.0, float(dest_node.x_px - self.quad_w // 2))
        self.picker_scroll_y: float = max(0.0, float(dest_node.y_px - self.quad_h // 2))
        self._clamp_picker_scroll()

        self._dragging_map: bool = False
        self._dragging_vscroll: bool = False
        self._dragging_hscroll: bool = False
        self._drag_start_pt: Tuple[int, int] = (0, 0)
        self._scroll_start_xy: Tuple[float, float] = (0.0, 0.0)
        self.hover_map_px: Optional[Tuple[int, int]] = None
        self.hover_node_id: Optional[int] = None

        # Các nút bấm trên thanh công cụ Ô 2 khi ở chế độ Chọn Điểm Đến:
        # [XAC NHAN (Enter)], [+], [-], [1:1], [Fit]
        bw, bh = 38, 24
        margin_r = 16
        x_fit_1 = self.quad_w - margin_r - bw
        x_11_1 = x_fit_1 - bw - 4
        x_minus_1 = x_11_1 - bw - 4
        x_plus_1 = x_minus_1 - bw - 4
        confirm_w = 148
        x_conf_1 = x_plus_1 - confirm_w - 8

        self._picker_buttons = [
            ("CONFIRM", x_conf_1, 4, x_conf_1 + confirm_w, 4 + bh),
            ("+", x_plus_1, 4, x_plus_1 + bw, 4 + bh),
            ("-", x_minus_1, 4, x_minus_1 + bw, 4 + bh),
            ("1:1", x_11_1, 4, x_11_1 + bw, 4 + bh),
            ("Fit", x_fit_1, 4, x_fit_1 + bw, 4 + bh),
        ]

        # Nút [Doi Diem Den (M)] ở góc phải trên của Ô 2 khi đã ở chế độ Quan Sát Lane Detect
        reopen_w = 165
        self._reopen_picker_btn = (self.quad_w - reopen_w - 10, 4, self.quad_w - 10, 4 + bh)

        # ---- Trạng thái Radar của Ô 4 (Quan sát Bản đồ động) ----
        self.radar_zoom: float = 1.35
        self._sim_progress_idx: float = 0.0
        self.mini_w = max(130, int(self.quad_w * 0.22))
        self.mini_h = max(115, int(self.mini_w * (self.map_h / float(max(1, self.map_w)))))
        self._cached_mini_map: np.ndarray = cv2.resize(
            self.navigator.route_map_bgr, (self.mini_w, self.mini_h), interpolation=cv2.INTER_AREA
        )

        # ---- Lịch sử FPS & Tốc độ cho đồ thị Sparkline ở Ô 3 ----
        self.fps_history: deque = deque(maxlen=80)
        self.speed_history: deque = deque(maxlen=80)

        self._windows_initialized: bool = False
        self._affinity_applied: bool = False

    def confirm_destination(self) -> None:
        """Xác nhận điểm đến đã chọn -> chuyển Ô 2 sang chế độ Quan Sát Lane Detect và focus lại cửa sổ Game."""
        self.dest_confirmed = True
        _try_focus_fh4_game_window()

    def reopen_destination_picker(self) -> None:
        """Mở lại bản đồ cuộn Chọn Điểm Đến trên Ô 2."""
        self.dest_confirmed = False

    def refresh_route_cache(self) -> None:
        """Cập nhật lại bản đồ thu nhỏ sau khi người dùng chọn điểm đến mới."""
        self._cached_mini_map = cv2.resize(
            self.navigator.route_map_bgr, (self.mini_w, self.mini_h), interpolation=cv2.INTER_AREA
        )

    def _clamp_picker_scroll(self) -> None:
        vis_w = self.quad_w / max(0.1, self.picker_zoom)
        vis_h = self.quad_h / max(0.1, self.picker_zoom)
        max_x = max(0.0, float(self.map_w) - vis_w)
        max_y = max(0.0, float(self.map_h) - vis_h)
        self.picker_scroll_x = float(np.clip(self.picker_scroll_x, 0.0, max_x))
        self.picker_scroll_y = float(np.clip(self.picker_scroll_y, 0.0, max_y))

    def scroll_picker_by(self, dx_map: float, dy_map: float) -> None:
        """Cuộn bản đồ ở Ô 2 theo độ lệch pixel bản đồ (dx_map, dy_map)."""
        self.picker_scroll_x += dx_map
        self.picker_scroll_y += dy_map
        self._clamp_picker_scroll()

    def zoom_picker_at(
        self,
        new_zoom: float,
        anchor_local_x: Optional[int] = None,
        anchor_local_y: Optional[int] = None,
    ) -> None:
        """Phóng to / thu nhỏ bản đồ ở Ô 2 giữ nguyên điểm neo dưới con trỏ chuột."""
        old_z = self.picker_zoom
        new_z = float(np.clip(new_zoom, self.picker_min_zoom, self.picker_max_zoom))
        ax = self.quad_w * 0.5 if anchor_local_x is None else float(anchor_local_x)
        ay = self.quad_h * 0.5 if anchor_local_y is None else float(anchor_local_y)

        map_anchor_x = self.picker_scroll_x + ax / old_z
        map_anchor_y = self.picker_scroll_y + ay / old_z

        self.picker_zoom = new_z
        self.picker_scroll_x = map_anchor_x - ax / new_z
        self.picker_scroll_y = map_anchor_y - ay / new_z
        self._clamp_picker_scroll()

    def _screen_to_map_coords(self, local_x: int, local_y: int) -> Tuple[int, int]:
        """Chuyển tọa độ chuột (local_x, local_y) trong Ô 2 sang tọa độ pixel trên bản đồ gốc."""
        vis_w = min(self.map_w, int(round(self.quad_w / self.picker_zoom)))
        vis_h = min(self.map_h, int(round(self.quad_h / self.picker_zoom)))
        x0 = int(np.clip(round(self.picker_scroll_x), 0, max(0, self.map_w - vis_w)))
        y0 = int(np.clip(round(self.picker_scroll_y), 0, max(0, self.map_h - vis_h)))

        mx = int(np.clip(round(x0 + (local_x / float(max(1, self.quad_w))) * vis_w), 0, self.map_w - 1))
        my = int(np.clip(round(y0 + (local_y / float(max(1, self.quad_h))) * vis_h), 0, self.map_h - 1))
        return mx, my

    def handle_picker_mouse(self, event: int, local_x: int, local_y: int, flags: int) -> None:
        """
        Xử lý thao tác chuột trên Ô 2:
          - Khi đang ở chế độ Quan Sát Lane Detect (`dest_confirmed=True`):
            Click vào nút `[Doi Diem Den (M)]` sẽ mở lại bản đồ chọn điểm đến.
          - Khi đang ở chế độ Chọn Điểm Đến (`dest_confirmed=False`):
            Hỗ trợ cuộn, kéo thanh trượt, zoom, click chọn đích, và bấm `[XAC NHAN (Enter)]`
            (hoặc double-click) để xác nhận và chuyển sang Ô Quan Sát Lane Detect.
        """
        local_x = int(np.clip(local_x, 0, self.quad_w - 1))
        local_y = int(np.clip(local_y, 0, self.quad_h - 1))

        # Nếu đã xác nhận điểm đến (đang ở chế độ Quan Sát Lane Detect):
        if self.dest_confirmed:
            if event == cv2.EVENT_LBUTTONDOWN:
                rx1, ry1, rx2, ry2 = self._reopen_picker_btn
                if rx1 <= local_x <= rx2 and ry1 <= local_y <= ry2:
                    self.reopen_destination_picker()
            return

        # Đang ở chế độ Chọn Điểm Đến trên bản đồ cuộn:
        mx, my = self._screen_to_map_coords(local_x, local_y)
        self.hover_map_px = (mx, my)
        self.hover_node_id = self.navigator.find_nearest_node(mx, my)

        sb_thick = 14
        in_vscroll = local_x >= self.quad_w - sb_thick and 34 <= local_y <= self.quad_h - sb_thick
        in_hscroll = local_y >= self.quad_h - sb_thick and 0 <= local_x <= self.quad_w - sb_thick

        if event == cv2.EVENT_LBUTTONDBLCLK and local_y > 32:
            # Double-click chuột trái vào bản đồ -> Chọn điểm đến & Xác nhận chuyển sang Lane Detect ngay!
            new_route = self.navigator.set_destination(
                dest_px=(mx, my),
                start_node_id=self.navigator.start_node_id,
                export_chart=False,
            )
            self._sim_progress_idx = 0.0
            self.refresh_route_cache()
            self.confirm_destination()
            if self.on_route_changed is not None:
                self.on_route_changed(new_route)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Kiểm tra nếu click vào nút [XAC NHAN], [+], [-], [1:1], [Fit] trên thanh công cụ
            for label, bx1, by1, bx2, by2 in self._picker_buttons:
                if bx1 <= local_x <= bx2 and by1 <= local_y <= by2:
                    if label == "CONFIRM":
                        self.confirm_destination()
                    elif label == "+":
                        self.zoom_picker_at(self.picker_zoom * 1.25)
                    elif label == "-":
                        self.zoom_picker_at(self.picker_zoom * 0.80)
                    elif label == "1:1":
                        self.zoom_picker_at(1.0)
                    elif label == "Fit":
                        self.zoom_picker_at(self.picker_min_zoom)
                    return

            # 2. Kiểm tra nếu kéo thanh cuộn dọc / ngang
            if in_vscroll:
                self._dragging_vscroll = True
                ratio = (local_y - 34) / float(max(1, self.quad_h - 34 - sb_thick))
                vis_h = self.quad_h / self.picker_zoom
                self.picker_scroll_y = ratio * max(0.0, self.map_h - vis_h)
                self._clamp_picker_scroll()
                return
            if in_hscroll:
                self._dragging_hscroll = True
                ratio = local_x / float(max(1, self.quad_w - sb_thick))
                vis_w = self.quad_w / self.picker_zoom
                self.picker_scroll_x = ratio * max(0.0, self.map_w - vis_w)
                self._clamp_picker_scroll()
                return

            # 3. Bắt đầu kéo bản đồ (hoặc click chọn điểm đến nếu thả chuột tại chỗ)
            self._dragging_map = True
            self._drag_start_pt = (local_x, local_y)
            self._scroll_start_xy = (self.picker_scroll_x, self.picker_scroll_y)

        elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            self._dragging_map = True
            self._drag_start_pt = (local_x, local_y)
            self._scroll_start_xy = (self.picker_scroll_x, self.picker_scroll_y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging_vscroll:
                ratio = (local_y - 34) / float(max(1, self.quad_h - 34 - sb_thick))
                vis_h = self.quad_h / self.picker_zoom
                self.picker_scroll_y = ratio * max(0.0, self.map_h - vis_h)
                self._clamp_picker_scroll()
            elif self._dragging_hscroll:
                ratio = local_x / float(max(1, self.quad_w - sb_thick))
                vis_w = self.quad_w / self.picker_zoom
                self.picker_scroll_x = ratio * max(0.0, self.map_w - vis_w)
                self._clamp_picker_scroll()
            elif self._dragging_map:
                dx = local_x - self._drag_start_pt[0]
                dy = local_y - self._drag_start_pt[1]
                self.picker_scroll_x = self._scroll_start_xy[0] - dx / self.picker_zoom
                self.picker_scroll_y = self._scroll_start_xy[1] - dy / self.picker_zoom
                self._clamp_picker_scroll()

        elif event == cv2.EVENT_LBUTTONUP:
            was_dragging_map = self._dragging_map
            self._dragging_map = False
            self._dragging_vscroll = False
            self._dragging_hscroll = False

            if was_dragging_map:
                move_dist = math.hypot(
                    local_x - self._drag_start_pt[0], local_y - self._drag_start_pt[1]
                )
                # Click tại chỗ (< 6px) -> Chọn Điểm Đến trên bản đồ (bấm Enter hoặc nút XÁC NHẬN để chuyển sang Lane Detect)
                if move_dist < 6.0 and local_y > 32 and local_x < self.quad_w - sb_thick and local_y < self.quad_h - sb_thick:
                    new_route = self.navigator.set_destination(
                        dest_px=(mx, my),
                        start_node_id=self.navigator.start_node_id,
                        export_chart=False,
                    )
                    self._sim_progress_idx = 0.0
                    self.refresh_route_cache()
                    if self.on_route_changed is not None:
                        self.on_route_changed(new_route)

        elif event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            was_dragging = self._dragging_map
            self._dragging_map = False
            if event == cv2.EVENT_RBUTTONUP and was_dragging:
                move_dist = math.hypot(
                    local_x - self._drag_start_pt[0], local_y - self._drag_start_pt[1]
                )
                if move_dist < 6.0 and local_y > 32:
                    new_start_id = self.navigator.find_nearest_node(mx, my)
                    if self.navigator.dest_node_id is not None:
                        new_route = self.navigator.set_destination(
                            dest_node_id=self.navigator.dest_node_id,
                            start_node_id=new_start_id,
                            export_chart=False,
                        )
                        self._sim_progress_idx = 0.0
                        self.refresh_route_cache()
                        if self.on_route_changed is not None:
                            self.on_route_changed(new_route)

        elif event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 120 if flags > 0 else -120
            step = 75.0 / self.picker_zoom
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                factor = 1.18 if delta > 0 else 0.85
                self.zoom_picker_at(self.picker_zoom * factor, local_x, local_y)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.scroll_picker_by(-step if delta > 0 else step, 0.0)
            else:
                self.scroll_picker_by(0.0, -step if delta > 0 else step)

        elif event == cv2.EVENT_MOUSEHWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 120 if flags > 0 else -120
            step = 75.0 / self.picker_zoom
            self.scroll_picker_by(-step if delta > 0 else step, 0.0)

    def _on_quad_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Callback chuột cho chế độ cửa sổ hợp nhất 4 Ô (2x2)."""
        if x >= self.quad_w and y < self.quad_h:
            self.handle_picker_mouse(event, x - self.quad_w, y, flags)
        elif x >= self.quad_w and y >= self.quad_h and event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 120 if flags > 0 else -120
            self.radar_zoom = float(np.clip(self.radar_zoom * (1.15 if delta > 0 else 0.87), 0.7, 3.0))

    def _on_q2_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Callback chuột riêng cho cửa sổ Ô 2 khi ở chế độ 4windows."""
        self.handle_picker_mouse(event, x, y, flags)

    def _on_q4_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Callback chuột riêng cho cửa sổ Ô 4 khi ở chế độ 4windows."""
        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 120 if flags > 0 else -120
            self.radar_zoom = float(np.clip(self.radar_zoom * (1.15 if delta > 0 else 0.87), 0.7, 3.0))

    def handle_key(self, key: int) -> bool:
        """
        Xử lý phím tắt:
          - Enter (13) hoặc Space (32): Xác nhận điểm đến -> Chuyển Ô 2 thành Ô Quan Sát Lane Detect!
          - M / m: Mở lại bản đồ Chọn Điểm Đến trên Ô 2.
          - W/A/S/D, +/-, R: Cuộn và Zoom bản đồ khi Ô 2 đang ở chế độ Chọn Điểm Đến.
          - q hoặc ESC (27): Thoát chương trình.
        """
        if key in (ord("q"), 27):
            return False
        if key in (13, 32):  # Enter hoặc Space -> Xác nhận Điểm Đến, chuyển Ô 2 sang Quan Sát Lane Detect
            self.confirm_destination()
            return True
        if key in (ord("m"), ord("M")):  # Phím M -> Mở lại bản đồ Chọn Điểm Đến
            self.reopen_destination_picker()
            return True

        if not self.dest_confirmed:
            step = 65.0 / self.picker_zoom
            if key in (ord("w"), ord("W"), 82):
                self.scroll_picker_by(0.0, -step)
            elif key in (ord("s"), ord("S"), 84):
                self.scroll_picker_by(0.0, step)
            elif key in (ord("a"), ord("A"), 81):
                self.scroll_picker_by(-step, 0.0)
            elif key in (ord("d"), ord("D"), 83):
                self.scroll_picker_by(step, 0.0)
            elif key in (ord("+"), ord("=")):
                self.zoom_picker_at(self.picker_zoom * 1.2)
            elif key in (ord("-"), ord("_")):
                self.zoom_picker_at(self.picker_zoom * 0.83)
            elif key in (ord("r"), ord("R")):
                self.zoom_picker_at(1.0)
        return True

    # =========================================================================
    # Ô 1 (Góc Trên-Trái): CỬA SỔ GAME (RAW GAME WINDOW FEED)
    # =========================================================================
    def render_game_quadrant(
        self,
        raw_frame: np.ndarray,
        path_data: Any,
        detector_name: str,
        fps: float,
        n_lanes: int,
        control_enabled: bool,
    ) -> np.ndarray:
        """
        Vẽ Ô 1: Khung hình Cửa sổ Game Forza Horizon 4 sạch sẽ (chỉ kèm thanh trạng thái mỏng ở trên),
        nhường phần hiển thị chi tiết vạch làn đường B-Spline & ROI cho Ô 2 (sau khi xác nhận điểm đến).
        """
        tile = cv2.resize(raw_frame, (self.quad_w, self.quad_h), interpolation=cv2.INTER_LINEAR)

        # Thanh Header mỏng của Ô 1
        cv2.rectangle(tile, (0, 0), (self.quad_w, 32), (18, 22, 28), -1)
        cv2.line(tile, (0, 32), (self.quad_w, 32), (0, 200, 255), 1)
        mode_tag = "AUTO-CONTROL" if control_enabled else "VIEW-ONLY"
        fps_color = (60, 255, 100) if fps >= 30.0 else (0, 200, 255)
        dest_status = "LOCKED -> LANE DETECT ON [O 2]" if self.dest_confirmed else "SELECTING DEST ON [O 2]..."
        header_txt = f"[O 1] CUA SO GAME ({mode_tag}) | {dest_status}"
        cv2.putText(tile, header_txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 245, 255), 1, cv2.LINE_AA)
        fps_badge = f"{fps:4.1f} FPS"
        cv2.putText(tile, fps_badge, (self.quad_w - 95, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 2, cv2.LINE_AA)
        return tile

    # =========================================================================
    # Ô 2A (Góc Trên-Phải - Trước khi xác nhận): CHỌN ĐIỂM ĐẾN (CUỘN / ZOOM / CLICK)
    # =========================================================================
    def render_destination_picker_quadrant(self) -> np.ndarray:
        """
        Vẽ Ô 2 ở trạng thái Chọn Điểm Đến: Bản đồ FH4 có thanh cuộn dọc/ngang, lăn chuột cuộn/zoom,
        kéo chuột pan, click chuột trái chọn Đích và nút [XAC NHAN (Enter)] để chuyển thành Ô Quan Sát Lane Detect.
        """
        vis_w = min(self.map_w, int(round(self.quad_w / self.picker_zoom)))
        vis_h = min(self.map_h, int(round(self.quad_h / self.picker_zoom)))
        x0 = int(np.clip(round(self.picker_scroll_x), 0, max(0, self.map_w - vis_w)))
        y0 = int(np.clip(round(self.picker_scroll_y), 0, max(0, self.map_h - vis_h)))

        crop = self.navigator.picker_map_bgr[y0 : y0 + vis_h, x0 : x0 + vis_w]
        tile = cv2.resize(crop, (self.quad_w, self.quad_h), interpolation=cv2.INTER_LINEAR)

        # Vẽ vòng tròn highlight nút giao gần con trỏ chuột nhất khi hover
        if self.hover_node_id is not None and self.hover_node_id in self.navigator.nodes:
            hnode = self.navigator.nodes[self.hover_node_id]
            if x0 <= hnode.x_px <= x0 + vis_w and y0 <= hnode.y_px <= y0 + vis_h:
                hx = int(round((hnode.x_px - x0) / float(max(1, vis_w)) * self.quad_w))
                hy = int(round((hnode.y_px - y0) / float(max(1, vis_h)) * self.quad_h))
                cv2.circle(tile, (hx, hy), 13, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(tile, (hx, hy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)

        # Vẽ Thanh Cuộn Dọc (Vertical Scrollbar) bên cạnh phải
        sb_t = 12
        top_bar_h = 32
        track_h = self.quad_h - top_bar_h - sb_t
        cv2.rectangle(tile, (self.quad_w - sb_t, top_bar_h), (self.quad_w, self.quad_h - sb_t), (25, 28, 32), -1)
        thumb_h = max(24, int(round((vis_h / float(max(1, self.map_h))) * track_h)))
        max_sy = max(1.0, float(self.map_h - vis_h))
        thumb_y = top_bar_h + int(round((y0 / max_sy) * max(0, track_h - thumb_h)))
        cv2.rectangle(
            tile,
            (self.quad_w - sb_t + 2, thumb_y),
            (self.quad_w - 2, min(self.quad_h - sb_t, thumb_y + thumb_h)),
            (0, 200, 255),
            -1,
        )

        # Vẽ Thanh Cuộn Ngang (Horizontal Scrollbar) ở cạnh đáy
        track_w = self.quad_w - sb_t
        cv2.rectangle(tile, (0, self.quad_h - sb_t), (track_w, self.quad_h), (25, 28, 32), -1)
        thumb_w = max(24, int(round((vis_w / float(max(1, self.map_w))) * track_w)))
        max_sx = max(1.0, float(self.map_w - vis_w))
        thumb_x = int(round((x0 / max_sx) * max(0, track_w - thumb_w)))
        cv2.rectangle(
            tile,
            (thumb_x, self.quad_h - sb_t + 2),
            (min(track_w, thumb_x + thumb_w), self.quad_h - 2),
            (0, 200, 255),
            -1,
        )

        # Vẽ Header Ô 2 và các nút bấm [XAC NHAN (Enter)], [+], [-], [1:1], [Fit]
        cv2.rectangle(tile, (0, 0), (self.quad_w, top_bar_h), (18, 22, 28), -1)
        cv2.line(tile, (0, top_bar_h), (self.quad_w, top_bar_h), (0, 200, 255), 1)
        dest_id = self.navigator.dest_node_id if self.navigator.dest_node_id is not None else 12
        dest_lbl = self.navigator.nodes[dest_id].label
        cv2.putText(
            tile,
            f"[O 2] CHON DIEM DEN ({self.picker_zoom:.1f}x): [{dest_id}] {dest_lbl[:18]}",
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 255, 180),
            1,
            cv2.LINE_AA,
        )

        for label, bx1, by1, bx2, by2 in self._picker_buttons:
            if label == "CONFIRM":
                cv2.rectangle(tile, (bx1, by1), (bx2, by2), (25, 155, 55), -1)
                cv2.rectangle(tile, (bx1, by1), (bx2, by2), (100, 255, 140), 1)
                btn_txt = "XAC NHAN (Enter)"
                tw = cv2.getTextSize(btn_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)[0][0]
                tx = bx1 + (bx2 - bx1 - tw) // 2
                cv2.putText(tile, btn_txt, (tx, by1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(tile, (bx1, by1), (bx2, by2), (45, 55, 68), -1)
                cv2.rectangle(tile, (bx1, by1), (bx2, by2), (0, 200, 255), 1)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
                tx = bx1 + (bx2 - bx1 - tw) // 2
                cv2.putText(tile, label, (tx, by1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        footer_txt = "Wheel/Keo: Cuon | Click Trai: Chon Dich | Bam [XAC NHAN] hoac Enter de chuyen sang LANE DETECT"
        cv2.rectangle(tile, (0, self.quad_h - sb_t - 22), (self.quad_w - sb_t, self.quad_h - sb_t), (15, 18, 22), -1)
        cv2.putText(
            tile,
            footer_txt,
            (8, self.quad_h - sb_t - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (120, 255, 180),
            1,
            cv2.LINE_AA,
        )
        return tile

    # =========================================================================
    # Ô 2B (Góc Trên-Phải - Sau khi xác nhận): QUAN SÁT LANE DETECT (UK LANE MONITOR)
    # =========================================================================
    def render_lane_detect_quadrant(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        path_data: Any,
        detector_name: str,
        fps: float,
        n_lanes: int,
        steering_cmd: float,
    ) -> np.ndarray:
        """
        Vẽ Ô 2 sau khi đã xác nhận điểm đến: Chuyển thành Ô QUAN SÁT LANE DETECT chuyên sâu:
          - Hiển thị khung vùng quan tâm Polygon ROI
          - Tô màu hành lang làn đường bên trái chuẩn Vương Quốc Anh (UK Left Corridor)
          - Vẽ vạch lề trái UK (`uk_left_edge`), vạch phân cách giữa (`uk_center_divider`),
            đường dẫn tâm làn (`centerline`) và điểm nhìn trước (`lookahead_point`)
          - Đính kèm cửa sổ thu nhỏ (PiP) hiển thị mặt nạ biên nhị phân trong vùng ROI (ROI Lane & Edge Mask)
          - Có nút `[Doi Diem Den (M)]` ở góc phải trên để mở lại bản đồ chọn đích bất kỳ lúc nào.
        """
        base_img = raw_frame if (raw_frame is not None and raw_frame.size > 0) else annotated_frame
        tile = cv2.resize(base_img, (self.quad_w, self.quad_h), interpolation=cv2.INTER_LINEAR)
        orig_h, orig_w = base_img.shape[:2]
        sx = self.quad_w / float(max(1, orig_w))
        sy = self.quad_h / float(max(1, orig_h))

        # 1. Vẽ khung vùng quan tâm Polygon ROI trên Ô 2
        roi_pts = np.array(
            [
                [
                    (int(self.quad_w * 0.05), self.quad_h - 28),
                    (int(self.quad_w * 0.28), int(self.quad_h * 0.38)),
                    (int(self.quad_w * 0.72), int(self.quad_h * 0.38)),
                    (int(self.quad_w * 0.98), self.quad_h - 28),
                ]
            ],
            dtype=np.int32,
        )
        cv2.polylines(tile, roi_pts, True, (180, 120, 60), 1, cv2.LINE_AA)

        # 2. Tô màu hành lang làn đường bên trái UK và vẽ các đường cong B-Spline đã phân loại
        labeled = getattr(path_data, "classified_lanes", None) or {}
        scaled_labeled: Dict[str, np.ndarray] = {}
        for ltype, curve in labeled.items():
            pts = np.asarray(curve, dtype=np.float32).copy()
            if pts.ndim == 2 and len(pts) >= 2:
                pts[:, 0] *= sx
                pts[:, 1] *= sy
                scaled_labeled[ltype] = pts.astype(np.int32)

        if "uk_left_edge" in scaled_labeled and "uk_center_divider" in scaled_labeled:
            corridor = tile.copy()
            # Đảm bảo 2 nhánh đa giác cùng chiều dọc từ dưới lên trên trước khi nối khép kín
            left_c = scaled_labeled["uk_left_edge"]
            right_c = scaled_labeled["uk_center_divider"]
            if left_c[0, 1] < left_c[-1, 1]:
                left_c = np.flipud(left_c)
            if right_c[0, 1] < right_c[-1, 1]:
                right_c = np.flipud(right_c)
            poly_corr = np.vstack([left_c, np.flipud(right_c)])
            cv2.fillPoly(corridor, [poly_corr.reshape((-1, 1, 2))], (0, 195, 85))
            cv2.addWeighted(corridor, 0.26, tile, 0.74, 0, tile)

        for ltype, pts_i32 in scaled_labeled.items():
            pts_draw = pts_i32.reshape((-1, 1, 2))
            if ltype == "uk_left_edge":
                cv2.polylines(tile, [pts_draw], False, (255, 225, 0), 4, cv2.LINE_AA)
            elif ltype == "uk_center_divider":
                cv2.polylines(tile, [pts_draw], False, (0, 165, 255), 4, cv2.LINE_AA)
            else:
                cv2.polylines(tile, [pts_draw], False, (180, 105, 255), 3, cv2.LINE_AA)

        # 3. Vẽ đường dẫn tâm làn trái UK (Centerline) & Điểm nhìn trước (Lookahead Point)
        ego_x = self.quad_w // 2
        if getattr(path_data, "valid", False) and getattr(path_data, "centerline", None):
            cl = np.asarray(path_data.centerline, dtype=np.float32).copy()
            if len(cl) >= 2:
                cl[:, 0] *= sx
                cl[:, 1] *= sy
                ego_x = int(round(float(cl[0, 0])))
                cv2.polylines(tile, [cl.astype(np.int32).reshape((-1, 1, 2))], False, (0, 255, 255), 3, cv2.LINE_AA)

        if getattr(path_data, "valid", False) and getattr(path_data, "lookahead_point", None) is not None:
            lx = int(round(path_data.lookahead_point[0] * sx))
            ly = int(round(path_data.lookahead_point[1] * sy))
            ego_pt = (ego_x, self.quad_h - 28)
            cv2.line(tile, ego_pt, (lx, ly), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(tile, (lx, ly), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(tile, (lx, ly), 10, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Đính kèm cửa sổ thu nhỏ (PiP) hiển thị Mặt nạ Biên ROI (ROI Lane & Edge Mask) ở góc trên trái Ô 2
        pip_w = max(130, int(self.quad_w * 0.23))
        pip_h = max(78, int(self.quad_h * 0.21))
        small_bgr = cv2.resize(tile, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        if getattr(self, "_cached_pip_roi_shape", None) != (pip_h, pip_w):
            self._cached_pip_roi_pts = np.array(
                [
                    [
                        (int(pip_w * 0.05), pip_h - 3),
                        (int(pip_w * 0.28), int(pip_h * 0.38)),
                        (int(pip_w * 0.72), int(pip_h * 0.38)),
                        (int(pip_w * 0.98), pip_h - 3),
                    ]
                ],
                dtype=np.int32,
            )
            mask = np.zeros((pip_h, pip_w), dtype=np.uint8)
            cv2.fillPoly(mask, self._cached_pip_roi_pts, 255)
            self._cached_pip_roi_mask = mask
            self._cached_pip_roi_shape = (pip_h, pip_w)

        small_roi_mask = self._cached_pip_roi_mask
        small_edges = cv2.bitwise_and(cv2.Canny(small_gray, 50, 140), small_roi_mask)
        small_ridge = cv2.subtract(small_gray, cv2.boxFilter(small_gray, cv2.CV_8U, (15, 5)))
        small_paint = cv2.bitwise_and(
            np.where((small_ridge >= 12) & (small_gray >= 110), np.uint8(255), np.uint8(0)),
            small_roi_mask,
        )
        pip_bgr = np.full((pip_h, pip_w, 3), (14, 18, 22), dtype=np.uint8)
        cv2.polylines(pip_bgr, self._cached_pip_roi_pts, True, (90, 70, 45), 1, cv2.LINE_AA)
        pip_bgr[small_edges > 0] = (180, 160, 0)
        pip_bgr[small_paint > 0] = (0, 255, 180)
        # Vẽ luôn 2 biên làn phát hiện được lên PiP để dễ đối chiếu
        pip_sx = pip_w / float(max(1, self.quad_w))
        pip_sy = pip_h / float(max(1, self.quad_h))
        for ltype, pts_i32 in scaled_labeled.items():
            p_pip = pts_i32.astype(np.float32).copy()
            p_pip[:, 0] *= pip_sx
            p_pip[:, 1] *= pip_sy
            c_pip = (255, 225, 0) if ltype == "uk_left_edge" else (0, 165, 255)
            cv2.polylines(pip_bgr, [p_pip.astype(np.int32).reshape((-1, 1, 2))], False, c_pip, 2, cv2.LINE_AA)
        cv2.putText(pip_bgr, "ROI LANE MASK", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 255), 1, cv2.LINE_AA)
        px0, py0 = 10, 40
        tile[py0 : py0 + pip_h, px0 : px0 + pip_w] = pip_bgr
        cv2.rectangle(tile, (px0, py0), (px0 + pip_w, py0 + pip_h), (0, 210, 255), 1)

        # 5. Thanh Header Ô 2 + Nút bấm [Doi Diem Den (M)]
        top_bar_h = 32
        cv2.rectangle(tile, (0, 0), (self.quad_w, top_bar_h), (18, 22, 28), -1)
        cv2.line(tile, (0, top_bar_h), (self.quad_w, top_bar_h), (0, 200, 255), 1)
        cv2.putText(
            tile,
            f"[O 2] QUAN SAT LANE DETECT ({detector_name.upper()}) | Lanes: {n_lanes} ({getattr(path_data, 'uk_lane_type', 'none')})",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (80, 255, 160),
            1,
            cv2.LINE_AA,
        )

        rx1, ry1, rx2, ry2 = self._reopen_picker_btn
        cv2.rectangle(tile, (rx1, ry1), (rx2, ry2), (55, 70, 95), -1)
        cv2.rectangle(tile, (rx1, ry1), (rx2, ry2), (0, 215, 255), 1)
        btn_lbl = "Doi Diem Den (M)"
        tw = cv2.getTextSize(btn_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)[0][0]
        cv2.putText(
            tile,
            btn_lbl,
            (rx1 + (rx2 - rx1 - tw) // 2, ry1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 6. Thanh Footer thông số Lane Detect ở đáy Ô 2
        heading_deg = (
            math.degrees(path_data.heading_error_rad)
            if getattr(path_data, "heading_error_rad", None) is not None
            else 0.0
        )
        curv = float(getattr(path_data, "curvature", 0.0) or 0.0)
        conf = float(getattr(path_data, "confidence", 0.0) or 0.0)
        footer_txt = (
            f"Cyan=Le Trai UK | Orange=Tim Duong | Yellow=Centerline | "
            f"Err: {heading_deg:+5.1f} deg | Curv: {curv:.4f} | Steer: {steering_cmd:+5.1f} | Conf: {conf:.2f}"
        )
        cv2.rectangle(tile, (0, self.quad_h - 24), (self.quad_w, self.quad_h), (15, 18, 22), -1)
        cv2.putText(
            tile,
            footer_txt,
            (8, self.quad_h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (225, 240, 255),
            1,
            cv2.LINE_AA,
        )
        return tile

    def render_top_right_quadrant(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        path_data: Any,
        detector_name: str,
        fps: float,
        n_lanes: int,
        steering_cmd: float,
    ) -> np.ndarray:
        """
        Tự động vẽ Ô 2 tùy theo trạng thái xác nhận điểm đến (`self.dest_confirmed`):
          - Chưa xác nhận (`False`): Hiển thị Bản đồ Cuộn Chọn Điểm Đến.
          - Đã xác nhận (`True`)  : Hiển thị Ô Quan Sát Lane Detect.
        """
        if self.dest_confirmed:
            return self.render_lane_detect_quadrant(
                raw_frame=raw_frame,
                annotated_frame=annotated_frame,
                path_data=path_data,
                detector_name=detector_name,
                fps=fps,
                n_lanes=n_lanes,
                steering_cmd=steering_cmd,
            )
        return self.render_destination_picker_quadrant()

    # =========================================================================
    # Ô 3 (Góc Dưới-Trái): QUAN SÁT TELEMETRY & ĐIỀU KHIỂN
    # =========================================================================
    def render_telemetry_quadrant(
        self,
        fps: float,
        current_speed_kmh: float,
        target_speed_kmh: float,
        steering_cmd: float,
        throttle: float,
        brake: float,
        path_data: Any,
        telemetry_pkt: Optional[Dict[str, Any]],
        control_enabled: bool,
        dist_to_goal_m: float,
    ) -> np.ndarray:
        """Vẽ Ô 3: Bảng đồng hồ Telemetry số hóa (Speedometer, RPM, Gear, Steer, Throttle/Brake, 3D Pos, Sparkline FPS)."""
        self.fps_history.append(float(fps))
        self.speed_history.append(float(current_speed_kmh))

        tile = np.full((self.quad_h, self.quad_w, 3), (18, 20, 25), dtype=np.uint8)

        # Header Ô 3
        cv2.rectangle(tile, (0, 0), (self.quad_w, 32), (24, 28, 36), -1)
        cv2.line(tile, (0, 32), (self.quad_w, 32), (0, 200, 255), 1)
        udp_live = telemetry_pkt is not None and bool(telemetry_pkt.get("IsRaceOn", 0))
        udp_badge = "UDP TELEMETRY: LIVE" if udp_live else ("UDP: WAITING" if control_enabled else "TELEMETRY: SIMULATED")
        udp_color = (60, 255, 100) if udp_live else (0, 215, 255)
        cv2.putText(
            tile,
            f"[O 3] QUAN SAT TELEMETRY & DIEU KHIEN UK  |  {udp_badge}",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            udp_color,
            1,
            cv2.LINE_AA,
        )

        # ---- 1. Đồng hồ Tốc độ Bán nguyệt (Speedometer Arc) bên trái ----
        cx = int(self.quad_w * 0.20)
        cy = int(self.quad_h * 0.37)
        radius = min(82, int(min(self.quad_w, self.quad_h) * 0.18))
        cv2.ellipse(tile, (cx, cy), (radius, radius), 0, 135, 405, (55, 62, 72), 10, cv2.LINE_AA)
        speed_ratio = float(np.clip(current_speed_kmh / 200.0, 0.0, 1.0))
        end_angle = int(round(135 + 270 * speed_ratio))
        arc_color = (0, 230, 110) if current_speed_kmh <= target_speed_kmh + 8 else (0, 140, 255)
        if end_angle > 135:
            cv2.ellipse(tile, (cx, cy), (radius, radius), 0, 135, end_angle, arc_color, 10, cv2.LINE_AA)

        tgt_ratio = float(np.clip(target_speed_kmh / 200.0, 0.0, 1.0))
        tgt_rad = math.radians(135 + 270 * tgt_ratio)
        tx1 = int(round(cx + (radius - 12) * math.cos(tgt_rad)))
        ty1 = int(round(cy + (radius - 12) * math.sin(tgt_rad)))
        tx2 = int(round(cx + (radius + 8) * math.cos(tgt_rad)))
        ty2 = int(round(cy + (radius + 8) * math.sin(tgt_rad)))
        cv2.line(tile, (tx1, ty1), (tx2, ty2), (0, 255, 255), 3, cv2.LINE_AA)

        spd_txt = f"{current_speed_kmh:4.0f}"
        cv2.putText(tile, spd_txt, (cx - 34, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, f"km/h (Tgt {target_speed_kmh:.0f})", (cx - 52, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 210, 240), 1, cv2.LINE_AA)

        gear = str((telemetry_pkt or {}).get("Gear", "D" if current_speed_kmh > 1 else "N"))
        drivetrain = str((telemetry_pkt or {}).get("Drivetrain", "AWD"))
        rpm = float((telemetry_pkt or {}).get("CurrentEngineRpm", current_speed_kmh * 68.0 + 850.0))
        max_rpm = max(1000.0, float((telemetry_pkt or {}).get("EngineMaxRpm", 8500.0)))
        cv2.putText(
            tile,
            f"GEAR: {gear} ({drivetrain}) | RPM: {rpm:.0f}",
            (max(10, cx - 78), cy + radius + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

        # ---- 2. Thước Góc Lái (Steering Gauge) & Thanh Ga/Phanh ở vùng giữa ----
        mid_x = int(self.quad_w * 0.42)
        bar_w = int(self.quad_w * 0.52)

        rpm_y = 52
        cv2.putText(tile, "ENGINE RPM", (mid_x, rpm_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (180, 195, 210), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (mid_x, rpm_y), (mid_x + bar_w, rpm_y + 14), (45, 50, 60), -1)
        rpm_fill = int(round(bar_w * float(np.clip(rpm / max_rpm, 0.0, 1.0))))
        rpm_col = (0, 220, 120) if rpm < max_rpm * 0.78 else (0, 80, 255)
        cv2.rectangle(tile, (mid_x, rpm_y), (mid_x + rpm_fill, rpm_y + 14), rpm_col, -1)

        st_y = 92
        cv2.putText(
            tile,
            f"STEERING ANGLE: {steering_cmd:+6.1f} / 120 ({'LEFT' if steering_cmd < -1 else ('RIGHT' if steering_cmd > 1 else 'CENTER')})",
            (mid_x, st_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(tile, (mid_x, st_y), (mid_x + bar_w, st_y + 16), (45, 50, 60), -1)
        center_bx = mid_x + bar_w // 2
        cv2.line(tile, (center_bx, st_y - 2), (center_bx, st_y + 18), (255, 255, 255), 2)
        steer_norm = float(np.clip(steering_cmd / 120.0, -1.0, 1.0))
        steer_px = int(round(center_bx + steer_norm * (bar_w // 2)))
        x1_s, x2_s = min(center_bx, steer_px), max(center_bx, steer_px)
        cv2.rectangle(tile, (x1_s, st_y + 2), (x2_s, st_y + 14), (255, 195, 0), -1)

        tb_y = 132
        half_bw = (bar_w - 16) // 2
        cv2.putText(tile, f"THROTTLE: {throttle * 100:4.0f}%", (mid_x, tb_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (100, 255, 140), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (mid_x, tb_y), (mid_x + half_bw, tb_y + 14), (45, 50, 60), -1)
        cv2.rectangle(tile, (mid_x, tb_y), (mid_x + int(round(half_bw * np.clip(throttle, 0.0, 1.0))), tb_y + 14), (40, 220, 90), -1)

        bx_brake = mid_x + half_bw + 16
        cv2.putText(tile, f"BRAKE: {brake * 100:4.0f}%", (bx_brake, tb_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 150, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (bx_brake, tb_y), (bx_brake + half_bw, tb_y + 14), (45, 50, 60), -1)
        cv2.rectangle(tile, (bx_brake, tb_y), (bx_brake + int(round(half_bw * np.clip(brake, 0.0, 1.0))), tb_y + 14), (0, 60, 255), -1)

        # ---- 3. Bảng Tọa độ 3D & Thông số Làn đường UK ----
        info_y = 166
        pos_x = float((telemetry_pkt or {}).get("PositionX", 0.0))
        pos_y = float((telemetry_pkt or {}).get("PositionY", 0.0))
        pos_z = float((telemetry_pkt or {}).get("PositionZ", 0.0))
        yaw_deg = math.degrees(float((telemetry_pkt or {}).get("Yaw", 0.0)))
        pitch_deg = math.degrees(float((telemetry_pkt or {}).get("Pitch", 0.0)))
        roll_deg = math.degrees(float((telemetry_pkt or {}).get("Roll", 0.0)))
        heading_err_deg = (
            math.degrees(path_data.heading_error_rad)
            if getattr(path_data, "heading_error_rad", None) is not None
            else 0.0
        )
        curv = float(getattr(path_data, "curvature", 0.0) or 0.0)
        conf = float(getattr(path_data, "confidence", 0.0) or 0.0)
        uk_type = str(getattr(path_data, "uk_lane_type", "none"))

        lines = [
            f"World Pos (m) : X={pos_x:+7.1f} | Y={pos_y:+6.1f} | Z={pos_z:+7.1f}",
            f"Orientation   : Yaw={yaw_deg:+6.1f} deg | Pitch={pitch_deg:+5.1f} | Roll={roll_deg:+5.1f}",
            f"UK Lane Mode  : {uk_type} (Conf: {conf:.2f})",
            f"Path Geometry : Err={heading_err_deg:+5.1f} deg | Curv={curv:.4f} | Goal={dist_to_goal_m / 1000.0:.2f} km",
        ]
        for idx_l, txt in enumerate(lines):
            cv2.putText(
                tile,
                txt,
                (mid_x, info_y + idx_l * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (215, 225, 235),
                1,
                cv2.LINE_AA,
            )

        # ---- 4. Biểu đồ Sparkline FPS & Tốc độ ở phần đáy Ô 3 ----
        chart_top = max(260, int(self.quad_h * 0.66))
        chart_bot = self.quad_h - 12
        chart_left = 14
        chart_right = self.quad_w - 14
        chart_w = max(20, chart_right - chart_left)
        chart_h = max(30, chart_bot - chart_top)

        cv2.rectangle(tile, (chart_left, chart_top), (chart_right, chart_bot), (26, 30, 38), -1)
        cv2.rectangle(tile, (chart_left, chart_top), (chart_right, chart_bot), (55, 65, 80), 1)

        y_30fps = chart_bot - int(round((30.0 / 120.0) * chart_h))
        cv2.line(tile, (chart_left, y_30fps), (chart_right, y_30fps), (0, 180, 220), 1)
        cv2.putText(tile, "30 FPS min", (chart_left + 6, max(chart_top + 12, y_30fps - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 210, 255), 1, cv2.LINE_AA)

        if len(self.fps_history) >= 2:
            pts_fps = []
            for i, val in enumerate(self.fps_history):
                px = chart_left + int(round((i / float(max(1, len(self.fps_history) - 1))) * (chart_w - 4)))
                py = chart_bot - int(round(np.clip(val / 120.0, 0.0, 1.0) * (chart_h - 6))) - 2
                pts_fps.append([px, py])
            cv2.polylines(tile, [np.array(pts_fps, dtype=np.int32).reshape((-1, 1, 2))], False, (60, 255, 110), 2, cv2.LINE_AA)

        if len(self.speed_history) >= 2:
            pts_spd = []
            for i, val in enumerate(self.speed_history):
                px = chart_left + int(round((i / float(max(1, len(self.speed_history) - 1))) * (chart_w - 4)))
                py = chart_bot - int(round(np.clip(val / 180.0, 0.0, 1.0) * (chart_h - 6))) - 2
                pts_spd.append([px, py])
            cv2.polylines(tile, [np.array(pts_spd, dtype=np.int32).reshape((-1, 1, 2))], False, (255, 190, 0), 2, cv2.LINE_AA)

        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else fps
        cv2.putText(
            tile,
            f"REAL-TIME PERFORMANCE: {fps:5.1f} FPS (Avg: {avg_fps:5.1f} FPS) [Green=FPS, Cyan=Speed]",
            (chart_left + 8, chart_top + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 255, 150),
            1,
            cv2.LINE_AA,
        )
        return tile

    # =========================================================================
    # Ô 4 (Góc Dưới-Phải): QUAN SÁT BẢN ĐỒ & LỘ TRÌNH ĐỘNG (LIVE MAP & RADAR)
    # =========================================================================
    def render_live_map_quadrant(
        self,
        telemetry_pkt: Optional[Dict[str, Any]],
        route_info: Dict[str, Any],
        active_wp: Dict[str, Any],
        turn_instruction: str,
        dist_to_goal_m: float,
    ) -> np.ndarray:
        """
        Vẽ Ô 4: Bản đồ dẫn đường động tự động trượt theo vị trí hiện tại của xe (Radar),
        hiển thị hướng đầu xe, đường lộ trình A* phía trước và bản đồ thu nhỏ toàn cục.
        """
        poly = self.navigator.route_polyline_px
        has_live_pos = telemetry_pkt is not None and bool(telemetry_pkt.get("IsRaceOn", 0))

        if has_live_pos:
            wx = float(telemetry_pkt.get("PositionX", 0.0))
            wz = float(telemetry_pkt.get("PositionZ", 0.0))
            car_px, car_py = self.navigator.analyzer.world_to_pixel(wx, wz, self.map_w, self.map_h)
            car_yaw = float(telemetry_pkt.get("Yaw", 0.0))
        elif poly and len(poly) >= 2:
            self._sim_progress_idx = (self._sim_progress_idx + 0.65) % max(1, len(poly) - 1)
            idx0 = int(self._sim_progress_idx)
            idx1 = min(len(poly) - 1, idx0 + 1)
            frac = self._sim_progress_idx - idx0
            car_px = int(round(poly[idx0][0] * (1.0 - frac) + poly[idx1][0] * frac))
            car_py = int(round(poly[idx0][1] * (1.0 - frac) + poly[idx1][1] * frac))
            dx_p = poly[idx1][0] - poly[idx0][0]
            dy_p = poly[idx0][1] - poly[idx1][1]
            car_yaw = math.atan2(dx_p, dy_p)
        else:
            sn = self.navigator.nodes[self.navigator.start_node_id]
            car_px, car_py = sn.x_px, sn.y_px
            car_yaw = 0.0

        rad_w = min(self.map_w, int(round(self.quad_w / self.radar_zoom)))
        rad_h = min(self.map_h, int(round(self.quad_h / self.radar_zoom)))
        rx0 = int(np.clip(car_px - rad_w // 2, 0, max(0, self.map_w - rad_w)))
        ry0 = int(np.clip(car_py - rad_h // 2, 0, max(0, self.map_h - rad_h)))

        crop = self.navigator.route_map_bgr[ry0 : ry0 + rad_h, rx0 : rx0 + rad_w]
        tile = cv2.resize(crop, (self.quad_w, self.quad_h), interpolation=cv2.INTER_LINEAR)

        ego_x = int(round((car_px - rx0) / float(max(1, rad_w)) * self.quad_w))
        ego_y = int(round((car_py - ry0) / float(max(1, rad_h)) * self.quad_h))
        cv2.circle(tile, (ego_x, ego_y), 24, (255, 220, 0), 1, cv2.LINE_AA)

        dir_x = math.sin(car_yaw)
        dir_y = -math.cos(car_yaw)
        perp_x = -dir_y
        perp_y = dir_x
        tip = (int(round(ego_x + dir_x * 16)), int(round(ego_y + dir_y * 16)))
        left_w = (int(round(ego_x - dir_x * 10 + perp_x * 9)), int(round(ego_y - dir_y * 10 + perp_y * 9)))
        right_w = (int(round(ego_x - dir_x * 10 - perp_x * 9)), int(round(ego_y - dir_y * 10 - perp_y * 9)))
        arrow_pts = np.array([tip, left_w, right_w], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(tile, [arrow_pts], (0, 255, 120), cv2.LINE_AA)
        cv2.polylines(tile, [arrow_pts], True, (255, 255, 255), 2, cv2.LINE_AA)

        mini = self._cached_mini_map.copy()
        mx_box0 = int(round((rx0 / float(max(1, self.map_w))) * self.mini_w))
        my_box0 = int(round((ry0 / float(max(1, self.map_h))) * self.mini_h))
        mx_box1 = int(round(((rx0 + rad_w) / float(max(1, self.map_w))) * self.mini_w))
        my_box1 = int(round(((ry0 + rad_h) / float(max(1, self.map_h))) * self.mini_h))
        cv2.rectangle(mini, (mx_box0, my_box0), (mx_box1, my_box1), (0, 255, 255), 2)
        mcx = int(round((car_px / float(max(1, self.map_w))) * self.mini_w))
        mcy = int(round((car_py / float(max(1, self.map_h))) * self.mini_h))
        cv2.circle(mini, (mcx, mcy), 4, (0, 255, 0), -1)

        pad = 8
        y_m1 = self.quad_h - self.mini_h - pad
        y_m2 = self.quad_h - pad
        x_m1 = self.quad_w - self.mini_w - pad
        x_m2 = self.quad_w - pad
        tile[y_m1:y_m2, x_m1:x_m2] = mini
        cv2.rectangle(tile, (x_m1, y_m1), (x_m2, y_m2), (255, 255, 255), 1)

        cv2.rectangle(tile, (0, 0), (self.quad_w, 32), (18, 22, 28), -1)
        cv2.line(tile, (0, 32), (self.quad_w, 32), (0, 200, 255), 1)
        cv2.putText(
            tile,
            f"[O 4] QUAN SAT BAN DO & LO TRINH A* ({route_info['start_node']} -> {route_info['dest_node']})",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 220, 110),
            1,
            cv2.LINE_AA,
        )

        banner_w = max(200, x_m1 - 16)
        cv2.rectangle(tile, (8, self.quad_h - 54), (8 + banner_w, self.quad_h - 8), (15, 20, 26), -1)
        cv2.rectangle(tile, (8, self.quad_h - 54), (8 + banner_w, self.quad_h - 8), (0, 200, 255), 1)
        cv2.putText(
            tile,
            f"Next: {active_wp['label']}  |  Dist to Goal: {dist_to_goal_m:.0f} m",
            (16, self.quad_h - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"UK Nav Rule: {turn_instruction}",
            (16, self.quad_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (80, 255, 160),
            1,
            cv2.LINE_AA,
        )
        return tile

    def compose_quad_canvas(
        self,
        q1_game: np.ndarray,
        q2_top_right: np.ndarray,
        q3_telemetry: np.ndarray,
        q4_map: np.ndarray,
    ) -> np.ndarray:
        """Ghép 4 Ô thành 1 khung hình 2x2 chuẩn chia 4 phần màn hình."""
        top_row = np.hstack((q1_game, q2_top_right))
        bot_row = np.hstack((q3_telemetry, q4_map))
        canvas = np.vstack((top_row, bot_row))

        h_total, w_total = canvas.shape[:2]
        cv2.line(canvas, (self.quad_w, 0), (self.quad_w, h_total), (0, 200, 255), 2)
        cv2.line(canvas, (0, self.quad_h), (w_total, self.quad_h), (0, 200, 255), 2)
        return canvas

    def show(
        self,
        q1_game: np.ndarray,
        q2_top_right: np.ndarray,
        q3_telemetry: np.ndarray,
        q4_map: np.ndarray,
        quad_canvas: Optional[np.ndarray] = None,
    ) -> int:
        """
        Hiển thị 4 phần lên màn hình máy tính:
          - `game`: Dành cho khi chạy trực tiếp với Game Forza Horizon 4!
            Chừa trống Ô 1 (Góc Trên-Trái) cho cửa sổ Forza Horizon 4 thật (tự động đưa cửa sổ game về Ô 1),
            và đặt 3 cửa sổ OpenCV vào đúng Ô 2 (Trên-Phải), Ô 3 (Dưới-Trái), Ô 4 (Dưới-Phải).
          - `4windows`: Mở cả 4 cửa sổ độc lập ở 4 góc màn hình.
          - `quad`: Ghép cả 4 ô vào 1 cửa sổ 2x2 (tiện khi chạy mô phỏng với `--video`).
        Trả về mã phím từ `cv2.waitKey(1)`.
        """
        if self.layout in ("game", "4windows"):
            if not self._windows_initialized:
                windows_to_create = [
                    (self.WIN_Q2, self.screen_geom["q2_picker"]),
                    (self.WIN_Q3, self.screen_geom["q3_telemetry"]),
                    (self.WIN_Q4, self.screen_geom["q4_map"]),
                ]
                if self.layout == "4windows":
                    windows_to_create.insert(0, (self.WIN_Q1, self.screen_geom["q1_game"]))

                for wname, (wx, wy, ww, wh) in windows_to_create:
                    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(wname, ww, wh)
                    cv2.moveWindow(wname, wx, wy)
                cv2.setMouseCallback(self.WIN_Q2, self._on_q2_mouse)
                cv2.setMouseCallback(self.WIN_Q4, self._on_q4_mouse)
                _try_snap_fh4_game_window(*self.screen_geom["q1_game"])
                self._windows_initialized = True

            if self.layout == "4windows":
                cv2.imshow(self.WIN_Q1, q1_game)
                if not self._affinity_applied:
                    _try_exclude_window_from_capture(self.WIN_Q1)
                    self._affinity_applied = True

            cv2.imshow(self.WIN_Q2, q2_top_right)
            cv2.imshow(self.WIN_Q3, q3_telemetry)
            cv2.imshow(self.WIN_Q4, q4_map)
        else:
            canvas = quad_canvas if quad_canvas is not None else self.compose_quad_canvas(
                q1_game, q2_top_right, q3_telemetry, q4_map
            )
            if not self._windows_initialized:
                cv2.namedWindow(self.MAIN_WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.MAIN_WINDOW, self.quad_w * 2, self.quad_h * 2)
                cv2.moveWindow(
                    self.MAIN_WINDOW,
                    self.screen_geom["screen_left"],
                    self.screen_geom["screen_top"],
                )
                cv2.setMouseCallback(self.MAIN_WINDOW, self._on_quad_mouse)
                _try_snap_fh4_game_window(*self.screen_geom["q1_game"])
                self._windows_initialized = True

            cv2.imshow(self.MAIN_WINDOW, canvas)
            if not self._affinity_applied:
                _try_exclude_window_from_capture(self.MAIN_WINDOW)
                self._affinity_applied = True

        return cv2.waitKey(1) & 0xFF
