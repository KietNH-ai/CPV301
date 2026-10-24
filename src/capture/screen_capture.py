"""
screen_capture.py — Wrapper mss tốc độ cao (>= 60 FPS) để chụp vùng game Forza Horizon 4.

Cách dùng:
    cap = ScreenCapture(region=(0, 0, 960, 540))    # left, top, width, height
    frame = cap.grab()                              # np.ndarray BGR liên tục (contiguous), sẵn sàng cho OpenCV
    print(cap.fps)
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mss
except ImportError:  # pragma: no cover - phụ thuộc môi trường
    mss = None


def get_screen_quadrant_geometry(monitor: int = 1, taskbar_margin: int = 48) -> dict:
    """
    Trả về kích thước màn hình và tọa độ 4 Ô (2x2 Quadrant Grid):
      - q1_game:      Góc Trên-Trái  (Cửa sổ Game)
      - q2_picker:    Góc Trên-Phải  (Chọn điểm đến - có thanh cuộn/zoom)
      - q3_telemetry: Góc Dưới-Trái  (Quan sát Telemetry & Điều khiển)
      - q4_map:       Góc Dưới-Phải  (Quan sát Bản đồ & Lộ trình A*)
    """
    left, top, width, height = 0, 0, 1920, 1080
    if mss is not None:
        try:
            with mss.mss() as sct:
                mon = sct.monitors[monitor] if monitor < len(sct.monitors) else sct.monitors[0]
                left = int(mon["left"])
                top = int(mon["top"])
                width = int(mon["width"])
                height = int(mon["height"])
        except Exception:
            pass

    usable_h = max(480, height - taskbar_margin)
    qw = max(320, width // 2)
    qh = max(240, usable_h // 2)

    return {
        "screen_left": left,
        "screen_top": top,
        "screen_width": width,
        "screen_height": height,
        "usable_height": usable_h,
        "quad_width": qw,
        "quad_height": qh,
        "q1_game": (left, top, qw, qh),
        "q2_picker": (left + qw, top, qw, qh),
        "q3_telemetry": (left, top + qh, qw, qh),
        "q4_map": (left + qw, top + qh, qw, qh),
    }


def get_fh4_window_client_region() -> Optional[Tuple[int, int, int, int]]:
    """
    Tự động tìm vùng Client Area thực tế (không tính thanh tiêu đề/viền cửa sổ hay thanh VS Code bên ngoài)
    của cửa sổ 'Forza Horizon 4' trên Windows nếu game đang mở.
    """
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "Forza Horizon 4")
        if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return None

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        class POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        rect = RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        pt = POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None

        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        if w >= 320 and h >= 240:
            return (int(pt.x), int(pt.y), w, h)
    except Exception:
        pass
    return None


class ScreenCapture:
    """Chụp màn hình liên tục tốc độ cao, đo FPS, hỗ trợ cắt vùng (ROI) game."""

    def __init__(
        self,
        monitor: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
        fps_window: int = 30,
        auto_track_fh4_window: bool = True,
    ):
        """
        Args:
            monitor: Số thứ tự màn hình (1 = màn chính).
            region: (left, top, width, height) vùng cần cắt. None = toàn màn hình.
            fps_window: Số frame để tính FPS trung bình trượt.
            auto_track_fh4_window: Tự động bám theo vùng Client của cửa sổ 'Forza Horizon 4' nếu tìm thấy.
        """
        if mss is None:
            raise ImportError(
                "Thiếu thư viện 'mss'. Cài: pip install mss (đã có trong requirements.txt)."
            )
        self._sct = mss.mss()
        self.monitor = monitor
        self.region = region  # (left, top, w, h)
        self.auto_track_fh4_window = auto_track_fh4_window
        self._times: deque = deque(maxlen=fps_window)
        self._frame_idx: int = 0
        self._fh4_client_region: Optional[Tuple[int, int, int, int]] = (
            get_fh4_window_client_region() if auto_track_fh4_window else None
        )
        self._monitor_info = (
            self._sct.monitors[monitor]
            if monitor < len(self._sct.monitors)
            else self._sct.monitors[0]
        )

    @property
    def capture_geometry(self) -> dict:
        """Dict {left, top, width, height} dùng cho mss.grab."""
        if self.auto_track_fh4_window and self._fh4_client_region is not None:
            left, top, w, h = self._fh4_client_region
            return {"left": int(left), "top": int(top), "width": int(w), "height": int(h)}
        if self.region is not None:
            left, top, w, h = self.region
            return {"left": int(left), "top": int(top), "width": int(w), "height": int(h)}
        return {
            "left": self._monitor_info["left"],
            "top": self._monitor_info["top"],
            "width": self._monitor_info["width"],
            "height": self._monitor_info["height"],
        }

    def grab(self) -> np.ndarray:
        """Chụp 1 frame -> np.ndarray BGR (uint8) liên tục trong bộ nhớ (SIMD BGRA->BGR)."""
        if self.auto_track_fh4_window and (self._frame_idx % 60 == 0):
            self._fh4_client_region = get_fh4_window_client_region()
        self._frame_idx += 1
        geom = self.capture_geometry
        raw = self._sct.grab(geom)
        frame = cv2.cvtColor(np.asarray(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
        now = time.perf_counter()
        self._times.append(now)
        return frame

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0:
            return 0.0
        return (len(self._times) - 1) / dt

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
