"""
screen_capture.py — Wrapper mss để chụp vùng game Forza Horizon 4.

Cách dùng:
    cap = ScreenCapture(region=(0, 0, 1280, 720))   # left, top, width, height
    frame = cap.grab()                              # np.ndarray BGR, sẵn sàng cho OpenCV
    print(cap.fps)
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

try:
    import mss
except ImportError:  # pragma: no cover - phụ thuộc môi trường
    mss = None


class ScreenCapture:
    """Chụp màn hình liên tục, đo FPS, hỗ trợ cắt vùng (ROI) game."""

    def __init__(
        self,
        monitor: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
        fps_window: int = 30,
    ):
        """
        Args:
            monitor: Số thứ tự màn hình (1 = màn chính).
            region: (left, top, width, height) vùng cần cắt. None = toàn màn hình.
            fps_window: Số frame để tính FPS trung bình trượt.
        """
        if mss is None:
            raise ImportError(
                "Thiếu thư viện 'mss'. Cài: pip install mss (đã có trong requirements.txt)."
            )
        self._sct = mss.mss()
        self.monitor = monitor
        self.region = region  # (left, top, w, h)
        self._times: deque = deque(maxlen=fps_window)
        self._monitor_info = self._sct.monitors[monitor] if monitor < len(self._sct.monitors) else self._sct.monitors[0]

    @property
    def capture_geometry(self) -> dict:
        """Dict {left, top, width, height} dùng cho mss.grab."""
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
        """Chụp 1 frame -> np.ndarray BGR (uint8)."""
        geom = self.capture_geometry
        raw = self._sct.grab(geom)
        # mss trả về BGRA; bỏ alpha -> BGR
        frame = np.array(raw, dtype=np.uint8)[..., :3]
        now = time.monotonic()
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
