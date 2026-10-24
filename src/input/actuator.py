"""
src/input/actuator.py — Bộ truyền động điều khiển xe trong Forza Horizon 4.
Hỗ trợ 3 backend:
  1. "gamepad"  : Tay cầm Xbox 360 ảo qua `vgamepad` (khuyên dùng trên Windows + ViGEmBus)
  2. "keyboard" : Giả lập phím A/D/W/S qua `pydirectinput` (fallback)
  3. "null"     : Chế độ an toàn (`dry_run=True` / `--view-only` / unit test), không gửi lệnh thật
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger("FH4Controller")


class VehicleActuator:
    """
    Send steering, throttle, and brake controls to Forza Horizon 4.

    Steering convention:
        Negative (-120.0 .. 0.0) = steer left
        Positive (0.0 .. +120.0) = steer right
    Throttle / Brake range:
        0.0 .. 1.0
    """

    def __init__(
        self,
        prefer_gamepad: bool = True,
        dry_run: bool = False,
        allow_null_fallback: bool = False,
    ):
        self.backend: Optional[str] = None
        self.gamepad = None
        self.pdi = None
        self.dry_run = dry_run
        self.last_applied: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        if dry_run:
            self.backend = "null"
            return

        if prefer_gamepad:
            try:
                import vgamepad as vg

                self.gamepad = vg.VX360Gamepad()
                self.backend = "gamepad"
                logger.warning("Using virtual Xbox gamepad.")
            except Exception as exc:
                logger.warning("Could not initialize vgamepad: %s", exc)

        if self.backend is None:
            try:
                import pydirectinput as pdi

                self.pdi = pdi
                self.backend = "keyboard"
                logger.warning(
                    "Using keyboard fallback. Analog steering is unavailable."
                )
            except Exception as exc:
                logger.warning("Could not initialize keyboard controls: %s", exc)

        if self.backend is None:
            if allow_null_fallback:
                self.backend = "null"
                logger.warning("Using null actuator backend (dry-run fallback).")
            else:
                raise RuntimeError(
                    "No actuator backend available. Install vgamepad and ViGEmBus, "
                    "install pydirectinput, or pass dry_run=True for view-only mode."
                )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def apply(self, steering: float, throttle: float, brake: float) -> None:
        steering = self._clamp(steering, -120.0, 120.0)
        throttle = self._clamp(throttle, 0.0, 1.0)
        brake = self._clamp(brake, 0.0, 1.0)
        self.last_applied = (steering, throttle, brake)

        if self.backend == "gamepad" and self.gamepad is not None:
            stick_x = int((steering / 120.0) * 32767)
            self.gamepad.left_joystick(x_value=stick_x, y_value=0)
            self.gamepad.right_trigger_float(value_float=throttle)
            self.gamepad.left_trigger_float(value_float=brake)
            self.gamepad.update()
        elif self.backend == "keyboard" and self.pdi is not None:
            self._apply_keyboard(steering, throttle, brake)

    def _apply_keyboard(self, steering: float, throttle: float, brake: float) -> None:
        if steering < -8.0:
            self.pdi.keyUp("d")
            self.pdi.keyDown("a")
        elif steering > 8.0:
            self.pdi.keyUp("a")
            self.pdi.keyDown("d")
        else:
            self.pdi.keyUp("a")
            self.pdi.keyUp("d")

        if throttle > 0.05:
            self.pdi.keyDown("w")
        else:
            self.pdi.keyUp("w")

        if brake > 0.05:
            self.pdi.keyDown("s")
        else:
            self.pdi.keyUp("s")

    def release_all(self) -> None:
        """Release throttle, brake, and steering controls."""
        self.last_applied = (0.0, 0.0, 0.0)
        if self.backend == "gamepad" and self.gamepad is not None:
            self.gamepad.reset()
            self.gamepad.update()
        elif self.backend == "keyboard" and self.pdi is not None:
            for key in ("a", "d", "w", "s"):
                self.pdi.keyUp(key)
