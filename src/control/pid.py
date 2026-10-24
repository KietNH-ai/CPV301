"""
src/control/pid.py — Bộ điều khiển tốc độ PID (Proportional-Integral-Derivative)
và các bộ điều khiển `SpeedGovernor`, `SteeringController` tương thích với `test_fh4/`.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple


class PIDSpeedController:
    """
    PID speed controller with curvature-based target speed reduction and anti-windup.
    Outputs `(throttle, brake)` in `[0.0, 1.0]`.
    """

    def __init__(
        self,
        kp: float = 0.06,
        ki: float = 0.01,
        kd: float = 0.012,
        base_target_speed_kmh: float = 60.0,
        min_target_speed_kmh: float = 25.0,
        max_target_speed_kmh: float = 60.0,
        curvature_sensitivity: float = 20.0,
        speed_deadband_kmh: float = 1.5,
        integral_limit: float = 15.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_target_speed_kmh = base_target_speed_kmh
        self.min_target_speed_kmh = min_target_speed_kmh
        self.max_target_speed_kmh = max_target_speed_kmh
        self.curvature_sensitivity = curvature_sensitivity
        self.speed_deadband_kmh = speed_deadband_kmh
        self.integral_limit = integral_limit

        self._integral: float = 0.0
        self._prev_error: Optional[float] = None
        self._prev_time: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def calculate_target_speed(
        self,
        curvature: Optional[float],
        map_target_speed_kmh: Optional[float] = None,
        path_target_speed_kmh: Optional[float] = None,
    ) -> float:
        target = self.base_target_speed_kmh
        if path_target_speed_kmh is not None:
            target = path_target_speed_kmh
        elif map_target_speed_kmh is not None:
            target = map_target_speed_kmh
        elif curvature is not None:
            target = target / (1.0 + self.curvature_sensitivity * abs(curvature))

        return max(
            self.min_target_speed_kmh,
            min(self.max_target_speed_kmh, target),
        )

    def calculate(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
        dt: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Return `(throttle, brake)` in `[0.0, 1.0]` using PID control."""
        now = time.monotonic()
        if dt is None:
            if self._prev_time is None:
                dt = 1.0 / 30.0
            else:
                dt = max(1e-3, min(0.5, now - self._prev_time))
        self._prev_time = now

        error = float(target_speed_kmh) - float(current_speed_kmh)
        if abs(error) <= self.speed_deadband_kmh:
            self._prev_error = error
            return 0.0, 0.0

        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral + error * dt),
        )
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / max(dt, 1e-3)
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        if u >= 0.0:
            throttle = min(1.0, max(0.0, u))
            return throttle, 0.0
        else:
            brake = min(1.0, max(0.0, -u * 0.85))
            return 0.0, brake


class SpeedGovernor:
    """Basic proportional speed controller with curvature-based speed reduction."""

    def __init__(
        self,
        base_target_speed_kmh: float = 60.0,
        min_target_speed_kmh: float = 25.0,
        max_target_speed_kmh: float = 60.0,
        curvature_sensitivity: float = 20.0,
        speed_deadband_kmh: float = 2.0,
    ):
        self.base_target_speed_kmh = base_target_speed_kmh
        self.min_target_speed_kmh = min_target_speed_kmh
        self.max_target_speed_kmh = max_target_speed_kmh
        self.curvature_sensitivity = curvature_sensitivity
        self.speed_deadband_kmh = speed_deadband_kmh

    def calculate_target_speed(
        self,
        curvature: Optional[float],
        map_target_speed_kmh: Optional[float] = None,
        path_target_speed_kmh: Optional[float] = None,
    ) -> float:
        target = self.base_target_speed_kmh
        if path_target_speed_kmh is not None:
            target = path_target_speed_kmh
        elif map_target_speed_kmh is not None:
            target = map_target_speed_kmh
        elif curvature is not None:
            target = target / (1.0 + self.curvature_sensitivity * abs(curvature))

        return max(
            self.min_target_speed_kmh,
            min(self.max_target_speed_kmh, target),
        )

    def calculate(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
    ) -> Tuple[float, float]:
        """Return (throttle, brake)."""
        speed_error = target_speed_kmh - current_speed_kmh
        if speed_error > self.speed_deadband_kmh:
            throttle = min(1.0, max(0.0, speed_error / 20.0))
            return throttle, 0.0
        if speed_error < -self.speed_deadband_kmh:
            brake = min(1.0, max(0.0, -speed_error / 25.0))
            return 0.0, brake
        return 0.0, 0.0


class SteeringController:
    """Convert heading error into a smoothed steering command."""

    def __init__(
        self,
        kp: float = 70.0,
        smoothing: float = 0.35,
        deadband_deg: float = 1.5,
        max_steering: float = 120.0,
    ):
        self.kp = kp
        self.smoothing = smoothing
        self.deadband_deg = deadband_deg
        self.max_steering = max_steering
        self._smoothed_error_deg: Optional[float] = None

    def reset(self) -> None:
        self._smoothed_error_deg = None

    def calculate(self, heading_error_rad: float, speed_kmh: float = 60.0) -> float:
        error_deg = math.degrees(heading_error_rad)
        if self._smoothed_error_deg is None:
            self._smoothed_error_deg = error_deg
        else:
            self._smoothed_error_deg = (
                self.smoothing * error_deg
                + (1.0 - self.smoothing) * self._smoothed_error_deg
            )
        error = self._smoothed_error_deg
        if abs(error) < self.deadband_deg:
            return 0.0
        steering = self.kp * error
        return max(-self.max_steering, min(self.max_steering, steering))
