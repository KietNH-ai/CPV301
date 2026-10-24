
"""
fh4_controller.py

Forza Horizon 4 Autonomous Controller
------------------------------------
Responsibilities:
    1. Read vehicle telemetry.
    2. Receive path data from a future vision module.
    3. Receive route data from a future map module.
    4. Calculate steering, throttle and brake.
    5. Send commands through a virtual gamepad or keyboard.
    6. Print status changes and save control data to CSV.

Important:
    - This file does not implement YOLO or line fitting.
    - Vision/map data must be supplied by another module.
    - The controller does not steer without a valid heading.
    - Default target speed: 60 km/h.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import threading
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("FH4Controller")


# ============================================================
# CSV LOGGER
# ============================================================

class FH4CSVLogger:
    """Save controller status and control values to a CSV file."""

    FIELDS = [
        "timestamp",
        "elapsed_s",
        "status",
        "message",
        "speed_kmh",
        "target_speed_kmh",
        "steering",
        "throttle",
        "brake",
        "source",
        "path_confidence",
        "heading_error_rad",
        "curvature",
    ]

    def __init__(self, filename: str = "fh4_log.csv"):
        self.filename = os.path.abspath(filename)
        self._lock = threading.Lock()

        file_exists = (
            os.path.exists(self.filename)
            and os.path.getsize(self.filename) > 0
        )

        self._file = open(
            self.filename,
            "a",
            newline="",
            encoding="utf-8",
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDS,
        )

        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def write(self, **kwargs: Any) -> None:
        row = {field: "" for field in self.FIELDS}
        row.update(kwargs)

        with self._lock:
            if self._file.closed:
                return

            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


# ============================================================
# DATA INTERFACES
# ============================================================

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
    uk_lane_type: str = "unknown"


@dataclass
class MapData:
    """Interface for output from a future route/map module."""

    timestamp: float = 0.0
    valid: bool = False

    heading_error_rad: Optional[float] = None
    target_speed_kmh: Optional[float] = None
    waypoint: Optional[tuple[float, float]] = None
    distance_to_goal_m: Optional[float] = None
    turn_hint: str = "straight"
    uk_instruction: str = ""


@dataclass
class VehicleState:
    """Vehicle state extracted from telemetry."""

    timestamp: float = 0.0
    valid: bool = False

    speed_kmh: float = 0.0
    is_race_on: bool = False

    heading: Optional[float] = None
    yaw_rate: Optional[float] = None
    vehicle_type: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    position_z: Optional[float] = None


@dataclass
class ControlCommand:
    """Final control command sent to the actuator."""

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    target_speed_kmh: float = 0.0
    source: str = "none"


# ============================================================
# THREAD-SAFE INPUT BUFFER
# ============================================================

class InputBuffer:
    """Store the latest path and map data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._path = PathData()
        self._map = MapData()

    def update_path(self, data: PathData) -> None:
        if data.timestamp <= 0:
            data.timestamp = time.monotonic()

        with self._lock:
            self._path = data

    def update_map(self, data: MapData) -> None:
        if data.timestamp <= 0:
            data.timestamp = time.monotonic()

        with self._lock:
            self._map = data

    def get_latest(self) -> tuple[PathData, MapData]:
        with self._lock:
            return self._path, self._map


# ============================================================
# VEHICLE ACTUATOR
# ============================================================

class VehicleActuator:
    """
    Send controls to FH4.

    Preferred backend:
        vgamepad

    Fallback:
        pydirectinput

    Steering convention:
        Negative = left
        Positive = right
    """

    def __init__(self, prefer_gamepad: bool = True):
        self.backend: Optional[str] = None
        self.gamepad = None
        self.pdi = None

        if prefer_gamepad:
            try:
                import vgamepad as vg

                self.gamepad = vg.VX360Gamepad()
                self.backend = "gamepad"

                logger.warning("Using virtual Xbox gamepad.")

            except Exception as exc:
                logger.warning(
                    "Could not initialize vgamepad: %s", exc
                )

        if self.backend is None:
            try:
                import pydirectinput as pdi

                self.pdi = pdi
                self.backend = "keyboard"

                logger.warning(
                    "Using keyboard fallback. "
                    "Analog steering is unavailable."
                )

            except Exception as exc:
                logger.error(
                    "Could not initialize keyboard controls: %s",
                    exc,
                )

        if self.backend is None:
            raise RuntimeError(
                "No actuator backend available. Install vgamepad "
                "and ViGEmBus, or install pydirectinput."
            )

    @staticmethod
    def _clamp(
        value: float,
        low: float,
        high: float,
    ) -> float:
        return max(low, min(high, value))

    def apply(
        self,
        steering: float,
        throttle: float,
        brake: float,
    ) -> None:
        steering = self._clamp(steering, -120.0, 120.0)
        throttle = self._clamp(throttle, 0.0, 1.0)
        brake = self._clamp(brake, 0.0, 1.0)

        if self.backend == "gamepad":
            stick_x = int((steering / 120.0) * 32767)

            self.gamepad.left_joystick(
                x_value=stick_x,
                y_value=0,
            )

            self.gamepad.right_trigger_float(
                value_float=throttle
            )

            self.gamepad.left_trigger_float(
                value_float=brake
            )

            self.gamepad.update()

        elif self.backend == "keyboard":
            self._apply_keyboard(
                steering,
                throttle,
                brake,
            )

    def _apply_keyboard(
        self,
        steering: float,
        throttle: float,
        brake: float,
    ) -> None:
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
        """Release throttle, brake and steering controls."""

        if self.backend == "gamepad":
            self.gamepad.reset()
            self.gamepad.update()

        elif self.backend == "keyboard":
            for key in ("a", "d", "w", "s"):
                self.pdi.keyUp(key)


# ============================================================
# SPEED GOVERNOR
# ============================================================

class SpeedGovernor:
    """Basic speed controller with curvature-based speed reduction."""

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
            target = target / (
                1.0
                + self.curvature_sensitivity * abs(curvature)
            )

        return max(
            self.min_target_speed_kmh,
            min(self.max_target_speed_kmh, target),
        )

    def calculate(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
    ) -> tuple[float, float]:
        """Return (throttle, brake)."""

        speed_error = target_speed_kmh - current_speed_kmh

        if speed_error > self.speed_deadband_kmh:
            throttle = min(
                1.0,
                max(0.0, speed_error / 20.0),
            )
            return throttle, 0.0

        if speed_error < -self.speed_deadband_kmh:
            brake = min(
                1.0,
                max(0.0, -speed_error / 25.0),
            )
            return 0.0, brake

        return 0.0, 0.0


# ============================================================
# STEERING CONTROLLER
# ============================================================

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

    def calculate(self, heading_error_rad: float) -> float:
        error_deg = math.degrees(heading_error_rad)

        if self._smoothed_error_deg is None:
            self._smoothed_error_deg = error_deg
        else:
            self._smoothed_error_deg = (
                self.smoothing * error_deg
                + (1.0 - self.smoothing)
                * self._smoothed_error_deg
            )

        error = self._smoothed_error_deg

        if abs(error) < self.deadband_deg:
            return 0.0

        steering = self.kp * error

        return max(
            -self.max_steering,
            min(self.max_steering, steering),
        )


# ============================================================
# MAIN AUTONOMOUS CONTROLLER
# ============================================================

class FH4AutonomousController:
    """Combine telemetry, path data, control logic and actuation."""

    def __init__(
        self,
        telemetry_receiver: Any,
        actuator: Optional[VehicleActuator] = None,
        input_buffer: Optional[InputBuffer] = None,
        max_path_age: float = 0.35,
        max_map_age: float = 1.0,
        min_path_confidence: float = 0.50,
        base_target_speed_kmh: float = 60.0,
    ):
        self.telemetry = telemetry_receiver
        self.actuator = actuator or VehicleActuator()
        self.inputs = input_buffer or InputBuffer()

        self.max_path_age = max_path_age
        self.max_map_age = max_map_age
        self.min_path_confidence = min_path_confidence

        self.steering_controller = SteeringController()

        self.speed_governor = SpeedGovernor(
            base_target_speed_kmh=base_target_speed_kmh,
        )

        self.vehicle_state = VehicleState()
        self.last_command = ControlCommand()

        self.running = False
        self._stopped = False

        self._start_time = time.monotonic()
        self._last_status: Optional[str] = None
        self._last_message = ""

        self.csv_logger = FH4CSVLogger("fh4_log.csv")

    # --------------------------------------------------------
    # STATUS AND CSV LOGGING
    # --------------------------------------------------------

    def _set_status(
        self,
        status: str,
        message: str = "",
    ) -> None:
        """Print only when the status or message changes."""

        changed = (
            status != self._last_status
            or message != self._last_message
        )

        if not changed:
            return

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"[{status}] {message}",
            flush=True,
        )

        self.csv_logger.write(
            timestamp=datetime.now().isoformat(
                timespec="milliseconds"
            ),
            elapsed_s=round(
                time.monotonic() - self._start_time, 3
            ),
            status=status,
            message=message,
        )

        self._last_status = status
        self._last_message = message

    def _log_control_step(
        self,
        command: ControlCommand,
        path: PathData,
    ) -> None:
        """Write one control iteration to CSV."""

        self.csv_logger.write(
            timestamp=datetime.now().isoformat(
                timespec="milliseconds"
            ),
            elapsed_s=round(
                time.monotonic() - self._start_time, 3
            ),
            status=self._last_status or "UNKNOWN",
            message=self._last_message,
            speed_kmh=round(
                self.vehicle_state.speed_kmh, 2
            ),
            target_speed_kmh=round(
                command.target_speed_kmh, 2
            ),
            steering=round(command.steering, 2),
            throttle=round(command.throttle, 3),
            brake=round(command.brake, 3),
            source=command.source,
            path_confidence=round(path.confidence, 3),
            heading_error_rad=(
                path.heading_error_rad
                if path.heading_error_rad is not None
                else ""
            ),
            curvature=(
                path.curvature
                if path.curvature is not None
                else ""
            ),
        )

    # --------------------------------------------------------
    # INPUT INTERFACES
    # --------------------------------------------------------

    def update_path(self, data: PathData) -> None:
        """Called by the vision/path module."""
        self.inputs.update_path(data)

    def update_map(self, data: MapData) -> None:
        """Called by a future map/route module."""
        self.inputs.update_map(data)

    # --------------------------------------------------------
    # TELEMETRY
    # --------------------------------------------------------

    @staticmethod
    def _get_field(
        packet: Any,
        *names: str,
        default: Any = None,
    ) -> Any:
        """Read a field from a dictionary or an object."""

        for name in names:
            if isinstance(packet, dict) and name in packet:
                return packet[name]

            if hasattr(packet, name):
                return getattr(packet, name)

        return default

    @staticmethod
    def _speed_is_mps(packet: Any) -> bool:
        """Only convert speed fields explicitly named in m/s."""

        if isinstance(packet, dict):
            return (
                "speed_mps" in packet
                and "SpeedKmh" not in packet
                and "speed_kmh" not in packet
            )

        return (
            hasattr(packet, "speed_mps")
            and not hasattr(packet, "SpeedKmh")
            and not hasattr(packet, "speed_kmh")
        )

    def _read_vehicle_state(self) -> VehicleState:
        try:
            result = self.telemetry.get_latest()

            if not isinstance(result, tuple) or len(result) < 3:
                raise ValueError(
                    "Expected get_latest() to return "
                    "(packet, is_fresh, age)."
                )

            packet, is_fresh, age = result

            if packet is None or not is_fresh:
                return VehicleState(valid=False)

            speed = self._get_field(
                packet,
                "SpeedKmh",
                "speed_kmh",
                "speed_mps",
                "speed",
                default=None,
            )

            race_on = self._get_field(
                packet,
                "IsRaceOn",
                "is_race_on",
                default=None,
            )

            if speed is None:
                logger.debug(
                    "Telemetry packet has no speed field."
                )
                return VehicleState(valid=False)

            # Fail safe if the race state is unavailable.
            if race_on is None:
                race_on = False

            speed = float(speed)

            if not math.isfinite(speed) or speed < 0:
                logger.debug("Invalid telemetry speed.")
                return VehicleState(valid=False)

            speed_kmh = speed

            if self._speed_is_mps(packet):
                speed_kmh *= 3.6

            return VehicleState(
                timestamp=time.monotonic() - float(age),
                valid=True,
                speed_kmh=speed_kmh,
                is_race_on=bool(race_on),
                heading=self._get_field(
                    packet,
                    "Yaw",
                    "yaw",
                    "heading",
                ),
                yaw_rate=self._get_field(
                    packet,
                    "AngularVelocityY",
                    "yaw_rate",
                ),
                vehicle_type=self._get_field(
                    packet,
                    "VehicleType",
                    "vehicle_type",
                ),
                position_x=self._get_field(
                    packet,
                    "PositionX",
                    "position_x",
                ),
                position_y=self._get_field(
                    packet,
                    "PositionY",
                    "position_y",
                ),
                position_z=self._get_field(
                    packet,
                    "PositionZ",
                    "position_z",
                ),
            )

        except Exception as exc:
            logger.debug(
                "Telemetry read failed: %s", exc
            )
            return VehicleState(valid=False)

    # --------------------------------------------------------
    # PATH SOURCE SELECTION
    # --------------------------------------------------------

    def _select_heading(
        self,
        path: PathData,
        map_data: MapData,
        now: float,
    ) -> tuple[Optional[float], Optional[float], str]:
        """Prefer fresh vision data; use map data as fallback."""

        path_age = now - path.timestamp

        if (
            path.valid
            and path.heading_error_rad is not None
            and math.isfinite(path.heading_error_rad)
            and path_age >= 0
            and path_age <= self.max_path_age
            and path.confidence >= self.min_path_confidence
        ):
            curvature = path.curvature

            if (
                curvature is not None
                and not math.isfinite(curvature)
            ):
                curvature = None

            return (
                path.heading_error_rad,
                curvature,
                "vision",
            )

        map_age = now - map_data.timestamp

        if (
            map_data.valid
            and map_data.heading_error_rad is not None
            and math.isfinite(map_data.heading_error_rad)
            and map_age >= 0
            and map_age <= self.max_map_age
        ):
            return (
                map_data.heading_error_rad,
                None,
                "map",
            )

        return None, None, "none"

    # --------------------------------------------------------
    # SAFE STOP
    # --------------------------------------------------------

    def _safe_stop(self, reason: str) -> None:
        """
        Release all controls when required data is unavailable.

        Releasing controls is not the same as emergency braking.
        """

        self.actuator.release_all()
        self.steering_controller.reset()

        self.last_command = ControlCommand(
            source="safe_stop"
        )

        self._set_status("SAFE_STOP", reason)

        path, _ = self.inputs.get_latest()
        self._log_control_step(
            self.last_command,
            path,
        )

    # --------------------------------------------------------
    # ONE CONTROL ITERATION
    # --------------------------------------------------------

    def step(self) -> ControlCommand:
        """Read inputs, calculate controls, actuate and log."""

        now = time.monotonic()

        # 1. Read vehicle telemetry.
        self.vehicle_state = self._read_vehicle_state()

        # 2. Read the latest vision and map data.
        path, map_data = self.inputs.get_latest()

        # 3. Validate telemetry.
        if not self.vehicle_state.valid:
            self._safe_stop("Telemetry unavailable")
            return self.last_command

        if not self.vehicle_state.is_race_on:
            self._safe_stop(
                "Race/game state not confirmed"
            )
            return self.last_command

        # 4. Select a fresh, valid heading.
        heading_error, curvature, source = (
            self._select_heading(
                path,
                map_data,
                now,
            )
        )

        if heading_error is None:
            self._safe_stop(
                "No fresh, valid path data"
            )
            return self.last_command

        # 5. Calculate steering.
        steering = self.steering_controller.calculate(
            heading_error
        )

        # 6. Only use the speed recommendation from the
        # source that supplied the selected heading.
        path_speed = (
            path.target_speed_kmh
            if source == "vision"
            else None
        )

        map_speed = (
            map_data.target_speed_kmh
            if source == "map"
            else None
        )

        target_speed = (
            self.speed_governor.calculate_target_speed(
                curvature=curvature,
                map_target_speed_kmh=map_speed,
                path_target_speed_kmh=path_speed,
            )
        )

        # 7. Calculate throttle and brake.
        throttle, brake = self.speed_governor.calculate(
            current_speed_kmh=self.vehicle_state.speed_kmh,
            target_speed_kmh=target_speed,
        )

        # 8. Build the command.
        command = ControlCommand(
            steering=steering,
            throttle=throttle,
            brake=brake,
            target_speed_kmh=target_speed,
            source=source,
        )

        # 9. Send controls to FH4.
        self.actuator.apply(
            steering=command.steering,
            throttle=command.throttle,
            brake=command.brake,
        )

        self.last_command = command

        # 10. Print only when status/message changes.
        self._set_status(
            "RUNNING",
            f"Control source: {source}",
        )

        # 11. Save this iteration to CSV.
        self._log_control_step(
            command,
            path,
        )

        return command

    # --------------------------------------------------------
    # CONTROLLER LIFECYCLE
    # --------------------------------------------------------

    def run(self, loop_hz: float = 30.0) -> None:
        """Run the controller at the requested loop frequency."""

        if self._stopped:
            raise RuntimeError(
                "This controller has already been stopped."
            )

        interval = 1.0 / max(1.0, loop_hz)
        self.running = True

        self._set_status(
            "STARTED",
            f"Loop={loop_hz:.1f} Hz, "
            f"backend={self.actuator.backend}",
        )

        try:
            while self.running:
                started = time.monotonic()

                try:
                    self.step()

                except Exception as exc:
                    logger.exception(
                        "Controller step failed: %s", exc
                    )

                    try:
                        self._safe_stop(
                            "Controller step exception"
                        )
                    except Exception:
                        logger.exception(
                            "Failed to release controls."
                        )
                        self.running = False

                elapsed = time.monotonic() - started
                time.sleep(
                    max(0.0, interval - elapsed)
                )

        except KeyboardInterrupt:
            self._set_status(
                "INTERRUPTED",
                "Keyboard interrupt received",
            )

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop safely and close resources only once."""

        if self._stopped:
            return

        self._stopped = True
        self.running = False

        try:
            self.actuator.release_all()

        except Exception:
            logger.exception(
                "Failed to release actuator controls."
            )

        try:
            if hasattr(self.telemetry, "stop"):
                self.telemetry.stop()

        except Exception:
            logger.exception(
                "Failed to stop telemetry receiver."
            )

        finally:
            try:
                self._set_status(
                    "STOPPED",
                    "Controller stopped",
                )
            finally:
                self.csv_logger.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Start telemetry and the autonomous controller."""

    from fh4_telemetry import TelemetryReceiver

    receiver = TelemetryReceiver(
        host="0.0.0.0",
        port=9999,
        stale_timeout=0.4,
    )

    receiver.start()

    controller = None

    try:
        controller = FH4AutonomousController(
            telemetry_receiver=receiver,
            max_path_age=0.35,
            max_map_age=1.0,
            min_path_confidence=0.50,
            base_target_speed_kmh=60.0,
        )

        # Vision/map integration will be added separately.
        #
        # Example only; this is NOT real model output:
        #
        # controller.update_path(
        #     PathData(
        #         timestamp=time.monotonic(),
        #         valid=True,
        #         confidence=0.90,
        #         heading_error_rad=0.05,
        #         curvature=0.01,
        #     )
        # )

        controller.run(loop_hz=30.0)

    finally:
        if controller is not None:
            controller.stop()
        elif hasattr(receiver, "stop"):
            receiver.stop()


if __name__ == "__main__":
    main()
