"""
src/control/controller.py — Bộ điều khiển tự hành trung tâm `FH4AutonomousController`
tích hợp:
  - Phân loại làn đường chuẩn Anh Quốc (UK Left-Hand Traffic) từ `PathData`
  - Định tuyến bản đồ toàn cục từ `FH4MapNavigator` (`MapData`) + tọa độ `PositionX/Z, Yaw`
  - Pure Pursuit + PID + VehicleActuator + CSV Logger (xuất tại `ROOT_DIR / fh4_log.csv`)
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
from typing import Any, Optional, Tuple

from src.control.pure_pursuit import PathData, PurePursuitController
from src.control.pid import PIDSpeedController, SpeedGovernor, SteeringController
from src.input.actuator import VehicleActuator

logger = logging.getLogger("FH4Controller")


class FH4CSVLogger:
    """Save controller status, telemetry coordinates, and control values to a CSV file."""

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
        "uk_lane_type",
        "position_x",
        "position_z",
        "distance_to_goal_m",
        "turn_instruction",
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
            extrasaction="ignore",
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


@dataclass
class MapData:
    """Interface for output from the FH4 UK route/map module (`FH4MapNavigator`)."""

    timestamp: float = 0.0
    valid: bool = False
    heading_error_rad: Optional[float] = None
    target_speed_kmh: Optional[float] = None
    waypoint: Optional[Tuple[float, float]] = None
    turn_instruction: str = "KEEP_LEFT_LANE_UK"
    distance_to_goal_m: Optional[float] = None


@dataclass
class VehicleState:
    """Vehicle state extracted from FH4 UDP telemetry (including 3D world coordinates)."""

    timestamp: float = 0.0
    valid: bool = False
    speed_kmh: float = 0.0
    is_race_on: bool = False
    heading: Optional[float] = None
    yaw_rate: Optional[float] = None
    vehicle_type: Optional[str] = None
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0


@dataclass
class ControlCommand:
    """Final control command sent to the actuator."""

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed_kmh: float = 0.0
    source: str = "none"


class InputBuffer:
    """Thread-safe store for the latest path and map data."""

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

    def get_latest(self) -> Tuple[PathData, MapData]:
        with self._lock:
            return self._path, self._map


class FH4AutonomousController:
    """
    Combine FH4 UDP telemetry (with PositionX/Z & Yaw), UK vision lane keeping (`PathData`),
    global map route planning (`FH4MapNavigator` / `MapData`), Pure Pursuit + PID control, and actuation.
    """

    def __init__(
        self,
        telemetry_receiver: Any,
        actuator: Optional[VehicleActuator] = None,
        input_buffer: Optional[InputBuffer] = None,
        map_navigator: Optional[Any] = None,
        max_path_age: float = 0.35,
        max_map_age: float = 1.0,
        min_path_confidence: float = 0.50,
        base_target_speed_kmh: float = 60.0,
        arrival_tolerance_m: float = 15.0,
        use_pure_pursuit: bool = True,
        use_pid: bool = True,
        dry_run: bool = False,
        log_file: str = "fh4_log.csv",
    ):
        self.telemetry = telemetry_receiver
        self.actuator = actuator or VehicleActuator(dry_run=dry_run, allow_null_fallback=dry_run)
        self.inputs = input_buffer or InputBuffer()
        self.map_navigator = map_navigator

        self.max_path_age = max_path_age
        self.max_map_age = max_map_age
        self.min_path_confidence = min_path_confidence
        self.arrival_tolerance_m = arrival_tolerance_m

        if use_pure_pursuit:
            self.steering_controller: Any = PurePursuitController()
        else:
            self.steering_controller = SteeringController()

        if use_pid:
            self.speed_governor: Any = PIDSpeedController(
                base_target_speed_kmh=base_target_speed_kmh,
                max_target_speed_kmh=max(60.0, base_target_speed_kmh),
            )
        else:
            self.speed_governor = SpeedGovernor(
                base_target_speed_kmh=base_target_speed_kmh,
                max_target_speed_kmh=max(60.0, base_target_speed_kmh),
            )

        self.vehicle_state = VehicleState()
        self.last_command = ControlCommand()

        self.running = False
        self._stopped = False

        self._start_time = time.monotonic()
        self._last_status: Optional[str] = None
        self._last_message = ""

        self.csv_logger = FH4CSVLogger(log_file)

    def _set_status(self, status: str, message: str = "") -> None:
        changed = status != self._last_status or message != self._last_message
        if not changed:
            return

        print(f"[{time.strftime('%H:%M:%S')}] [{status}] {message}", flush=True)
        self.csv_logger.write(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            elapsed_s=round(time.monotonic() - self._start_time, 3),
            status=status,
            message=message,
        )
        self._last_status = status
        self._last_message = message

    def _log_control_step(self, command: ControlCommand, path: PathData, map_data: Optional[MapData] = None) -> None:
        self.csv_logger.write(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            elapsed_s=round(time.monotonic() - self._start_time, 3),
            status=self._last_status or "UNKNOWN",
            message=self._last_message,
            speed_kmh=round(self.vehicle_state.speed_kmh, 2),
            target_speed_kmh=round(command.target_speed_kmh, 2),
            steering=round(command.steering, 2),
            throttle=round(command.throttle, 3),
            brake=round(command.brake, 3),
            source=command.source,
            path_confidence=round(path.confidence, 3),
            heading_error_rad=path.heading_error_rad if path.heading_error_rad is not None else "",
            curvature=path.curvature if path.curvature is not None else "",
            uk_lane_type=getattr(path, "uk_lane_type", ""),
            position_x=round(self.vehicle_state.position_x, 2),
            position_z=round(self.vehicle_state.position_z, 2),
            distance_to_goal_m=(
                round(map_data.distance_to_goal_m, 2)
                if map_data and map_data.distance_to_goal_m is not None
                else ""
            ),
            turn_instruction=map_data.turn_instruction if map_data else "",
        )

    def update_path(self, data: PathData) -> None:
        self.inputs.update_path(data)

    def update_map(self, data: MapData) -> None:
        self.inputs.update_map(data)

    @staticmethod
    def _get_field(packet: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(packet, dict) and name in packet:
                return packet[name]
            if hasattr(packet, name):
                return getattr(packet, name)
        return default

    @staticmethod
    def _speed_is_mps(packet: Any) -> bool:
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
            if self.telemetry is None:
                return VehicleState(valid=False)
            result = self.telemetry.get_latest()
            if not isinstance(result, tuple) or len(result) < 3:
                raise ValueError("Expected get_latest() to return (packet, is_fresh, age).")
            packet, is_fresh, age = result
            if packet is None or not is_fresh:
                return VehicleState(valid=False)

            speed = self._get_field(
                packet, "SpeedKmh", "speed_kmh", "speed_mps", "speed", default=None
            )
            race_on = self._get_field(packet, "IsRaceOn", "is_race_on", default=None)
            if speed is None:
                return VehicleState(valid=False)
            if race_on is None:
                race_on = False

            speed = float(speed)
            if not math.isfinite(speed) or speed < 0:
                return VehicleState(valid=False)

            speed_kmh = speed * 3.6 if self._speed_is_mps(packet) else speed
            pos_x = float(self._get_field(packet, "PositionX", "position_x", default=0.0) or 0.0)
            pos_y = float(self._get_field(packet, "PositionY", "position_y", default=0.0) or 0.0)
            pos_z = float(self._get_field(packet, "PositionZ", "position_z", default=0.0) or 0.0)

            return VehicleState(
                timestamp=time.monotonic() - float(age),
                valid=True,
                speed_kmh=speed_kmh,
                is_race_on=bool(race_on),
                heading=self._get_field(packet, "Yaw", "yaw", "heading"),
                yaw_rate=self._get_field(packet, "AngularVelocityY", "yaw_rate"),
                vehicle_type=self._get_field(packet, "VehicleType", "vehicle_type"),
                position_x=pos_x,
                position_y=pos_y,
                position_z=pos_z,
            )
        except Exception as exc:
            logger.debug("Telemetry read failed: %s", exc)
            return VehicleState(valid=False)

    def _refresh_map_from_navigator(self) -> None:
        """Nếu có `map_navigator` và xe có tọa độ telemetry hợp lệ, tự động cập nhật `MapData`."""
        if self.map_navigator is None or not self.vehicle_state.valid:
            return
        if not getattr(self.map_navigator, "route_waypoints", None):
            return
        yaw = float(self.vehicle_state.heading or 0.0)
        guidance = self.map_navigator.compute_map_guidance(
            position_x=self.vehicle_state.position_x,
            position_z=self.vehicle_state.position_z,
            yaw_rad=yaw,
        )
        if guidance.get("valid"):
            self.update_map(
                MapData(
                    timestamp=guidance["timestamp"],
                    valid=True,
                    heading_error_rad=guidance["heading_error_rad"],
                    target_speed_kmh=guidance["target_speed_kmh"],
                    waypoint=guidance["waypoint"],
                    turn_instruction=guidance["turn_instruction"],
                    distance_to_goal_m=guidance["distance_to_goal_m"],
                )
            )

    def _select_heading(
        self,
        path: PathData,
        map_data: MapData,
        now: float,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Kết hợp dữ liệu Giữ làn UK (`path`) và Định tuyến Bản đồ (`map_data`):
          - Nếu cả hai đều hợp lệ: trộn có trọng số (`vision+map_uk`) để xe vừa bám làn bên trái
            chuẩn Anh Quốc vừa đi đúng hướng lộ trình tại các ngã rẽ/bùng binh.
          - Nếu chỉ có `vision`: dùng `vision`.
          - Nếu chỉ có `map`: dùng `map`.
        """
        path_age = now - path.timestamp
        vision_ok = (
            path.valid
            and path.heading_error_rad is not None
            and math.isfinite(path.heading_error_rad)
            and 0 <= path_age <= self.max_path_age
            and path.confidence >= self.min_path_confidence
        )

        map_age = now - map_data.timestamp
        map_ok = (
            map_data.valid
            and map_data.heading_error_rad is not None
            and math.isfinite(map_data.heading_error_rad)
            and 0 <= map_age <= self.max_map_age
        )

        if vision_ok and map_ok:
            curvature = path.curvature
            if curvature is not None and not math.isfinite(curvature):
                curvature = None
            # Khi tới giao lộ cần rẽ (góc lệch bản đồ lớn), tăng trọng số của bản đồ
            w_map = 0.55 if abs(float(map_data.heading_error_rad)) > 0.22 else 0.25
            fused_heading = (
                (1.0 - w_map) * float(path.heading_error_rad)
                + w_map * float(map_data.heading_error_rad)
            )
            return fused_heading, curvature, "vision+map_uk"

        if vision_ok:
            curvature = path.curvature
            if curvature is not None and not math.isfinite(curvature):
                curvature = None
            return path.heading_error_rad, curvature, "vision"

        if map_ok:
            return map_data.heading_error_rad, None, "map"

        return None, None, "none"

    def _safe_stop(self, reason: str, status: str = "SAFE_STOP") -> None:
        self.actuator.release_all()
        self.steering_controller.reset()
        if hasattr(self.speed_governor, "reset"):
            self.speed_governor.reset()
        self.last_command = ControlCommand(source=status.lower())
        self._set_status(status, reason)
        path, map_data = self.inputs.get_latest()
        self._log_control_step(self.last_command, path, map_data)

    def step(self) -> ControlCommand:
        """Read inputs, update map navigation from telemetry, calculate controls, actuate and log."""
        now = time.monotonic()
        self.vehicle_state = self._read_vehicle_state()

        if not self.vehicle_state.valid:
            self._safe_stop("Telemetry unavailable")
            return self.last_command

        if not self.vehicle_state.is_race_on:
            self._safe_stop("Race/game state not confirmed")
            return self.last_command

        # Tự động cập nhật lộ trình bản đồ từ tọa độ Telemetry nếu có map_navigator
        self._refresh_map_from_navigator()
        path, map_data = self.inputs.get_latest()

        # Kiểm tra xem xe đã tới điểm đích được đánh dấu trên bản đồ chưa
        if (
            map_data.valid
            and map_data.distance_to_goal_m is not None
            and map_data.distance_to_goal_m <= self.arrival_tolerance_m
        ):
            self._safe_stop(
                f"Arrived at destination (dist={map_data.distance_to_goal_m:.1f}m)",
                status="ARRIVED",
            )
            return self.last_command

        heading_error, curvature, source = self._select_heading(path, map_data, now)
        if heading_error is None:
            self._safe_stop("No fresh, valid path data")
            return self.last_command

        try:
            steering = self.steering_controller.calculate(
                heading_error, speed_kmh=self.vehicle_state.speed_kmh
            )
        except TypeError:
            steering = self.steering_controller.calculate(heading_error)

        path_speed = path.target_speed_kmh if "vision" in source else None
        map_speed = map_data.target_speed_kmh if "map" in source else None
        target_speed = self.speed_governor.calculate_target_speed(
            curvature=curvature,
            map_target_speed_kmh=map_speed,
            path_target_speed_kmh=path_speed,
        )

        throttle, brake = self.speed_governor.calculate(
            current_speed_kmh=self.vehicle_state.speed_kmh,
            target_speed_kmh=target_speed,
        )

        command = ControlCommand(
            steering=steering,
            throttle=throttle,
            brake=brake,
            target_speed_kmh=target_speed,
            source=source,
        )
        self.actuator.apply(
            steering=command.steering,
            throttle=command.throttle,
            brake=command.brake,
        )
        self.last_command = command
        self._set_status("RUNNING", f"Control source: {source}")
        self._log_control_step(command, path, map_data)
        return command

    def run(self, loop_hz: float = 30.0) -> None:
        if self._stopped:
            raise RuntimeError("This controller has already been stopped.")
        interval = 1.0 / max(1.0, loop_hz)
        self.running = True
        self._set_status("STARTED", f"Loop={loop_hz:.1f} Hz, backend={self.actuator.backend}")
        try:
            while self.running:
                started = time.monotonic()
                try:
                    self.step()
                except Exception as exc:
                    logger.exception("Controller step failed: %s", exc)
                    try:
                        self._safe_stop("Controller step exception")
                    except Exception:
                        self.running = False
                elapsed = time.monotonic() - started
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            self._set_status("INTERRUPTED", "Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        try:
            self.actuator.release_all()
        except Exception:
            logger.exception("Failed to release actuator controls.")
        try:
            if hasattr(self.telemetry, "stop"):
                self.telemetry.stop()
        except Exception:
            logger.exception("Failed to stop telemetry receiver.")
        finally:
            try:
                self._set_status("STOPPED", "Controller stopped")
            finally:
                self.csv_logger.close()
