"""
Forza Horizon 4 Telemetry Module.

Receives FH4 UDP Data Out packets.
Parses the complete packet layout and exposes vehicle state, speed,
drivetrain, and 3D world coordinates (PositionX/Y/Z, Yaw/Pitch/Roll)
needed by the autonomous driving controller and UK Map Navigator.
"""

import socket
import struct
import threading
import time
from typing import Dict, Optional, Tuple, Any


# FH4 Data Out packet layout.
# Original layout: 323 parsed bytes + 1 trailing byte = 324 bytes.
PACKET_FIELDS = [
    ("s32", "IsRaceOn"),
    ("u32", "TimestampMS"),
    ("f32", "EngineMaxRpm"),
    ("f32", "EngineIdleRpm"),
    ("f32", "CurrentEngineRpm"),

    ("f32", "AccelerationX"),
    ("f32", "AccelerationY"),
    ("f32", "AccelerationZ"),

    ("f32", "VelocityX"),
    ("f32", "VelocityY"),
    ("f32", "VelocityZ"),

    ("f32", "AngularVelocityX"),
    ("f32", "AngularVelocityY"),
    ("f32", "AngularVelocityZ"),

    ("f32", "Yaw"),
    ("f32", "Pitch"),
    ("f32", "Roll"),

    ("f32", "NormalizedSuspensionTravelFrontLeft"),
    ("f32", "NormalizedSuspensionTravelFrontRight"),
    ("f32", "NormalizedSuspensionTravelRearLeft"),
    ("f32", "NormalizedSuspensionTravelRearRight"),

    ("f32", "TireSlipRatioFrontLeft"),
    ("f32", "TireSlipRatioFrontRight"),
    ("f32", "TireSlipRatioRearLeft"),
    ("f32", "TireSlipRatioRearRight"),

    ("f32", "WheelRotationSpeedFrontLeft"),
    ("f32", "WheelRotationSpeedFrontRight"),
    ("f32", "WheelRotationSpeedRearLeft"),
    ("f32", "WheelRotationSpeedRearRight"),

    ("s32", "WheelOnRumbleStripFrontLeft"),
    ("s32", "WheelOnRumbleStripFrontRight"),
    ("s32", "WheelOnRumbleStripRearLeft"),
    ("s32", "WheelOnRumbleStripRearRight"),

    ("f32", "WheelInPuddleDepthFrontLeft"),
    ("f32", "WheelInPuddleDepthFrontRight"),
    ("f32", "WheelInPuddleDepthRearLeft"),
    ("f32", "WheelInPuddleDepthRearRight"),

    ("f32", "SurfaceRumbleFrontLeft"),
    ("f32", "SurfaceRumbleFrontRight"),
    ("f32", "SurfaceRumbleRearLeft"),
    ("f32", "SurfaceRumbleRearRight"),

    ("f32", "TireSlipAngleFrontLeft"),
    ("f32", "TireSlipAngleFrontRight"),
    ("f32", "TireSlipAngleRearLeft"),
    ("f32", "TireSlipAngleRearRight"),

    ("f32", "TireCombinedSlipFrontLeft"),
    ("f32", "TireCombinedSlipFrontRight"),
    ("f32", "TireCombinedSlipRearLeft"),
    ("f32", "TireCombinedSlipRearRight"),

    ("f32", "SuspensionTravelMetersFrontLeft"),
    ("f32", "SuspensionTravelMetersFrontRight"),
    ("f32", "SuspensionTravelMetersRearLeft"),
    ("f32", "SuspensionTravelMetersRearRight"),

    ("s32", "CarOrdinal"),
    ("s32", "CarClass"),
    ("s32", "CarPerformanceIndex"),
    ("s32", "DrivetrainType"),
    ("s32", "NumCylinders"),

    ("hzn", "HorizonPlaceholder"),

    ("f32", "PositionX"),
    ("f32", "PositionY"),
    ("f32", "PositionZ"),

    ("f32", "Speed"),
    ("f32", "Power"),
    ("f32", "Torque"),

    ("f32", "TireTempFrontLeft"),
    ("f32", "TireTempFrontRight"),
    ("f32", "TireTempRearLeft"),
    ("f32", "TireTempRearRight"),

    ("f32", "Boost"),
    ("f32", "Fuel"),
    ("f32", "DistanceTraveled"),

    ("f32", "BestLap"),
    ("f32", "LastLap"),
    ("f32", "CurrentLap"),
    ("f32", "CurrentRaceTime"),

    ("u16", "LapNumber"),
    ("u8", "RacePosition"),

    ("u8", "Accel"),
    ("u8", "Brake"),
    ("u8", "Clutch"),
    ("u8", "HandBrake"),
    ("u8", "Gear"),

    ("s8", "Steer"),
    ("s8", "NormalizedDrivingLine"),
    ("s8", "NormalizedAIBrakeDifference"),
]


TYPE_INFO = {
    "s32": ("<i", 4),
    "u32": ("<I", 4),
    "f32": ("<f", 4),
    "u16": ("<H", 2),
    "u8":  ("<B", 1),
    "s8":  ("<b", 1),
    "hzn": (None, 12),
}

EXPECTED_PARSED_BYTES = sum(
    TYPE_INFO[kind][1] for kind, _ in PACKET_FIELDS
)


def decode_fh4_packet(data: bytes) -> Dict[str, Any]:
    """Decode the complete FH4 packet and return selected fields (including 3D coordinates & orientation)."""

    if len(data) < EXPECTED_PARSED_BYTES:
        raise ValueError(
            f"Packet too short: {len(data)} bytes; "
            f"expected at least {EXPECTED_PARSED_BYTES}"
        )

    raw: Dict[str, Any] = {}
    offset = 0

    # Parse every field to preserve the original byte offsets.
    for kind, name in PACKET_FIELDS:
        fmt, size = TYPE_INFO[kind]

        if offset + size > len(data):
            raise ValueError(
                f"Packet truncated at {name}, offset={offset}"
            )

        if kind == "hzn":
            raw[name] = data[offset:offset + size].hex()
        else:
            raw[name] = struct.unpack_from(fmt, data, offset)[0]

        offset += size

    # Convert only the fields required by the controller and map navigator.
    speed_mps = float(raw["Speed"])
    gear_raw = int(raw["Gear"])
    drivetrain_type = int(raw["DrivetrainType"])

    if gear_raw == 0:
        gear_label = "R"
    else:
        gear_label = str(gear_raw)

    drivetrain_map = {
        0: "FWD",
        1: "RWD",
        2: "AWD",
    }

    return {
        "IsRaceOn": int(raw["IsRaceOn"]),
        "TimestampMS": int(raw["TimestampMS"]),
        "Steer": int(raw["Steer"]),
        "CurrentEngineRpm": float(raw["CurrentEngineRpm"]),
        "Gear": gear_label,
        "GearRaw": gear_raw,
        "Speed": speed_mps,
        "SpeedKmh": max(0.0, speed_mps * 3.6),
        "DrivetrainType": drivetrain_type,
        "Drivetrain": drivetrain_map.get(
            drivetrain_type, "Unknown"
        ),
        "PositionX": float(raw["PositionX"]),
        "PositionY": float(raw["PositionY"]),
        "PositionZ": float(raw["PositionZ"]),
        "Yaw": float(raw["Yaw"]),
        "Pitch": float(raw["Pitch"]),
        "Roll": float(raw["Roll"]),
        "VelocityX": float(raw["VelocityX"]),
        "VelocityY": float(raw["VelocityY"]),
        "VelocityZ": float(raw["VelocityZ"]),
        "AngularVelocityY": float(raw["AngularVelocityY"]),
        "NormalizedDrivingLine": int(raw["NormalizedDrivingLine"]),
        "NormalizedAIBrakeDifference": int(raw["NormalizedAIBrakeDifference"]),
    }


class TelemetryReceiver:
    """Thread-safe UDP telemetry receiver for FH4."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9999,
        stale_timeout: float = 0.4,
    ):
        self.host = host
        self.port = port
        self.stale_timeout = stale_timeout

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        self._latest_packet: Optional[Dict[str, Any]] = None
        self._last_received_time = 0.0
        self._packet_count = 0
        self._error_msg: Optional[str] = None

    def start(self) -> bool:
        """Start the background UDP listener."""

        if self._running:
            return True

        try:
            sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            )
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
            self._sock = sock

        except OSError as exc:
            self._error_msg = (
                f"Failed to bind UDP port {self.port}: {exc}"
            )
            print(f"[TelemetryReceiver] {self._error_msg}")

            if self._sock:
                self._sock.close()
                self._sock = None

            return False

        self._running = True
        self._error_msg = None

        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="FH4-Telemetry-Worker",
        )
        self._thread.start()

        print(
            f"[TelemetryReceiver] Listening on "
            f"UDP {self.host}:{self.port}"
        )
        print(
            f"[TelemetryReceiver] Expected parsed bytes: "
            f"{EXPECTED_PARSED_BYTES}"
        )

        return True

    def stop(self) -> None:
        """Stop the listener and release its socket."""

        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        print("\n[TelemetryReceiver] Stopped.")

    def _listen_loop(self) -> None:
        while self._running:
            sock = self._sock
            if sock is None:
                break

            try:
                data, addr = sock.recvfrom(2048)

                # Decode before updating the latest valid packet.
                packet = decode_fh4_packet(data)

                now = time.monotonic()

                with self._lock:
                    self._latest_packet = packet
                    self._last_received_time = now
                    self._packet_count += 1
                    self._error_msg = None

            except socket.timeout:
                continue

            except OSError as exc:
                if self._running:
                    with self._lock:
                        self._error_msg = str(exc)
                    print(f"\n[UDP ERROR] {exc}")
                break

            except Exception as exc:
                message = (
                    f"{type(exc).__name__}: {exc}; "
                    f"packet bytes={len(data) if 'data' in locals() else 'unknown'}"
                )
                with self._lock:
                    self._error_msg = message

                print(f"\n[TELEMETRY DECODE ERROR] {message}")

    def get_latest(
        self,
    ) -> Tuple[Optional[Dict[str, Any]], bool, float]:
        """
        Returns:
            packet, is_fresh, age_in_seconds
        """

        now = time.monotonic()

        with self._lock:
            packet = (
                self._latest_packet.copy()
                if self._latest_packet is not None
                else None
            )
            last_time = self._last_received_time

        if packet is None:
            return None, False, float("inf")

        age = now - last_time

        is_fresh = (
            age <= self.stale_timeout
            and packet.get("IsRaceOn", 0) == 1
        )

        return packet, is_fresh, age

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packet_count

    @property
    def error_message(self) -> Optional[str]:
        with self._lock:
            return self._error_msg


def run_diagnostic() -> None:
    """Standalone diagnostic CLI."""

    print("=" * 65)
    print("FH4 Telemetry Receiver Diagnostic")
    print(f"Parsed layout size: {EXPECTED_PARSED_BYTES} bytes")
    print("FH4 Data Out = ON")
    print("Data Out IP = 127.0.0.1")
    print("Data Out Port = 9999")
    print("Press Ctrl+C to exit.")
    print("=" * 65)

    receiver = TelemetryReceiver()

    if not receiver.start():
        print("Could not start receiver. Check UDP port 9999.")
        return

    try:
        while True:
            packet, is_fresh, age = receiver.get_latest()

            if packet is None:
                print(
                    "\rWaiting for UDP telemetry packets...",
                    end="",
                    flush=True,
                )

                error = receiver.error_message
                if error:
                    print(f"\nLast error: {error}")

            else:
                status = (
                    "FRESH"
                    if is_fresh
                    else (
                        "STALE"
                        if packet["IsRaceOn"] == 1
                        else "PAUSED/MENU"
                    )
                )

                print(
                    f"\r[{status:10s}] "
                    f"Speed: {packet['SpeedKmh']:6.1f} km/h | "
                    f"Pos: ({packet['PositionX']:7.1f}, {packet['PositionZ']:7.1f}) | "
                    f"Yaw: {packet['Yaw']:+5.2f} | "
                    f"Gear: {packet['Gear']:>2s} | "
                    f"Steer: {packet['Steer']:4d} | "
                    f"Packets: {receiver.packet_count:6d} | "
                    f"Age: {age:.2f}s   ",
                    end="",
                    flush=True,
                )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping diagnostic...")

    finally:
        receiver.stop()


if __name__ == "__main__":
    run_diagnostic()
