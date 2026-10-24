"""
src/control/telemetry.py — Nhận và giải mã gói tin UDP Data Out (324 bytes) từ Forza Horizon 4.
Tái xuất và đồng bộ hoàn toàn với `test_fh4/fh4_telemetry.py`.
"""
from test_fh4.fh4_telemetry import (
    PACKET_FIELDS,
    TYPE_INFO,
    EXPECTED_PARSED_BYTES,
    decode_fh4_packet,
    TelemetryReceiver,
    run_diagnostic,
)

__all__ = [
    "PACKET_FIELDS",
    "TYPE_INFO",
    "EXPECTED_PARSED_BYTES",
    "decode_fh4_packet",
    "TelemetryReceiver",
    "run_diagnostic",
]
