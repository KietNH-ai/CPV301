"""
src.map — Gói phân tích bản đồ Forza Horizon 4 (UK), trích xuất đồ thị đường đi,
định tuyến tự hành A* kết hợp tọa độ Telemetry và giao diện chia 4 ô màn hình (Quad Dashboard).
"""
from src.map.map_analyzer import (
    RoadNode,
    RoadEdge,
    FH4MapAnalyzer,
    ensure_sample_fh4_map,
    DEFAULT_FH4_LANDMARKS,
    DEFAULT_FH4_ROAD_SEGMENTS,
)
from src.map.navigator import FH4MapNavigator
from src.map.quad_dashboard import FH4QuadDashboard

__all__ = [
    "RoadNode",
    "RoadEdge",
    "FH4MapAnalyzer",
    "FH4MapNavigator",
    "FH4QuadDashboard",
    "ensure_sample_fh4_map",
    "DEFAULT_FH4_LANDMARKS",
    "DEFAULT_FH4_ROAD_SEGMENTS",
]
