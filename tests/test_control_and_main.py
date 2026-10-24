import math
import struct
import time
import cv2
import numpy as np
try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def approx(val, abs=1e-6, rel=1e-6):
            class _Approx:
                def __eq__(self, other):
                    return math.isclose(other, val, abs_tol=abs, rel_tol=rel)
            return _Approx()
        @staticmethod
        def skip(reason=""):
            pass
    pytest = _MockPytest()

from src import (
    PerspectiveTransformer,
    PipelineBuilder,
    VehicleActuator,
    PathData,
    MapData,
    PurePursuitController,
    classify_uk_lanes,
    extract_path_from_lanes,
    PIDSpeedController,
    FH4AutonomousController,
    FH4MapAnalyzer,
    FH4MapNavigator,
    FH4QuadDashboard,
    decode_fh4_packet,
    EXPECTED_PARSED_BYTES,
    PACKET_FIELDS,
    TYPE_INFO,
    ROOT_DIR,
    CONFIG_DIR,
    RAW_DATA_DIR,
    PROCESSING_DATA_DIR,
    FH4_MAP_RAW_DIR,
    FH4_MAP_PROCESSING_DIR,
)
from src.pipelinebuilder import PipelineBuilder as LegacyPipelineBuilder
from src.preprocessing.batch_processor import preprocess_single_road_image
from src.tusimple_to_yolo import lane_to_polygons, polygon_to_yolo_line
from main import parse_args, run_pipeline_loop


def test_pipelinebuilder_legacy_import():
    """Ensure `from src.pipelinebuilder import PipelineBuilder` matches `src.pipeline_builder`."""
    assert LegacyPipelineBuilder is PipelineBuilder


def test_bev_yaml_and_perspective_transformer():
    bev_yaml = CONFIG_DIR / "bev.yaml"
    assert bev_yaml.exists()

    for preset in ("default", "fh4_bumper", "tusimple"):
        transformer = PerspectiveTransformer.from_yaml(bev_yaml, preset=preset)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        warped = transformer.warp_to_birds_eye(frame)
        unwarped = transformer.unwarp_to_camera_view(warped)
        assert warped.shape == (720, 1280, 3)
        assert unwarped.shape == (720, 1280, 3)

        pts = np.array([[400, 650], [880, 650]], dtype=np.float32)
        bev_pts = transformer.warp_points(pts, 720, 1280, inverse=False)
        cam_pts = transformer.warp_points(bev_pts, 720, 1280, inverse=True)
        assert bev_pts.shape == (2, 2)
        assert np.allclose(pts, cam_pts, atol=1.0)


def test_extract_path_from_lanes_and_pure_pursuit():
    height, width = 720, 1280
    y_vals = np.linspace(height - 15, int(height * 0.45), 30)

    # Symmetric straight lanes -> near-zero heading error
    left_lane = np.column_stack((np.linspace(260, 520, 30), y_vals)).astype(np.int32)
    right_lane = np.column_stack((np.linspace(1020, 760, 30), y_vals)).astype(np.int32)

    path_straight = extract_path_from_lanes([left_lane, right_lane], (height, width))
    assert path_straight.valid is True
    assert path_straight.confidence >= 0.90
    assert abs(path_straight.heading_error_rad) < 0.05
    assert path_straight.lookahead_point is not None
    assert len(path_straight.centerline) > 5

    # Right-curving lanes -> positive heading error and positive steering command
    t = (height - 15 - y_vals) / float(height * 0.55)
    left_curve = np.column_stack((260 + 280 * t + 140 * (t ** 2), y_vals)).astype(np.int32)
    right_curve = np.column_stack((980 + 180 * t + 140 * (t ** 2), y_vals)).astype(np.int32)

    path_right = extract_path_from_lanes([left_curve, right_curve], (height, width))
    assert path_right.valid is True
    assert path_right.heading_error_rad > 0.05
    assert path_right.curvature > 0.0

    pp = PurePursuitController()
    steer_right = pp.calculate(path_right.heading_error_rad, speed_kmh=50.0)
    assert steer_right > 0.0

    # Single-lane fallback
    path_single = extract_path_from_lanes([left_lane], (height, width))
    assert path_single.valid is True
    assert 0.5 <= path_single.confidence < 0.9

    # Empty lanes -> invalid path
    path_empty = extract_path_from_lanes([], (height, width))
    assert path_empty.valid is False


def test_pid_speed_controller():
    pid = PIDSpeedController(base_target_speed_kmh=60.0, min_target_speed_kmh=25.0)

    # High curvature reduces target speed
    v_straight = pid.calculate_target_speed(curvature=0.0)
    v_curve = pid.calculate_target_speed(curvature=0.10)
    assert v_straight == pytest.approx(60.0)
    assert 25.0 <= v_curve < v_straight

    # Below target speed -> positive throttle, zero brake
    throttle, brake = pid.calculate(current_speed_kmh=30.0, target_speed_kmh=60.0, dt=0.05)
    assert throttle > 0.0
    assert brake == 0.0

    # Above target speed -> zero throttle, positive brake
    pid.reset()
    throttle2, brake2 = pid.calculate(current_speed_kmh=75.0, target_speed_kmh=50.0, dt=0.05)
    assert throttle2 == 0.0
    assert brake2 > 0.0


def test_decode_fh4_packet_and_autonomous_controller(tmp_path):
    # Construct a valid 324-byte FH4 UDP packet
    buf = bytearray(324)
    offset = 0
    for kind, name in PACKET_FIELDS:
        fmt, size = TYPE_INFO[kind]
        if kind == "hzn":
            pass
        elif name == "IsRaceOn":
            struct.pack_into(fmt, buf, offset, 1)
        elif name == "Speed":
            struct.pack_into(fmt, buf, offset, 15.0)  # 15 m/s = 54 km/h
        elif name == "Gear":
            struct.pack_into(fmt, buf, offset, 3)
        elif name == "DrivetrainType":
            struct.pack_into(fmt, buf, offset, 2)  # AWD
        offset += size

    assert offset == EXPECTED_PARSED_BYTES
    decoded = decode_fh4_packet(bytes(buf))
    assert decoded["IsRaceOn"] == 1
    assert decoded["SpeedKmh"] == pytest.approx(54.0, abs=0.1)
    assert decoded["Gear"] == "3"
    assert decoded["Drivetrain"] == "AWD"

    class MockTelemetry:
        def get_latest(self):
            return decoded, True, 0.02

        def stop(self):
            pass

    log_csv = tmp_path / "test_fh4_log.csv"
    actuator = VehicleActuator(dry_run=True)
    controller = FH4AutonomousController(
        telemetry_receiver=MockTelemetry(),
        actuator=actuator,
        base_target_speed_kmh=60.0,
        dry_run=True,
        log_file=str(log_csv),
    )

    controller.update_path(
        PathData(
            timestamp=time.monotonic(),
            valid=True,
            confidence=0.95,
            heading_error_rad=0.12,
            curvature=0.02,
        )
    )
    cmd = controller.step()
    assert cmd.source == "vision"
    assert cmd.steering > 0.0
    controller.stop()
    assert log_csv.exists()


def test_tusimple_to_yolo_helpers():
    h_samples = list(range(300, 700, 20))
    lane_left = [int(250 + (700 - y) * 0.5) for y in h_samples]
    polys = lane_to_polygons([lane_left], h_samples, img_h=720, img_w=1280)
    assert len(polys) >= 1
    line = polygon_to_yolo_line(polys[0], img_w=1280, img_h=720, class_id=0)
    assert line.startswith("0 ")
    coords = [float(v) for v in line.split()[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_main_video_view_only_loop(tmp_path):
    video_path = RAW_DATA_DIR / "test_driving.mp4"
    if not video_path.exists():
        pytest.skip("Sample test_driving.mp4 not present")

    out_video = tmp_path / "sim_out.mp4"
    args = parse_args(
        [
            "--view-only",
            "--no-display",
            "--detector",
            "hough",
            "--dest-node",
            "12",
            "--video",
            str(video_path),
            "--save",
            str(out_video),
            "--max-frames",
            "3",
        ]
    )
    stats = run_pipeline_loop(args)
    assert stats["frames"] == 3
    assert stats["route"]["dest_node"] == "Edinburgh City Centre"
    assert out_video.exists()


def test_uk_lane_classification_and_turn_hints():
    height, width = 720, 1280
    y_vals = np.linspace(height - 15, int(height * 0.45), 30)

    # 3 lanes (2-lane UK road: left edge, center divider, right opposite edge)
    left_edge = np.column_stack((np.linspace(180, 420, 30), y_vals)).astype(np.int32)
    center_div = np.column_stack((np.linspace(640, 640, 30), y_vals)).astype(np.int32)
    right_opp = np.column_stack((np.linspace(1100, 860, 30), y_vals)).astype(np.int32)

    uk_info = classify_uk_lanes([left_edge, center_div, right_opp], (height, width))
    assert uk_info["uk_lane_type"] == "uk_multi_lane_left_corridor"
    assert "uk_left_edge" in uk_info["labeled_curves"]
    assert "uk_center_divider" in uk_info["labeled_curves"]
    assert "uk_oncoming_right_edge" in uk_info["labeled_curves"]

    path_uk = extract_path_from_lanes(
        [left_edge, center_div, right_opp],
        (height, width),
        driving_side="uk_left",
        map_turn_hint="turn_left",
    )
    assert path_uk.valid is True
    assert path_uk.uk_lane_type == "uk_multi_lane_left_corridor"
    assert path_uk.heading_error_rad < 0.0  # Left lane + left turn hint shifts target left


def test_fh4_map_analyzer_and_navigator(tmp_path):
    analyzer = FH4MapAnalyzer()
    graph = analyzer.build_road_graph(save_outputs=True)
    assert len(graph["nodes"]) >= 10
    assert len(graph["edges"]) >= 10
    assert (FH4_MAP_RAW_DIR / "fh4_uk_map.png").exists()
    assert (FH4_MAP_PROCESSING_DIR / "road_graph_overlay.png").exists()
    assert (ROOT_DIR / "fh4_map_road_graph.csv").exists()

    nav = FH4MapNavigator(analyzer=analyzer)
    route_info = nav.set_destination(dest_node_id=12, start_node_id=0)
    assert route_info["start_node"] == "Horizon Festival Site"
    assert route_info["dest_node"] == "Edinburgh City Centre"
    assert route_info["total_distance_m"] > 0.0
    assert (FH4_MAP_PROCESSING_DIR / "planned_route_map.png").exists()
    assert (ROOT_DIR / "fh4_route_waypoints.csv").exists()
    assert (ROOT_DIR / "fh4_map_route_chart.png").exists()

    # Check live guidance from vehicle telemetry coordinates near start node
    start_node = nav.nodes[0]
    guidance = nav.compute_map_guidance(
        position_x=start_node.world_x,
        position_z=start_node.world_z,
        yaw_rad=0.0,
    )
    assert guidance["valid"] is True
    assert guidance["distance_to_goal_m"] > 50.0
    assert "UK" in guidance["turn_instruction"] or "LEFT" in guidance["turn_instruction"]


def test_preprocess_single_road_image_pipeline():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw synthetic white UK left lane and center divider
    cv2.line(img, (220, 710), (520, 350), (255, 255, 255), 10)
    cv2.line(img, (720, 710), (660, 350), (255, 255, 255), 10)
    res = preprocess_single_road_image(img)
    assert "highlighted" in res
    assert res["highlighted"].shape == img.shape
    assert res["num_lanes"] >= 1
    assert res["uk_lane_type"].startswith("uk_")


def test_quad_dashboard_scroll_zoom_click_and_30fps():
    nav = FH4MapNavigator(save_graph_outputs=False)
    route_info = nav.set_destination(dest_node_id=12, start_node_id=0, export_chart=False)
    dashboard = FH4QuadDashboard(navigator=nav, quad_width=640, quad_height=360)

    # Test zoom and scroll on Quadrant 2 (Destination Picker)
    initial_scroll_y = dashboard.picker_scroll_y
    dashboard.scroll_picker_by(25.0, 40.0)
    assert dashboard.picker_scroll_y != initial_scroll_y or dashboard.picker_zoom > 0
    dashboard.zoom_picker_at(1.5, 320, 180)
    assert dashboard.picker_zoom == pytest.approx(1.5, abs=0.05)

    # Simulate left-click on Quadrant 2 to pick a new destination
    dashboard.handle_picker_mouse(cv2.EVENT_LBUTTONDOWN, 200, 180, 0)
    dashboard.handle_picker_mouse(cv2.EVENT_LBUTTONUP, 200, 180, 0)
    assert nav.dest_node_id is not None
    assert dashboard.dest_confirmed is False

    # Render all 4 quadrants before confirmation (Q2 = Destination Picker) and after confirmation (Q2 = Lane Detect)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    path_data = PathData(
        timestamp=time.monotonic(),
        valid=True,
        confidence=0.92,
        heading_error_rad=0.03,
        curvature=0.005,
        centerline=[(400, 700), (450, 450)],
        lookahead_point=(450, 450),
        uk_lane_type="uk_standard_left_lane",
        classified_lanes={
            "uk_left_edge": np.array([[220, 700], [380, 450]], dtype=np.int32),
            "uk_center_divider": np.array([[640, 700], [540, 450]], dtype=np.int32),
        },
    )
    q2_picker = dashboard.render_top_right_quadrant(dummy_frame, dummy_frame, path_data, "hough", 60.0, 2, 8.0)
    assert q2_picker.shape == (360, 640, 3)

    # Confirm destination (press Enter key = 13) -> Q2 switches to Lane Detect view!
    assert dashboard.handle_key(13) is True
    assert dashboard.dest_confirmed is True

    t0 = time.perf_counter()
    for _ in range(15):
        q1 = dashboard.render_game_quadrant(dummy_frame, path_data, "hough", 60.0, 2, False)
        q2 = dashboard.render_top_right_quadrant(dummy_frame, dummy_frame, path_data, "hough", 60.0, 2, 8.0)
        q3 = dashboard.render_telemetry_quadrant(60.0, 55.0, 60.0, 8.0, 0.5, 0.0, path_data, None, False, 1200.0)
        q4 = dashboard.render_live_map_quadrant(
            None, route_info, route_info["waypoints"][0], "KEEP_LEFT_LANE_UK", 1200.0
        )
        canvas = dashboard.compose_quad_canvas(q1, q2, q3, q4)
    elapsed = max(1e-6, time.perf_counter() - t0)
    render_fps = 15.0 / elapsed

    assert canvas.shape == (720, 1280, 3)
    assert render_fps >= 30.0

    # Press 'm' to reopen destination picker on Q2
    assert dashboard.handle_key(ord("m")) is True
    assert dashboard.dest_confirmed is False


