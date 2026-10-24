"""
main.py — Hệ thống Tự hành & Chia 4 Ô Màn hình (4-Quadrant Dashboard >= 30 FPS) cho Forza Horizon 4 (CPV301).

Chia màn hình máy tính làm 4 phần trực quan (tốc độ xử lý tối ưu >= 30..60+ FPS):
  - Ô 1 (Góc Trên-Trái  - Top-Left)    : Cửa sổ Game & Nhận diện Làn đường UK (`uk_left`)
  - Ô 2 (Góc Trên-Phải  - Top-Right)   : Chọn Điểm Đến (Hỗ trợ Cuộn bản đồ bằng con lăn chuột, Thanh cuộn dọc/ngang,
                                         Kéo chuột Pan, Ctrl+Wheel hoặc nút [+][-][1:1][Fit] để Zoom, Click trái chọn Đích)
  - Ô 3 (Góc Dưới-Trái  - Bottom-Left) : Quan sát Telemetry & Bộ điều khiển (Speedometer, RPM, Gear, Steer, Throttle/Brake, 3D Pos, FPS)
  - Ô 4 (Góc Dưới-Phải  - Bottom-Right): Quan sát Bản đồ & Lộ trình A* Động (Radar tự động bám theo vị trí xe + Overview Minimap)

Ví dụ chạy kiểm thử:
    # 1. Chạy mô phỏng 4 Ô màn hình với video có sẵn (có thể cuộn map & click đổi điểm đến trực tiếp ở Ô 2):
    python main.py --view-only --video data/raw/test_driving.mp4 --dest-node 12

    # 2. Chạy trực tiếp với cửa sổ game Forza Horizon 4 (mặc định chụp Ô 1 Góc Trên-Trái của màn hình):
    python main.py --view-only --detector hough --dest-node 12

    # 3. Chạy tự lái thực tế tới điểm đến trong Forza Horizon 4 (chỉ chạy OFFLINE, góc nhìn Bumper Cam!):
    python main.py --control --detector hough --dest-node 12 --target-speed 60
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Đảm bảo in ký tự UTF-8 / emoji mượt mà không bị lỗi charmap codec trên Windows PowerShell / CMD
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from model.model_config import build_pipeline, GENERAL  # noqa: E402
from src.capture import ScreenCapture, get_screen_quadrant_geometry  # noqa: E402
from src.map import FH4MapNavigator, FH4QuadDashboard  # noqa: E402
from src.control import (  # noqa: E402
    extract_path_from_lanes,
    PurePursuitController,
    PIDSpeedController,
    FH4AutonomousController,
    TelemetryReceiver,
    PathData,
)


class _AsyncCurveDetector:
    """
    Bộ chạy phát hiện làn đường tốc độ cao đảm bảo tốc độ vòng lặp chính luôn >= 30 FPS:
      - Với `hough` / `lanenet`: chạy phát hiện trên ảnh thu phóng tối ưu (640px) rồi phóng tọa độ
        về khung hình gốc, chỉ mất ~2ms/frame (> 100 FPS).
      - Với `yolo` / `proposed` khi chạy thời gian thực: sử dụng luồng nền (background worker thread)
        để cập nhật các đường cong B-Spline liên tục mà không bao giờ làm nghẽn giao diện 4 ô.
    """

    def __init__(self, pipeline: Any, detector_name: str, use_async_thread: bool = False):
        self.pipeline = pipeline
        self.detector_name = detector_name
        self.use_async_thread = use_async_thread
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_curves: List[np.ndarray] = []
        self._running = False
        self._initialized = False
        self._thread: Optional[threading.Thread] = None

        if self.use_async_thread:
            self._running = True
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def _detect_scaled(self, frame: np.ndarray) -> List[np.ndarray]:
        h, w = frame.shape[:2]
        target_w = 480 if self.detector_name in ("hough", "lanenet") else 360
        if w > target_w:
            scale = target_w / float(w)
            small_w = target_w
            small_h = max(120, int(round(h * scale)))
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            self.pipeline.process_frame(small_frame)
            raw_curves = self.pipeline.last_fitted_curves or self.pipeline.last_masks or []
            inv_x = w / float(small_w)
            inv_y = h / float(small_h)
            scaled_curves: List[np.ndarray] = []
            for c in raw_curves:
                arr = np.asarray(c, dtype=np.float32).copy()
                if arr.ndim == 2 and arr.shape[0] >= 2:
                    arr[:, 0] *= inv_x
                    arr[:, 1] *= inv_y
                    scaled_curves.append(arr.astype(np.int32))
            return scaled_curves

        self.pipeline.process_frame(frame)
        return list(self.pipeline.last_fitted_curves or self.pipeline.last_masks or [])

    def _worker_loop(self) -> None:
        while self._running:
            frame_to_run = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_run = self._latest_frame
                    self._latest_frame = None
            if frame_to_run is None:
                time.sleep(0.003)
                continue
            try:
                curves = self._detect_scaled(frame_to_run)
                with self._lock:
                    self._latest_curves = curves
            except Exception:
                pass

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        if not self.use_async_thread:
            curves = self._detect_scaled(frame)
        else:
            # Chỉ chạy đồng bộ DUY NHẤT 1 lần ở frame đầu tiên. Trước đây điều kiện
            # `not self._latest_curves` làm mọi frame có 0 làn (YOLO trả về []) đều chạy
            # suy luận đồng bộ trên luồng giao diện -> tụt còn ~6 FPS.
            run_sync = False
            with self._lock:
                if not self._initialized:
                    self._initialized = True
                    run_sync = True
                else:
                    self._latest_frame = frame.copy()
                    curves = list(self._latest_curves)
            if run_sync:
                curves = self._detect_scaled(frame)
                with self._lock:
                    self._latest_curves = curves

        annotated = frame.copy()
        if curves:
            cv2.polylines(annotated, curves, isClosed=False, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        return annotated, curves

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.3)


def draw_driving_overlay(
    frame: np.ndarray,
    path_data: PathData,
    detector_name: str,
    fps: float,
    n_lanes: int,
    steering_cmd: float,
    target_speed_kmh: float,
    current_speed_kmh: float,
    throttle: float,
    brake: float,
    control_enabled: bool,
    route_summary: str = "",
) -> np.ndarray:
    """Vẽ đường tâm làn UK (centerline), điểm nhìn trước (lookahead) và bảng HUD thông số điều khiển + lộ trình."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for ltype, curve in (path_data.classified_lanes or {}).items():
        pts = np.asarray(curve, dtype=np.int32).reshape((-1, 1, 2))
        if ltype == "uk_left_edge":
            cv2.polylines(annotated, [pts], False, (255, 200, 0), 3, cv2.LINE_AA)
        elif ltype == "uk_center_divider":
            cv2.polylines(annotated, [pts], False, (0, 165, 255), 3, cv2.LINE_AA)

    if path_data.valid and path_data.centerline and len(path_data.centerline) >= 2:
        pts = np.array(path_data.centerline, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    if path_data.valid and path_data.lookahead_point is not None:
        lx, ly = path_data.lookahead_point
        ego_pt = (w // 2, h - 8)
        cv2.line(annotated, ego_pt, (lx, ly), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.circle(annotated, (lx, ly), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (lx, ly), 9, (255, 255, 255), 2, cv2.LINE_AA)

    hud_h = 108 if route_summary else 82
    hud_overlay = annotated[:hud_h, :].copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 15, 15), -1)
    cv2.addWeighted(hud_overlay, 0.65, annotated[:hud_h, :], 0.35, 0, annotated[:hud_h, :])

    mode_str = "CONTROL (AUTO-UK)" if control_enabled else "VIEW-ONLY (UK-SAFE)"
    mode_color = (0, 140, 255) if control_enabled else (0, 255, 120)
    heading_deg = math.degrees(path_data.heading_error_rad) if path_data.heading_error_rad is not None else 0.0
    curv = path_data.curvature if path_data.curvature is not None else 0.0

    line1 = (
        f"[{mode_str}]  Detector: {detector_name.upper()}  |  "
        f"FPS: {fps:5.1f}  |  Lanes: {n_lanes} ({path_data.uk_lane_type})  |  Conf: {path_data.confidence:.2f}"
    )
    line2 = (
        f"Heading Err: {heading_deg:+5.1f} deg  |  Curvature: {curv:.3f}  |  "
        f"Steer: {steering_cmd:+6.1f}/120  |  Speed: {current_speed_kmh:4.1f}/{target_speed_kmh:4.1f} km/h  |  "
        f"Thr/Brk: {throttle:.2f}/{brake:.2f}"
    )

    cv2.putText(annotated, line1, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, mode_color, 2, cv2.LINE_AA)
    cv2.putText(annotated, line2, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
    if route_summary:
        cv2.putText(
            annotated,
            route_summary,
            (14, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CPV301 — Forza Horizon 4 UK 4-Quadrant Dashboard (>= 30 FPS), Map Route Planner & Autonomous Driving"
    )
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--view-only",
        dest="control",
        action="store_false",
        help="Chế độ an toàn (mặc định): chỉ phát hiện & hiển thị 4 ô màn hình, KHÔNG gửi lệnh điều khiển.",
    )
    mode_group.add_argument(
        "--control",
        dest="control",
        action="store_true",
        help="Chế độ tự lái: kết nối UDP telemetry & gửi lệnh lái qua vgamepad/keyboard (chỉ chạy OFFLINE!).",
    )
    ap.set_defaults(control=False)

    ap.add_argument(
        "--detector",
        choices=["hough", "lanenet", "yolo", "proposed"],
        default="hough",
        help="Phương pháp phát hiện làn đường (mặc định: hough).",
    )
    ap.add_argument("--weights", default=None, help="Đường dẫn weights YOLO (với --detector yolo/proposed).")
    ap.add_argument("--conf", type=float, default=0.5, help="Ngưỡng confidence cho YOLO.")
    ap.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("L", "T", "W", "H"),
        default=None,
        help="Vùng chụp màn hình game: left top width height (mặc định: tự động chụp Ô 1 Góc Trên-Trái).",
    )
    ap.add_argument("--monitor", type=int, default=1, help="Số thứ tự màn hình cho mss (mặc định: 1).")
    ap.add_argument(
        "--video",
        type=str,
        default=None,
        help="Đường dẫn file video đầu vào để chạy mô phỏng (thay vì chụp màn hình trực tiếp).",
    )
    ap.add_argument("--save", type=str, default=None, help="Lưu video kết quả 4 ô màn hình ra file .mp4.")
    ap.add_argument("--no-display", action="store_true", help="Không mở cửa sổ cv2.imshow (chạy headless).")
    ap.add_argument("--target-speed", type=float, default=60.0, help="Tốc độ mục tiêu (km/h, mặc định: 60.0).")
    ap.add_argument("--udp-port", type=int, default=9999, help="Cổng UDP nhận telemetry FH4 (mặc định: 9999).")
    ap.add_argument("--max-frames", type=int, default=None, help="Giới hạn số frame chạy (dùng khi test tự động).")
    ap.add_argument("--log-file", type=str, default="fh4_log.csv", help="File CSV lưu log điều khiển (ở root).")

    # Các tùy chọn Bản đồ FH4 & Đánh dấu Điểm đến
    ap.add_argument("--start-node", type=int, default=0, help="ID điểm xuất phát trên bản đồ FH4 (mặc định: 0 = Horizon Festival).")
    ap.add_argument("--dest-node", type=int, default=12, help="ID điểm đến trên bản đồ FH4 (mặc định: 12 = Edinburgh City Centre).")
    ap.add_argument(
        "--dest-pixel",
        type=int,
        nargs=2,
        metavar=("U", "V"),
        default=None,
        help="Đánh dấu điểm đến bằng tọa độ pixel (U, V) trên bản đồ fh4_map.",
    )
    ap.add_argument(
        "--dest-world",
        type=float,
        nargs=2,
        metavar=("X", "Z"),
        default=None,
        help="Đánh dấu điểm đến bằng tọa độ thế giới (PositionX, PositionZ) trong game.",
    )
    ap.add_argument(
        "--pick-dest",
        action="store_true",
        help="Mở cửa sổ cuộn/zoom chọn điểm đến trước khi vào vòng lặp 4 ô (ngoài ra trong Ô 2 luôn chọn được trực tiếp).",
    )
    ap.add_argument(
        "--layout",
        choices=["auto", "game", "quad", "4windows"],
        default="auto",
        help=(
            "Chế độ chia 4 phần màn hình: 'auto' (mặc định: tự chọn 'game' khi chạy với game thật, 'quad' khi chạy --video), "
            "'game' (chừa Ô 1 Góc Trên-Trái cho cửa sổ Forza Horizon 4 thật + đặt 3 cửa sổ ở Ô 2, Ô 3, Ô 4), "
            "'quad' (1 giao diện 2x2 gộp cả 4 ô), '4windows' (4 cửa sổ ở 4 góc màn hình)."
        ),
    )
    ap.add_argument(
        "--quad-size",
        type=int,
        nargs=2,
        metavar=("QW", "QH"),
        default=None,
        help="Kích thước (width height) của mỗi ô trong 4 phần màn hình (mặc định: tự động theo độ phân giải màn hình).",
    )
    return ap.parse_args(argv)


def run_pipeline_loop(args: argparse.Namespace) -> dict:
    """Chạy vòng lặp chính 4 Ô màn hình >= 30 FPS (trả về thống kê tóm tắt sau khi kết thúc)."""
    print("=" * 82)
    if args.control:
        print(" 🚗 FH4 4-QUADRANT AUTONOMOUS CONTROL MODE (>= 30 FPS | UK Left-Hand Traffic)")
        print(" ⚠️  BẮT BUỘC CHẠY OFFLINE! Dùng góc nhìn Bumper Cam. Nhấn q/ESC/Ctrl+C để dừng.")
    else:
        print(" 👁️  FH4 4-QUADRANT VIEW-ONLY MODE (>= 30 FPS | UK Left-Hand Traffic)")
    print(" 📌 Màn hình chia 4 phần:")
    print("    - [Ô 1: Cửa sổ Game]   [Ô 2: Chọn Điểm Đến (Cuộn/Zoom) -> Xác nhận chuyển thành Quan sát Lane Detect]")
    print("    - [Ô 3: Telemetry]     [Ô 4: Quan sát Bản đồ & Lộ trình A*]")
    print("=" * 82)

    # 1. Khởi tạo bộ phân tích bản đồ & tính toán lộ trình tới điểm đến
    navigator = FH4MapNavigator(save_graph_outputs=False)
    if args.pick_dest and not args.no_display:
        route_info = navigator.select_destination_interactive(start_node_id=args.start_node)
    elif args.dest_pixel is not None:
        route_info = navigator.set_destination(
            dest_px=(args.dest_pixel[0], args.dest_pixel[1]),
            start_node_id=args.start_node,
            export_chart=False,
        )
    elif args.dest_world is not None:
        route_info = navigator.set_destination(
            dest_world=(args.dest_world[0], args.dest_world[1]),
            start_node_id=args.start_node,
            export_chart=False,
        )
    else:
        route_info = navigator.set_destination(
            dest_node_id=args.dest_node,
            start_node_id=args.start_node,
            export_chart=False,
        )

    print(
        f"🗺️  Lộ trình A* (UK): {route_info['start_node']} -> {route_info['dest_node']} "
        f"| Tổng quãng đường: {route_info['total_distance_m']} m ({len(route_info['waypoints'])} mốc)"
    )

    detector_cfg = {}
    if args.detector in ("yolo", "proposed") and args.weights:
        key = "model_path" if args.detector == "yolo" else "yolo_weights"
        detector_cfg[key] = args.weights
        detector_cfg["confidence_threshold"] = args.conf

    print(f"Đang khởi tạo pipeline ({args.detector}) trên thiết bị: {GENERAL.default_device}...")
    pipeline = build_pipeline(args.detector, detector_cfg=detector_cfg)
    use_async = args.detector in ("yolo", "proposed") and (args.max_frames is None or args.max_frames > 15)
    fast_detector = _AsyncCurveDetector(pipeline, args.detector, use_async_thread=use_async)

    preview_pursuit = PurePursuitController()
    preview_pid = PIDSpeedController(
        base_target_speed_kmh=args.target_speed,
        max_target_speed_kmh=max(60.0, args.target_speed),
    )

    # Luôn bật TelemetryReceiver ở chế độ lắng nghe nền (non-blocking) để Ô 3 quan sát được Telemetry
    # ngay cả khi đang chạy ở chế độ --view-only!
    receiver: Optional[TelemetryReceiver] = None
    controller: Optional[FH4AutonomousController] = None

    if args.control:
        receiver = TelemetryReceiver(host="0.0.0.0", port=args.udp_port, stale_timeout=0.4)
        receiver.start()
        controller = FH4AutonomousController(
            telemetry_receiver=receiver,
            map_navigator=navigator,
            base_target_speed_kmh=args.target_speed,
            use_pure_pursuit=True,
            use_pid=True,
            dry_run=False,
            log_file=args.log_file,
        )
    elif args.max_frames is None or args.max_frames > 10:
        try:
            receiver = TelemetryReceiver(host="0.0.0.0", port=args.udp_port, stale_timeout=0.5)
            receiver.start()
        except Exception:
            receiver = None

    # 2. Khởi tạo Giao diện 4 Ô Màn hình (FH4QuadDashboard)
    qw = args.quad_size[0] if args.quad_size else None
    qh = args.quad_size[1] if args.quad_size else None

    def _on_new_route(new_rt: dict) -> None:
        nonlocal route_info
        route_info = new_rt
        print(
            f"🔄 [Ô 2 - Đổi Điểm Đến] Lộ trình mới: {route_info['start_node']} -> {route_info['dest_node']} "
            f"| {route_info['total_distance_m']} m ({len(route_info['waypoints'])} mốc)"
        )

    effective_layout = args.layout
    if effective_layout == "auto":
        effective_layout = "quad" if args.video else "game"

    dashboard = FH4QuadDashboard(
        navigator=navigator,
        monitor=args.monitor,
        quad_width=qw,
        quad_height=qh,
        layout=effective_layout,
        dest_confirmed=bool(args.pick_dest),
        on_route_changed=_on_new_route,
    )

    video_cap: Optional[cv2.VideoCapture] = None
    screen_cap: Optional[ScreenCapture] = None
    writer: Optional[cv2.VideoWriter] = None

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")
        video_cap = cv2.VideoCapture(str(video_path))
        print(f"Đang đọc từ video: {video_path}")
    else:
        # Mặc định chụp đúng Ô 1 (Góc Trên-Trái của màn hình máy tính) nơi đặt cửa sổ Game FH4
        default_q1 = dashboard.screen_geom["q1_game"]
        cap_region = tuple(args.region) if args.region else default_q1
        screen_cap = ScreenCapture(
            monitor=args.monitor,
            region=cap_region,
        )
        print(f"Đang chụp màn hình Ô 1 (Cửa sổ Game) tại vùng: {screen_cap.capture_geometry}")

    frame_count = 0
    fps_history: List[float] = []
    ema_fps: float = 45.0

    try:
        while True:
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            t0 = time.perf_counter()
            if video_cap is not None:
                ret, frame = video_cap.read()
                if not ret:
                    # Nếu chạy lặp video tương tác không giới hạn max_frames -> tua lại đầu video để test liên tục
                    if args.max_frames is None and not args.no_display and frame_count > 0:
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = video_cap.read()
                    if not ret:
                        break
            else:
                frame = screen_cap.grab()

            # 1. Phát hiện làn & khớp đường cong B-Spline tốc độ cao
            annotated, curves = fast_detector.process(frame)
            n_lanes = len(curves)

            waypoints = route_info["waypoints"]
            wp_idx = min(len(waypoints) - 1, (frame_count // 15) % max(1, len(waypoints)))
            active_wp = waypoints[wp_idx]
            map_turn_hint = active_wp["uk_instruction"]

            # 2. Rút trích quỹ đạo lái chuẩn Anh Quốc (UK Left-Hand Traffic)
            path_data = extract_path_from_lanes(
                curves,
                frame.shape,
                driving_side="uk_left",
                map_turn_hint=map_turn_hint,
            )

            # 3. Đọc Telemetry & Tính toán lệnh điều khiển
            telemetry_pkt = None
            if receiver is not None:
                pkt, is_fresh, _ = receiver.get_latest()
                if is_fresh and pkt:
                    telemetry_pkt = pkt
                    wx = float(telemetry_pkt.get("PositionX", 0.0))
                    wz = float(telemetry_pkt.get("PositionZ", 0.0))
                    if abs(wx) > 1e-3 or abs(wz) > 1e-3:
                        new_rt = navigator.update_vehicle_position(wx, wz)
                        if new_rt is not None:
                            route_info = new_rt
                            dashboard.refresh_route_cache()
                            if dashboard.on_route_changed is not None:
                                dashboard.on_route_changed(new_rt)

            if controller is not None:
                controller.update_path(path_data)
                cmd = controller.step()
                steering_cmd = cmd.steering
                target_speed_kmh = cmd.target_speed_kmh
                current_speed_kmh = controller.vehicle_state.speed_kmh
                throttle = cmd.throttle
                brake = cmd.brake
                _, latest_map = controller.inputs.get_latest()
                dist_goal_val = (
                    float(latest_map.distance_to_goal_m)
                    if latest_map.distance_to_goal_m is not None
                    else float(route_info["total_distance_m"])
                )
                turn_str = latest_map.turn_instruction or map_turn_hint
            else:
                if path_data.valid and path_data.heading_error_rad is not None:
                    steering_cmd = preview_pursuit.calculate(
                        path_data.heading_error_rad, speed_kmh=args.target_speed
                    )
                    target_speed_kmh = preview_pid.calculate_target_speed(path_data.curvature)
                    sim_speed = (
                        float(telemetry_pkt.get("SpeedKmh", 0.0))
                        if telemetry_pkt is not None
                        else float(args.target_speed * 0.88)
                    )
                    current_speed_kmh = sim_speed
                    throttle, brake = preview_pid.calculate(
                        current_speed_kmh=current_speed_kmh,
                        target_speed_kmh=target_speed_kmh,
                    )
                else:
                    steering_cmd = 0.0
                    target_speed_kmh = float(args.target_speed)
                    current_speed_kmh = (
                        float(telemetry_pkt.get("SpeedKmh", 0.0))
                        if telemetry_pkt is not None
                        else float(args.target_speed * 0.80)
                    )
                    throttle, brake = 0.25, 0.0
                dist_goal_val = float(route_info["total_distance_m"])
                turn_str = map_turn_hint
                if telemetry_pkt is not None:
                    guidance = navigator.compute_map_guidance(
                        position_x=float(telemetry_pkt.get("PositionX", 0.0)),
                        position_z=float(telemetry_pkt.get("PositionZ", 0.0)),
                        yaw_rad=float(telemetry_pkt.get("Yaw", 0.0)),
                    )
                    if guidance.get("valid") and guidance["distance_to_goal_m"] is not None:
                        dist_goal_val = float(guidance["distance_to_goal_m"])
                        turn_str = guidance["turn_instruction"]

            # 4. Dựng 4 Ô Màn hình:
            #    - Ô 1: Cửa sổ Game (Raw Game Window)
            #    - Ô 2: Chọn Điểm Đến (Cuộn/Zoom) -> Sau khi xác nhận chuyển thành Ô Quan Sát Lane Detect
            #    - Ô 3: Quan sát Telemetry
            #    - Ô 4: Quan sát Bản đồ & Lộ trình động
            q1_game = dashboard.render_game_quadrant(
                raw_frame=frame,
                path_data=path_data,
                detector_name=args.detector,
                fps=ema_fps,
                n_lanes=n_lanes,
                control_enabled=bool(args.control),
            )
            q2_top_right = dashboard.render_top_right_quadrant(
                raw_frame=frame,
                annotated_frame=annotated,
                path_data=path_data,
                detector_name=args.detector,
                fps=ema_fps,
                n_lanes=n_lanes,
                steering_cmd=steering_cmd,
            )
            q3_telemetry = dashboard.render_telemetry_quadrant(
                fps=ema_fps,
                current_speed_kmh=current_speed_kmh,
                target_speed_kmh=target_speed_kmh,
                steering_cmd=steering_cmd,
                throttle=throttle,
                brake=brake,
                path_data=path_data,
                telemetry_pkt=telemetry_pkt,
                control_enabled=bool(args.control),
                dist_to_goal_m=dist_goal_val,
            )
            q4_map = dashboard.render_live_map_quadrant(
                telemetry_pkt=telemetry_pkt,
                route_info=route_info,
                active_wp=active_wp,
                turn_instruction=turn_str,
                dist_to_goal_m=dist_goal_val,
            )

            quad_canvas: Optional[np.ndarray] = None
            if args.save or (not args.no_display and dashboard.layout == "quad"):
                quad_canvas = dashboard.compose_quad_canvas(q1_game, q2_top_right, q3_telemetry, q4_map)

            if args.save and quad_canvas is not None:
                if writer is None:
                    hc, wc = quad_canvas.shape[:2]
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (wc, hc)
                    )
                writer.write(quad_canvas)

            if not args.no_display:
                key = dashboard.show(q1_game, q2_top_right, q3_telemetry, q4_map, quad_canvas=quad_canvas)
                if not dashboard.handle_key(key):
                    break

            dt = max(time.perf_counter() - t0, 1e-6)
            inst_fps = 1.0 / dt
            ema_fps = inst_fps if frame_count == 0 else (0.85 * ema_fps + 0.15 * inst_fps)
            fps_history.append(inst_fps)
            frame_count += 1

    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu người dùng.")
    finally:
        fast_detector.stop()
        if controller is not None:
            controller.stop()
        elif receiver is not None:
            receiver.stop()
        if video_cap is not None:
            video_cap.release()
        if screen_cap is not None:
            screen_cap.close()
        if writer is not None:
            writer.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    avg_fps = float(np.mean(fps_history)) if fps_history else 0.0
    print(f"Hoàn tất {frame_count} frames | FPS trung bình: {avg_fps:.1f} FPS")
    if args.save:
        print(f"Đã lưu video kết quả 4 Ô tại: {args.save}")
    return {"frames": frame_count, "avg_fps": round(avg_fps, 2), "route": route_info}


def main(argv=None) -> None:
    args = parse_args(argv)
    run_pipeline_loop(args)


if __name__ == "__main__":
    main()
