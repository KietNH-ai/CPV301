"""
src/map/navigator.py — Bộ định tuyến A* (A-Star) trên bản đồ Forza Horizon 4,
cho phép người dùng đánh dấu điểm đến (Destination Marking), hỗ trợ cuộn/phóng to
bản đồ (Scroll/Pan/Zoom), tính toán đường đi kết hợp phân loại làn đường chuẩn
Anh Quốc (UK Left-Hand Traffic) và tọa độ từ `fh4_telemetry.py` để dẫn đường tự động.
"""
from __future__ import annotations

import csv
import heapq
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.path import ROOT_DIR, FH4_MAP_PROCESSING_DIR
from src.map.map_analyzer import FH4MapAnalyzer, RoadNode, _draw_label_pill


def _wrap_angle_rad(angle: float) -> float:
    """Chuẩn hóa góc về đoạn [-pi, +pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class FH4MapNavigator:
    """
    Hệ thống lập lộ trình & dẫn đường tự hành trên bản đồ Forza Horizon 4 (UK):
      - Nạp đồ thị đường đi từ `FH4MapAnalyzer`.
      - Cho phép người dùng đánh dấu điểm đến (bằng tọa độ pixel, tọa độ thế giới, node ID hoặc click chuột).
      - Tìm đường đi tối ưu bằng thuật toán A* (A-Star).
      - Tạo lộ trình chi tiết kèm hướng dẫn làn đường nước Anh (đi làn trái, vào bùng binh chiều kim đồng hồ).
      - Lưu đệm (cache) ảnh bản đồ lộ trình (`route_map_bgr`) và đường dẫn (`route_polyline_px`)
        để hiển thị thời gian thực >= 30 FPS trên giao diện 4 ô màn hình.
    """

    def __init__(
        self,
        map_path: Optional[Path] = None,
        analyzer: Optional[FH4MapAnalyzer] = None,
        save_graph_outputs: bool = True,
    ):
        self.analyzer = analyzer or FH4MapAnalyzer()
        self.graph = self.analyzer.build_road_graph(map_path=map_path, save_outputs=save_graph_outputs)
        self.nodes: Dict[int, RoadNode] = self.graph["nodes"]
        self.adjacency = self.graph["adjacency"]
        self.width: int = self.graph["width"]
        self.height: int = self.graph["height"]

        graph_overlay_path = FH4_MAP_PROCESSING_DIR / "road_graph_overlay.png"
        if graph_overlay_path.exists():
            loaded_overlay = cv2.imread(str(graph_overlay_path))
            self.graph_overlay_bgr: np.ndarray = (
                loaded_overlay if loaded_overlay is not None else self.graph["map_bgr"].copy()
            )
        else:
            self.graph_overlay_bgr = self.graph["map_bgr"].copy()

        self.route_map_bgr: np.ndarray = self.graph_overlay_bgr.copy()
        self.picker_map_bgr: np.ndarray = self.graph_overlay_bgr.copy()
        self.route_polyline_px: List[Tuple[int, int]] = []

        self.start_node_id: int = 0
        self.dest_node_id: Optional[int] = None
        self.dest_pixel: Optional[Tuple[int, int]] = None
        self.route_node_ids: List[int] = []
        self.route_waypoints: List[Dict[str, Any]] = []

        # --- Map-matching theo vị trí telemetry thật của xe ---
        # follow_vehicle_start=True: khi có telemetry, lộ trình luôn xuất phát từ vị trí xe thật
        # (chiếu xe lên đoạn đường gần nhất) thay vì node xuất phát cố định (Horizon Festival).
        self.follow_vehicle_start: bool = True
        self.last_vehicle_world: Optional[Tuple[float, float]] = None
        self.route_from_vehicle_active: bool = False
        self.reroute_off_route_m: float = 120.0
        self.reroute_cooldown_s: float = 6.0
        self._last_reroute_t: float = -1e9
        self._off_route_since: Optional[float] = None
        self._lead_in_px: List[Tuple[int, int]] = []
        # Hình học lộ trình dày đặc trong hệ tọa độ thế giới (dùng cho dẫn đường bám tuyến)
        self._route_world: Optional[np.ndarray] = None
        self._route_cum_m: Optional[np.ndarray] = None
        self._wp_along_m: List[float] = []
        self._last_seg_idx: Optional[int] = None

    def find_nearest_node(self, x_px: float, y_px: float) -> int:
        """Tìm node trên đồ thị đường đi gần vị trí pixel (x_px, y_px) nhất."""
        best_id = 0
        best_dist = float("inf")
        for nid, node in self.nodes.items():
            d = math.hypot(node.x_px - x_px, node.y_px - y_px)
            if d < best_dist:
                best_dist = d
                best_id = nid
        return best_id

    def find_nearest_node_world(self, world_x: float, world_z: float) -> int:
        """Tìm node gần tọa độ thế giới (PositionX, PositionZ) nhất."""
        px, py = self.analyzer.world_to_pixel(world_x, world_z, self.width, self.height)
        return self.find_nearest_node(px, py)

    def set_destination(
        self,
        dest_px: Optional[Tuple[int, int]] = None,
        dest_world: Optional[Tuple[float, float]] = None,
        dest_node_id: Optional[int] = None,
        start_px: Optional[Tuple[int, int]] = None,
        start_world: Optional[Tuple[float, float]] = None,
        start_node_id: Optional[int] = None,
        export_chart: bool = True,
        use_vehicle_start: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Người dùng đánh dấu điểm đến (và điểm xuất phát tùy chọn), hệ thống tự động
        chạy A* tìm lộ trình, xuất ảnh bản đồ lộ trình vào `fh4_map/processing/`
        và xuất biểu đồ + CSV lộ trình ra thư mục gốc `ROOT_DIR`.
        Khi gọi trực tiếp trong vòng lặp 30+ FPS, có thể đặt `export_chart=False`
        để đổi lộ trình tức thì (< 2ms) mà không bị khựng bởi Matplotlib.

        `use_vehicle_start=None` (tự động): nếu đã nhận được tọa độ telemetry thật của xe và người dùng
        không chủ động đổi điểm xuất phát, lộ trình sẽ bắt đầu ngay tại vị trí xe trên bản đồ.
        """
        if use_vehicle_start is None:
            use_vehicle_start = (
                self.follow_vehicle_start
                and self.last_vehicle_world is not None
                and start_px is None
                and start_world is None
                and (start_node_id is None or start_node_id == self.start_node_id)
            )

        if start_node_id is not None and start_node_id in self.nodes:
            self.start_node_id = start_node_id
        elif start_world is not None:
            self.start_node_id = self.find_nearest_node_world(start_world[0], start_world[1])
        elif start_px is not None:
            self.start_node_id = self.find_nearest_node(start_px[0], start_px[1])

        if dest_node_id is not None and dest_node_id in self.nodes:
            self.dest_node_id = dest_node_id
            dn = self.nodes[dest_node_id]
            self.dest_pixel = (dn.x_px, dn.y_px)
        elif dest_world is not None:
            self.dest_node_id = self.find_nearest_node_world(dest_world[0], dest_world[1])
            dn = self.nodes[self.dest_node_id]
            self.dest_pixel = (dn.x_px, dn.y_px)
        elif dest_px is not None:
            self.dest_node_id = self.find_nearest_node(dest_px[0], dest_px[1])
            self.dest_pixel = (int(dest_px[0]), int(dest_px[1]))
        else:
            raise ValueError("Cần cung cấp một trong: dest_px, dest_world, hoặc dest_node_id")

        if use_vehicle_start and self.last_vehicle_world is not None:
            return self.route_from_vehicle(
                self.last_vehicle_world[0],
                self.last_vehicle_world[1],
                dest_node_id=self.dest_node_id,
                export_chart=export_chart,
            )

        self._lead_in_px = []
        self.route_from_vehicle_active = False
        return self.plan_route(self.start_node_id, self.dest_node_id, export_chart=export_chart)

    # ------------------------------------------------------------------
    # MAP-MATCHING: chiếu vị trí xe thật lên mạng lưới đường & lập lộ trình từ chính vị trí xe
    # ------------------------------------------------------------------
    def world_to_pixel_f(self, world_x: float, world_z: float) -> Tuple[float, float]:
        """Quy đổi tọa độ thế giới -> pixel (số thực, không làm tròn/kẹp) trên ảnh bản đồ đang dùng."""
        min_x, max_x, min_z, max_z = self.analyzer.world_bounds
        u = (float(world_x) - min_x) / max(1e-6, max_x - min_x) * self.width
        v = (max_z - float(world_z)) / max(1e-6, max_z - min_z) * self.height
        return u, v

    def pixel_to_world_f(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Quy đổi pixel (số thực) -> tọa độ thế giới FH4 (PositionX, PositionZ)."""
        min_x, max_x, min_z, max_z = self.analyzer.world_bounds
        wx = min_x + (float(x_px) / max(1.0, float(self.width))) * (max_x - min_x)
        wz = max_z - (float(y_px) / max(1.0, float(self.height))) * (max_z - min_z)
        return wx, wz

    def snap_to_nearest_road(self, world_x: float, world_z: float) -> Dict[str, Any]:
        """
        Chiếu vị trí xe lên đoạn đường (cạnh đồ thị) gần nhất trên bản đồ.
        Trả về cạnh `(u, v)`, chỉ số điểm trên polyline, điểm chiếu (pixel) và khoảng cách lệch (mét).
        """
        px, py = self.world_to_pixel_f(world_x, world_z)
        best: Dict[str, Any] = {"dist_px": float("inf"), "edge": None, "index": 0, "snap_px": (px, py)}
        for edge in self.graph.get("edges", []):
            poly = edge.polyline_px
            if not poly or len(poly) < 2:
                continue
            arr = np.asarray(poly, dtype=np.float32)
            d = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
            i = int(np.argmin(d))
            if float(d[i]) < best["dist_px"]:
                best = {
                    "dist_px": float(d[i]),
                    "edge": (edge.u, edge.v),
                    "index": i,
                    "snap_px": (int(arr[i, 0]), int(arr[i, 1])),
                }
        best["dist_m"] = best["dist_px"] * self.analyzer.meters_per_pixel(self.width)
        return best

    def _route_length_px(self, node_ids: List[int]) -> float:
        polys = self.graph.get("edge_polylines", {})
        total = 0.0
        for a, b in zip(node_ids[:-1], node_ids[1:]):
            seg = polys.get((a, b))
            if seg and len(seg) >= 2:
                arr = np.asarray(seg, dtype=np.float32)
                total += float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))))
            else:
                na, nb = self.nodes[a], self.nodes[b]
                total += math.hypot(na.x_px - nb.x_px, na.y_px - nb.y_px)
        return total

    def route_from_vehicle(
        self,
        world_x: float,
        world_z: float,
        dest_node_id: Optional[int] = None,
        export_chart: bool = False,
    ) -> Dict[str, Any]:
        """
        Lập lộ trình xuất phát NGAY TẠI VỊ TRÍ XE THẬT (từ telemetry):
          1. Chiếu xe lên đoạn đường gần nhất (u -> v).
          2. Thử xuất phát qua cả 2 đầu mút u và v, chọn phương án có tổng quãng đường ngắn nhất.
          3. Ghép đoạn "dẫn nhập" (xe -> node đầu mút) vào đầu lộ trình để đường xanh bắt đầu đúng chỗ xe đứng.
        """
        goal = self.dest_node_id if dest_node_id is None else dest_node_id
        if goal is None:
            goal = self.find_nearest_node_world(world_x, world_z)
        self.last_vehicle_world = (float(world_x), float(world_z))
        snap = self.snap_to_nearest_road(world_x, world_z)
        car_px = self.world_to_pixel_f(world_x, world_z)
        car_pt = (int(round(car_px[0])), int(round(car_px[1])))

        best_start = self.find_nearest_node_world(world_x, world_z)
        best_lead: List[Tuple[int, int]] = [car_pt]
        if snap["edge"] is not None:
            u, v = snap["edge"]
            poly = self.graph["edge_polylines"][(u, v)]
            i = int(snap["index"])
            best_cost = float("inf")
            for start, part in ((u, list(reversed(poly[: i + 1]))), (v, list(poly[i:]))):
                path = self._astar_shortest_path(start, goal)
                part_arr = np.asarray(part, dtype=np.float32)
                part_len = (
                    float(np.sum(np.hypot(np.diff(part_arr[:, 0]), np.diff(part_arr[:, 1]))))
                    if len(part) >= 2
                    else 0.0
                )
                cost = part_len + self._route_length_px(path) + snap["dist_px"]
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
                    best_lead = [car_pt] + [(int(p[0]), int(p[1])) for p in part]

        self._lead_in_px = best_lead
        self.route_from_vehicle_active = True
        self._last_reroute_t = time.monotonic()
        self._off_route_since = None
        return self.plan_route(best_start, goal, export_chart=export_chart)

    def update_vehicle_position(self, world_x: float, world_z: float) -> Optional[Dict[str, Any]]:
        """
        Gọi mỗi frame khi có telemetry thật. Tự động lập lại lộ trình từ vị trí xe khi:
          - Lần đầu nhận được tọa độ xe (đồng bộ điểm xuất phát với vị trí thật), hoặc
          - Xe đi lệch khỏi lộ trình > `reroute_off_route_m` liên tục > 2 giây (có thời gian chờ).
        Trả về route_info mới nếu đã lập lại lộ trình, ngược lại trả về None.
        """
        if abs(world_x) < 1e-3 and abs(world_z) < 1e-3:
            return None
        self.last_vehicle_world = (float(world_x), float(world_z))
        if not self.follow_vehicle_start or self.dest_node_id is None:
            return None
        now = time.monotonic()
        if not self.route_from_vehicle_active:
            return self.route_from_vehicle(world_x, world_z, export_chart=False)

        loc = self.locate_on_route(world_x, world_z)
        if loc is None:
            return None
        if loc["cross_track_m"] > self.reroute_off_route_m:
            if self._off_route_since is None:
                self._off_route_since = now
            elif now - self._off_route_since > 2.0 and now - self._last_reroute_t > self.reroute_cooldown_s:
                return self.route_from_vehicle(world_x, world_z, export_chart=False)
        else:
            self._off_route_since = None
        return None

    def select_destination_interactive(
        self,
        start_node_id: int = 0,
        window_name: str = "FH4 Map - Cuon/Zoom & Click Chon Diem Den (Enter/q de xac nhan)",
        viewport_size: Tuple[int, int] = (920, 520),
    ) -> Dict[str, Any]:
        """
        Mở cửa sổ bản đồ FH4 có hỗ trợ CUỘN (Scrollbars, Kéo chuột, Lăn chuột Zoom)
        để người dùng đánh dấu điểm đến trực quan ngay cả trên màn hình nhỏ.
        """
        base_map = self.graph_overlay_bgr.copy()
        sn = self.nodes[start_node_id]
        cv2.circle(base_map, (sn.x_px, sn.y_px), 12, (0, 255, 0), -1, cv2.LINE_AA)
        clicked: List[Tuple[int, int]] = [(self.nodes[12].x_px, self.nodes[12].y_px)]

        vw, vh = viewport_size
        state = {
            "zoom": 1.0,
            "scroll_x": max(0.0, float(sn.x_px - vw // 2)),
            "scroll_y": max(0.0, float(sn.y_px - vh // 2)),
            "dragging": False,
            "drag_start": (0, 0),
            "scroll_start": (0.0, 0.0),
        }

        def _clamp_scroll():
            z = state["zoom"]
            vis_w = vw / z
            vis_h = vh / z
            state["scroll_x"] = max(0.0, min(float(self.width) - vis_w, state["scroll_x"]))
            state["scroll_y"] = max(0.0, min(float(self.height) - vis_h, state["scroll_y"]))

        _clamp_scroll()

        def _on_mouse(event, x, y, flags, param):
            z = state["zoom"]
            if event == cv2.EVENT_LBUTTONDOWN:
                state["dragging"] = True
                state["drag_start"] = (x, y)
                state["scroll_start"] = (state["scroll_x"], state["scroll_y"])
            elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
                dx = x - state["drag_start"][0]
                dy = y - state["drag_start"][1]
                state["scroll_x"] = state["scroll_start"][0] - dx / z
                state["scroll_y"] = state["scroll_start"][1] - dy / z
                _clamp_scroll()
            elif event == cv2.EVENT_LBUTTONUP:
                dx = abs(x - state["drag_start"][0])
                dy = abs(y - state["drag_start"][1])
                state["dragging"] = False
                if dx < 6 and dy < 6:
                    map_x = int(np.clip(state["scroll_x"] + x / z, 0, self.width - 1))
                    map_y = int(np.clip(state["scroll_y"] + y / z, 0, self.height - 1))
                    clicked[0] = (map_x, map_y)
            elif event == cv2.EVENT_MOUSEWHEEL:
                delta = cv2.getMouseWheelDelta(flags)
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    old_z = state["zoom"]
                    new_z = float(np.clip(old_z * (1.15 if delta > 0 else 0.87), 0.55, 2.5))
                    state["zoom"] = new_z
                elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                    state["scroll_x"] -= (60.0 if delta > 0 else -60.0) / z
                else:
                    state["scroll_y"] -= (60.0 if delta > 0 else -60.0) / z
                _clamp_scroll()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, vw, vh)
        cv2.setMouseCallback(window_name, _on_mouse)
        while True:
            preview = base_map.copy()
            cx, cy = clicked[0]
            near_id = self.find_nearest_node(cx, cy)
            near_node = self.nodes[near_id]
            cv2.drawMarker(preview, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
            cv2.circle(preview, (near_node.x_px, near_node.y_px), 14, (0, 255, 255), 3, cv2.LINE_AA)

            z = state["zoom"]
            vis_w = int(min(self.width, round(vw / z)))
            vis_h = int(min(self.height, round(vh / z)))
            x0 = int(np.clip(round(state["scroll_x"]), 0, max(0, self.width - vis_w)))
            y0 = int(np.clip(round(state["scroll_y"]), 0, max(0, self.height - vis_h)))
            crop = preview[y0 : y0 + vis_h, x0 : x0 + vis_w]
            view = cv2.resize(crop, (vw, vh), interpolation=cv2.INTER_LINEAR)
            cv2.putText(
                view,
                f"Dest: [{near_id}] {near_node.label} | Wheel: Cuon | Keo chuot: Pan | Click: Chon | Enter/q: OK",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, view)
            key = cv2.waitKey(16) & 0xFF
            if key in (13, 32, ord("q"), 27):
                break
        cv2.destroyWindow(window_name)
        return self.set_destination(dest_px=clicked[0], start_node_id=start_node_id)

    def _astar_shortest_path(self, start_id: int, goal_id: int) -> List[int]:
        """Thuật toán A* tìm đường đi ngắn nhất (ưu tiên cao tốc/quốc lộ có thời gian di chuyển nhỏ nhất)."""
        if start_id == goal_id:
            return [start_id]

        goal_node = self.nodes[goal_id]

        def _heuristic(nid: int) -> float:
            n = self.nodes[nid]
            return math.hypot(n.x_px - goal_node.x_px, n.y_px - goal_node.y_px) / 110.0

        open_heap: List[Tuple[float, int]] = [(0.0, start_id)]
        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {nid: float("inf") for nid in self.nodes}
        g_score[start_id] = 0.0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_id:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, dist_px, rclass, speed_lim in self.adjacency.get(current, []):
                travel_cost = dist_px / max(30.0, float(speed_lim))
                tentative_g = g_score[current] + travel_cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + _heuristic(neighbor)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return [start_id, goal_id]

    def plan_route(
        self,
        start_id: int,
        goal_id: int,
        export_chart: bool = True,
    ) -> Dict[str, Any]:
        """
        Tính toán lộ trình chi tiết giữa `start_id` và `goal_id`, nội suy các waypoints trung gian,
        phân loại làn đường chuẩn UK (bên trái), và lưu biểu đồ + CSV ở cả `fh4_map/processing/` và `ROOT_DIR`.
        """
        self.start_node_id = start_id
        self.dest_node_id = goal_id
        self.route_node_ids = self._astar_shortest_path(start_id, goal_id)

        waypoints: List[Dict[str, Any]] = []
        cumulative_dist_m = 0.0

        for idx in range(len(self.route_node_ids)):
            nid = self.route_node_ids[idx]
            node = self.nodes[nid]

            if idx < len(self.route_node_ids) - 1:
                next_node = self.nodes[self.route_node_ids[idx + 1]]
                dx_w = next_node.world_x - node.world_x
                dz_w = next_node.world_z - node.world_z
                heading_rad = math.atan2(dx_w, dz_w)
                seg_len_m = math.hypot(dx_w, dz_w)

                road_class = "primary_uk"
                speed_lim = 75.0
                for nb, _, rc, sp in self.adjacency.get(nid, []):
                    if nb == next_node.node_id:
                        road_class = rc
                        speed_lim = sp
                        break
            else:
                heading_rad = waypoints[-1]["heading_rad"] if waypoints else 0.0
                seg_len_m = 0.0
                road_class = node.road_type
                speed_lim = 0.0

            if idx == len(self.route_node_ids) - 1:
                uk_instruction = "ARRIVED_PULL_OVER_LEFT"
            elif node.road_type == "roundabout":
                uk_instruction = "ROUNDABOUT_CLOCKWISE_KEEP_LEFT"
            elif 0 < idx < len(self.route_node_ids) - 1:
                prev_heading = waypoints[-1]["heading_rad"]
                turn_delta = _wrap_angle_rad(heading_rad - prev_heading)
                if turn_delta < -0.25:
                    uk_instruction = "TURN_LEFT_INTO_UK_LEFT_LANE"
                elif turn_delta > 0.25:
                    uk_instruction = "TURN_RIGHT_CROSS_TRAFFIC_TO_UK_LEFT_LANE"
                else:
                    uk_instruction = "KEEP_LEFT_LANE_UK"
            else:
                uk_instruction = "KEEP_LEFT_LANE_UK"

            waypoints.append(
                {
                    "step": idx,
                    "node_id": nid,
                    "label": node.label,
                    "x_px": node.x_px,
                    "y_px": node.y_px,
                    "world_x": node.world_x,
                    "world_z": node.world_z,
                    "road_class": road_class,
                    "uk_lane": "left_lane_uk",
                    "uk_instruction": uk_instruction,
                    "heading_rad": round(heading_rad, 5),
                    "target_speed_kmh": round(speed_lim, 1),
                    "cumulative_distance_m": round(cumulative_dist_m, 1),
                }
            )
            cumulative_dist_m += seg_len_m

        # Quãng đường thực tế dọc theo đường cong trên bản đồ (thay vì khoảng cách chim bay giữa các node)
        self._build_route_geometry(self._compose_route_polyline())
        if self._route_cum_m is not None and len(self._wp_along_m) == len(waypoints):
            for wp, along in zip(waypoints, self._wp_along_m):
                wp["cumulative_distance_m"] = round(float(along), 1)
            cumulative_dist_m = float(self._route_cum_m[-1])

        self.route_waypoints = waypoints
        self._last_seg_idx = None
        self._export_route_visuals_and_csv(export_chart=export_chart)

        return {
            "start_node": self.nodes[start_id].label,
            "dest_node": self.nodes[goal_id].label,
            "node_ids": self.route_node_ids,
            "total_distance_m": round(cumulative_dist_m, 1),
            "waypoints": self.route_waypoints,
        }

    def _compose_route_polyline(self) -> List[Tuple[int, int]]:
        """
        Ghép toàn bộ chuỗi pixel (polyline) của các đoạn đường trong lộ trình hiện tại,
        bao gồm đoạn dẫn nhập (lead-in) từ vị trí xe thật (nếu có).
        """
        edge_polys = self.graph.get("edge_polylines", {})
        curved_pts: List[Tuple[int, int]] = []
        if getattr(self, "_lead_in_px", None):
            curved_pts.extend(self._lead_in_px)

        for idx in range(len(self.route_node_ids) - 1):
            u = self.route_node_ids[idx]
            v = self.route_node_ids[idx + 1]
            seg_pts = edge_polys.get((u, v))
            if seg_pts:
                if not curved_pts:
                    curved_pts.extend(seg_pts)
                else:
                    curved_pts.extend(seg_pts[1:])
            else:
                nu = self.nodes[u]
                nv = self.nodes[v]
                if not curved_pts:
                    curved_pts.append((nu.x_px, nu.y_px))
                curved_pts.append((nv.x_px, nv.y_px))

        if not curved_pts and self.route_node_ids:
            curved_pts = [(self.nodes[nid].x_px, self.nodes[nid].y_px) for nid in self.route_node_ids]

        return curved_pts

    def _build_route_geometry(self, curved_pts: List[Tuple[int, int]]) -> None:
        """
        Chuyển đổi chuỗi pixel của lộ trình sang tọa độ thế giới (world_x, world_z)
        và tính khoảng cách tích lũy (mét) dọc theo đường cong thực tế.
        """
        if not curved_pts or len(curved_pts) < 2:
            self._route_world = None
            self._route_cum_m = None
            self._wp_along_m = []
            return

        world_pts = []
        for px, py in curved_pts:
            wx, wz = self.pixel_to_world_f(px, py)
            world_pts.append([wx, wz])

        arr = np.asarray(world_pts, dtype=np.float32)
        diffs = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(diffs)])

        self._route_world = arr
        self._route_cum_m = cum

        # Tính khoảng cách dọc theo lộ trình cho từng waypoint (mốc)
        wp_along: List[float] = []
        for wp in self.route_waypoints:
            wpx, wpz = wp["world_x"], wp["world_z"]
            dists = np.hypot(arr[:, 0] - wpx, arr[:, 1] - wpz)
            closest_idx = int(np.argmin(dists))
            wp_along.append(float(cum[closest_idx]))
        self._wp_along_m = wp_along

    def _export_route_visuals_and_csv(self, export_chart: bool = True) -> None:
        """Xuất ảnh bản đồ lộ trình vào `fh4_map/processing/` và xuất CSV + Biểu đồ ra `ROOT_DIR`."""
        if not self.route_waypoints:
            return

        # 1. Xuất CSV lộ trình tại ROOT_DIR
        csv_path = ROOT_DIR / "fh4_route_waypoints.csv"
        fieldnames = list(self.route_waypoints[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.route_waypoints)

        # 2. Vẽ lộ trình cong bám theo mặt đường thực tế lên ảnh trong fh4_map/processing/planned_route_map.png
        route_img = self.graph["map_bgr"].copy()
        dark_bg = np.zeros_like(route_img)
        cv2.addWeighted(route_img, 0.88, dark_bg, 0.12, 0, route_img)

        edge_polys = self.graph.get("edge_polylines", {})
        curved_pts: List[Tuple[int, int]] = list(self.route_polyline_px)
        if not curved_pts:
            curved_pts = self._compose_route_polyline()

        if not curved_pts:
            curved_pts = [(wp["x_px"], wp["y_px"]) for wp in self.route_waypoints]

        self.route_polyline_px = curved_pts
        pts = np.array(curved_pts, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) >= 2:
            cv2.polylines(route_img, [pts], isClosed=False, color=(15, 20, 25), thickness=13, lineType=cv2.LINE_AA)
            cv2.polylines(route_img, [pts], isClosed=False, color=(255, 195, 0), thickness=8, lineType=cv2.LINE_AA)
            cv2.polylines(route_img, [pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        for idx, wp in enumerate(self.route_waypoints):
            p = (wp["x_px"], wp["y_px"])
            cv2.circle(route_img, p, 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(route_img, p, 7, (0, 140, 255), -1, cv2.LINE_AA)
            if 0 < idx < len(self.route_waypoints) - 1:
                _draw_label_pill(
                    route_img,
                    f"#{idx} {wp['label']}",
                    (p[0] + 14, p[1] + 5),
                    text_color=(255, 255, 255),
                    border_color=(255, 200, 0),
                    font_scale=0.42,
                )

        # Đánh dấu START (Xanh lá) và DESTINATION (Đỏ cam nổi bật)
        s_wp = self.route_waypoints[0]
        d_wp = self.route_waypoints[-1]
        cv2.circle(route_img, (s_wp["x_px"], s_wp["y_px"]), 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(route_img, (s_wp["x_px"], s_wp["y_px"]), 11, (0, 220, 60), -1, cv2.LINE_AA)
        _draw_label_pill(
            route_img,
            f"START: {s_wp['label']}",
            (max(10, s_wp["x_px"] - 80), min(self.height - 15, s_wp["y_px"] + 34)),
            text_color=(140, 255, 140),
            border_color=(0, 230, 80),
            font_scale=0.52,
            thickness=2,
        )

        cv2.circle(route_img, (d_wp["x_px"], d_wp["y_px"]), 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(route_img, (d_wp["x_px"], d_wp["y_px"]), 11, (0, 30, 255), -1, cv2.LINE_AA)
        _draw_label_pill(
            route_img,
            f"DEST: {d_wp['label']}",
            (max(10, d_wp["x_px"] - 90), max(28, d_wp["y_px"] - 20)),
            text_color=(180, 230, 255),
            border_color=(0, 80, 255),
            font_scale=0.52,
            thickness=2,
        )

        self.route_map_bgr = route_img

        # Tạo bản đồ riêng cho Ô 2 (Chọn Điểm Đến): hiển thị đầy đủ 20 địa danh + mạng lưới đường + lộ trình hiện tại
        picker_img = self.graph_overlay_bgr.copy()
        if len(pts) >= 2:
            cv2.polylines(picker_img, [pts], isClosed=False, color=(15, 20, 25), thickness=12, lineType=cv2.LINE_AA)
            cv2.polylines(picker_img, [pts], isClosed=False, color=(255, 195, 0), thickness=7, lineType=cv2.LINE_AA)
            cv2.polylines(picker_img, [pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(picker_img, (s_wp["x_px"], s_wp["y_px"]), 14, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(picker_img, (s_wp["x_px"], s_wp["y_px"]), 10, (0, 220, 60), -1, cv2.LINE_AA)
        cv2.circle(picker_img, (d_wp["x_px"], d_wp["y_px"]), 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(picker_img, (d_wp["x_px"], d_wp["y_px"]), 11, (0, 30, 255), -1, cv2.LINE_AA)
        self.picker_map_bgr = picker_img

        if not export_chart:
            return

        FH4_MAP_PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(FH4_MAP_PROCESSING_DIR / "planned_route_map.png"), route_img)

        # 3. Xuất biểu đồ tổng hợp phân tích Đồ thị Bản đồ + Lộ trình A* + Tốc độ lộ trình tại ROOT_DIR
        graph_overlay_bgr = self.graph_overlay_bgr

        fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        axes[0].imshow(cv2.cvtColor(graph_overlay_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("1. Full FH4 UK Road Graph (fh4_map/processing)", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(route_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"2. A* Route: {s_wp['label']} -> {d_wp['label']}\n(UK Left-Hand Traffic)",
            fontweight="bold",
        )
        axes[1].axis("off")

        dists = [wp["cumulative_distance_m"] / 1000.0 for wp in self.route_waypoints]
        speeds = [wp["target_speed_kmh"] for wp in self.route_waypoints]
        labels = [wp["label"][:14] for wp in self.route_waypoints]
        axes[2].plot(dists, speeds, marker="o", linewidth=2.6, color="#1f77b4", label="Target Speed (km/h)")
        axes[2].fill_between(dists, speeds, alpha=0.18, color="#1f77b4")
        for idx_pt, (x_d, y_s, lbl) in enumerate(zip(dists, speeds, labels)):
            y_offset = 10 if idx_pt % 2 == 0 else -16
            axes[2].annotate(
                lbl,
                (x_d, y_s),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        axes[2].set_title("3. UK Route Speed Profile vs Distance", fontweight="bold")
        axes[2].set_xlabel("Distance along Planned Route (km)")
        axes[2].set_ylabel("Recommended Speed (km/h)")
        axes[2].set_ylim(0, 195)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        chart_root_path = ROOT_DIR / "fh4_map_route_chart.png"
        fig.savefig(str(chart_root_path), dpi=140)
        plt.close(fig)

    def locate_on_route(self, world_x: float, world_z: float) -> Optional[Dict[str, Any]]:
        """
        Xác định vị trí hiện tại của xe trên lộ trình cong thế giới:
          - Chiếu trực giao xe lên từng đoạn thẳng của polyline lộ trình.
          - Tính sai số lệch ngang (cross_track_m), khoảng cách đã đi dọc lộ trình (along_m),
            và khoảng cách còn lại tới đích dọc theo mặt đường (dist_to_goal_m).
        """
        if self._route_world is None or len(self._route_world) < 2 or self._route_cum_m is None:
            return None

        p = np.array([world_x, world_z], dtype=np.float32)
        p0 = self._route_world[:-1]
        p1 = self._route_world[1:]
        seg_vec = p1 - p0
        seg_lens_sq = np.sum(seg_vec ** 2, axis=1)
        seg_lens_sq = np.maximum(seg_lens_sq, 1e-6)

        v_to_p = p - p0
        t = np.sum(v_to_p * seg_vec, axis=1) / seg_lens_sq
        t_clamped = np.clip(t, 0.0, 1.0)[:, np.newaxis]

        proj = p0 + t_clamped * seg_vec
        dists = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
        best_idx = int(np.argmin(dists))

        best_proj = proj[best_idx]
        cross_track_m = float(dists[best_idx])
        seg_t = float(t_clamped[best_idx, 0])
        seg_len_m = math.hypot(seg_vec[best_idx, 0], seg_vec[best_idx, 1])
        along_m = float(self._route_cum_m[best_idx] + seg_t * seg_len_m)
        total_len_m = float(self._route_cum_m[-1])
        dist_to_goal_m = max(0.0, total_len_m - along_m)

        return {
            "seg_idx": best_idx,
            "snap_world": (float(best_proj[0]), float(best_proj[1])),
            "cross_track_m": round(cross_track_m, 2),
            "along_m": round(along_m, 2),
            "dist_to_goal_m": round(dist_to_goal_m, 2),
            "total_len_m": round(total_len_m, 2),
        }

    def compute_map_guidance(
        self,
        position_x: float,
        position_z: float,
        yaw_rad: float,
    ) -> Dict[str, Any]:
        """
        Dựa trên tọa độ thực `PositionX`, `PositionZ` và góc hướng `Yaw` từ `fh4_telemetry.py`,
        tính toán waypoint tiếp theo trên lộ trình đã chọn và trả về góc lệch hướng `heading_error_rad`,
        tốc độ mục tiêu `target_speed_kmh`, khoảng cách tới đích và chỉ dẫn làn UK.
        """
        if not self.route_waypoints:
            return {
                "valid": False,
                "heading_error_rad": None,
                "target_speed_kmh": None,
                "waypoint": None,
                "turn_instruction": "NO_ROUTE",
                "distance_to_goal_m": None,
            }

        # Cố gắng định vị bám theo hình học lộ trình cong chi tiết
        loc = self.locate_on_route(position_x, position_z)
        if loc is not None and self._route_world is not None and self._route_cum_m is not None:
            # Điểm ngắm phía trước (lookahead) 25m dọc theo đường cong lộ trình
            target_dist = min(loc["total_len_m"], loc["along_m"] + 25.0)
            target_idx = int(np.searchsorted(self._route_cum_m, target_dist))
            target_idx = min(len(self._route_world) - 1, max(0, target_idx))
            target_world = self._route_world[target_idx]

            dx = float(target_world[0]) - position_x
            dz = float(target_world[1]) - position_z
            desired_yaw = math.atan2(dx, dz)
            heading_error_rad = _wrap_angle_rad(desired_yaw - float(yaw_rad))

            # Xác định waypoint hiện tại để lấy speed_limit và instruction
            cur_wp = self.route_waypoints[-1]
            for wp in reversed(self.route_waypoints):
                if loc["along_m"] >= (wp.get("cumulative_distance_m") or 0.0) - 25.0:
                    cur_wp = wp
                    break

            return {
                "timestamp": time.monotonic(),
                "valid": True,
                "heading_error_rad": round(heading_error_rad, 5),
                "target_speed_kmh": float(cur_wp["target_speed_kmh"] or 45.0),
                "waypoint": (float(target_world[0]), float(target_world[1])),
                "turn_instruction": cur_wp["uk_instruction"],
                "distance_to_goal_m": round(loc["dist_to_goal_m"], 2),
                "current_segment_label": cur_wp["label"],
            }

        # Phương án dự phòng (fallback khi chưa có hình học lộ trình): tìm waypoint gần nhất
        dists = [
            math.hypot(wp["world_x"] - position_x, wp["world_z"] - position_z)
            for wp in self.route_waypoints
        ]
        nearest_idx = int(np.argmin(dists))
        goal_wp = self.route_waypoints[-1]
        dist_to_goal = math.hypot(goal_wp["world_x"] - position_x, goal_wp["world_z"] - position_z)

        target_idx = min(len(self.route_waypoints) - 1, nearest_idx + 1)
        if dists[nearest_idx] > 120.0 and nearest_idx == 0:
            target_idx = 0
        target_wp = self.route_waypoints[target_idx]

        dx = target_wp["world_x"] - position_x
        dz = target_wp["world_z"] - position_z
        desired_yaw = math.atan2(dx, dz)
        heading_error_rad = _wrap_angle_rad(desired_yaw - float(yaw_rad))

        return {
            "timestamp": time.monotonic(),
            "valid": True,
            "heading_error_rad": round(heading_error_rad, 5),
            "target_speed_kmh": float(target_wp["target_speed_kmh"] or 45.0),
            "waypoint": (float(target_wp["world_x"]), float(target_wp["world_z"])),
            "turn_instruction": target_wp["uk_instruction"],
            "distance_to_goal_m": round(dist_to_goal, 2),
            "current_segment_label": target_wp["label"],
        }
