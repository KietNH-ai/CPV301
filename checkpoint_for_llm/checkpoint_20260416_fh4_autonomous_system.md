# Checkpoint: Forza Horizon 4 Autonomous System — 4-Quadrant Dashboard (>= 30 FPS) & UK Rural Road Curve/Boundary Fix

- **Timestamp**: 2026-10-24T06:55:00+07:00
- **Project Root**: `h:\FPT\4\CPV\project\CPV301`

---

## 1. Summary of Completed Work
1. **Diagnosed 4 Root Causes from Live FH4 UK Country Road Screenshot (`media_1792798473475_0baf0d87.png`)**:
   - **Root Cause 1 (Straight-line fitting `deg=1`)**: `HoughLaneDetector.detect_from_mask()` previously fitted `np.polyfit(..., deg=1)` to all Hough points, turning curved roads over hill crests into a rigid straight line (`Curv: 0.0000`) even when `SplineFitter` ran afterward.
   - **Root Cause 2 (Missing Unmarked Left Boundary `uk_left_edge` on UK Country Roads)**: On UK rural roads in Forza Horizon 4, only the center divider (`uk_center_divider`) has white paint; the left road edge is an unmarked asphalt-to-grass/curb boundary where standard `cv2.Canny` found zero lane lines.
   - **Root Cause 3 (Yellow `centerline` & Red `lookahead_point` shooting into the left grass hillside)**: When `left_curve` was missing (`None`), `extract_path_from_lanes()` subtracted a large `0.35 * base_half_width` offset at the top of the ROI (`y_top`) from the steeply angled center divider, pushing `centerline` and `lookahead_point` completely off the road onto the left grass bank.
   - **Root Cause 4 (Window Client Area & PiP Mask)**: Captured frames included non-game left borders (VS Code activity bar at `x < 45px`), and the PiP mask in Quadrant 2 ran Canny across the entire frame (including sky and trees) instead of the road ROI.

2. **Implemented Complete Fixes Across 5 Modules**:
   - [`src/preprocessing/preprocessor.py`](file:///h:/FPT/4/CPV/project/CPV301/src/preprocessing/preprocessor.py):
     - Added **Horizontal Local-Contrast Top-Hat Ridge Filtering** (`cv2.subtract(gray, cv2.boxFilter(gray, cv2.CV_8U, (31, 7)))`) combined with Canny edges in `detect_edges()`, preserving thin distant curved white/yellow lane markings up to `roi_top_y_ratio = 0.38`.
     - Adjusted `roi_bottom_left_x_ratio = 0.05` to exclude left-edge window border artifacts.
   - [`src/detection/hough_transform.py`](file:///h:/FPT/4/CPV/project/CPV301/src/detection/hough_transform.py):
     - Replaced `deg=1` straight-line fitting with `_fit_curved_lane()` (slice-median binning + MAD outlier rejection + blended degree-2/3 polynomial & smoothed empirical medians) and `_refine_curve_with_paint_ridge()` (sliding-window tracking along white/yellow road paint ridges all the way to the hill crest).
     - Added `_detect_uk_left_road_edge()`: automatically detects the **unmarked UK Left Asphalt-vs-Grass Road Boundary (`uk_left_edge`)** by sampling the reference cool-grey asphalt immediately to the left of `uk_center_divider` and scanning leftward for the grass/soil color-warmth jump (`(R - B) + 0.65*(G - B)`), color dissimilarity, and Sobel edge transition within the 3D camera perspective lane width corridor.
   - [`src/control/pure_pursuit.py`](file:///h:/FPT/4/CPV/project/CPV301/src/control/pure_pursuit.py):
     - Upgraded `_interp_lane_x()` with smooth tangent extrapolation and updated `extract_path_from_lanes()` so `y_top`/`y_bottom` adapt to the detected curves, perspective width converges toward `y_vanish = 0.31 * H`, and `x_center` is strictly bounded inside `[x_left + margin, x_right - margin]`.
   - [`src/map/quad_dashboard.py`](file:///h:/FPT/4/CPV/project/CPV301/src/map/quad_dashboard.py):
     - Updated `render_lane_detect_quadrant()` to render cleanly on `raw_frame`, fill the green UK Left Corridor properly, and display a cached, high-FPS **ROI Lane & Edge Mask PiP** showing painted ridges and fitted `uk_left_edge`/`uk_center_divider` curves without sky/tree clutter.
   - [`src/capture/screen_capture.py`](file:///h:/FPT/4/CPV/project/CPV301/src/capture/screen_capture.py):
     - Added `get_fh4_window_client_region()` using Win32 `GetClientRect` + `ClientToScreen` to automatically lock onto the clean client area of the `Forza Horizon 4` window when running on Windows.

3. **Verification Results**:
   - Tested directly on the road scene from `media_1792798473475_0baf0d87.png` (`scratch/fixed_fh4_uk_road.png`):
     - `Detected curves: 2 | uk_lane_type: uk_standard_left_lane | valid: True | conf: 0.95`
     - `Classified keys: ['uk_left_edge', 'uk_center_divider']`
     - `Heading error deg: -10.11 | Curvature: 0.0798 | Steer: -19.61`
   - All unit tests in `tests/test_control_and_main.py` pass (`11/11 PASSED`, including `test_quad_dashboard_scroll_zoom_click_and_30fps`).

---

## 2. Current Task & Progress
- **Status**: Completed and verified.

---

## 3. Modified & Created Files
- `src/preprocessing/preprocessor.py` (Modified: Horizontal local-contrast ridge + Canny edge detection, updated ROI ratios)
- `src/detection/hough_transform.py` (Modified: Curved lane fitting `_fit_curved_lane`, sliding-window paint ridge tracker `_refine_curve_with_paint_ridge`, and unmarked UK left asphalt-to-grass boundary detector `_detect_uk_left_road_edge`)
- `src/control/pure_pursuit.py` (Modified: Smooth tangent interpolation `_interp_lane_x`, perspective-consistent path extraction & corridor clamping in `extract_path_from_lanes`)
- `src/map/quad_dashboard.py` (Modified: Clean `render_lane_detect_quadrant` corridor rendering + cached high-FPS `ROI LANE MASK` PiP)
- `src/capture/screen_capture.py` (Modified: Automatic Win32 `Forza Horizon 4` client-area tracking `get_fh4_window_client_region`)
- `checkpoint_for_llm/checkpoint_20260416_fh4_autonomous_system.md` (Updated)

---

## 4. Next Action Steps
1. Run `.\.venv\Scripts\python.exe main.py --view-only --detector hough --dest-node 12` (or `--control` in FH4 Offline mode with Bumper Cam) to test live on Forza Horizon 4.
2. Fine-tune `--target-speed` or `PurePursuitController` gains if driving high-speed S-curves.

---

## 5. Known Issues & Blockers
- None.
