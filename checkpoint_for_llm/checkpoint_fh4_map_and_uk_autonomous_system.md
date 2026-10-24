# Checkpoint — FH4 Map Navigation (UK Left-Hand Traffic), Preprocessing (`data/processing`) & Training (`data/output`)

## 1. Summary of Completed Work
- **Created `fh4_map/` Directory Structure**:
  - `fh4_map/raw/fh4_uk_map.png` (raw Forza Horizon 4 Great Britain map).
  - `fh4_map/processing/` (`road_mask.png`, `road_skeleton.png`, `road_graph_overlay.png`, `road_graph.json`, `planned_route_map.png`, `fh4_map_route_chart.png`).
- **Implemented Map Analysis, Destination Marking & UK Left-Hand Traffic Autonomous Driving**:
  - `src/map/map_analyzer.py` (`FH4MapAnalyzer`) and `src/map/navigator.py` (`FH4MapNavigator`) extract road network graphs, support interactive (`--pick-dest`) or CLI (`--dest-node`, `--dest-pixel`, `--dest-world`) destination marking, compute A* optimal routes, and provide live navigation guidance using 3D telemetry (`PositionX`, `PositionY`, `PositionZ`, `Yaw`).
  - `src/control/pure_pursuit.py` implements `classify_uk_lanes()` and UK Left-Hand Traffic path extraction (`driving_side="uk_left"`), keeping the vehicle in the left lane and adjusting lookahead for UK turns and clockwise roundabouts.
  - `src/control/controller.py` (`FH4AutonomousController`) fuses vision + map guidance (`source="vision+map_uk"`) and stops automatically upon arriving at the destination.
  - Updated `test_fh4/fh4_telemetry.py`, `test_fh4/FH4_controler.py`, and `main.py`.
- **Upgraded `notebooks/01_pre_processing.ipynb` & `src/preprocessing/batch_processor.py`**:
  - Implemented 6-stage road preprocessing pipeline (CLAHE + Gaussian blur + Canny + HLS white/yellow mask + UK polygon ROI + blob filtering + B-Spline + UK lane classification).
  - Exported processed images and YOLO-seg dataset splits (`train/`, `valid/`, `highlighted/`) into `data/processing/`.
- **Upgraded `notebooks/02_train.ipynb` & `src/evaluation/train_and_export.py`**:
  - Configured `config/data.yaml` to train from `data/processing/`.
  - Trained YOLO11-seg (`yolo11n-seg.pt`) and exported weights (`best.pt`), validation predictions (`predictions/`), and evaluation reports to `data/output/`.
- **Placed All Charts (`.png`) and CSV Metrics (`.csv`) at Project Root (`ROOT_DIR`)**:
  - `fh4_map_road_graph.csv`, `fh4_route_waypoints.csv`, `fh4_map_route_chart.png`
  - `preprocessing_metrics.csv`, `preprocessing_pipeline_chart.png`
  - `training_results.csv`, `evaluation_metrics.csv`, `training_and_evaluation_chart.png`
  - `fh4_log.csv`
- **Verification**:
  - All 21/21 unit tests in `pytest -v` pass 100%.

## 2. Current Task & Progress
- All 6 user requirements are 100% completed and verified.

## 3. Modified & Created Files
- `src/path.py`
- `src/map/__init__.py`, `src/map/map_analyzer.py`, `src/map/navigator.py`
- `src/control/pure_pursuit.py`, `src/control/controller.py`, `src/control/__init__.py`
- `src/preprocessing/batch_processor.py`, `src/preprocessing/__init__.py`
- `src/evaluation/train_and_export.py`
- `test_fh4/fh4_telemetry.py`, `test_fh4/FH4_controler.py`
- `main.py`, `config/data.yaml`
- `notebooks/01_pre_processing.ipynb`, `notebooks/02_train.ipynb`
- `tests/test_control_and_main.py`
- `docs/explain.md`, `README.md`
- Root CSVs & Charts: `fh4_map_road_graph.csv`, `fh4_route_waypoints.csv`, `fh4_map_route_chart.png`, `preprocessing_metrics.csv`, `preprocessing_pipeline_chart.png`, `training_results.csv`, `evaluation_metrics.csv`, `training_and_evaluation_chart.png`, `fh4_log.csv`.

## 4. Next Action Steps
1. Open `notebooks/01_pre_processing.ipynb` or `notebooks/02_train.ipynb` in Jupyter / VS Code using `.venv` kernel to inspect interactive outputs.
2. Run `python main.py --control --detector hough --pick-dest` when playing Forza Horizon 4 offline (bumper cam, UDP telemetry enabled on port 5300).

## 5. Known Issues & Blockers
- None.
