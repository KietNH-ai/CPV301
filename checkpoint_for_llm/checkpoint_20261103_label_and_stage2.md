# Checkpoint — TuSimple Label Analysis, Stage 2 Evaluation, FH4 Stabilization, Speed Optimization & Anti-Hallucination Road Scanning

**Created/Updated:** 2026-11-04 02:28 +07:00

## 1. Summary of Completed Work
1. **Full TuSimple Label Analysis (`notebooks/00_label_analysis.ipynb` + `data/processing/label_analysis/`)**:
   - Audited 100% of TuSimple JSONL files (`6,408` clips total: `3,626` in `train_set` + `2,782` in `test_set`).
   - Verified `0` broken image paths, `0` overlapping clips between train (`2,901`) and valid (`725`), and `100%` `h_samples` compliance (`160..710`, 56 rows).
2. **Stage 1 Preprocessing (`notebooks/01_pre_processing.ipynb` + `data/processing/`)**:
   - Exported `2,901` CLAHE-enhanced training images/labels (`data/processing/train/`) and `725` validation images/labels (`data/processing/valid/`).
3. **Stage 2 Training & Real Ground-Truth Evaluation (`notebooks/02_train.ipynb`)**:
   - Evaluated against real TuSimple validation Ground-Truth polygons (`data/processing/valid/labels/*.txt`) and exported metrics to `data/output/evaluation_metrics.csv`.
4. **Synchronized Imports & Dependency Files**:
   - Updated `tests/test_control_and_main.py` after removal of `src/pipelinebuilder.py` (`from src.pipeline_builder import PipelineBuilder`).
   - Created `requirements(testFH4).txt` (full dependencies with FH4 live capture/gamepad) and updated `requirements.txt` (core processing/evaluation dependencies) and `README.md`.
5. **Anti-Hallucination & Ghost Lane Elimination (`src/detection/hough_transform.py`, `src/proposed_method/yolo_spline.py`, `main.py`)**:
   - **Mandatory Road Paint Ridge Support**: In `_refine_curve_with_paint_ridge`, replaced falling back to raw Canny edges (`initial_pts`) with `return None` when genuine paint ridge support `< 2` points. Grass, fences, walls, shadows, and desktop text no longer trigger ghost lines.
   - **Calibrated Paint Mask Thresholds & Density Sanity Check**: In `_extract_paint_ridge_mask`, set white paint (`ridge >= 8`, `gray >= 80`, `chroma <= 42`) and yellow paint (`ridge >= 7`, `r >= 95`, `g >= 85`). Added excessive paint density rejection (`paint_nz > 0.085 * roi_area`) to reject dense grids, office walls, and white glare.
   - **Position & Projection-Based Segment Classification**: Replaced naive `slopes < 0` vs `slopes > 0` classification with bottom projection $x_{proj} = x_{mean} + (H - y_{mean}) / \text{slope}$ and spatial centroid gating. On sharp bends (where both left and right lane boundaries tilt in the same direction and have negative or positive slope), segments are now properly separated into left/right without false syntheses or lane dropping.
   - **Geometric Plausibility Gating**: Added vertical reach gating (`np.max(curve[:, 1]) >= height * 0.62`), lane width limits (`0.14*W <= sep_bot <= 0.92*W`), and ego side gating ($x_{l,bot} \le mid\_x \cdot 1.15$ and $x_{r,bot} \ge mid\_x \cdot 0.85$).
   - **Eliminated YOLO Phantom Coordinates**: In `_extract_yolo_guided_ego_masks`, removed the fallback that scanned fixed highway positions ($x = 0.46*W / 0.58*W$) when YOLO detected no lane boxes.
   - **Zero Phantom Hold in Dashboard Loop**: In `_stabilize_curves` (`main.py`), when no lane boundaries are observed, state is immediately cleared (`_ema_left_x = None`, `_ema_right_x = None`, returning `[]`) rather than holding old ghost lanes when off-road.
   - **Optimized 4-Quadrant Dashboard Render Speed**: Downscaled PiP image directly from `tile` (already downscaled) rather than `base_img` (720p) in `src/map/quad_dashboard.py`, keeping performance $> 30$ FPS.
   - **Validation Results**: Non-road scenes (blank, grass lawn, mud ruts, wall grid) detect **`0`** curves. Real road (`data/raw/20.jpg`) and synthetic curves detect exactly **`2`** valid curves. All **`23/23`** unit tests pass in `11.4s`.

## 2. Current Task & Progress
- All anti-hallucination fixes and road detection stability improvements have been implemented, verified, and tested with 100% green test suite.

## 3. Modified & Created Files
- `src/detection/hough_transform.py`: Paint ridge support requirement, excessive paint density check, curved line classification, geometric gating.
- `src/proposed_method/yolo_spline.py`: Removed hallucinating nominal scan when YOLO detection is empty.
- `src/map/quad_dashboard.py`: Optimized PiP downscaling for $\ge 30$ FPS.
- `main.py`: Immediate state reset on no road in `_stabilize_curves`.
- `tests/test_preprocessing_and_hough.py`: Updated curved road benchmark comparison assertion.
- `requirements(testFH4).txt`
- `requirements.txt`
- `README.md`
- `checkpoint_for_llm/checkpoint_20261103_label_and_stage2.md`

## 4. Next Action Steps
1. Run `python main.py --view-only --video data/raw/test_driving.mp4 --dest-node 12` to visually inspect lane detection on video.
2. Run live with Forza Horizon 4 (`python main.py --view-only --detector hough --dest-node 12`) to verify off-road behavior (lawn/dirt returns 0 lanes, asphalt road returns accurate ego lane).

## 5. Known Issues & Blockers
- None. All 23 unit tests pass.
