import cv2
import sys
from pathlib import Path
from typing import List, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.path import RAW_DATA_DIR, OUTPUT_DATA_DIR


class PipelineBuilder:
    def __init__(self, detector=None, spline_fitter=None):
        """
        Builds the Lane Detection Pipeline, allowing you to easily swap methods
        (like YOLO, LaneNet, or Hough Transform) for research comparison.
        """
        self.detector = detector
        self.spline_fitter = spline_fitter
        self.last_results: Any = None
        self.last_masks: List[Any] = []
        self.last_fitted_curves: List[Any] = []

    def process_frame(self, frame):
        """
        Xử lý 1 frame: detect -> (spline fit) -> trả về frame đã chú thích.
        Lưu lại `self.last_masks` và `self.last_fitted_curves` để vòng lặp điều khiển
        (Pure Pursuit / HUD) tái sử dụng mà không cần chạy detect hay spline fit lần 2.
        """
        masks = []
        results = None
        if self.detector:
            results, masks = self.detector.detect(frame)

        self.last_results = results
        self.last_masks = masks or []

        if self.spline_fitter and masks:
            self.last_fitted_curves = self.spline_fitter.fit_multiple_lanes(masks)
            if hasattr(self.spline_fitter, "draw_fitted_curves"):
                return self.spline_fitter.draw_fitted_curves(frame, self.last_fitted_curves)
            return self.spline_fitter.fit_and_draw(frame, masks)

        self.last_fitted_curves = self.last_masks
        if self.detector and masks:
            return self.detector.draw_lanes(frame, masks)
        return frame.copy()

    def process_video(self, input_filename, output_filename):
        """Reads a video, runs detection & fitting on each frame, and saves the output."""
        input_path = Path(input_filename)
        if not input_path.is_absolute() and not input_path.exists():
            input_path = RAW_DATA_DIR / input_filename
        output_path = Path(output_filename)
        if not output_path.is_absolute() and output_path.parent == Path("."):
            output_path = OUTPUT_DATA_DIR / output_filename

        if not input_path.exists():
            raise FileNotFoundError(f"Cannot find input video: {input_path}")
        cap = cv2.VideoCapture(str(input_path))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Starting pipeline processing for {input_filename}...")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.process_frame(frame)
            out.write(annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        cap.release()
        out.release()
        print(f"Processing complete! Saved to {output_path}")


if __name__ == "__main__":
    print("Pipeline Builder ready.")
