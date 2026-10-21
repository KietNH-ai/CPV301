import cv2
import sys
from pathlib import Path

# bắt nó về root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.path import RAW_DATA_DIR, OUTPUT_DATA_DIR


class PipelineBuilder:
    def __init__(self, detector=None, spline_fitter=None):
        """
        Builds the Lane Detection Pipeline, allowing you to easily swap methods
        (like YOLO, LaneNet, or Hough Transform) for research comparison.

        Args:
            detector: The lane detection model (e.g., YOLOLaneDetector).
            spline_fitter: The post-processing spline curve fitter.
        """
        self.detector = detector
        self.spline_fitter = spline_fitter

    def process_frame(self, frame):
        """
        Xử lý 1 frame: detect -> (spline fit) -> trả về frame đã chú thích.
        Dùng chung cho process_video() và vòng lặp thời gian thực (FH4 view-only / control).
        """
        masks = []
        if self.detector:
            # YOLO outputs bounding boxes/masks; Hough/LaneNet trả (lines, masks)
            results, masks = self.detector.detect(frame)

        # Step 2: Post-processing (Spline Fitting)
        if self.spline_fitter and masks:
            # Smooth the lane and draw it
            return self.spline_fitter.fit_and_draw(frame, masks)
        if self.detector and masks:
            # Fallback: detector tự vẽ (ví dụ proposed method đã vẽ spline + vùng đi được nội bộ)
            return self.detector.draw_lanes(frame, masks)
        return frame

    def process_video(self, input_filename, output_filename):
        """
        Reads a video, runs detection & fitting on each frame, and saves the output.
        """
        input_path = RAW_DATA_DIR / input_filename
        output_path = OUTPUT_DATA_DIR / output_filename
        if not input_path.exists():
            raise FileNotFoundError(f"Cannot find input video: {input_path}")
        cap = cv2.VideoCapture(str(input_path))

        # Get video properties for saving the output
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Using mp4v codec for standard mp4 video files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    # Example Usage:
    # from src.yolo_detector import YOLOLaneDetector
    # from src.fitting.spline_fitter import SplineFitter
    # detector = YOLOLaneDetector('weights/yolo11n-seg.pt')
    # spline_fitter = SplineFitter(spline_order=3, smoothing=3.0)
    # pipeline = PipelineBuilder(detector=detector, spline_fitter=spline_fitter)
    # pipeline.process_video('my_driving_video.mp4', 'output_result.mp4')
