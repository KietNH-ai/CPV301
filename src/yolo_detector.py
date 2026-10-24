import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOLaneDetector:
    def __init__(self, model_path='yolo11m-seg.pt', confidence_threshold: float = 0.5):
        """
        Initialize the YOLOv11 instance segmentation model for lane detection.
        
        Args:
            model_path (str): Path to the trained YOLO weights (e.g., 'weights/best.pt').
                              Defaults to the nano segmentation model for quick testing.
            confidence_threshold (float): Minimum confidence threshold for detections.
        """
        if YOLO is None:
            raise ImportError(
                "The 'ultralytics' package is required for YOLOLaneDetector. "
                "Install it with: pip install ultralytics"
            )
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        """
        Run inference on a single image frame.
        
        Args:
            frame (np.ndarray): The image frame (from cv2.imread or VideoCapture).
            
        Returns:
            results: Raw YOLO results object.
            lane_masks: A list of numpy arrays containing (x,y) coordinates of the lane pixels.
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        lane_masks = []
        if results and results[0].masks is not None:
            for mask in results[0].masks.xy:
                if len(mask) >= 2:
                    lane_masks.append(np.array(mask, dtype=np.int32))
                
        return results, lane_masks

    def draw_lanes(self, frame, lane_masks):
        """
        Helper method to draw the raw YOLO masks directly onto the frame.
        (Later, Spline fitting will replace this basic drawing method).
        """
        annotated_frame = frame.copy()
        if lane_masks:
            cv2.polylines(annotated_frame, lane_masks, isClosed=False, color=(0, 255, 0), thickness=3)
        return annotated_frame
