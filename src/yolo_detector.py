import cv2
import numpy as np
from ultralytics import YOLO

class YOLOLaneDetector:
    def __init__(self, model_path='yolo11n-seg.pt'):
        """
        Initialize the YOLOv11 instance segmentation model for lane detection.
        
        Args:
            model_path (str): Path to the trained YOLO weights (e.g., 'weights/best.pt').
                              Defaults to the nano segmentation model for quick testing.
        """
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
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        lane_masks = []
        if results[0].masks is not None:
            # Extract polygon points from the segmentation mask
            for mask in results[0].masks.xy:
                lane_masks.append(np.array(mask, dtype=np.int32))
                
        return results, lane_masks

    def draw_lanes(self, frame, lane_masks):
        """
        Helper method to draw the raw YOLO masks directly onto the frame.
        (Later, Spline fitting will replace this basic drawing method).
        """
        annotated_frame = frame.copy()
        cv2.polylines(annotated_frame, lane_masks, isClosed=False, color=(0, 255, 0), thickness=3)
        return annotated_frame
