"""
Lane Detection Project (CPV301)
"""

import sys
from pathlib import Path
#bắt nó về root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.path import (
    ROOT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DATA_DIR,
    MODEL_DIR,
    WEIGHTS_DIR,
    CONFIG_DIR,
    DOCS_DIR,
    CHECKPOINT_FOR_LLM_DIR
)
from src.yolo_detector import YOLOLaneDetector
from src.fitting.spline_fitter import SplineFitter
from src.pipelinebuilder import PipelineBuilder

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUTPUT_DATA_DIR",
    "MODEL_DIR",
    "WEIGHTS_DIR",
    "CONFIG_DIR",
    "DOCS_DIR",
    "CHECKPOINT_FOR_LLM_DIR",
    "YOLOLaneDetector",
    "SplineFitter",
    "PipelineBuilder"
]
