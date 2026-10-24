import os
from pathlib import Path

# Get the absolute path to the project root (assuming path.py is in CPV301/src)
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Define standard data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSING_DATA_DIR = DATA_DIR / "processing"
OUTPUT_DATA_DIR = DATA_DIR / "output"
OUTPUT_DIR = OUTPUT_DATA_DIR

# Define Forza Horizon 4 Map directories
FH4_MAP_DIR = ROOT_DIR / "fh4_map"
FH4_MAP_RAW_DIR = FH4_MAP_DIR / "raw"
FH4_MAP_PROCESSING_DIR = FH4_MAP_DIR / "processing"

# Define other important project directories
MODEL_DIR = ROOT_DIR / "model"
WEIGHTS_DIR = ROOT_DIR / "weights"
CONFIG_DIR = ROOT_DIR / "config"
DOCS_DIR = ROOT_DIR / "docs"
CHECKPOINT_FOR_LLM_DIR = ROOT_DIR / "checkpoint_for_llm"


def ensure_dir(path: Path):
    """Utility to ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Automatically create active directories if they are missing
ensure_dir(RAW_DATA_DIR)
ensure_dir(PROCESSING_DATA_DIR)
ensure_dir(OUTPUT_DATA_DIR)
ensure_dir(FH4_MAP_RAW_DIR)
ensure_dir(FH4_MAP_PROCESSING_DIR)
ensure_dir(CONFIG_DIR)
ensure_dir(CHECKPOINT_FOR_LLM_DIR)
