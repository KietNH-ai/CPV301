import os
from pathlib import Path

# Get the absolute path to the project root (assuming path.py is in CPV301/src)
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Define standard data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"

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

# Automatically create directories if they are missing
ensure_dir(RAW_DATA_DIR)
ensure_dir(PROCESSED_DATA_DIR)
ensure_dir(OUTPUT_DATA_DIR)
ensure_dir(WEIGHTS_DIR)
ensure_dir(CONFIG_DIR)
ensure_dir(CHECKPOINT_FOR_LLM_DIR)

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()

print("File hiện tại:", __file__)
print("Thư mục cha:", Path(__file__).parent)
print("ROOT_DIR:", ROOT_DIR)
print("ROOT_DIR tồn tại:", ROOT_DIR.exists())
