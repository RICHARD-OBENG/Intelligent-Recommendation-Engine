import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python import path
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import run_pipeline

if __name__ == "__main__":
    run_pipeline()