import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / ".index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 5

SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")
SARVAM_REASONING_EFFORT = os.getenv("SARVAM_REASONING_EFFORT", "low")
SARVAM_MAX_TOKENS = int(os.getenv("SARVAM_MAX_TOKENS", "1500"))
