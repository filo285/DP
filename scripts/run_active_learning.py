"""Run active-learning experiments and evaluate their acquisition efficiency."""
import os
from pathlib import Path
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dp2.active_learning.cli import main

if __name__ == "__main__":
    main()
