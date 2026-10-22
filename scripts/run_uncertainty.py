"""Run MC dropout; implementation is isolated in dp2.mc_dropout."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dp2.mc_dropout.cli import main
if __name__ == "__main__":
    main()
