"""Run from the project root: python src/dp2/deep_ensembles {train,evaluate}."""
from pathlib import Path
import sys

# Also support direct directory execution without installing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python src/dp2/deep_ensembles {train,evaluate} [options]")
        print("  train     Train independent ensemble members")
        print("  evaluate  Export uncertainty from a trained ensemble")
        print("Use train --help or evaluate --help for options.")
        return
    command = sys.argv.pop(1)
    if command == "train":
        from dp2.deep_ensembles.train import main as run
    elif command == "evaluate":
        from dp2.deep_ensembles.cli import main as run
    else:
        raise SystemExit(f"Unknown command: {command}. Choose train or evaluate.")
    run()


if __name__ == "__main__":
    main()
