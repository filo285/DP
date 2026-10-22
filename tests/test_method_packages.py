"""Include method-local tests in the project's standard unittest command."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))


def load_tests(loader, tests, pattern):
    for path in sorted((SOURCE / "dp2").glob("*/tests/test_*.py")):
        name = ".".join(path.relative_to(SOURCE).with_suffix("").parts)
        tests.addTests(loader.loadTestsFromName(name))
    return tests
