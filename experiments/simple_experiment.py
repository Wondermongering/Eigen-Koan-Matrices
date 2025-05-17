"""Run a tiny Eigen-Koan Matrix experiment with a dummy model."""

import sys
import types
from pathlib import Path

# Ensure repository root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub optional dependencies so the example works without extra packages
sys.modules.setdefault('numpy', types.ModuleType('numpy'))
console_mod = types.ModuleType('rich.console')
class DummyConsole:
    def print(self, *args, **kwargs):
        pass
console_mod.Console = DummyConsole
sys.modules['rich.console'] = console_mod

table_mod = types.ModuleType('rich.table')
class DummyTable:
    def __init__(self, *args, **kwargs):
        pass
    def add_column(self, *args, **kwargs):
        pass
    def add_row(self, *args, **kwargs):
        pass
table_mod.Table = DummyTable
sys.modules['rich.table'] = table_mod
sys.modules.setdefault('rich', types.ModuleType('rich'))

from eigen_koan_matrix import create_philosophical_ekm


def dummy_model(prompt: str) -> str:
    """Return a short acknowledgement for the given prompt."""
    return f"[dummy] Received: {prompt[:30]}..."


def run():
    matrix = create_philosophical_ekm()
    path = [0] * matrix.size
    result = matrix.traverse(dummy_model, path=path)
    print("Prompt:", result["prompt"])
    print("Response:", result["response"])
    print("Main strength:", result["main_diagonal_strength"])
    print("Anti strength:", result["anti_diagonal_strength"])


if __name__ == "__main__":
    run()
