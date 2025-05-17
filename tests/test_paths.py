import sys
import types

import pytest

# Stub external dependencies
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

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect, create_random_ekm


def test_get_diagonal_sequences():
    ekm = create_random_ekm(3)
    main_seq, anti_seq = ekm.get_diagonal_sequences()
    assert main_seq == ekm.main_diagonal.tokens
    for i, token in enumerate(anti_seq):
        j = ekm.size - 1 - i
        if i == j:
            assert token == ekm.main_diagonal.tokens[i]
        else:
            assert token == ekm.anti_diagonal.tokens[i]


def test_generate_all_paths_small_matrix():
    ekm = create_random_ekm(2)
    paths = ekm.generate_all_paths()
    assert len(paths) == 4  # 2^2 possible paths
    assert paths[0] == [0, 0]
    assert paths[-1] == [1, 1]


def test_analyze_path_paradox_counts():
    tasks = ["Do A", "Do B"]
    constraints = ["be precise", "speak metaphorically"]
    main_affect = DiagonalAffect(name="pos", tokens=["m1", "m2"], description="", valence=0.2, arousal=0.2)
    anti_affect = DiagonalAffect(name="neg", tokens=["a1", "a2"], description="", valence=-0.2, arousal=0.3)
    ekm = EigenKoanMatrix(2, tasks, constraints, main_affect, anti_affect)

    analysis = ekm.analyze_path_paradox([0, 1])
    assert analysis["tension_count"] == 1
    assert analysis["main_diagonal_count"] == 2
    assert analysis["anti_diagonal_count"] == 0
