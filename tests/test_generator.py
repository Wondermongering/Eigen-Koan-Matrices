import sys, types

# ensure minimal sklearn stubs for import
sk = types.ModuleType('sklearn')
cluster = types.ModuleType('sklearn.cluster')
class KMeans:
    def __init__(self, *a, **k):
        pass
    def fit_predict(self, X):
        return [0] * len(X)
    @property
    def cluster_centers_(self):
        return [[0 for _ in range(len(X[0]))] for X in [X] if X]
cluster.KMeans = KMeans
metrics = types.ModuleType('sklearn.metrics')
pair = types.ModuleType('sklearn.metrics.pairwise')
pair.cosine_similarity = lambda X, Y=None: [[0 for _ in range(len(X))] for _ in range(len(X))]
metrics.pairwise = pair
sk.cluster = cluster
sk.metrics = metrics
sys.modules['sklearn'] = sk
sys.modules['sklearn.cluster'] = cluster
sys.modules['sklearn.metrics'] = metrics
sys.modules['sklearn.metrics.pairwise'] = pair

from eigen_koan_matrix import EigenKoanMatrix
from ekm_generator import EKMGenerator

class DummyGenerator(EKMGenerator):
    def _select_diverse_elements(self, elements, count, embedding_key=None):
        return elements[:count]

    def _find_contrastive_pair(self, elements):
        return elements[0], elements[1]

    def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
        return [f"{emotion_name}_{i}" for i in range(count)]

def test_generate_ekm_basic():
    gen = DummyGenerator()
    ekm = gen.generate_ekm(size=2)
    assert isinstance(ekm, EigenKoanMatrix)
    assert ekm.size == 2
    assert len(ekm.task_rows) == 2
    assert len(ekm.constraint_cols) == 2
    for i in range(2):
        assert ekm.cells[i][i] != "{NULL}"
        assert ekm.cells[i][1 - i] != "{NULL}"
    assert ekm.name == "Generated EKM 2x2"

def test_generate_themed_matrices_basic():
    gen = DummyGenerator()
    themes = ["ethics", "custom"]
    mats = gen.generate_themed_matrices(themes, size=2)
    assert set(mats.keys()) == set(themes)
    for matrix in mats.values():
        assert isinstance(matrix, EigenKoanMatrix)
        assert matrix.size == 2
