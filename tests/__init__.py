from contextlib import contextmanager
from unittest import mock
import types
import sys

@contextmanager
def patch_external_libs():
    modules = {}

    # rich stubs
    rich_mod = types.ModuleType('rich')
    console_mod = types.ModuleType('rich.console')
    class Console:
        def print(self, *args, **kwargs):
            pass
    console_mod.Console = Console
    table_mod = types.ModuleType('rich.table')
    class Table:
        def __init__(self, *a, **k):
            pass
        def add_column(self, *a, **k):
            pass
        def add_row(self, *a, **k):
            pass
    table_mod.Table = Table
    progress_mod = types.ModuleType('rich.progress')
    class Progress:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def track(self, iterable, *args, **kwargs):
            return iterable
    progress_mod.Progress = Progress
    rich_mod.console = console_mod
    rich_mod.table = table_mod
    rich_mod.progress = progress_mod
    modules.update({
        'rich': rich_mod,
        'rich.console': console_mod,
        'rich.table': table_mod,
        'rich.progress': progress_mod,
    })

    # sklearn stubs
    sk_mod = types.ModuleType('sklearn')
    cluster_mod = types.ModuleType('sklearn.cluster')
    class KMeans:
        def __init__(self, *a, **k):
            pass
        def fit_predict(self, X):
            return [0] * len(X)
        @property
        def cluster_centers_(self):
            return [[0]]
    cluster_mod.KMeans = KMeans
    metrics_mod = types.ModuleType('sklearn.metrics')
    pair_mod = types.ModuleType('sklearn.metrics.pairwise')
    class SimMat(list):
        def tolist(self):
            return self
    pair_mod.cosine_similarity = lambda X, Y=None: SimMat([[0 for _ in range(len(X))] for _ in range(len(X))])
    metrics_mod.pairwise = pair_mod
    sk_mod.cluster = cluster_mod
    sk_mod.metrics = metrics_mod
    sk_mod.decomposition = types.ModuleType('sklearn.decomposition')
    sk_mod.decomposition.PCA = lambda *a, **k: None
    sk_mod.manifold = types.ModuleType('sklearn.manifold')
    sk_mod.manifold.TSNE = lambda *a, **k: None
    fe_mod = types.ModuleType('sklearn.feature_extraction')
    fe_text_mod = types.ModuleType('sklearn.feature_extraction.text')
    class TF:
        def __init__(self, *a, **k):
            pass
        def fit_transform(self, X):
            class Mat(list):
                def tolist(self):
                    return self
            return Mat([[1 for _ in X] for _ in X])
    fe_text_mod.TfidfVectorizer = TF
    fe_mod.text = fe_text_mod
    modules.update({
        'sklearn': sk_mod,
        'sklearn.cluster': cluster_mod,
        'sklearn.metrics': metrics_mod,
        'sklearn.metrics.pairwise': pair_mod,
        'sklearn.feature_extraction': fe_mod,
        'sklearn.feature_extraction.text': fe_text_mod,
        'sklearn.decomposition': sk_mod.decomposition,
        'sklearn.manifold': sk_mod.manifold,
    })

    # pandas stub
    pandas_mod = types.ModuleType('pandas')
    pandas_mod.DataFrame = lambda *a, **k: None
    modules['pandas'] = pandas_mod

    # matplotlib stub
    plt_mod = types.ModuleType('matplotlib.pyplot')
    plt_mod.figure = lambda *a, **k: None
    plt_mod.subplots = lambda *a, **k: ((types.SimpleNamespace(), types.SimpleNamespace()), None)
    plt_mod.close = lambda *a, **k: None
    plt_mod.savefig = lambda *a, **k: None
    modules['matplotlib'] = types.ModuleType('matplotlib')
    modules['matplotlib.pyplot'] = plt_mod

    # seaborn stub
    modules['seaborn'] = types.ModuleType('seaborn')

    # wordcloud stub
    wc_mod = types.ModuleType('wordcloud')
    wc_mod.WordCloud = lambda **k: types.SimpleNamespace(generate_from_frequencies=lambda f: None)
    modules['wordcloud'] = wc_mod

    # numpy stub
    np_mod = types.ModuleType('numpy')
    class ndarray(list):
        pass
    np_mod.ndarray = ndarray
    class random_cls:
        @staticmethod
        def random(shape):
            if isinstance(shape, tuple):
                if len(shape) == 2:
                    return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
                return [0 for _ in range(shape[0])]
            return [0]
    np_mod.random = random_cls()
    np_mod.array = lambda x: x
    np_mod.where = lambda condition: []
    np_mod.argsort = lambda arr: []
    linalg_mod = types.ModuleType('numpy.linalg')
    linalg_mod.norm = lambda x, axis=None: 0
    np_mod.linalg = linalg_mod
    class Arr(list):
        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                r, c = idx
                return super().__getitem__(r)[c]
            return super().__getitem__(idx)
    np_mod.corrcoef = lambda a, b: Arr([[0, 0], [0, 0]])
    modules.update({
        'numpy': np_mod,
        'numpy.linalg': linalg_mod,
    })

    # nltk stub
    nltk_mod = types.ModuleType('nltk')
    nltk_mod.download = lambda *a, **k: None
    class SIA:
        def polarity_scores(self, text):
            return {'pos': 0.0, 'neg': 0.0, 'compound': 0.0}
    nltk_mod.sentiment = types.ModuleType('nltk.sentiment')
    nltk_mod.sentiment.SentimentIntensityAnalyzer = SIA
    nltk_mod.corpus = types.ModuleType('nltk.corpus')
    nltk_mod.corpus.stopwords = types.SimpleNamespace(words=lambda lang: set())
    nltk_mod.word_tokenize = lambda text: text.split()
    nltk_mod.sent_tokenize = lambda text: [text]
    modules.update({
        'nltk': nltk_mod,
        'nltk.sentiment': nltk_mod.sentiment,
        'nltk.corpus': nltk_mod.corpus,
    })

    # textblob stub
    textblob_mod = types.ModuleType('textblob')
    textblob_mod.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0, subjectivity=0.0))
    modules['textblob'] = textblob_mod

    with mock.patch.dict(sys.modules, modules):
        yield
