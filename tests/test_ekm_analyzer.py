from tests import pytest
import os, json, shutil, sys, types

def setup_modules():
    plt = types.ModuleType('matplotlib.pyplot')
    plt.figure = lambda *a, **k: None
    plt.subplots = lambda *a, **k: ((types.SimpleNamespace(), types.SimpleNamespace()), None)
    plt.close = lambda *a, **k: None
    plt.savefig = lambda *a, **k: None
    sys.modules['matplotlib'] = types.ModuleType('matplotlib')
    sys.modules['matplotlib.pyplot'] = plt
    sys.modules['seaborn'] = types.ModuleType('seaborn')
    wc = types.ModuleType('wordcloud')
    wc.WordCloud = lambda **k: types.SimpleNamespace(generate_from_frequencies=lambda f: None)
    sys.modules['wordcloud'] = wc

    sk = types.ModuleType('sklearn')
    fe = types.ModuleType('sklearn.feature_extraction')
    fe_text = types.ModuleType('sklearn.feature_extraction.text')
    class TF:
        def __init__(self, *a, **k):
            pass
        def fit_transform(self, X):
            class Mat(list):
                def tolist(self):
                    return self
            return Mat([[1 for _ in X] for _ in X])
    fe_text.TfidfVectorizer = TF
    fe.text = fe_text
    metrics = types.ModuleType('sklearn.metrics')
    class SimMat(list):
        def tolist(self):
            return self
    pair = types.ModuleType('sklearn.metrics.pairwise')
    pair.cosine_similarity = lambda X: SimMat([[1 for _ in X] for _ in X])
    metrics.pairwise = pair
    sk.feature_extraction = fe
    sk.metrics = metrics
    sk.decomposition = types.ModuleType('sklearn.decomposition')
    sk.decomposition.PCA = lambda *a, **k: None
    sk.manifold = types.ModuleType('sklearn.manifold')
    sk.manifold.TSNE = lambda *a, **k: None
    sys.modules['sklearn'] = sk
    sys.modules['sklearn.feature_extraction'] = fe
    sys.modules['sklearn.feature_extraction.text'] = fe_text
    sys.modules['sklearn.metrics'] = metrics
    sys.modules['sklearn.metrics.pairwise'] = pair
    sys.modules['sklearn.decomposition'] = sk.decomposition
    sys.modules['sklearn.manifold'] = sk.manifold

    nltk = types.ModuleType('nltk')
    nltk.download = lambda *a, **k: None
    class SIA:
        def polarity_scores(self, text):
            return {'pos': 0.0, 'neg': 0.0, 'compound': 0.0}
    nltk.sentiment = types.ModuleType('nltk.sentiment')
    nltk.sentiment.SentimentIntensityAnalyzer = SIA
    nltk.corpus = types.ModuleType('nltk.corpus')
    nltk.corpus.stopwords = types.SimpleNamespace(words=lambda lang: set())
    nltk.word_tokenize = lambda text: text.split()
    nltk.sent_tokenize = lambda text: [text]
    sys.modules['nltk'] = nltk
    sys.modules['nltk.sentiment'] = nltk.sentiment
    sys.modules['nltk.corpus'] = nltk.corpus

    textblob = types.ModuleType('textblob')
    textblob.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0, subjectivity=0.0))
    sys.modules['textblob'] = textblob

    import numpy as np
    if not hasattr(np, 'corrcoef'):
        class Arr(list):
            def __getitem__(self, idx):
                if isinstance(idx, tuple):
                    r, c = idx
                    return super().__getitem__(r)[c]
                return super().__getitem__(idx)
        np.corrcoef = lambda a, b: Arr([[0, 0], [0, 0]])


def test_analyze_single_result_basic():
    setup_modules()
    from ekm_analyzer import EKMAnalyzer
    os.makedirs('mock_results', exist_ok=True)
    try:
        with open('mock_results/test.json', 'w') as f:
            json.dump({
                'matrix_name': 'M',
                'model_name': 'model',
                'results': [{
                    'response': 'text',
                    'prompt': 'p',
                    'path': [0],
                    'main_diagonal_affect': 'A',
                    'main_diagonal_strength': 1.0,
                    'anti_diagonal_affect': 'B',
                    'anti_diagonal_strength': 0.0
                }]
            }, f)
        analyzer = EKMAnalyzer(results_dir='mock_results')
        result = analyzer.analyze_single_result(0)
        assert result['matrix_name'] == 'M'
        assert result['response_count'] == 1
    finally:
        shutil.rmtree('mock_results', ignore_errors=True)
