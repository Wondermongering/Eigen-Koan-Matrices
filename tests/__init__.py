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

    # httpx stub
    httpx_mod = types.ModuleType('httpx')

    # Define a more complete Response class that httpx.AsyncClient would return
    class MockHttpResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data # Store the dict
            self.content = json.dumps(self._json_data).encode('utf-8') # bytes
            self.body = self.content # for resp.body.decode() compatibility
            self.text = self.content.decode('utf-8') # string

        def json(self):
            # Return the original dict, not a re-parsed one, to preserve object identity if needed
            return self._json_data

    class SimpleClient:
        # Class attribute to hold the state of the matrix data
        # This allows a simple way for swap_tasks to modify data that get_matrix retrieves
        _matrix_data_state = {
            'size': 2,
            'task_rows': ['Task Row 0 Initial', 'Task Row 1 Initial'],
            'constraint_cols': ['Constraint Col 0 Initial', 'Constraint Col 1 Initial'],
            'cells': [[{'id': 'c00'}, {'id': 'c01'}], [{'id': 'c10'}, {'id': 'c11'}]]
        }

        @classmethod
        def reset_state(cls): # Helper to ensure tests start with fresh state if needed
            cls._matrix_data_state = {
                'size': 2,
                'task_rows': ['Task Row 0 Initial', 'Task Row 1 Initial'],
                'constraint_cols': ['Constraint Col 0 Initial', 'Constraint Col 1 Initial'],
                'cells': [[{'id': 'c00'}, {'id': 'c01'}], [{'id': 'c10'}, {'id': 'c11'}]]
            }

        def __init__(self, *args, **kwargs):
            # Ensure state is reset for each new client instance if tests create them per operation.
            # Or, rely on explicit reset if client is a singleton.
            # For now, let's assume tests might reuse instances or class state is okay.
            pass

        def _handle_get_matrix(self):
            # Return a copy to prevent direct modification of the state dict through the response object
            return MockHttpResponse(200, self.__class__._matrix_data_state.copy())

        def _handle_swap_tasks(self, json_payload: dict):
            r1_idx = json_payload.get('row1')
            r2_idx = json_payload.get('row2')

            current_tasks = self.__class__._matrix_data_state['task_rows']
            if r1_idx is not None and r2_idx is not None and \
               0 <= r1_idx < len(current_tasks) and \
               0 <= r2_idx < len(current_tasks):

                current_tasks[r1_idx], current_tasks[r2_idx] = current_tasks[r2_idx], current_tasks[r1_idx]
                return MockHttpResponse(200, {'status': 'success', 'message': 'Tasks swapped'})
            else:
                return MockHttpResponse(400, {'status': 'error', 'message': 'Invalid row indices for swap'})

        def request(self, method: str, url: str, json: dict = None, **kwargs): # Added json for POST
            # Basic routing: if URL suggests swap and method is POST, handle as swap.
            if method.upper() == "POST" and "swap" in str(url).lower(): # Check url for 'swap'
                return self._handle_swap_tasks(json if json else {})
            # Default to GET matrix data for other GET requests or unspecific URLs
            elif method.upper() == "GET":
                 return self._handle_get_matrix()
            # Fallback for other methods/unrecognized URLs
            return MockHttpResponse(404, {"status": "error", "message": "Mock endpoint not found"})

        def get(self, url: str, **kwargs):
            return self.request(method="GET", url=url, **kwargs)

        def post(self, url: str, json: dict = None, **kwargs):
            return self.request(method="POST", url=url, json=json, **kwargs)

        async def __aenter__(self): # Async context manager support
            return self

        async def __aexit__(self, exc_type, exc, tb): # Async context manager support
            pass

        def close(self): # Synchronous close for Client
            pass

        async def aclose(self): # Asynchronous close for AsyncClient
            pass

    httpx_mod.AsyncClient = SimpleClient
    httpx_mod.Client = SimpleClient # Also mock sync client

    # httpx.Request and httpx.Response stubs (can be basic if not heavily used by tested code)
    class Request:
        def __init__(self, method, url, headers=None, content=None):
            self.method = method
            self.url = url
            self.headers = headers or {}
            self.content = content
    # Make httpx.Response an alias for our MockHttpResponse for consistency
    httpx_mod.Response = MockHttpResponse
    httpx_mod.Request = Request

    # Other httpx attributes that might be accessed
    httpx_mod.TimeoutException = type('TimeoutException', (Exception,), {})
    httpx_mod.ConnectTimeout = type('ConnectTimeout', (httpx_mod.TimeoutException,), {})
    httpx_mod.ReadTimeout = type('ReadTimeout', (httpx_mod.TimeoutException,), {})
    httpx_mod.WriteTimeout = type('WriteTimeout', (httpx_mod.TimeoutException,), {})
    httpx_mod.PoolTimeout = type('PoolTimeout', (httpx_mod.TimeoutException,), {})
    httpx_mod.NetworkError = type('NetworkError', (Exception,), {}) # Base for connection errors
    httpx_mod.ConnectError = type('ConnectError', (httpx_mod.NetworkError,), {})
    httpx_mod.ReadError = type('ReadError', (httpx_mod.NetworkError,), {})
    httpx_mod.WriteError = type('WriteError', (httpx_mod.NetworkError,), {})
    httpx_mod.CloseError = type('CloseError', (httpx_mod.NetworkError,), {})
    httpx_mod.HTTPStatusError = type('HTTPStatusError', (Exception,), {'request': None, 'response': None}) # Stub for error responses
    httpx_mod.InvalidURL = type('InvalidURL', (ValueError,), {})
    httpx_mod.CookieConflict = type('CookieConflict', (ValueError,), {})


    modules['httpx'] = httpx_mod

    # Ensure json is available for the above stubs
    if 'json' not in sys.modules:
        import json as json_module
        modules['json'] = json_module


    with mock.patch.dict(sys.modules, modules):
        yield
