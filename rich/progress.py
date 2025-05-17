class Progress:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def track(self, iterable, *args, **kwargs):
        return iterable
