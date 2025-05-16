import sys
import importlib
import pathlib
from contextlib import ContextDecorator

class RaisesContext(ContextDecorator):
    def __init__(self, exc):
        self.exc = exc
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"{self.exc} not raised")
        if not issubclass(exc_type, self.exc):
            raise AssertionError(f"Expected {self.exc}, got {exc_type}")
        return True

def raises(exc):
    return RaisesContext(exc)

def main(args=None):
    args = args or []
    quiet = '-q' in args
    test_dir = pathlib.Path('tests')
    test_files = sorted(p for p in test_dir.glob('test_*.py'))
    total = 0
    failed = 0
    for file in test_files:
        mod = importlib.import_module(f"tests.{file.stem}")
        for name in dir(mod):
            if name.startswith('test_') and callable(getattr(mod, name)):
                total += 1
                try:
                    getattr(mod, name)()
                    if quiet:
                        print('.', end='')
                    else:
                        print(f"{name}: PASSED")
                except Exception:
                    failed += 1
                    if quiet:
                        print('F', end='')
                    else:
                        print(f"{name}: FAILED")
                        import traceback; traceback.print_exc()
    if quiet:
        print()
        print(f"{total-failed} passed, {failed} failed")
    else:
        print(f"Ran {total} tests")
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
