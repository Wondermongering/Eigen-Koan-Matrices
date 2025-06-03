class Console:
    def print(self, *args, **kwargs):
        pass

class Table:
    def __init__(self, *args, **kwargs):
        pass
    def add_column(self, *args, **kwargs):
        pass
    def add_row(self, *args, **kwargs):
        pass

class Progress:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def track(self, iterable, *args, **kwargs):
        return iterable

class Panel:
    def __init__(self, *args, **kwargs):
        pass
    @staticmethod
    def fit(*args, **kwargs):
        return None

class Layout:
    def __init__(self, *args, **kwargs):
        pass

class Syntax:
    def __init__(self, *args, **kwargs):
        pass

# Expose submodules for compatibility
import types, sys
console_mod = types.ModuleType('rich.console')
console_mod.Console = Console
sys.modules['rich.console'] = console_mod

table_mod = types.ModuleType('rich.table')
table_mod.Table = Table
sys.modules['rich.table'] = table_mod

progress_mod = types.ModuleType('rich.progress')
progress_mod.Progress = Progress
sys.modules['rich.progress'] = progress_mod

panel_mod = types.ModuleType('rich.panel')
panel_mod.Panel = Panel
sys.modules['rich.panel'] = panel_mod

layout_mod = types.ModuleType('rich.layout')
layout_mod.Layout = Layout
sys.modules['rich.layout'] = layout_mod

syntax_mod = types.ModuleType('rich.syntax')
syntax_mod.Syntax = Syntax
sys.modules['rich.syntax'] = syntax_mod

__all__ = ['Console', 'Table', 'Progress', 'Panel', 'Layout', 'Syntax']
