from .config import RefocusConfig

def __getattr__(name):
    if name == 'RefocusProcessor':
        from .processor import RefocusProcessor
        globals()['RefocusProcessor'] = RefocusProcessor
        return RefocusProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['RefocusProcessor', 'RefocusConfig']
