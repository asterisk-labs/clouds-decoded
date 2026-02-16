from .config import CloudHeightConfig

def __getattr__(name):
    if name == 'CloudHeightProcessor':
        from .processor import CloudHeightProcessor
        globals()['CloudHeightProcessor'] = CloudHeightProcessor
        return CloudHeightProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CloudHeightProcessor', 'CloudHeightConfig']
