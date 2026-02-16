from .config import AlbedoEstimatorConfig

def __getattr__(name):
    if name == 'AlbedoEstimator':
        from .processor import AlbedoEstimator
        globals()['AlbedoEstimator'] = AlbedoEstimator
        return AlbedoEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['AlbedoEstimator', 'AlbedoEstimatorConfig']
