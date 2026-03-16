from .config import CloudMaskConfig

def __getattr__(name):
    if name in ('CloudMaskProcessor', 'ThresholdCloudMaskProcessor'):
        from .processor import CloudMaskProcessor, ThresholdCloudMaskProcessor
        globals()['CloudMaskProcessor'] = CloudMaskProcessor
        globals()['ThresholdCloudMaskProcessor'] = ThresholdCloudMaskProcessor
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CloudMaskProcessor', 'ThresholdCloudMaskProcessor', 'CloudMaskConfig']
