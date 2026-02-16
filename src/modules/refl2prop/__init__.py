from .config import Refl2PropConfig, ShadingRefl2PropConfig

def __getattr__(name):
    if name in ('CloudPropertyInverter', 'ShadingPropertyInverter'):
        from .processor import CloudPropertyInverter, ShadingPropertyInverter
        globals()['CloudPropertyInverter'] = CloudPropertyInverter
        globals()['ShadingPropertyInverter'] = ShadingPropertyInverter
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'CloudPropertyInverter',
    'ShadingPropertyInverter',
    'Refl2PropConfig',
    'ShadingRefl2PropConfig',
]
