"""Visualisation tools for clouds-decoded outputs."""
from __future__ import annotations

from .layers import Layer, RenderConfig, RGBConfig
from .loader import load_scene_layers
from .static import plot_layer, plot_overview, save_figure
from .viewer import InteractiveViewer
from .visualiser import Visualiser

# Lazy imports for classes that depend on optional viz extras (panel, holoviews, bokeh).
# Importing these triggers a DeprecationWarning for deprecated names.
_LAZY_VIZ = {"ProjectVisualiser": ".project_visualiser", "WebViewer": ".web_viewer"}
_DEPRECATED = {"SceneData": "._deprecated", "ViserViewer": "._deprecated"}

def __getattr__(name: str):
    if name in _LAZY_VIZ:
        import importlib
        mod = importlib.import_module(_LAZY_VIZ[name], __name__)
        return getattr(mod, name)
    if name in _DEPRECATED:
        from ._deprecated import SceneData, ViserViewer  # noqa: F811
        return {"SceneData": SceneData, "ViserViewer": ViserViewer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InteractiveViewer",
    "Layer",
    "ProjectVisualiser",
    "RenderConfig",
    "RGBConfig",
    "Visualiser",
    "WebViewer",
    "load_scene_layers",
    "plot_layer",
    "plot_overview",
    "save_figure",
]
