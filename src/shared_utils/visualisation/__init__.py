"""Visualisation tools for clouds-decoded outputs."""
from __future__ import annotations

from .layers import Layer, RenderConfig, RGBConfig
from .loader import load_scene_layers
from .static import plot_layer, plot_overview, save_figure
from .viewer import InteractiveViewer
from .visualiser import Visualiser
from .project_visualiser import ProjectVisualiser
from .web_viewer import WebViewer

# Backward-compat re-exports from deprecated 3D module.
# Importing these triggers a DeprecationWarning.
def __getattr__(name: str):
    if name in ("SceneData", "ViserViewer"):
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
