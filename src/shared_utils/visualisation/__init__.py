"""Visualisation tools for clouds-decoded outputs."""
import logging as _logging
import warnings as _warnings

_warnings.warn(
    "clouds_decoded.visualisation is a work in progress — API and outputs may change in future releases.",
    stacklevel=2,
)
_logging.getLogger(__name__).warning(
    "Visualisation module is a work in progress — API and outputs may change in future releases."
)

from .layers import Layer, RenderConfig, RGBConfig
from .loader import load_scene_layers
from .static import plot_layer, plot_overview, save_figure
from .viewer import InteractiveViewer
from .viser_viewer import SceneData, ViserViewer
from .web_viewer import WebViewer

__all__ = [
    "InteractiveViewer",
    "Layer",
    "RenderConfig",
    "RGBConfig",
    "SceneData",
    "ViserViewer",
    "WebViewer",
    "load_scene_layers",
    "plot_layer",
    "plot_overview",
    "save_figure",
]
