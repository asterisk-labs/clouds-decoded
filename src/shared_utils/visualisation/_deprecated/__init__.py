"""Deprecated visualisation modules.

These modules are retained for backward compatibility but are no longer
actively developed. Use :class:`~clouds_decoded.visualisation.visualiser.Visualiser`
and :class:`~clouds_decoded.visualisation.project_visualiser.ProjectVisualiser` instead.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "clouds_decoded.visualisation._deprecated contains legacy 3D viewers. "
    "Use Visualiser / ProjectVisualiser for 2D workflows.",
    DeprecationWarning,
    stacklevel=2,
)

from .viser_viewer import SceneData, ViserViewer

__all__ = ["SceneData", "ViserViewer"]
