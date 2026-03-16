"""High-level OOP wrapper for visualising clouds-decoded outputs.

The :class:`Visualiser` takes GeoRasterData objects (or paths to GeoTIFF
files) and auto-detects how to render them based on their metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from matplotlib.figure import Figure

from .layers import (
    Layer,
    RGBConfig,
    downsample_layer,
    layer_from_albedo,
    layer_from_cloud_height,
    layer_from_cloud_mask,
    layer_from_rgb,
    layers_from_cloud_properties,
)
from .static import (
    _build_cmap,
    _build_norm,
    _inset_colorbar,
    plot_layer,
    plot_overview,
    render_to_axes,
    save_figure,
)

#: Default display resolution in metres.  60 m keeps arrays small and
#: matches the coarsest Sentinel-2 band resolution.
DEFAULT_DISPLAY_RESOLUTION_M: float = 60.0

logger = logging.getLogger(__name__)


class Visualiser:
    """Visualise a single scene's outputs with auto-detection and phase masking.

    Accepts GeoRasterData objects or paths to ``.tif`` files.  The data type
    is detected automatically and the appropriate render settings are applied.

    When :class:`~clouds_decoded.data.CloudPropertiesData` is added, phase
    masking is applied: ``r_eff_liq`` is set to NaN where ``ice_liq_ratio > 0.5``
    and ``r_eff_ice`` is set to NaN where ``ice_liq_ratio < 0.5``.

    Args:
        scene: Optional :class:`~clouds_decoded.data.Sentinel2Scene` for RGB
            composites (true colour and ice composite).
        rgb_config: Override :class:`RGBConfig` for the true-colour composite.
        ice_rgb_config: Override :class:`RGBConfig` for the ice composite.
            Defaults to per-channel stretch enabled.
        display_resolution_m: Target pixel size in metres for display.
            Defaults to :data:`DEFAULT_DISPLAY_RESOLUTION_M` (60 m).
            Set to ``None`` to keep native resolution.
    """

    def __init__(
        self,
        scene=None,
        rgb_config: Optional[RGBConfig] = None,
        ice_rgb_config: Optional[RGBConfig] = None,
        display_resolution_m: Optional[float] = DEFAULT_DISPLAY_RESOLUTION_M,
    ):
        self._scene = scene
        self._rgb_config = rgb_config
        self._ice_rgb_config = ice_rgb_config
        self._display_resolution_m = display_resolution_m
        self._layers: Dict[str, Layer] = {}

        if scene is not None:
            self._add_scene_layers(scene)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, data_or_path: Union[object, str, Path]) -> "Visualiser":
        """Add a data object or ``.tif`` file path for visualisation.

        Auto-detects the data type from the object class or GeoTIFF metadata.

        Args:
            data_or_path: A :class:`~clouds_decoded.data.base.GeoRasterData`
                subclass instance, or a path to a ``.tif`` file.

        Returns:
            ``self`` for chaining.
        """
        if isinstance(data_or_path, (str, Path)):
            data = self._load_from_path(Path(data_or_path))
        else:
            data = data_or_path

        if data is not None:
            self._detect_and_add(data)

        return self

    @property
    def layers(self) -> List[Layer]:
        """All layers in insertion order."""
        return list(self._layers.values())

    @property
    def layer_names(self) -> List[str]:
        """Names of all loaded layers."""
        return list(self._layers.keys())

    def plot(self, layer_name: str, ax=None, figsize=(8, 8)) -> Figure:
        """Plot a single named layer.

        Args:
            layer_name: Key from :attr:`layer_names`.
            ax: Optional matplotlib Axes to draw into.
            figsize: Figure size when creating a new figure.

        Returns:
            The matplotlib Figure.
        """
        if layer_name not in self._layers:
            raise KeyError(f"Unknown layer {layer_name!r}. Available: {self.layer_names}")
        return plot_layer(self._layers[layer_name], ax=ax, figsize=figsize)

    def overview(
        self,
        ncols: int = 3,
        figsize=(18, 14),
        properties_alpha: float = 0.7,
    ) -> Figure:
        """Plot all layers in a grid overview.

        Properties layers are overlaid on the True Colour RGB rather than
        shown as standalone panels.

        Args:
            ncols: Number of columns in the grid.
            figsize: Figure size.
            properties_alpha: Transparency for properties overlays on RGB.

        Returns:
            The matplotlib Figure.
        """
        if not self._layers:
            raise ValueError("No layers to display. Add data first.")

        import math
        import matplotlib.pyplot as plt

        rgb_layer = self._layers.get("True Colour")
        _OVERLAY_PREFIXES = ("Properties: ", "Cloud Height")
        overlay_layers = [l for n, l in self._layers.items()
                          if any(n.startswith(p) for p in _OVERLAY_PREFIXES)]
        standalone = [l for n, l in self._layers.items()
                      if not any(n.startswith(p) for p in _OVERLAY_PREFIXES)]

        # Scalar layers get overlaid on RGB; RGB/categorical are standalone
        n_panels = len(standalone) + len(overlay_layers)
        if n_panels == 0:
            return plot_overview(self.layers, ncols=ncols, figsize=figsize)

        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        idx = 0
        # Standalone layers (RGB composites, cloud mask, etc.)
        for layer in standalone:
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            im = render_to_axes(layer, ax)
            if im is not None and not layer.render.categorical:
                _inset_colorbar(im, ax)
            idx += 1

        # Scalar layers overlaid on RGB
        for layer in overlay_layers:
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            if rgb_layer is not None:
                render_to_axes(rgb_layer, ax, show_title=False)
            im = render_to_axes(layer, ax, alpha=properties_alpha)
            if im is not None:
                # Build a full-opacity ScalarMappable for a clean colorbar.
                import matplotlib.colors as mcolors
                render = layer.render
                cmap = _build_cmap(render)
                norm = _build_norm(render) or mcolors.Normalize(
                    vmin=render.vmin, vmax=render.vmax)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                _inset_colorbar(sm, ax)
            idx += 1

        # Hide unused axes
        for i in range(idx, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout(pad=0.5)
        return fig

    def composite(
        self,
        base_name: str,
        overlay_name: str,
        alpha: float = 0.5,
        ax=None,
        figsize=(10, 10),
    ) -> Figure:
        """Composite two layers with transparency.

        Renders *base_name* first, then *overlay_name* with the given alpha.

        Args:
            base_name: Base layer name.
            overlay_name: Overlay layer name.
            alpha: Transparency of the overlay (0 = invisible, 1 = opaque).
            ax: Optional matplotlib Axes.
            figsize: Figure size when creating a new figure.

        Returns:
            The matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        for name in (base_name, overlay_name):
            if name not in self._layers:
                raise KeyError(f"Unknown layer {name!r}. Available: {self.layer_names}")

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        render_to_axes(self._layers[base_name], ax, show_title=False)
        # Overlay on the same axes with transparency
        overlay = self._layers[overlay_name]
        render_to_axes(overlay, ax, alpha=alpha)

        fig.tight_layout(pad=0.5)
        return fig

    def save(self, path: str, layer_name: Optional[str] = None, dpi: int = 150) -> None:
        """Save a figure to disk.

        Args:
            path: Output file path.
            layer_name: If given, save that single layer; otherwise save the overview.
            dpi: Resolution.
        """
        fig = self.plot(layer_name) if layer_name else self.overview()
        save_figure(fig, path, dpi=dpi)

    @classmethod
    def from_directory(
        cls,
        scene_dir: str,
        scene_path: Optional[str] = None,
        rgb_config: Optional[RGBConfig] = None,
        ice_rgb_config: Optional[RGBConfig] = None,
        display_resolution_m: Optional[float] = DEFAULT_DISPLAY_RESOLUTION_M,
    ) -> "Visualiser":
        """Load all outputs from a project scene directory.

        Discovers GeoTIFFs (cloud_mask.tif, cloud_height.tif, albedo.tif,
        properties.tif) and optionally loads the .SAFE scene for RGB composites.

        Args:
            scene_dir: Path to the scene output directory.
            scene_path: Path to the ``.SAFE`` directory for RGB composites.
                Falls back to ``manifest.json`` if not provided.
            rgb_config: Override for true-colour RGB.
            ice_rgb_config: Override for ice composite RGB.
            display_resolution_m: Target pixel size in metres for display.
                Defaults to 60 m.  Set to ``None`` for native resolution.

        Returns:
            A configured :class:`Visualiser` instance.
        """
        import json

        scene_dir_p = Path(scene_dir)

        # Resolve scene_path from manifest if needed
        if scene_path is None:
            manifest_path = scene_dir_p / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                scene_path = manifest.get("scene_path")

        # Load Sentinel2Scene if available
        scene = None
        if scene_path and Path(scene_path).exists():
            try:
                from clouds_decoded.data import Sentinel2Scene

                scene = Sentinel2Scene()
                scene.read(scene_path)
            except Exception as e:
                logger.warning(f"Could not load scene for RGB composites: {e}")
                scene = None

        vis = cls(
            scene=scene,
            rgb_config=rgb_config,
            ice_rgb_config=ice_rgb_config,
            display_resolution_m=display_resolution_m,
        )

        # Load pipeline outputs
        _TIF_LOADERS = {
            "cloud_mask.tif": "CloudMaskData",
            "cloud_height.tif": "CloudHeightGridData",
            "albedo.tif": "AlbedoData",
            "properties.tif": "CloudPropertiesData",
        }
        for filename, class_name in _TIF_LOADERS.items():
            tif_path = scene_dir_p / filename
            if not tif_path.exists():
                continue
            try:
                vis.add(tif_path)
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        return vis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_scene_layers(self, scene) -> None:
        """Build RGB composite layers from a Sentinel2Scene."""
        try:
            layer = layer_from_rgb(
                scene, "true_color",
                rgb_config=self._rgb_config,
                display_resolution_m=self._display_resolution_m,
            )
            self._layers[layer.name] = layer
        except Exception as e:
            logger.warning(f"Could not build true_color RGB: {e}")

        try:
            layer = layer_from_rgb(
                scene, "ice",
                rgb_config=self._ice_rgb_config,
                display_resolution_m=self._display_resolution_m,
            )
            self._layers[layer.name] = layer
        except Exception as e:
            logger.warning(f"Could not build ice composite: {e}")

    def _load_from_path(self, path: Path):
        """Load a GeoRasterData subclass from a .tif file path.

        Uses the filename to guess the data class. Falls back to reading
        the ``clouds_decoded`` metadata tag to detect the type.
        """
        from clouds_decoded.data import (
            AlbedoData,
            CloudHeightGridData,
            CloudMaskData,
            CloudPropertiesData,
        )

        name = path.stem.lower()
        loaders = {
            "cloud_mask": CloudMaskData,
            "cloud_height": CloudHeightGridData,
            "albedo": AlbedoData,
            "properties": CloudPropertiesData,
        }

        cls = loaders.get(name)
        if cls is not None:
            return cls.from_file(str(path))

        # Fallback: try each class until one succeeds
        for cls in (CloudMaskData, CloudPropertiesData, CloudHeightGridData, AlbedoData):
            try:
                return cls.from_file(str(path))
            except Exception:
                continue

        logger.warning(f"Could not determine data type for {path}")
        return None

    def _detect_and_add(self, data) -> None:
        """Dispatch a data object to the appropriate layer factory."""
        from clouds_decoded.data import (
            AlbedoData,
            CloudHeightGridData,
            CloudMaskData,
            CloudPropertiesData,
        )

        if isinstance(data, CloudMaskData):
            layer = layer_from_cloud_mask(data)
            self._layers[layer.name] = self._maybe_downsample(layer)

        elif isinstance(data, CloudHeightGridData):
            layer = layer_from_cloud_height(data)
            self._layers[layer.name] = self._maybe_downsample(layer)

        elif isinstance(data, CloudPropertiesData):
            # Apply phase masking at full resolution, then downsample.
            full_layers = {l.name: l for l in layers_from_cloud_properties(data)}
            self._apply_phase_masking(data, full_layers)
            for layer in full_layers.values():
                self._layers[layer.name] = self._maybe_downsample(layer)

        elif isinstance(data, AlbedoData):
            logger.debug("Albedo data skipped in visualiser (not displayed).")

        else:
            logger.warning(f"Unsupported data type: {type(data).__name__}")

    def _maybe_downsample(self, layer: Layer) -> Layer:
        """Downsample a layer to the display resolution if set."""
        if self._display_resolution_m is not None:
            return downsample_layer(layer, self._display_resolution_m)
        return layer

    def _apply_phase_masking(
        self,
        properties_data,
        layers: Optional[Dict[str, Layer]] = None,
    ) -> None:
        """Mask r_eff layers by dominant phase.

        ``r_eff_liq`` is set to NaN where ``ice_liq_ratio > 0.5`` (ice-dominated)
        and ``r_eff_ice`` is set to NaN where ``ice_liq_ratio < 0.5`` (liquid-dominated).

        Operates on *layers* dict in-place.  If *layers* is ``None``, falls
        back to ``self._layers``.
        """
        target = layers if layers is not None else self._layers

        band_names = list(properties_data.metadata.band_names)
        if "ice_liq_ratio" not in band_names:
            return

        phase_idx = band_names.index("ice_liq_ratio")
        phase = properties_data.data[phase_idx]

        masks = {
            "Properties: r_eff_liq": phase > 0.5,    # NaN where ice-dominated
            "Properties: r_eff_ice": phase < 0.5,     # NaN where liquid-dominated
        }

        for key, mask in masks.items():
            if key not in target:
                continue
            layer = target[key]
            masked = layer.data.copy()
            masked[mask] = np.nan
            target[key] = Layer(
                name=layer.name,
                data=masked,
                render=layer.render,
                extent=layer.extent,
                resolution_m=layer.resolution_m,
            )
