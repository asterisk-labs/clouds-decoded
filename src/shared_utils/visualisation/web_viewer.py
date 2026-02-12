"""Web-based interactive viewer using Panel + HoloViews.

Launch with::

    clouds-decoded serve ./mynewproject/scenes/S2A_MSIL1C_.../

Then port-forward if remote::

    ssh -L 5006:localhost:5006 user@server

And open http://localhost:5006 in your browser.
"""
import logging
from typing import Dict, List, Optional

import holoviews as hv
import numpy as np
import panel as pn
from holoviews import opts

from .layers import Layer, RGBConfig, _apply_rgb_config

logger = logging.getLogger(__name__)

hv.extension("bokeh")

FRAME_WIDTH = 700
MAX_DISPLAY_PX = 1000  # max pixels on longest side for web display

# Cloud mask categorical palette
_CLOUD_MASK_CMAP = {0: "#2ca02c", 1: "#d62728", 2: "#ff7f0e", 3: "#7f7f7f"}


def _downsample(data: np.ndarray, max_px: int = MAX_DISPLAY_PX) -> np.ndarray:
    """Stride-downsample an array so the longest side <= max_px."""
    h, w = data.shape[:2]
    factor = max(1, max(h, w) // max_px)
    if factor <= 1:
        return data
    if data.ndim == 3:
        return data[::factor, ::factor, :].copy()
    return data[::factor, ::factor].copy()


def _get_bounds(extent, w, h):
    """Convert Layer extent to HoloViews bounds (left, bottom, right, top)."""
    if extent:
        return (extent[0], extent[2], extent[1], extent[3])
    return (0, 0, w, h)


def _layer_to_hv(layer: Layer) -> hv.Element:
    """Convert a Layer to a HoloViews element (no layout opts applied)."""
    data = layer.data
    extent = layer.extent

    if layer.is_rgb:
        h, w, _ = data.shape
        bounds = _get_bounds(extent, w, h)
        return hv.RGB(
            (np.linspace(bounds[0], bounds[2], w),
             np.linspace(bounds[3], bounds[1], h),
             data[:, :, 0], data[:, :, 1], data[:, :, 2]),
        )

    # Scalar layer
    h, w = data.shape
    bounds = _get_bounds(extent, w, h)
    render = layer.render

    # Build colorbar label
    clabel = render.label or layer.name
    if render.units:
        clabel += f" ({render.units})"

    if render.categorical and render.category_labels:
        labels = render.category_labels
        n = max(labels.keys()) + 1
        colors = [_CLOUD_MASK_CMAP.get(i, "#333333") for i in range(n)]
        img = hv.Image(
            (np.linspace(bounds[0], bounds[2], w),
             np.linspace(bounds[3], bounds[1], h),
             data.astype(np.float32)),
            vdims=[hv.Dimension("z", label=clabel)],
        )
        return img.opts(cmap=colors, clim=(-0.5, n - 0.5), colorbar=True)

    # Continuous scalar
    plot_data = data.astype(np.float32).copy()
    clim = (
        float(render.vmin) if render.vmin is not None else float(np.nanmin(plot_data)),
        float(render.vmax) if render.vmax is not None else float(np.nanmax(plot_data)),
    )

    if render.log_scale:
        plot_data = np.where(plot_data > 0, plot_data, np.nan)
        plot_data = np.log10(plot_data)
        clim = (np.log10(max(clim[0], 1e-10)), np.log10(max(clim[1], 1e-10)))
        clabel += " [log\u2081\u2080]"

    img = hv.Image(
        (np.linspace(bounds[0], bounds[2], w),
         np.linspace(bounds[3], bounds[1], h),
         plot_data),
        vdims=[hv.Dimension("z", label=clabel)],
    )

    return img.opts(cmap=render.cmap, clim=clim, colorbar=True)


class WebViewer:
    """Panel-based web viewer for clouds-decoded layers.

    Displays a base layer (default: RGB composite) with an optional
    semi-transparent overlay for analysis products.

    Args:
        layers: List of Layer objects to display.
    """

    def __init__(self, layers: List[Layer], max_display_px: int = MAX_DISPLAY_PX):
        if not layers:
            raise ValueError("No layers to display")

        # Downsample all layers for display performance
        self.layers = [self._downsample_layer(l, max_display_px) for l in layers]
        self._layer_map: Dict[str, Layer] = {l.name: l for l in self.layers}

        # Cache raw (pre-gamma) RGB data for live slider re-rendering
        self._raw_rgb: Dict[str, np.ndarray] = {}
        for layer in self.layers:
            if layer.is_rgb and layer.rgb_config is not None:
                cfg = layer.rgb_config
                raw = layer.data.astype(np.float64)
                raw = np.clip(raw, 1e-10, None)
                raw = np.power(raw, 1.0 / cfg.gamma)
                raw = (raw - cfg.offset) / max(cfg.gain, 1e-10)
                self._raw_rgb[layer.name] = np.clip(raw, 0, None).astype(np.float32)

        logger.info(
            "WebViewer: downsampled %d layers to <=%dpx",
            len(self.layers), max_display_px,
        )

    @staticmethod
    def _downsample_layer(layer: Layer, max_px: int) -> Layer:
        """Return a display-resolution copy of a Layer."""
        small = _downsample(layer.data, max_px)
        if small is layer.data:
            return layer
        h_orig = layer.data.shape[0]
        h_small = small.shape[0]
        factor = h_orig / h_small
        return Layer(
            name=layer.name,
            data=small,
            render=layer.render,
            extent=layer.extent,
            resolution_m=layer.resolution_m * factor if layer.resolution_m else None,
            is_rgb=layer.is_rgb,
            rgb_config=layer.rgb_config,
        )

    def _frame_dims(self, layer: Layer):
        """Compute frame_width/frame_height preserving geographic aspect."""
        extent = layer.extent
        if extent:
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            if dx > 0 and dy > 0:
                fh = max(200, int(FRAME_WIDTH * dy / dx))
                return FRAME_WIDTH, fh
        return FRAME_WIDTH, FRAME_WIDTH

    def _build(self) -> pn.viewable.Viewable:
        """Build the Panel layout."""
        layer_names = [l.name for l in self.layers]

        # Default base = first RGB layer, fallback to first layer
        default_base = layer_names[0]
        for l in self.layers:
            if l.is_rgb:
                default_base = l.name
                break

        # --- Base layer widgets ---
        base_select = pn.widgets.Select(
            name="Base Layer", options=layer_names, value=default_base, width=250,
        )
        gamma_slider = pn.widgets.FloatSlider(
            name="Gamma", start=0.1, end=3.0, step=0.05, value=0.7, width=250,
        )
        gain_slider = pn.widgets.FloatSlider(
            name="Gain", start=0.1, end=5.0, step=0.05, value=1.0, width=250,
        )
        offset_slider = pn.widgets.FloatSlider(
            name="Offset", start=-0.3, end=0.3, step=0.01, value=0.0, width=250,
        )
        rgb_controls = pn.Column(
            pn.pane.Markdown("### RGB Controls"),
            gamma_slider, gain_slider, offset_slider,
        )

        # --- Overlay widgets ---
        overlay_toggle = pn.widgets.Checkbox(
            name="Show Overlay", value=False, width=250,
        )
        overlay_select = pn.widgets.Select(
            name="Overlay Layer", options=layer_names, value=layer_names[0], width=250,
        )
        overlay_opacity = pn.widgets.FloatSlider(
            name="Opacity", start=0.0, end=1.0, step=0.05, value=0.7, width=250,
        )
        overlay_controls = pn.Column(overlay_select, overlay_opacity)

        # --- Visibility side-effects (separate from plot callback) ---
        @pn.depends(base_select, overlay_toggle, watch=True)
        def _update_visibility(base_name, overlay_on):
            rgb_controls.visible = self._layer_map[base_name].is_rgb
            overlay_controls.visible = overlay_on

        # --- Plot callback (pure — returns HoloViews element only) ---
        def _plot(base_name, overlay_name, overlay_on, opacity, gamma, gain, offset):
            base_layer = self._layer_map[base_name]

            # Apply RGB adjustments if base is RGB
            if base_layer.is_rgb and base_name in self._raw_rgb:
                cfg = RGBConfig(gamma=gamma, gain=gain, offset=offset)
                new_data = _apply_rgb_config(self._raw_rgb[base_name], cfg)
                base_layer = Layer(
                    name=base_layer.name, data=new_data, render=base_layer.render,
                    extent=base_layer.extent, resolution_m=base_layer.resolution_m,
                    is_rgb=True, rgb_config=cfg,
                )

            fw, fh = self._frame_dims(base_layer)
            plot_kw = dict(
                frame_width=fw, frame_height=fh,
                xlabel="Easting (m)", ylabel="Northing (m)",
                tools=["hover"], active_tools=["wheel_zoom"],
            )

            base_elem = _layer_to_hv(base_layer).opts(
                title=base_layer.name, **plot_kw,
            )

            if overlay_on and overlay_name:
                overlay_layer = self._layer_map[overlay_name]
                overlay_elem = _layer_to_hv(overlay_layer).opts(
                    alpha=opacity, **plot_kw,
                )
                base_elem = base_elem.opts(
                    title=f"{base_layer.name} + {overlay_layer.name}",
                )
                return base_elem * overlay_elem

            # Always return Overlay so DynamicMap sees a consistent type
            return hv.Overlay([base_elem])

        # Use DynamicMap with streams for efficient in-place Bokeh updates
        streams = [
            hv.streams.Params(base_select, ["value"], rename={"value": "base_name"}),
            hv.streams.Params(overlay_select, ["value"], rename={"value": "overlay_name"}),
            hv.streams.Params(overlay_toggle, ["value"], rename={"value": "overlay_on"}),
            hv.streams.Params(overlay_opacity, ["value"], rename={"value": "opacity"}),
            hv.streams.Params(gamma_slider, ["value"], rename={"value": "gamma"}),
            hv.streams.Params(gain_slider, ["value"], rename={"value": "gain"}),
            hv.streams.Params(offset_slider, ["value"], rename={"value": "offset"}),
        ]
        dmap = hv.DynamicMap(_plot, streams=streams)
        plot_pane = pn.pane.HoloViews(dmap, sizing_mode="stretch_both")

        sidebar = pn.Column(
            pn.pane.Markdown("## Clouds Decoded"),
            base_select,
            rgb_controls,
            pn.pane.Markdown("### Overlay"),
            overlay_toggle,
            overlay_controls,
            width=280,
        )

        # Initial visibility
        rgb_controls.visible = self._layer_map[default_base].is_rgb
        overlay_controls.visible = False

        return pn.Row(sidebar, plot_pane, sizing_mode="stretch_both")

    def serve(self, port: int = 5006, show: bool = False, **kwargs):
        """Start the Panel server.

        Args:
            port: Port to serve on.
            show: Open browser automatically (set False for remote).
        """
        app = self._build()
        pn.serve(app, port=port, show=show, title="Clouds Decoded Viewer", **kwargs)

    def panel(self) -> pn.viewable.Viewable:
        """Return the Panel layout (for notebooks or embedding)."""
        return self._build()
