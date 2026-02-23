"""Interactive matplotlib viewer with layer switching and RGB controls."""
import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from .layers import Layer, RGBConfig, _apply_rgb_config
from .static import _build_cmap, _build_norm, _CLOUD_MASK_CMAP

logger = logging.getLogger(__name__)


class InteractiveViewer:
    """Interactive viewer for clouds-decoded output layers.

    Opens a matplotlib figure with radio buttons for layer switching.
    RGB layers get gamma/gain/offset sliders.

    Usage::

        from clouds_decoded.visualisation import InteractiveViewer, load_scene_layers

        layers = load_scene_layers("mynewproject/scenes/S2A_MSIL1C_.../")
        viewer = InteractiveViewer(layers)
        viewer.show()
    """

    def __init__(self, layers: List[Layer]):
        if not layers:
            raise ValueError("No layers to display")
        self.layers = layers
        self._layer_map = {layer.name: layer for layer in layers}
        self._current_layer: Optional[Layer] = None
        self._current_im = None
        self._colorbar = None

        # Store raw (pre-gamma) RGB data for live re-rendering
        self._raw_rgb_cache = {}
        for layer in layers:
            if layer.is_rgb and layer.rgb_config is not None:
                # Reverse the gamma/gain/offset to recover raw reflectance
                # Easier: just store None and recompute from scene later
                # For now, store the current data and config
                self._raw_rgb_cache[layer.name] = self._recover_raw_rgb(layer)

    @staticmethod
    def _recover_raw_rgb(layer: Layer) -> np.ndarray:
        """Approximate raw reflectance from a processed RGB layer."""
        cfg = layer.rgb_config or RGBConfig()
        # Invert: out = clip((raw * gain + offset) ^ gamma)
        # raw ~ (out ^ (1/gamma) - offset) / gain
        out = layer.data.astype(np.float64)
        out = np.clip(out, 1e-10, None)
        raw = np.power(out, 1.0 / cfg.gamma)
        raw = (raw - cfg.offset) / max(cfg.gain, 1e-10)
        return np.clip(raw, 0, None).astype(np.float32)

    def _setup_figure(self):
        """Create the figure, axes, and widgets."""
        self._fig = plt.figure(figsize=(14, 8))

        # Main image axes
        self._ax = self._fig.add_axes([0.25, 0.15, 0.70, 0.80])

        # Radio buttons for layer selection
        labels = [layer.name for layer in self.layers]
        # Calculate height needed for radio buttons
        btn_height = min(0.8, 0.04 * len(labels))
        btn_bottom = 0.5 - btn_height / 2
        radio_ax = self._fig.add_axes([0.01, btn_bottom, 0.18, btn_height])
        radio_ax.set_frame_on(False)
        self._radio = RadioButtons(radio_ax, labels, active=0)
        self._radio.on_clicked(self._on_layer_change)

        # RGB sliders (initially hidden)
        self._slider_axes = {}
        self._sliders = {}

        gamma_ax = self._fig.add_axes([0.25, 0.08, 0.40, 0.025])
        self._sliders["gamma"] = Slider(gamma_ax, "Gamma", 0.1, 3.0, valinit=0.7)
        self._slider_axes["gamma"] = gamma_ax

        gain_ax = self._fig.add_axes([0.25, 0.045, 0.40, 0.025])
        self._sliders["gain"] = Slider(gain_ax, "Gain", 0.1, 5.0, valinit=1.0)
        self._slider_axes["gain"] = gain_ax

        offset_ax = self._fig.add_axes([0.25, 0.01, 0.40, 0.025])
        self._sliders["offset"] = Slider(offset_ax, "Offset", -0.2, 0.2, valinit=0.0)
        self._slider_axes["offset"] = offset_ax

        for s in self._sliders.values():
            s.on_changed(self._on_slider_change)

        # Display first layer
        self._switch_to(self.layers[0])

    def _switch_to(self, layer: Layer):
        """Display a given layer."""
        self._current_layer = layer
        self._ax.clear()

        # Remove old colorbar
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        # Show/hide RGB sliders
        is_rgb = layer.is_rgb
        for ax in self._slider_axes.values():
            ax.set_visible(is_rgb)

        if is_rgb:
            # Update slider positions to match current config
            cfg = layer.rgb_config or RGBConfig()
            self._sliders["gamma"].set_val(cfg.gamma)
            self._sliders["gain"].set_val(cfg.gain)
            self._sliders["offset"].set_val(cfg.offset)

            self._current_im = self._ax.imshow(
                layer.data, extent=layer.extent, origin="upper", aspect="equal",
            )
        else:
            render = layer.render
            cmap = _build_cmap(render)
            norm = _build_norm(render)

            kwargs = dict(
                cmap=cmap, extent=layer.extent, origin="upper",
                aspect="equal", interpolation="nearest",
            )
            if norm is not None:
                kwargs["norm"] = norm
            else:
                if render.vmin is not None:
                    kwargs["vmin"] = render.vmin
                if render.vmax is not None:
                    kwargs["vmax"] = render.vmax

            self._current_im = self._ax.imshow(layer.data, **kwargs)

            # Colorbar or legend
            if render.categorical and render.category_labels:
                labels = render.category_labels
                cmap_obj = _build_cmap(render)
                handles = [
                    plt.Line2D([0], [0], marker="s", color="w",
                               markerfacecolor=cmap_obj(i), markersize=10, label=lbl)
                    for i, lbl in sorted(labels.items())
                ]
                self._ax.legend(handles=handles, loc="upper right", framealpha=0.8)
            else:
                label = render.label
                if render.units:
                    label += f" ({render.units})"
                self._colorbar = self._fig.colorbar(
                    self._current_im, ax=self._ax, label=label, shrink=0.8,
                )

        title = layer.render.label or layer.name
        if layer.render.units:
            title += f" ({layer.render.units})"
        self._ax.set_title(title)
        self._ax.set_xlabel("Easting (m)")
        self._ax.set_ylabel("Northing (m)")

        self._fig.canvas.draw_idle()

    def _on_layer_change(self, label: str):
        """Radio button callback."""
        layer = self._layer_map.get(label)
        if layer is not None:
            self._switch_to(layer)

    def _on_slider_change(self, _val):
        """RGB slider callback — recompute and redisplay."""
        layer = self._current_layer
        if layer is None or not layer.is_rgb:
            return

        raw = self._raw_rgb_cache.get(layer.name)
        if raw is None:
            return

        new_cfg = RGBConfig(
            gamma=self._sliders["gamma"].val,
            gain=self._sliders["gain"].val,
            offset=self._sliders["offset"].val,
        )
        new_data = _apply_rgb_config(raw, new_cfg)
        self._current_im.set_data(new_data)
        self._fig.canvas.draw_idle()

    def show(self):
        """Open the interactive viewer window."""
        self._setup_figure()
        plt.show()
