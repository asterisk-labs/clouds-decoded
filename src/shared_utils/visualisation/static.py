"""Static plotting functions for generating figures from Layer objects."""
import math
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.figure import Figure

from .layers import Layer

# Fixed categorical palette for the 4-class cloud mask
_CLOUD_MASK_COLORS = ["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]  # clear, thick, thin, shadow
_CLOUD_MASK_CMAP = ListedColormap(_CLOUD_MASK_COLORS, name="cloud_mask")
_CLOUD_MASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)


def _build_norm(render):
    """Build a matplotlib Normalize from a RenderConfig."""
    if render.categorical:
        labels = render.category_labels or {}
        n = max(labels.keys()) + 1 if labels else 4
        bounds = [i - 0.5 for i in range(n + 1)]
        return BoundaryNorm(bounds, ncolors=n)
    if render.log_scale:
        vmin = render.vmin if render.vmin and render.vmin > 0 else 1e-2
        vmax = render.vmax or 1.0
        return LogNorm(vmin=vmin, vmax=vmax)
    return None  # let imshow use linear with vmin/vmax


def _build_cmap(render):
    """Build a matplotlib Colormap from a RenderConfig."""
    if render.categorical:
        labels = render.category_labels or {}
        n = max(labels.keys()) + 1 if labels else 4
        if n == 4:
            return _CLOUD_MASK_CMAP
        # Generic categorical fallback
        base = plt.get_cmap("tab10")
        return ListedColormap([base(i) for i in range(n)], name="categorical")
    return plt.get_cmap(render.cmap)


def _inset_title(ax: plt.Axes, text: str) -> None:
    """Place a title inside the top-left of the axes with a semi-transparent box."""
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        fontsize=9, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
    )


def _inset_colorbar(im, ax: plt.Axes) -> None:
    """Add a compact colorbar inside the bottom-right of the axes."""
    cax = ax.inset_axes([0.62, 0.04, 0.35, 0.03])  # [x0, y0, width, height]
    plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=6, length=2, pad=1)


def render_to_axes(
    layer: Layer,
    ax: plt.Axes,
    alpha: Optional[float] = None,
    show_title: bool = True,
) -> Optional[plt.cm.ScalarMappable]:
    """Render a Layer onto an existing matplotlib Axes.

    Args:
        layer: The Layer to render.
        ax: Target Axes.
        alpha: Optional transparency (0–1). Useful for overlays.
        show_title: If True, draw the layer name as an inset label.

    Returns:
        The ScalarMappable (for colorbar) or None for RGB layers.
    """
    ax.set_xticks([])
    ax.set_yticks([])

    if layer.is_rgb:
        ax.imshow(layer.data, extent=layer.extent, origin="upper", aspect="equal",
                  alpha=alpha)
        if show_title:
            _inset_title(ax, layer.name)
        return None

    render = layer.render
    cmap = _build_cmap(render)
    norm = _build_norm(render)

    kwargs = dict(
        cmap=cmap,
        extent=layer.extent,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    if alpha is not None:
        kwargs["alpha"] = alpha
    if norm is not None:
        kwargs["norm"] = norm
    else:
        if render.vmin is not None:
            kwargs["vmin"] = render.vmin
        if render.vmax is not None:
            kwargs["vmax"] = render.vmax

    im = ax.imshow(layer.data, **kwargs)

    if show_title:
        _inset_title(ax, render.label or layer.name)

    return im


def plot_layer(layer: Layer, ax: Optional[plt.Axes] = None, figsize=(8, 8)) -> Figure:
    """Plot a single Layer to a matplotlib Figure.

    Args:
        layer: The Layer to plot.
        ax: Optional axes to draw on. If None, a new figure is created.
        figsize: Figure size when creating a new figure.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    im = render_to_axes(layer, ax)

    if im is not None:
        render = layer.render
        if render.categorical and render.category_labels:
            # Categorical legend instead of colorbar
            labels = render.category_labels
            cmap = _build_cmap(render)
            handles = [
                plt.Line2D([0], [0], marker="s", color="w",
                           markerfacecolor=cmap(i), markersize=10, label=lbl)
                for i, lbl in sorted(labels.items())
            ]
            ax.legend(handles=handles, loc="upper right", framealpha=0.8)
        else:
            _inset_colorbar(im, ax)

    fig.tight_layout(pad=0.5)
    return fig


def plot_overview(layers: List[Layer], ncols: int = 3, figsize=(16, 12)) -> Figure:
    """Plot all layers in a grid overview.

    Args:
        layers: List of Layers to plot.
        ncols: Number of columns in the grid.
        figsize: Figure size.

    Returns:
        The matplotlib Figure.
    """
    n = len(layers)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No layers to display", ha="center", va="center", transform=ax.transAxes)
        return fig

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, layer in enumerate(layers):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        im = render_to_axes(layer, ax)

        if im is not None and not layer.render.categorical:
            _inset_colorbar(im, ax)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(pad=0.5)
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 150):
    """Save a matplotlib Figure to disk.

    Args:
        fig: Figure to save.
        path: Output file path (e.g. ``overview.png``).
        dpi: Resolution in dots per inch.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
