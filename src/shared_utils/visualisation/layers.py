"""Layer abstraction for visualisation.

A Layer pairs a 2D numpy array with rendering metadata (colormap, scale, extent),
decoupling data loading from display. Factory functions convert the project's data
classes into ready-to-render layers.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class RenderConfig(BaseModel):
    """How to display a single-channel layer."""
    cmap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    log_scale: bool = False
    categorical: bool = False
    category_labels: Optional[Dict[int, str]] = None
    label: str = ""
    units: str = ""


class RGBConfig(BaseModel):
    """Contrast controls for RGB composite layers.

    The rendering pipeline is::

        out = clip((data * gain + offset) ** gamma, 0, 1)

    Use :func:`rgb_config_from_data` to auto-compute *gain* and *offset*
    from the data's percentiles so the result fills [0, 1] after gamma.
    The user can then tweak *gain* from that starting point.
    """
    gamma: float = 0.65
    gain: float = 1.0
    offset: float = 0.0


# ---------------------------------------------------------------------------
# Layer model
# ---------------------------------------------------------------------------

class Layer(BaseModel):
    """A displayable image layer with rendering metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    data: np.ndarray  # (H, W) for scalar layers, (H, W, 3) for RGB
    render: RenderConfig = Field(default_factory=RenderConfig)
    extent: Optional[Tuple[float, float, float, float]] = None  # (left, right, bottom, top)
    resolution_m: Optional[float] = None
    is_rgb: bool = False
    rgb_config: Optional[RGBConfig] = None  # only set for RGB layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extent_from_transform(transform, height: int, width: int) -> Tuple[float, float, float, float]:
    """Convert an affine transform + shape to matplotlib imshow extent (left, right, bottom, top)."""
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e  # transform.e is negative
    return (left, right, bottom, top)


def _squeeze_2d(data: np.ndarray) -> np.ndarray:
    """Ensure (1, H, W) -> (H, W)."""
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    return data


def _apply_rgb_config(rgb: np.ndarray, config: RGBConfig) -> np.ndarray:
    """Apply gain, offset, and gamma correction to an RGB array.

    Pipeline: ``clip((data * gain + offset) ** gamma, 0, 1)``
    """
    out = rgb * config.gain + config.offset
    out = np.clip(out, 0, None)
    out = np.power(out, config.gamma)
    return np.clip(out, 0, 1).astype(np.float32)


def rgb_config_from_data(
    rgb: np.ndarray,
    gamma: float = 0.65,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> RGBConfig:
    """Compute gain and offset so that percentile-clipped data fills [0, 1] after gamma.

    Given ``out = (data * gain + offset) ** gamma``, we want
    ``p_low`` to map to 0 and ``p_high`` to map to 1.  Before gamma::

        data * gain + offset = 0   at p_low   =>  offset = -p_low * gain
        data * gain + offset = 1   at p_high  =>  gain = 1 / (p_high - p_low)

    Args:
        rgb: Raw reflectance array (H, W, 3).
        gamma: Gamma exponent.
        low_pct: Lower percentile for clipping.
        high_pct: Upper percentile for clipping.

    Returns:
        An :class:`RGBConfig` with gain/offset calibrated to the data.
    """
    valid = rgb[rgb > 0]
    if valid.size == 0:
        return RGBConfig(gamma=gamma)
    p_low = float(np.percentile(valid, low_pct))
    p_high = float(np.percentile(valid, high_pct))
    span = max(p_high - p_low, 1e-10)
    gain = 1.0 / span
    offset = -p_low * gain
    return RGBConfig(gamma=gamma, gain=gain, offset=offset)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def layer_from_cloud_mask(data) -> Layer:
    """Create a Layer from a CloudMaskData instance."""
    arr = _squeeze_2d(data.data)
    extent = _extent_from_transform(data.transform, *arr.shape[:2]) if data.transform else None
    res = abs(data.transform.a) if data.transform else None

    labels = getattr(data.metadata, "classes", {0: "Clear", 1: "Thick Cloud", 2: "Thin Cloud", 3: "Cloud Shadow"})
    return Layer(
        name="Cloud Mask",
        data=arr,
        render=RenderConfig(
            cmap="cloud_mask",  # handled specially by the renderer
            categorical=True,
            category_labels=labels,
            label="Cloud Mask",
        ),
        extent=extent,
        resolution_m=res,
    )


def layer_from_cloud_height(data) -> Layer:
    """Create a Layer from a CloudHeightGridData instance."""
    arr = _squeeze_2d(data.data)
    extent = _extent_from_transform(data.transform, *arr.shape[:2]) if data.transform else None
    res = abs(data.transform.a) if data.transform else None

    return Layer(
        name="Cloud Height",
        data=arr.astype(np.float32),
        render=RenderConfig(
            cmap="turbo",
            vmin=0,
            label="Cloud Top Height",
            units="m",
        ),
        extent=extent,
        resolution_m=res,
    )


# Per-band render presets for cloud properties
_PROPERTIES_RENDER: Dict[str, RenderConfig] = {
    "tau": RenderConfig(cmap="viridis", log_scale=True, vmin=0.1, vmax=200, label="Optical Thickness", units=""),
    "ice_liq_ratio": RenderConfig(cmap="RdBu", vmin=0, vmax=1, label="Ice / Liquid Ratio", units=""),
    "r_eff_liq": RenderConfig(cmap="plasma", log_scale=True, vmin=1, vmax=50, label="Effective Radius (liquid)", units="\u03bcm"),
    "r_eff_ice": RenderConfig(cmap="plasma", log_scale=True, vmin=3, vmax=370, label="Effective Radius (ice)", units="\u03bcm"),
    "uncertainty": RenderConfig(cmap="jet", label="Uncertainty", units=""),
}


def layers_from_cloud_properties(data) -> List[Layer]:
    """Create one Layer per band from a CloudPropertiesData instance."""
    band_names = getattr(data.metadata, "band_names", ["tau", "ice_liq_ratio", "r_eff_liq", "r_eff_ice", "uncertainty"])
    arr = data.data  # (N, H, W)

    extent = _extent_from_transform(data.transform, arr.shape[-2], arr.shape[-1]) if data.transform else None
    res = abs(data.transform.a) if data.transform else None

    layers = []
    for i, name in enumerate(band_names):
        if i >= arr.shape[0]:
            break
        band_data = arr[i].astype(np.float32)
        render = _PROPERTIES_RENDER.get(name, RenderConfig(label=name))
        layers.append(Layer(
            name=f"Properties: {name}",
            data=band_data,
            render=render,
            extent=extent,
            resolution_m=res,
        ))
    return layers


def layer_from_albedo(data) -> Layer:
    """Create a single mean-albedo Layer from an AlbedoData instance."""
    arr = data.data  # (N, H, W)

    extent = _extent_from_transform(data.transform, arr.shape[-2], arr.shape[-1]) if data.transform else None
    res = abs(data.transform.a) if data.transform else None

    mean_albedo = np.nanmean(arr, axis=0).astype(np.float32)
    return Layer(
        name="Albedo",
        data=mean_albedo,
        render=RenderConfig(cmap="gray", vmin=0, vmax=1, label="Albedo (mean)", units=""),
        extent=extent,
        resolution_m=res,
    )


def layers_from_albedo(data) -> List[Layer]:
    """Create one Layer per band from an AlbedoData instance."""
    band_names = getattr(data.metadata, "band_names", [])
    arr = data.data  # (N, H, W)

    extent = _extent_from_transform(data.transform, arr.shape[-2], arr.shape[-1]) if data.transform else None
    res = abs(data.transform.a) if data.transform else None

    layers = []
    for i, name in enumerate(band_names):
        if i >= arr.shape[0]:
            break
        layers.append(Layer(
            name=f"Albedo: {name}",
            data=arr[i].astype(np.float32),
            render=RenderConfig(cmap="gray", vmin=0, label=f"Albedo ({name})", units=""),
            extent=extent,
            resolution_m=res,
        ))
    return layers


# RGB composite definitions: (red_band, green_band, blue_band)
RGB_COMPOSITES = {
    "true_color": ("B04", "B03", "B02"),
    "ice": ("B12", "B11", "B04"),
}


def downsample_layer(layer: Layer, target_resolution_m: float) -> Layer:
    """Stride-downsample a layer to approximately *target_resolution_m*.

    If the layer's resolution is already >= *target_resolution_m* the
    layer is returned unchanged.

    Args:
        layer: Input layer.
        target_resolution_m: Desired pixel size in metres.

    Returns:
        A (potentially) downsampled Layer with updated extent and resolution.
    """
    if layer.resolution_m is None or layer.resolution_m >= target_resolution_m:
        return layer

    stride = max(1, round(target_resolution_m / layer.resolution_m))
    if stride <= 1:
        return layer

    if layer.is_rgb or layer.data.ndim == 3:
        data = layer.data[::stride, ::stride]
    else:
        data = layer.data[::stride, ::stride]

    new_res = layer.resolution_m * stride

    # Recompute extent from new shape.
    extent = layer.extent
    if extent is not None:
        # Extent stays the same (geographic bounds don't change),
        # but we keep it consistent.
        pass

    return Layer(
        name=layer.name,
        data=data,
        render=layer.render,
        extent=extent,
        resolution_m=new_res,
        is_rgb=layer.is_rgb,
        rgb_config=layer.rgb_config,
    )


def layer_from_rgb(
    scene,
    composite: str = "true_color",
    rgb_config: Optional[RGBConfig] = None,
    display_resolution_m: Optional[float] = None,
) -> Layer:
    """Create an RGB composite Layer from a Sentinel2Scene.

    Uses get_band(reflectance=True) for calibrated reflectance values.
    For the ice composite, per-channel stretch is enabled by default.

    Args:
        scene: A :class:`~clouds_decoded.data.Sentinel2Scene`.
        composite: ``"true_color"`` or ``"ice"``.
        rgb_config: Override :class:`RGBConfig`.
        display_resolution_m: Target pixel size in metres.  When set,
            bands are loaded at this resolution (or the closest native
            resolution) to avoid reading full-resolution data.
    """
    if composite not in RGB_COMPOSITES:
        raise ValueError(f"Unknown composite '{composite}'. Choose from {list(RGB_COMPOSITES.keys())}")

    r_band, g_band, b_band = RGB_COMPOSITES[composite]

    res_kwarg: dict = {}
    if display_resolution_m is not None:
        res_kwarg["resolution"] = int(display_resolution_m)

    from skimage.transform import resize as sk_resize

    r = scene.get_band(r_band, reflectance=True, **res_kwarg)
    g = scene.get_band(g_band, reflectance=True, **res_kwarg)
    b = scene.get_band(b_band, reflectance=True, **res_kwarg)

    # Resample to common shape (largest)
    target_shape = max([r.shape, g.shape, b.shape], key=lambda s: s[0] * s[1])
    if r.shape != target_shape:
        r = sk_resize(r, target_shape, order=1, preserve_range=True).astype(np.float32)
    if g.shape != target_shape:
        g = sk_resize(g, target_shape, order=1, preserve_range=True).astype(np.float32)
    if b.shape != target_shape:
        b = sk_resize(b, target_shape, order=1, preserve_range=True).astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)

    # Auto-compute gain/offset from data percentiles if no config given.
    if rgb_config is None:
        rgb_config = rgb_config_from_data(rgb)

    rgb = _apply_rgb_config(rgb, rgb_config)

    actual_res = display_resolution_m or (abs(scene.transform.a) if scene.transform else None)
    # Build a transform that matches the actual pixel size of the output array.
    # scene.transform is the B02 (10 m) reference; when bands are loaded at a
    # coarser resolution the pixel size differs but the origin stays the same.
    if scene.transform and actual_res:
        from rasterio.transform import Affine
        display_transform = Affine(actual_res, 0, scene.transform.c,
                                   0, -actual_res, scene.transform.f)
        extent = _extent_from_transform(display_transform, *target_shape)
    elif scene.transform:
        extent = _extent_from_transform(scene.transform, *target_shape)
    else:
        extent = None

    display_name = {"true_color": "True Colour", "ice": "Ice Composite"}.get(composite, composite)
    return Layer(
        name=display_name,
        data=rgb,
        render=RenderConfig(label=display_name),
        extent=extent,
        resolution_m=actual_res,
        is_rgb=True,
        rgb_config=rgb_config,
    )
