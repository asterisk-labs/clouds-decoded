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
    """Contrast controls for RGB composite layers."""
    gamma: float = 0.7
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
    """Apply gain, offset, and gamma correction to an RGB array."""
    out = rgb * config.gain + config.offset
    out = np.clip(out, 0, None)
    out = np.power(out, config.gamma)
    return np.clip(out, 0, 1).astype(np.float32)


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
            cmap="inferno",
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
    "r_eff_liq": RenderConfig(cmap="viridis", log_scale=True, vmin=1, vmax=50, label="Effective Radius (liquid)", units="\u03bcm"),
    "r_eff_ice": RenderConfig(cmap="viridis", log_scale=True, vmin=3, vmax=370, label="Effective Radius (ice)", units="\u03bcm"),
    "uncertainty": RenderConfig(cmap="magma", label="Uncertainty", units=""),
}


def layers_from_cloud_properties(data) -> List[Layer]:
    """Create one Layer per band from a CloudPropertiesData instance."""
    band_names = getattr(data.metadata, "band_names", ["tau", "ice_liq_ratio", "r_eff_liq", "r_eff_ice"])
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


def layer_from_rgb(
    scene,
    composite: str = "true_color",
    rgb_config: Optional[RGBConfig] = None,
) -> Layer:
    """Create an RGB composite Layer from a Sentinel2Scene.

    Uses get_band(reflectance=True) for calibrated reflectance values.
    """
    if composite not in RGB_COMPOSITES:
        raise ValueError(f"Unknown composite '{composite}'. Choose from {list(RGB_COMPOSITES.keys())}")

    if rgb_config is None:
        rgb_config = RGBConfig()

    r_band, g_band, b_band = RGB_COMPOSITES[composite]

    from skimage.transform import resize as sk_resize

    r = scene.get_band(r_band, reflectance=True)
    g = scene.get_band(g_band, reflectance=True)
    b = scene.get_band(b_band, reflectance=True)

    # Resample to common shape (largest)
    target_shape = max([r.shape, g.shape, b.shape], key=lambda s: s[0] * s[1])
    if r.shape != target_shape:
        r = sk_resize(r, target_shape, order=1, preserve_range=True).astype(np.float32)
    if g.shape != target_shape:
        g = sk_resize(g, target_shape, order=1, preserve_range=True).astype(np.float32)
    if b.shape != target_shape:
        b = sk_resize(b, target_shape, order=1, preserve_range=True).astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = _apply_rgb_config(rgb, rgb_config)

    extent = _extent_from_transform(scene.transform, *target_shape) if scene.transform else None
    res = abs(scene.transform.a) if scene.transform else None

    display_name = {"true_color": "True Colour", "ice": "Ice Composite"}.get(composite, composite)
    return Layer(
        name=display_name,
        data=rgb,
        render=RenderConfig(label=display_name),
        extent=extent,
        resolution_m=res,
        is_rgb=True,
        rgb_config=rgb_config,
    )
