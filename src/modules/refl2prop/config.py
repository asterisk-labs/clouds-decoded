# refl2prop/config.py
"""Configuration for the Cloud Property Inversion (Refl2Prop) module."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import Field, computed_field, model_validator

from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import DEFAULT_SURFACE_ALBEDO

# Default bands for inversion (can be customized per config)
DEFAULT_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

# Fixed geometry features (order matters for model input)
GEOMETRY_FEATURES: Tuple[str, ...] = (
    'incidence_angle',   # Solar Zenith Angle
    'shading_ratio',
    'cloud_top_height',
    'mu',                # Cosine of View Zenith
    'phi'                # Relative Azimuth
)
NUM_GEOMETRY_FEATURES = len(GEOMETRY_FEATURES)


def get_input_feature_names(bands: List[str]) -> List[str]:
    """
    Generate ordered list of input feature names from bands.

    Input vector structure:
    - [0:N]           -> Reflectance bands
    - [N:2N]          -> Surface albedos (one per band)
    - [2N:2N+5]       -> Geometry features

    Args:
        bands: List of band names (e.g., ['B01', 'B02', ...])

    Returns:
        Ordered list of all input feature names
    """
    features = []
    features.extend(bands)  # Reflectance bands
    features.extend([f"{b}_surface_albedo" for b in bands])  # Albedos
    features.extend(GEOMETRY_FEATURES)  # Geometry
    return features


class OutputFeature(str, Enum):
    """
    Output vector order (Physical Properties).
    These are fixed regardless of input band configuration.
    """
    TAU = 'tau'
    ICE_LIQ_RATIO = 'ice_liq_ratio'
    R_EFF_LIQ = 'r_eff_liq'
    R_EFF_ICE = 'r_eff_ice'
    UNCERTAINTY = 'uncertainty'

    @classmethod
    def list(cls) -> List[str]:
        return [param.value for param in cls]


class Refl2PropConfig(BaseProcessorConfig):
    """
    Configuration for the Cloud Property Inversion (Refl2Prop) module.

    The `bands` parameter is the primary configuration - all size-related
    parameters (input_size, noise_output_size, etc.) are computed from it.
    """
    # Method selection
    method: Literal["standard", "shading"] = Field(
        default="standard",
        description="Inversion method: 'standard' (per-pixel) or 'shading' (bag-based with attention)"
    )

    # Primary band configuration - drives all derived values
    bands: List[str] = Field(
        default=DEFAULT_BANDS,
        description="Sentinel-2 bands for inversion. All sizes computed from this."
    )

    # Resources
    model_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the .pth model checkpoint. "
            "Defaults to the managed assets directory; run "
            "'clouds-decoded download refl2prop' to fetch weights."
        ),
    )

    # Processing Parameters
    return_uncertainty: bool = Field(
        default=True,
        description="If True, calculates and appends uncertainty channel"
    )
    mask_invalid_height: bool = Field(
        default=True,
        description="If True, masks pixels with Cloud Height <= 0 as NaN"
    )
    batch_size: int = Field(
        default=32768,
        ge=1,
        le=1000000,
        description="Inference batch size (pixels)"
    )
    working_resolution: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Resolution in metres at which inference is performed.",
    )
    output_resolution: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Output spatial resolution in meters/pixel (absolute, not relative to inputs)"
    )

    # Model Architecture - output_size is fixed (4 cloud properties)
    output_size: int = Field(
        default=4,
        gt=0,
        description="Number of output targets (cloud properties)"
    )

    # Normalization Parameters (height only — band/albedo normalization
    # is handled by Sentinel2Scene.get_band(reflectance=True))
    norm_height_offset: float = Field(
        default=0.0,
        description="Offset for height normalization"
    )
    norm_height_scale: float = Field(
        default=1000.0,
        gt=0,
        description="Scale for height normalization (meters)"
    )

    # Default Fallbacks
    default_albedo: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_SURFACE_ALBEDO),
        description="Default surface albedo per band when estimation fails [0-1]. "
                    "Bands not listed fall back to 0.05."
    )
    default_shading_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default shading ratio when unavailable [0-1]"
    )

    # Output Features
    output_feature_names: List[str] = Field(
        default=['tau', 'ice_liq_ratio', 'r_eff_liq', 'r_eff_ice'],
        description="Names of output features (order matches model output)"
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode='after')
    def _resolve_model_path(self) -> Refl2PropConfig:
        """If no explicit path is given, point at the managed asset location."""
        if self.model_path is None:
            from clouds_decoded.assets import get_asset
            object.__setattr__(self, "model_path", str(get_asset("models/refl2prop/default.pth")))
        return self

    # =========================================================================
    # Computed fields - derived from bands
    # =========================================================================

    @computed_field
    @property
    def num_bands(self) -> int:
        """Number of spectral bands."""
        return len(self.bands)

    @computed_field
    @property
    def input_size(self) -> int:
        """
        Total input size for neural network.
        Formula: num_bands (reflectance) + num_bands (albedo) + 5 (geometry)
        """
        return self.num_bands * 2 + NUM_GEOMETRY_FEATURES

    @computed_field
    @property
    def noise_output_size(self) -> int:
        """
        Noise output size for OOD detection.
        Noise is applied to reflectance bands only.
        """
        return self.num_bands

    @computed_field
    @property
    def noise_indices(self) -> List[int]:
        """
        Indices where noise is applied during training.
        Noise on reflectance bands only (first num_bands elements).
        """
        return list(range(self.num_bands))

    @computed_field
    @property
    def input_feature_names(self) -> List[str]:
        """Ordered list of all input feature names."""
        return get_input_feature_names(self.bands)



class ShadingRefl2PropConfig(Refl2PropConfig):
    """
    Configuration for shading-aware cloud property inversion.

    Extends base config with window/stride parameters for bag-based processing.
    """
    # Window parameters
    window_size: int = Field(
        default=24,
        ge=4,
        le=64,
        description="Size of square processing window (pixels). Bag size = window_size^2."
    )
    stride: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Step size between windows. If < window_size, windows overlap."
    )

    # Model architecture (must match trained model)
    hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dimension for shading model"
    )
    n_heads: int = Field(
        default=4,
        gt=0,
        description="Number of attention heads"
    )
    n_attention_layers: int = Field(
        default=2,
        gt=0,
        description="Number of self-attention layers"
    )

    # Override output feature names to include tau_shading
    output_feature_names: List[str] = Field(
        default=['tau', 'ice_liq_ratio', 'r_eff_liq', 'r_eff_ice', 'tau_shading'],
        description="Names of output features (order matches model output)"
    )

    @computed_field
    @property
    def bag_size(self) -> int:
        """Number of pixels per bag (window_size^2)."""
        return self.window_size * self.window_size
