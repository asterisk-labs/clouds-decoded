"""Configuration for albedo estimation."""
from typing import Dict, Literal, Optional
from pydantic import Field, model_validator
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import DEFAULT_SURFACE_ALBEDO


class AlbedoEstimatorConfig(BaseProcessorConfig):
    """Configuration for surface albedo estimation.

    Supports three methods:
    - 'gp': Fits a Gaussian Process to clear-sky pixels (requires cloud mask).
      Reverts to the mean albedo far from observed clear pixels, avoiding the
      divergence issues of polynomial extrapolation.
    - 'idw': Inverse-distance weighting with farthest-point sampling
      (requires cloud mask). Much faster than GP while producing smooth
      spatial interpolation.
    - 'datadriven': Predicts albedo using a trained MLP from physical conditions.
    """

    method: Literal["gp", "idw", "datadriven"] = Field(
        default="idw",
        description="Estimation method: 'gp' (Gaussian Process), "
                    "'idw' (inverse-distance weighting), or 'datadriven' (trained MLP)"
    )
    fallback: Literal["datadriven", "constant"] = Field(
        default="datadriven",
        description="Fallback when GP conditions aren't met: 'datadriven' (MLP) "
                    "or 'constant' (per-band defaults)"
    )

    # GP fitting parameters
    min_clear_fraction: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of clear-sky pixels required. Below this, fallback is used."
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for clear class when mask is non-categorical"
    )
    output_resolution: int = Field(
        default=300,
        ge=10,
        le=1000,
        description="Resolution of output albedo grid in meters/pixel"
    )
    max_samples: int = Field(
        default=1000,
        ge=10,
        description="Max clear-sky pixels to use for GP training. "
                    "GP is O(n³) so keep this low (100–500). Samples are drawn "
                    "randomly from clear pixels after dilation and edge-margin filtering."
    )
    gp_length_scale: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=10.0,
        description="GP RBF length scale in normalised [0,1] coords. "
                    "None = auto-select via marginal likelihood."
    )
    gp_window_m: float = Field(
        default=180.0,
        ge=0,
        description="Side length of spatial averaging window in metres. "
                    "GP training targets are the mean reflectance over this "
                    "window, suppressing pixel noise. 0 = single-pixel."
    )
    gp_dilation_pixels: int = Field(
        default=20,
        ge=0,
        le=50,
        description="Cloud mask dilation buffer in pixels (at B02 resolution). "
                    "Samples within this distance of a cloud edge are excluded."
    )

    # IDW parameters
    idw_k_neighbours: int = Field(
        default=8,
        ge=1,
        description="Number of nearest sample points used per output pixel "
                    "in IDW interpolation. Limits computation and keeps the "
                    "weight matrix sparse."
    )
    idw_smoothing_m: float = Field(
        default=2000.0,
        ge=0.0,
        description="Regularisation distance in metres. Weights are "
                    "1 / (d + d0) instead of 1 / d, preventing a spike "
                    "when a sample falls on an output pixel and blending "
                    "neighbours more smoothly."
    )

    # Data-driven model path (resolved via managed assets when None)
    model_path: Optional[str] = Field(
        default=None,
        description="Path to trained unconditional albedo model checkpoint. "
                    "When None, resolved from the managed assets directory."
    )
    # Constant fallback
    default_albedo: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_SURFACE_ALBEDO),
        description="Default albedo per band when estimation fails [0-1]. "
                    "Bands not listed fall back to 0.05."
    )

    @model_validator(mode="after")
    def _resolve_model_path(self) -> "AlbedoEstimatorConfig":
        if self.model_path is None:
            from clouds_decoded.assets import get_asset
            self.model_path = str(
                get_asset("models/albedo_datadriven/default.pth")
            )
        return self
