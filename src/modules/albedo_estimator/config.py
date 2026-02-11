"""Configuration for albedo estimation."""
from typing import Dict, Literal
from pydantic import Field
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import DEFAULT_SURFACE_ALBEDO


class AlbedoEstimatorConfig(BaseProcessorConfig):
    """Configuration for surface albedo estimation.

    Supports two methods:
    - 'polynomial': Fits a 2D polynomial to clear-sky pixels (requires cloud mask).
      Falls back to percentile if insufficient clear pixels are available.
    - 'percentile': Simple percentile-based constant albedo per band (legacy).
    """

    method: Literal["polynomial", "percentile"] = Field(
        default="polynomial",
        description="Estimation method: 'polynomial' (clear-sky fit) or 'percentile' (legacy constant)"
    )

    # Polynomial fitting parameters
    polynomial_order: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Order of 2D polynomial fit (1=linear, 2=quadratic, 3=cubic)"
    )
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
        default=100_000,
        ge=1000,
        description="Max clear pixels to sample for fitting (subsampled if exceeded, for speed)"
    )

    # Fallback / legacy
    default_albedo: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_SURFACE_ALBEDO),
        description="Default albedo per band when estimation fails [0-1]. "
                    "Bands not listed fall back to 0.05."
    )
    percentile: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Percentile for dark-object subtraction (used in 'percentile' method or fallback) [0-100]"
    )
