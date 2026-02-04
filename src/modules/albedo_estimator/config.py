"""Configuration for albedo estimation."""
from pydantic import Field
from clouds_decoded.config import BaseProcessorConfig


class AlbedoEstimatorConfig(BaseProcessorConfig):
    """Configuration for surface albedo estimation.

    Uses a simple percentile-based method to estimate constant albedo per band.
    This is a placeholder for more sophisticated methods.
    """

    percentile: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Percentile for dark object subtraction [0-100]. "
                    "Lower values (e.g., 1.0) estimate minimum reflectance for dark surfaces."
    )

    default_albedo: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Default albedo value if calculation fails or scene is all NaN [0-1]."
    )

    method: str = Field(
        default="percentile",
        description="Estimation method. Currently only 'percentile' is supported."
    )
