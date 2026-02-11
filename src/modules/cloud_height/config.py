from typing import List, Optional
import numpy as np
from pydantic import Field, field_validator
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import BANDS

class CloudHeightConfig(BaseProcessorConfig):
    """Configuration for Cloud Height Processor.

    Retrieves cloud top height from Sentinel-2 parallax using multi-band correlation.
    """
    # Core Parameters
    reference_band: str = Field(
        default='B02',
        description="Reference band (fixed while others shift for parallax)"
    )
    bands: List[str] = Field(
        default=['B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08'],
        description="Bands to use for correlation (minimum 2 required)"
    )

    # Thresholding
    cloudy_thresh: float = Field(
        default=1600.0,
        ge=0,
        le=10000,
        description="Reflectance threshold for cloud detection (DN, 0-10000)"
    )
    threshold_band: str = Field(
        default='B08',
        description="Band to use for cloud thresholding"
    )

    # Spatial / Convolution
    along_track_resolution: int = Field(
        default=3,
        ge=1,
        le=60,
        description="Pixel size along track during convolution (meters)"
    )
    across_track_resolution: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Pixel size across track during convolution (meters)"
    )
    stride: int = Field(
        default=180,
        ge=10,
        le=5000,
        description="Stride between retrieval points (meters)"
    )
    convolved_size_along_track: int = Field(
        default=200,
        ge=50,
        le=2000,
        description="Correlation window size along track (meters)"
    )
    convolved_size_across_track: int = Field(
        default=200,
        ge=50,
        le=2000,
        description="Correlation window size across track (meters)"
    )

    # Method
    correlation_weighting: bool = Field(
        default=True,
        description="Weight height estimates by correlation strength"
    )
    spatial_smoothing_sigma: float = Field(
        default=180.0,
        ge=0,
        le=5000,
        description="Gaussian smoothing kernel sigma (meters, 0=no smoothing)"
    )

    # Height Search Space
    max_height: int = Field(
        default=18000,
        ge=1000,
        le=25000,
        description="Maximum cloud height to search (meters, troposphere limit ~18km)"
    )
    height_step: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Height search step size (meters)"
    )

    # System
    n_workers: int = Field(
        default=48,
        ge=1,
        le=64,
        description="Number of parallel workers for processing"
    )
    temp_dir: Optional[str] = Field(
        default=None,
        description="Temporary directory for intermediate files (None=use /dev/shm)"
    )

    @field_validator('bands')
    @classmethod
    def validate_bands(cls, v):
        """Ensure at least 2 bands for correlation."""
        if len(v) < 2:
            raise ValueError("At least 2 bands required for parallax correlation")
        return v

    @field_validator('reference_band')
    @classmethod
    def validate_reference_band(cls, v):
        """Validate reference band is a valid Sentinel-2 band."""
        if v not in set(BANDS):
            raise ValueError(f"Invalid band: {v}. Must be one of {BANDS}")
        return v

    @property
    def heights(self) -> np.ndarray:
        """Derived property: Array of heights to search."""
        hs = np.arange(0, self.max_height, self.height_step)
        if hs[-1] != self.max_height:
            hs = np.append(hs, self.max_height)
        return hs
