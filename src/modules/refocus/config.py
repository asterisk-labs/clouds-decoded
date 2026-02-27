# refocus/config.py
"""Configuration for the Refocus (parallax correction) module."""
from typing import Optional, List
from pydantic import Field, field_validator

from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import BANDS, BAND_RESOLUTIONS


class RefocusConfig(BaseProcessorConfig):
    """
    Configuration for parallax correction (refocusing) of Sentinel-2 bands.

    The refocus module removes height-dependent misalignment between Sentinel-2
    bands caused by the push-broom acquisition and orbital motion. Each band is
    shifted to align with the reference band (B02, which has zero time delay).

    Resolution modes:
        - output_resolution=None  -> each band stays at its native resolution
        - output_resolution=10    -> all bands resampled to 10 m common grid
    """
    # Reference band (zero time delay)
    reference_band: str = Field(
        default='B02',
        description="Reference band with zero parallax (B02 for Sentinel-2)"
    )

    # Which bands to refocus (None = all bands in the scene)
    bands: Optional[List[str]] = Field(
        default=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        # B09 and B10 are intentionally excluded — refl2prop (the primary consumer of
        # refocused output) does not use them.  Set bands=None to include all bands.
        description="Bands to refocus. None = all bands in the scene."
    )

    # Output resolution
    output_resolution: Optional[int] = Field(
        default=60, # same default as refl2prop (primary user of refocus)
        ge=10,
        description=(
            "Output resolution in meters. None = preserve native resolution per band. "
            "If set, all bands are resampled to this common grid."
        )
    )

    # Output saving
    save_refocused: bool = Field(
        default=False,
        description="Save refocused bands as individual GeoTIFFs in the output directory"
    )

    # Interpolation
    interpolation_order: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Interpolation order for band warping (0=nearest, 1=bilinear, 3=cubic)"
    )
    height_interpolation_order: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Interpolation order for upsampling height map to band resolution"
    )

    @field_validator('reference_band')
    @classmethod
    def validate_reference_band(cls, v):
        valid_bands = set(BAND_RESOLUTIONS.keys())
        if v not in valid_bands:
            raise ValueError(f"Invalid reference band: {v}. Must be one of {valid_bands}")
        return v
