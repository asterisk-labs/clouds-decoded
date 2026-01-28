from typing import List, Optional
from pydantic import Field
from .base import GeoRasterData, Metadata

class CloudPropertiesMetadata(Metadata):
    """Metadata for cloud properties (optical thickness, particle size, etc.)"""
    description: str = "Cloud Properties Inversion Results"
    band_names: List[str] = Field(
        default=["tau", "ice_liq_ratio", "r_eff_liq", "r_eff_ice"],
        description="Names of the bands in the data array"
    )

class CloudPropertiesData(GeoRasterData):
    """
    Data model for Cloud Properties (Refl2Prop output).
    Includes bands for Optical Thickness (Tau) and Effective Radius (Reff).
    """
    metadata: CloudPropertiesMetadata = Field(default_factory=CloudPropertiesMetadata)
