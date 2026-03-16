from typing import List, Optional
from pydantic import Field
from .base import GeoRasterData, Metadata

class CloudPropertiesMetadata(Metadata):
    """Metadata for cloud optical and microphysical property outputs.

    Records the band names present in the output array (default:
    optical thickness, ice-liquid ratio, liquid effective radius,
    ice effective radius).
    """
    description: str = "Cloud Properties Inversion Results"
    band_names: List[str] = Field(
        default=["tau", "ice_liq_ratio", "r_eff_liq", "r_eff_ice", "uncertainty"],
        description="Names of the bands in the data array"
    )

class CloudPropertiesData(GeoRasterData):
    """5-band cloud property raster from reflectance-to-property inversion.

    Bands (in order): optical thickness (tau), ice-liquid ratio,
    liquid effective radius, ice effective radius, and uncertainty.
    NaN indicates pixels where inversion was not performed (e.g. clear sky).
    """
    metadata: CloudPropertiesMetadata = Field(default_factory=CloudPropertiesMetadata)
