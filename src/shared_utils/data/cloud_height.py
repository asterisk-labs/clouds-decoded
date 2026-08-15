from typing import Dict, Any, List, Optional, ClassVar
import numpy as np
from pydantic import Field
from .base import GeoRasterData, PointCloudData, Metadata

class CloudHeightMetadata(Metadata):
    """Metadata for cloud height outputs.

    Stores the processing configuration used to produce the height map
    (e.g. method, bands, parallax settings), and the names of the bands
    present in the array.
    """
    processing_config: Dict[str, Any] = Field(default_factory=lambda: dict(status='unknown'))
    band_names: List[str] = Field(
        default_factory=list,
        description=(
            "Names of the bands in the data array. Band 0 is always the height; "
            "downstream consumers index it directly. With cloud_height "
            "`quality_bands` enabled the retrieval appends peak_correlation, "
            "fwhm and tied_span, and names all four here.\n"
            "Empty for a plain single-band height raster, deliberately: the "
            "stats module emits flat keys ('mean') for a raster with no band "
            "names and prefixed ones ('cloud_top_height__mean') otherwise, so "
            "naming the single band would rename keys in every existing "
            "project's stats without the raster having changed."
        )
    )

class CloudHeightGridData(GeoRasterData):
    """Cloud top height on a raster grid, in metres above ground level.

    Band 0 is the height (NaN for missing/clear pixels). Further bands, when
    present, are per-cell quality metrics named in ``metadata.band_names``.
    """
    metadata: CloudHeightMetadata = Field(default_factory=CloudHeightMetadata)

    def validate(self) -> bool:
        """Validate that heights are non-negative.

        Only the height band is checked. The quality bands carry different
        quantities on different scales -- peak correlation is a signed
        correlation and may legitimately be negative -- so validating the whole
        array would reject sound output.
        """
        if self.data is None:
            return True
        heights = self.data[0] if self.data.ndim == 3 else self.data
        if np.all(np.isnan(heights)):
            return True
        if np.nanmin(heights) < 0:
            return False
        return True
