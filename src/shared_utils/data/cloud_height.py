from typing import Dict, Any, Optional, ClassVar
import numpy as np
from pydantic import Field
from .base import GeoRasterData, PointCloudData, Metadata

class CloudHeightMetadata(Metadata):
    """Metadata for cloud height outputs.

    Stores the processing configuration used to produce the height map
    (e.g. method, bands, parallax settings).
    """
    processing_config: Dict[str, Any] = Field(default_factory=lambda: dict(status='unknown'))

class CloudHeightGridData(GeoRasterData):
    """Cloud top height on a raster grid, in metres above ground level.

    Values are non-negative floats (NaN for missing/clear pixels).
    Validation rejects any negative height values.
    """
    metadata: CloudHeightMetadata = Field(default_factory=CloudHeightMetadata)
    
    def validate(self) -> bool:
        """Validate that heights are non-negative."""
        if self.data is None:
            return True
        # Ignore NaNs, check for negative values
        # Cloud height should be >= 0 (or >0?)
        # 0 might be ground? usually ground is 0 height AGL or surface relative?
        # Assuming >= 0.
        if np.nanmin(self.data) < 0:
            return False
        return True
