from typing import Dict, Any, Optional, ClassVar
import numpy as np
from pydantic import Field
from .base import GeoRasterData, PointCloudData, Metadata

class CloudHeightMetadata(Metadata):
    processing_config: Dict[str, Any] = Field(default_factory=lambda: dict(status='unknown'))

class CloudHeightGridData(GeoRasterData):
    """
    Data model for Cloud Height on a raster grid.
    Values represent cloud top height in meters.
    """
    metadata: CloudHeightMetadata = Field(default_factory=CloudHeightMetadata)
    cloud_mask: Optional[np.ndarray] = Field(default=None, description="Binary cloud mask (1=cloud, 0=clear)")
    
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
