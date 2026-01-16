from typing import Dict, Any, Optional
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

class CloudHeightPointsData(PointCloudData):
    """
    Data model for Cloud Height as sparse points.
    Expected columns: x, y, height
    Optional columns: correlation, etc.
    """
    REQUIRED_COLUMNS = ['x', 'y', 'height']
    metadata: CloudHeightMetadata = Field(default_factory=CloudHeightMetadata)

    def validate(self) -> bool:
        """Validate dataframe columns and value ranges."""
        if self.data is None or self.data.empty:
            return True
            
        # Check columns
        if not all(col in self.data.columns for col in self.REQUIRED_COLUMNS):
            return False
            
        # Check heights are non-negative
        if self.data['height'].min() < 0:
            return False
            
        return True
