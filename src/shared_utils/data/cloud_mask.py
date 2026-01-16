from typing import Dict
from pydantic import Field
import numpy as np
from .base import GeoRasterData, Metadata

class CloudMaskMetadata(Metadata):
    categorical: bool = True
    classes: Dict[int, str] = {
        0: 'Clear',
        1: 'Thick Cloud',
        2: 'Thin Cloud',
        3: 'Cloud Shadow'
    }

class CloudMaskData(GeoRasterData):
    """
    Data model for cloud masks using the SEnSeI-v2 variants.
    Supports either discrete 4-class classification or 4-channel probability maps.
    """
    metadata: CloudMaskMetadata = Field(default_factory=CloudMaskMetadata)

    def validate(self) -> bool:
        """Validate that data conforms to the categorical strictness."""
        if self.data is None:
            return True 
        
        if self.metadata.categorical:
            # Discrete values check
            # Expecting 2D array of integers
            if self.data.ndim != 2:
                 # It's possible to have (1, H, W) for single band categorical
                 if self.data.ndim == 3 and self.data.shape[0] == 1:
                     pass
                 else:
                     return False
            
            unique_vals = np.unique(self.data)
            for val in unique_vals:
                if val not in self.metadata.classes:
                    return False
        else:
            # Probabilities check
            # Expecting 3D array (4, H, W) with floats 0-1
            if self.data.ndim != 3:
                return False
            if self.data.shape[0] != 4:
                return False
            if not np.issubdtype(self.data.dtype, np.floating):
                 return False
            if self.data.min() < 0 or self.data.max() > 1:
                 return False
                 
        return True
