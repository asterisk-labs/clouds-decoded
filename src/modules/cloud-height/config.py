from typing import List, Optional
import numpy as np
from pydantic import Field
from clouds_decoded.shared_utils.config import BaseProcessorConfig

class CloudHeightConfig(BaseProcessorConfig):
    """
    Configuration for Cloud Height Processor.
    """
    # Core Parameters
    reference_band: str = Field('B02', description="Band that is fixed whilst others move.")
    bands: List[str] = Field(['B02','B03','B04','B05','B07','B08'], description="Bands to use for correlation.")
    
    # Thresholding
    cloudy_thresh: float = Field(1600.0, description="Basic thresholding for cloudiness in DN.")
    threshold_band: str = Field('B08', description="Band to use for thresholding.")
    
    # Spatial / Convolution
    along_track_resolution: int = Field(5, description="Pixel size used during convolution (m).")
    across_track_resolution: int = Field(10, description="Pixel size used during convolution (m).")
    stride: int = Field(300, description="Stride between points in metres.")
    convolved_size_along_track: int = Field(200, description="Correlation window size along track (m).")
    convolved_size_across_track: int = Field(200, description="Correlation window size across track (m).")
    
    # Method
    correlation_weighting: bool = Field(True, description="Weight height estimates by correlation value.")
    spatial_smoothing_sigma: float = Field(200.0, description="Gaussian kernel sigma for smoothing (m).")
    
    # Height Search Space
    max_height: int = Field(18000, description="Maximum height to search (m).")
    height_step: int = Field(100, description="Height step (m).")
    
    # System
    temp_dir: Optional[str] = Field(None, description="Temporary directory. If None, uses /dev/shm.")

    @property
    def heights(self) -> np.ndarray:
        """Derived property: Array of heights to search."""
        hs = np.arange(0, self.max_height, self.height_step)
        if hs[-1] != self.max_height:
            hs = np.append(hs, self.max_height)
        return hs
