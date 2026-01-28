
import numpy as np
import logging
from typing import Dict, Union, Optional
from skimage.transform import resize
import rasterio

from clouds_decoded.data import Sentinel2Scene, CloudMaskData
# from .config import CloudMaskConfig  # TODO: create if needed

logger = logging.getLogger(__name__)

class CloudMaskProcessor:
    def __init__(self, config=None):
        self.config = config

    def process(self, scene: Sentinel2Scene, threshold_band: str = "B08", threshold_value: float = 1600) -> CloudMaskData:
        """
        Creates a basic cloud mask by thresholding a specific band.
        Values GREATER than the threshold are considered cloud (1), others not (0).
        
        Args:
            scene: Sentinel2Scene object with loaded bands.
            threshold_band: The band ID to use (e.g., 'B08', 'B10').
            threshold_value: The cutoff value for cloud detection.
            
        Returns:
            CloudMaskData: Binary raster (1=Cloud, 0=Clear) matching B02 resolution.
        """
        # Ensure bands are loaded
        if threshold_band not in scene.bands:
             raise ValueError(f"Required band {threshold_band} not loaded in scene.")
             
        # Get raw data
        band_data = scene.bands[threshold_band]
        
        # Ensure 2D
        if isinstance(band_data, np.ndarray) and band_data.ndim == 3:
             band_data = band_data[0]
             
        logger.info(f"Generating cloud mask using: {threshold_band} > {threshold_value}")
        
        # Calculate Mask
        # Note: Depending on processing level (L1C vs L2A), reflectance values differ.
        # L1C is uint16 (0-10000). User threshold might be float (0.25) or int (2500).
        # We need to detect or enforce convention.
        # Assuming existing logical legacy was: 0.25 float implies 0..1 range.
        # Sentinel2Scene.read() loads using rasterio.read().
        # If dataset is uint16, we should verify. 
        # But `Column.getMask` just did: self.bands[threshold_band] > threshold.
        # We assume the user configures the threshold matching the data type.
        
        mask = (band_data > threshold_value).astype(np.uint8)
        
        # If threshold band is low res (e.g. B10 is 60m), we should upscale it to 
        # match the reference standard (usually 10m or 20m like B02).
        # Let's standardize on B02 (10m) resolution/crs for the output mask.
        
        ref_band = "B02"
        if ref_band in scene.bands:
             ref_arr = scene.bands[ref_band]
             if ref_arr.ndim == 3: ref_arr = ref_arr[0]
             
             if mask.shape != ref_arr.shape:
                  logger.info(f"Resizing mask from {mask.shape} to {ref_arr.shape} (Reference {ref_band})")
                  # Use order=0 (Nearest) to preserve binary nature
                  mask = resize(mask, ref_arr.shape, order=0, preserve_range=True).astype(np.uint8)
        
        # Construct CloudMaskData using Sentinel2Scene georeferencing
        out = CloudMaskData(
             data=mask,
             transform=scene.transform,
             crs=scene.crs,
             metadata={
                 "method": "simple_threshold", 
                 "band": threshold_band, 
                 "value": threshold_value
             }
        )
        return out
