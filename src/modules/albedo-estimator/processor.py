import numpy as np
import logging
from typing import Dict, Optional

from clouds_decoded.data import Sentinel2Scene, AlbedoData, GeoRasterData
# from .config import AlbedoConfig 

logger = logging.getLogger(__name__)

class AlbedoEstimator:
    def __init__(self, config=None):
        self.config = config

    def process(self, scene: Sentinel2Scene, percentile: float = 1.0) -> Dict[str, AlbedoData]:
        """
        Estimates albedo as a constant value per band based on the n-th percentile of reflectances.
        Returns a dictionary of AlbedoData objects (one per band), where each object contains 
        a raster filled with the estimated value.
        
        Args:
            scene: The Sentinel2Scene input.
            percentile: The percentile to extract (0-100). Default 1.0 (dark object).
            
        Returns:
             Dict[str, AlbedoData]: Dictionary of AlbedoData raasters per band.
        """
        logger.info(f"Estimating albedo using {percentile}th percentile method.")
        
        albedo_rasters = {}
        
        # We iterate over all loaded bands in the scene
        for band_name, band_data in scene.bands.items():
            if not isinstance(band_data, np.ndarray):
                logger.debug(f"Skipping band {band_name} (not a numpy array/not loaded).")
                continue
                
            try:
                # Calculate scalar albedo
                estimated_val = 0.05 # Default
                
                valid_mask = ~np.isnan(band_data)
                if np.any(valid_mask):
                    estimated_val = float(np.percentile(band_data[valid_mask], percentile))
                else:
                    logger.warning(f"Band {band_name} is all NaN. Defaulting to 0.05")
                
                # Create a full raster of this value
                # Using float32 for model compatibility
                albedo_map = np.full_like(band_data, estimated_val, dtype=np.float32)

                # Wrap in AlbedoData
                # We assume the scene has a valid transform/crs from B02
                # If these are None, GeoRasterData will just lack georef, which is fine for internal use
                # but might be issue for writing.
                
                albedo_obj = AlbedoData(
                    data=albedo_map,
                    transform=scene.transform,
                    crs=scene.crs,
                    metadata={"method": "percentile", "percentile": percentile, "band": band_name}
                )
                
                albedo_rasters[band_name] = albedo_obj
                    
            except Exception as e:
                logger.error(f"Failed to estimate albedo for {band_name}: {e}")
                # Create a fallback/dummy raster of same shape as band??
                # If calculation fails, we might not even have shape if band_data was the issue.
                # Skip.
                continue
        
        return albedo_rasters

            
        h, w = ref_raster.data.shape # Assuming 2D [H, W] for single band read
        
        # List of band names to process
        band_names = list(scene.bands.keys())
        n_bands = len(band_names)
        
        # Initialize output array (Bands, Height, Width)
        out_array = np.zeros((n_bands, h, w), dtype=np.float32)
        
        metadata = {}
        
        for idx, b_name in enumerate(band_names):
            # Read band
            b_path = scene.bands[b_name]
            # We treat it as 1D for percentile calculation to save memory if possible
            # But standard read loads it all.
            with GeoRasterData.from_file(b_path) as b_obj:
                # b_obj is not a context manager, it returns instance. 
                pass
            
            b_obj = GeoRasterData.from_file(b_path)
            if b_obj.data is None:
                continue
                
            # Calculate Percentile
            # Note: b_obj.data might contain NaNs or 0 for nodata
            data_valid = b_obj.data[np.isfinite(b_obj.data)]
            if data_valid.size == 0:
                p_val = 0.0
            else:
                p_val = np.percentile(data_valid, percentile)
            
            # Fill the plane
            out_array[idx, :, :] = p_val
            metadata[f"albedo_{b_name}"] = float(p_val)
            
        # Create AlbedoData
        output = AlbedoData()
        output.data = out_array
        output.transform = ref_raster.transform
        output.crs = ref_raster.crs
        output.metadata.extra = metadata # Or proper field
        
        logger.info("Albedo estimation complete.")
        return output
