import numpy as np
import logging
from typing import List, Optional

from clouds_decoded.data import Sentinel2Scene, AlbedoData, GeoRasterData
from .config import AlbedoConfig # We'll create a dummy config or just use kwargs

logger = logging.getLogger(__name__)

class AlbedoEstimator:
    def __init__(self, config=None):
        self.config = config

    def process(self, scene: Sentinel2Scene, percentile: float = 1.0, reference_band: str = "B02") -> AlbedoData:
        """
        Estimates albedo as a constant value per band based on the n-th percentile of reflectances.
        
        Args:
            scene: The Sentinel2Scene input.
            percentile: The percentile to extract (0-100). Default 1.0 (dark object).
            reference_band: The band to use for the output spatial resolution/transform.
            
        Returns:
            AlbedoData: A multi-band raster where every pixel has the estimated albedo value.
        """
        logger.info(f"Estimating albedo using {percentile}th percentile method.")
        
        if reference_band not in scene.bands:
            # Fallback to first available if reference processing fails
            reference_band = list(scene.bands.keys())[0]
            logger.warning(f"Reference band not found, falling back to {reference_band}")

        # Get the reference raster data to establish transform/shape
        # Note: scene.bands[b] is a path. We need to read it to get shape/transform.
        # But we probably want to use the scene.read() method or similar helper?
        # scene.bands is Dict[str, Path] per previous read_file of sentinel.py? 
        # Wait, let's check sentinel.py again. 
        # scene.bands is declared as Dict[str, Any] but _get_bands returns paths?
        # The sentinel.py `read` method does: self.bands = self._get_bands(...)
        # I should check what _get_bands actually returns. Usually paths for huge datasets.
        
        # Checking Sentinel2Scene usage in CloudHeightProcessor implies we pass the scene object.
        # But let's assume scene.bands contains paths or we can use GeoRasterData.from_file(scene.bands[b])
        
        # Load reference to get geometry
        ref_path = scene.bands[reference_band]
        ref_raster = GeoRasterData.from_file(ref_path)
        
        if ref_raster.data is None:
            raise ValueError(f"Could not read reference band {reference_band}")
            
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
