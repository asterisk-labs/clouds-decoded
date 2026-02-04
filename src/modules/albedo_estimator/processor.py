"""Albedo estimation processor."""
import numpy as np
import logging
from typing import List

from clouds_decoded.data import Sentinel2Scene, AlbedoData, Metadata
from .config import AlbedoEstimatorConfig

logger = logging.getLogger(__name__)


class AlbedoEstimator:
    """Estimates surface albedo from Sentinel-2 scenes.

    Uses a simple percentile-based method to estimate constant albedo per band.
    This is a placeholder approach suitable for quick estimates; more sophisticated
    methods (BRDF models, atmospheric correction) can be added in the future.
    """

    def __init__(self, config: AlbedoEstimatorConfig):
        """Initialize with configuration.

        Args:
            config: AlbedoEstimatorConfig instance with estimation parameters
        """
        self.config = config
        logger.info(f"Initialized AlbedoEstimator with {config.method} method "
                   f"(percentile={config.percentile})")

    def process(self, scene: Sentinel2Scene) -> AlbedoData:
        """Estimate surface albedo for all bands in scene.

        Args:
            scene: Sentinel-2 scene with loaded bands

        Returns:
            AlbedoData: Multi-band raster with shape (n_bands, height, width).
                       Each band contains a constant albedo value based on the
                       specified percentile of that band's reflectance.
        """
        logger.info(f"Estimating albedo using {self.config.percentile}th percentile method")

        # Get band names in consistent order
        band_names = sorted(scene.bands.keys())
        n_bands = len(band_names)

        if n_bands == 0:
            raise ValueError("Scene has no loaded bands. Call scene.read() first.")

        # Get reference shape from first band (assume all bands will be resampled to match)
        ref_band_data = scene.bands[band_names[0]]
        if ref_band_data.ndim == 3:
            # Handle (1, H, W) shape
            h, w = ref_band_data.shape[1:]
        else:
            # Handle (H, W) shape
            h, w = ref_band_data.shape

        # Initialize output array (n_bands, height, width)
        albedo_array = np.zeros((n_bands, h, w), dtype=np.float32)
        albedo_values = {}  # Track computed values for metadata

        # Estimate albedo for each band
        for idx, band_name in enumerate(band_names):
            try:
                band_data = scene.bands[band_name]

                # Handle different shapes
                if band_data.ndim == 3 and band_data.shape[0] == 1:
                    band_data = band_data[0]  # Extract 2D array from (1, H, W)

                # Resize to reference shape if needed
                if band_data.shape != (h, w):
                    from skimage.transform import resize
                    band_data = resize(band_data, (h, w), order=1, preserve_range=True)

                # Calculate percentile on valid data
                valid_mask = np.isfinite(band_data)
                if np.any(valid_mask):
                    estimated_val = float(np.percentile(band_data[valid_mask],
                                                        self.config.percentile))
                else:
                    logger.warning(f"Band {band_name} is all NaN/invalid. "
                                 f"Using default albedo {self.config.default_albedo}")
                    estimated_val = self.config.default_albedo

                # Fill band plane with constant value
                albedo_array[idx, :, :] = estimated_val
                albedo_values[band_name] = estimated_val

            except Exception as e:
                logger.error(f"Failed to estimate albedo for {band_name}: {e}. "
                           f"Using default {self.config.default_albedo}")
                albedo_array[idx, :, :] = self.config.default_albedo
                albedo_values[band_name] = self.config.default_albedo

        # Package output with metadata
        metadata = Metadata(
            method=self.config.method,
            percentile=self.config.percentile,
            band_names=band_names,  # Preserve band order
            albedo_values=albedo_values,  # Actual computed values
        )

        output = AlbedoData(
            data=albedo_array,
            transform=scene.transform,
            crs=scene.crs,
            metadata=metadata
        )

        logger.info(f"Albedo estimation complete for {n_bands} bands")
        return output
