# refocus/processor.py
"""Parallax correction (refocusing) for Sentinel-2 multi-band imagery."""
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from clouds_decoded.constants import BAND_RESOLUTIONS, BAND_TIME_DELAYS
from clouds_decoded.modules.cloud_height.physics import heightsToOffsets
from clouds_decoded.base_processor import BaseProcessor

from .config import RefocusConfig

logger = logging.getLogger(__name__)


class RefocusProcessor(BaseProcessor):
    """
    Corrects parallax misalignment in Sentinel-2 bands using cloud height data.

    Sentinel-2's push-broom sensor acquires bands at different times (up to ~2.6s apart).
    At cloud altitude, this temporal offset creates spatial misalignment along the
    satellite track. This processor reverses that shift to produce a co-registered scene.

    The reference band (B02, time_delay=0) is unchanged. All other bands are warped
    to align with it.
    """

    def __init__(self, config: RefocusConfig):
        self.config = config

    def process(
        self,
        scene: Sentinel2Scene,
        height_data: CloudHeightGridData,
    ) -> Sentinel2Scene:
        """Bypass base-class resampling — returns Sentinel2Scene, not GeoRasterData."""
        return self._process(scene, height_data)

    def _process(
        self,
        scene: Sentinel2Scene,
        height_data: CloudHeightGridData,
    ) -> Sentinel2Scene:
        """
        Refocus a Sentinel-2 scene using cloud height data.

        Args:
            scene: Input Sentinel-2 scene with bands and metadata.
            height_data: Cloud height grid (meters). Can be at coarser resolution
                         (e.g. 300m from cloud_height module); will be interpolated
                         to each band's resolution.

        Returns:
            A new Sentinel2Scene with parallax-corrected bands.
        """
        config = self.config

        # Determine which bands to process
        bands_to_process = config.bands or list(scene.bands.keys())
        logger.info(f"Refocusing {len(bands_to_process)} bands: {bands_to_process}")

        # Extract height map (2D)
        height_map = self._extract_height_map(height_data)
        height_resolution = self._get_height_resolution(height_data, scene)

        # Get along-track direction from scene geometry
        image_azimuth = scene.image_azimuth or 0.0

        # Build output scene
        output = Sentinel2Scene(
            scene_directory=scene.scene_directory,
            footprints=scene.footprints,
            sun_zenith=scene.sun_zenith,
            sun_azimuth=scene.sun_azimuth,
            view_zenith=scene.view_zenith,
            view_azimuth=scene.view_azimuth,
            image_azimuth=scene.image_azimuth,
            latitude=scene.latitude,
            longitude=scene.longitude,
            orbit_type=scene.orbit_type,
            crs=scene.crs,
            transform=scene.transform,
            quantification_value=scene.quantification_value,
            radio_add_offset=scene.radio_add_offset,
        )
        output.bands = {}

        valid_bands = []
        for band_name in bands_to_process:
            if band_name not in scene.bands:
                logger.warning(f"Band {band_name} not in scene, skipping")
            else:
                valid_bands.append(band_name)

        def _process_band(band_name: str):
            band_resolution = BAND_RESOLUTIONS[band_name]
            time_delay = BAND_TIME_DELAYS.get(band_name, 0.0)

            if time_delay == 0.0:
                result = scene.get_band(
                    band_name, reflectance=False, resolution=config.output_resolution
                )
                logger.debug(f"  {band_name}: reference band, passed through")
                return band_name, result

            band_data = scene.get_band(band_name, reflectance=False)

            band_height = self._interpolate_height_to_band(
                height_map, height_resolution, band_data.shape, band_resolution
            )
            offsets = heightsToOffsets(band_height, [band_name], band_resolution)
            direction_sign = self._get_direction_sign_map(
                scene, band_name, band_data.shape, band_resolution
            )
            offsets = offsets * direction_sign

            row_offsets = offsets * np.cos(image_azimuth)
            col_offsets = offsets * np.sin(image_azimuth)

            corrected = self._warp_band(band_data, row_offsets, col_offsets)

            if config.output_resolution and config.output_resolution != band_resolution:
                scale = band_resolution / config.output_resolution
                warp_target = (int(corrected.shape[0] * scale), int(corrected.shape[1] * scale))
                if warp_target != corrected.shape:
                    corrected = resize(
                        corrected, warp_target,
                        order=config.interpolation_order,
                        preserve_range=True,
                    ).astype(corrected.dtype)

            max_offset_m = np.nanmax(np.abs(offsets)) * band_resolution
            logger.info(
                f"  {band_name}: delay={time_delay:.3f}s, "
                f"max_offset={max_offset_m:.1f}m, shape={corrected.shape}"
            )
            return band_name, corrected

        with ThreadPoolExecutor(max_workers=len(valid_bands)) as pool:
            for band_name, result in pool.map(_process_band, valid_bands):
                output.bands[band_name] = result

        # Copy bands that weren't in the processing list (pass through)
        for band_name, band_data in scene.bands.items():
            if band_name not in output.bands:
                output.bands[band_name] = band_data

        output.is_refocused = True
        logger.info("Refocusing complete")
        return output

    def _extract_height_map(self, height_data: CloudHeightGridData) -> np.ndarray:
        """Extract 2D height array from CloudHeightGridData."""
        data = height_data.data
        if data.ndim == 3:
            data = data[0]
        return data.astype(np.float32)

    def _get_height_resolution(
        self, height_data: CloudHeightGridData, scene: Sentinel2Scene
    ) -> float:
        """Determine the pixel size of the height map in meters."""
        if height_data.transform is not None:
            return abs(height_data.transform.a)
        # Fallback: estimate from scene extent and height map shape
        scene_width_m, _ = scene.get_scene_size_meters()
        return scene_width_m / height_data.data.shape[-1]

    def _get_direction_sign_map(
        self,
        scene: Sentinel2Scene,
        band_name: str,
        band_shape: tuple,
        band_resolution: float,
    ) -> np.ndarray:
        """
        Build a per-pixel direction sign array from the footprint map.

        Even detector IDs → +1 ('up'), odd detector IDs → -1 ('down').
        Falls back to the reference band footprint if the band's own footprint
        is unavailable, and to +1.0 if no footprints exist at all.
        """
        # Try band's own footprint first, then reference band
        fp = None
        for candidate in (band_name, self.config.reference_band):
            if candidate in scene.footprints:
                fp = scene.footprints[candidate]
                break

        if fp is None:
            logger.warning(f"No footprint available for {band_name}, defaulting to +1")
            return np.ones(band_shape, dtype=np.float32)

        # Resize footprint to band shape if needed (nearest-neighbor to preserve IDs)
        if fp.shape != band_shape:
            fp = resize(
                fp.astype(np.float64), band_shape,
                order=0, preserve_range=True,
            ).astype(fp.dtype)

        # Even IDs → +1, odd IDs → -1, zero (no data) → +1
        sign_map = np.where(fp % 2 == 0, 1.0, -1.0).astype(np.float32)
        return sign_map

    def _interpolate_height_to_band(
        self,
        height_map: np.ndarray,
        height_resolution: float,
        band_shape: tuple,
        band_resolution: float,
    ) -> np.ndarray:
        """
        Interpolate the height map to match a band's pixel grid.

        Uses bilinear interpolation (or as configured) to upsample the
        coarser height grid to the band's native resolution.
        """
        h_h, h_w = height_map.shape
        b_h, b_w = band_shape

        # If shapes already match, return as-is
        if (h_h, h_w) == (b_h, b_w):
            return height_map

        # Build coordinate grids
        # Height map pixel centers in meters
        h_rows = np.arange(h_h) * height_resolution + height_resolution / 2
        h_cols = np.arange(h_w) * height_resolution + height_resolution / 2

        # Band pixel centers in meters
        b_rows = np.arange(b_h) * band_resolution + band_resolution / 2
        b_cols = np.arange(b_w) * band_resolution + band_resolution / 2

        # Map band coordinates to height map pixel coordinates
        row_coords = np.interp(b_rows, h_rows, np.arange(h_h))
        col_coords = np.interp(b_cols, h_cols, np.arange(h_w))

        # Create 2D coordinate arrays
        rr, cc = np.meshgrid(row_coords, col_coords, indexing='ij')
        coords = np.array([rr, cc])

        result = map_coordinates(
            height_map, coords,
            order=self.config.height_interpolation_order,
            mode='nearest',
        )
        return result.astype(np.float32)

    def _warp_band(
        self,
        band_data: np.ndarray,
        row_offsets: np.ndarray,
        col_offsets: np.ndarray,
    ) -> np.ndarray:
        """
        Warp a band by applying per-pixel offsets.

        For each output pixel (r, c), samples the input at (r + row_offset, c + col_offset).
        Pixels with NaN height (clear sky) get zero offset -> pass through unchanged.
        """
        h, w = band_data.shape

        # Replace NaN offsets with 0 (clear sky / invalid height -> no correction)
        row_offsets = np.where(np.isfinite(row_offsets), row_offsets, 0.0)
        col_offsets = np.where(np.isfinite(col_offsets), col_offsets, 0.0)

        # Build sampling coordinates: output pixel + offset = where to sample input
        rows, cols = np.meshgrid(np.arange(h, dtype=np.float64),
                                 np.arange(w, dtype=np.float64),
                                 indexing='ij')
        sample_rows = rows + row_offsets
        sample_cols = cols + col_offsets

        coords = np.array([sample_rows, sample_cols])

        result = map_coordinates(
            band_data.astype(np.float64),
            coords,
            order=self.config.interpolation_order,
            mode='nearest',
        )
        return result.astype(band_data.dtype)

