"""Albedo pixel sampler for building ML training datasets.

Samples N clear-sky locations from a Sentinel-2 L1C scene, extracting
physical conditions (geometry, wind, location, time) and TOA reflectance
per band. Each sample can be the mean over a configurable window
(default 180m x 180m) to smooth out wave noise.
Output is a PointCloudData (Parquet-backed DataFrame).
"""
import logging
from typing import Optional, Callable, List

import numpy as np
import pandas as pd
import pyproj
from pydantic import BaseModel, Field
from scipy.ndimage import maximum_filter

from clouds_decoded.modules.albedo_estimator.sampling import windowed_clear_mean

from clouds_decoded.data import (
    Sentinel2Scene, CloudMaskData, PointCloudData, Metadata,
)
from clouds_decoded.constants import BANDS, BAND_RESOLUTIONS

logger = logging.getLogger(__name__)


class AlbedoSamplerConfig(BaseModel):
    """Configuration for sampling clear-sky pixel data for ML albedo training."""

    n_samples: int = Field(
        default=10_000,
        ge=100,
        le=1_000_000,
        description="Number of clear-sky pixels to sample per scene",
    )
    bands: List[str] = Field(
        default_factory=lambda: list(BANDS),
        description="Bands to extract TOA reflectance for",
    )
    reference_resolution: float = Field(
        default=10.0,
        description="Resolution in meters of the reference grid (B02)",
    )
    dilation_pixels: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Cloud mask dilation buffer in pixels",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible sampling",
    )
    include_shadow: bool = Field(
        default=False,
        description="If True, also exclude cloud shadow pixels (class 3)",
    )
    min_clear_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum clear-sky fraction of valid pixels to accept a scene",
    )
    window_size_m: float = Field(
        default=180.0,
        ge=0,
        description=(
            "Side length of the averaging window in metres. Each sample "
            "is the mean reflectance over a window_size_m × window_size_m "
            "patch (exact multiple of 10m, 20m, 60m). Set to 0 for single-pixel."
        ),
    )


class AlbedoSamplerMetadata(Metadata):
    """Metadata for albedo training samples."""

    scene_id: Optional[str] = None
    sensing_time: Optional[str] = None
    n_samples: int = 0
    clear_fraction: float = 0.0
    bands_included: List[str] = Field(default_factory=list)


class AlbedoPixelSampler:
    """Samples clear-sky pixels with physical conditions from Sentinel-2 scenes.

    Usage::

        sampler = AlbedoPixelSampler(config)
        result = sampler.sample(scene, cloud_mask)
        result.write("samples.parquet")
    """

    def __init__(self, config: Optional[AlbedoSamplerConfig] = None):
        if config is None:
            config = AlbedoSamplerConfig()
        self.config = config

    def sample(
        self,
        scene: Sentinel2Scene,
        cloud_mask: CloudMaskData,
        bathymetry: Optional[Callable[[float, float], float]] = None,
    ) -> PointCloudData:
        """Sample clear-sky pixels from a scene.

        Args:
            scene: Loaded Sentinel2Scene with bands.
            cloud_mask: Cloud mask (will be converted to binary).
            bathymetry: Optional callable(lat, lon) -> depth in metres.
                If None, the bathymetry column is filled with NaN.

        Returns:
            PointCloudData with one row per sampled pixel.
        """
        # 1. Build clear-sky mask at B02 resolution
        sample_rows, sample_cols, clear_fraction = self._select_clear_pixels(
            scene, cloud_mask,
        )
        n_samples = len(sample_rows)
        logger.info(f"Sampling {n_samples} pixels (clear fraction: {clear_fraction:.1%})")

        # 2. Lat/lon from pixel coordinates
        lats, lons = self._pixel_to_latlon(scene, sample_rows, sample_cols)

        # 3. Interpolated sun and view angles
        angles = scene.get_angles_at_pixels(
            sample_rows, sample_cols,
            resolution=self.config.reference_resolution,
        )

        # 4. Wind speed and direction
        try:
            wind_speed, wind_direction = scene.get_wind_data()
        except Exception as e:
            logger.warning(f"Could not read wind data: {e}. Using NaN.")
            wind_speed, wind_direction = np.nan, np.nan

        # 5. Day of year
        sensing_time_str = None
        if scene.sensing_time is not None:
            day_of_year = scene.sensing_time.timetuple().tm_yday
            sensing_time_str = scene.sensing_time.isoformat()
        else:
            logger.warning("No sensing time available. day_of_year set to 0.")
            day_of_year = 0

        # 6. Bathymetry
        if bathymetry is not None:
            bathy_values = np.array(
                [bathymetry(lat, lon) for lat, lon in zip(lats, lons)],
                dtype=np.float32,
            )
        else:
            bathy_values = np.full(n_samples, np.nan, dtype=np.float32)

        # 7. Detector index from B02 footprint
        footprint = scene.footprints.get("B02")
        if footprint is not None:
            fp = footprint[0] if footprint.ndim == 3 else footprint
            detector = fp[sample_rows, sample_cols].astype(np.int16)
        else:
            logger.warning("B02 footprint not available — detector set to 0")
            detector = np.zeros(n_samples, dtype=np.int16)

        # 8. TOA reflectance per band
        band_data = self._extract_reflectances(scene, sample_rows, sample_cols)

        # 9. Assemble DataFrame
        df = pd.DataFrame({
            'latitude': lats.astype(np.float64),
            'longitude': lons.astype(np.float64),
            'sun_zenith': angles['sun_zenith'],
            'sun_azimuth': angles['sun_azimuth'],
            'view_zenith': angles['view_zenith'],
            'view_azimuth': angles['view_azimuth'],
            'wind_speed': np.full(n_samples, wind_speed, dtype=np.float32),
            'wind_direction': np.full(n_samples, wind_direction, dtype=np.float32),
            'bathymetry': bathy_values,
            'day_of_year': np.full(n_samples, day_of_year, dtype=np.int16),
            'detector': detector,
        })
        for band_name, values in band_data.items():
            df[band_name] = values

        metadata = AlbedoSamplerMetadata(
            scene_id=scene.product_uri,
            sensing_time=sensing_time_str,
            n_samples=n_samples,
            clear_fraction=float(clear_fraction),
            bands_included=list(band_data.keys()),
        )

        return PointCloudData(data=df, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_clear_pixels(
        self,
        scene: Sentinel2Scene,
        cloud_mask: CloudMaskData,
    ) -> tuple:
        """Identify clear pixels and randomly sample, rejecting any near cloud.

        Uses ``maximum_filter`` to dilate the cloud mask in one vectorised pass,
        then samples from the remaining clear pixels.

        Returns:
            (sample_rows, sample_cols, clear_fraction)
        """
        positive_classes = [1, 2, 3] if self.config.include_shadow else [1, 2]
        binary_mask = cloud_mask.to_binary(
            positive_classes=positive_classes,
            dilation_pixels=0,
        )
        cloud_mask_2d = binary_mask.data
        if cloud_mask_2d.ndim == 3:
            cloud_mask_2d = cloud_mask_2d[0]

        # Resize to B02 reference grid if needed
        ref_band = scene.bands.get('B02')
        if ref_band is None:
            raise ValueError("B02 must be loaded for the reference grid")
        ref_h, ref_w = ref_band.shape if ref_band.ndim == 2 else ref_band.shape[1:]

        if cloud_mask_2d.shape != (ref_h, ref_w):
            from skimage.transform import resize
            cloud_mask_2d = resize(
                cloud_mask_2d.astype(np.float32), (ref_h, ref_w),
                order=0, preserve_range=True,
            ) > 0.5

        cloud_bool = cloud_mask_2d.astype(bool)

        # Exclude nodata pixels (DN=0 at scene edges / detector gaps)
        ref_raw = ref_band[0] if ref_band.ndim == 3 else ref_band
        valid_mask = (ref_raw != 0)

        clear_mask = ~cloud_bool & valid_mask

        n_valid = valid_mask.sum()
        clear_fraction = clear_mask.sum() / n_valid if n_valid > 0 else 0.0
        if clear_fraction < self.config.min_clear_fraction:
            raise ValueError(
                f"Clear fraction {clear_fraction:.1%} is below minimum "
                f"{self.config.min_clear_fraction:.0%} — skipping scene"
            )

        # Dilate cloud mask to reject pixels near cloud edges
        buf = self.config.dilation_pixels
        if buf > 0:
            dilated_cloud = maximum_filter(
                cloud_bool.view(np.uint8), size=2 * buf + 1,
            ).astype(bool)
            clear_mask &= ~dilated_cloud

        # Exclude edge pixels so averaging windows stay within bounds
        edge_margin = int(self.config.window_size_m / (2 * self.config.reference_resolution))
        if edge_margin > 0:
            clear_mask[:edge_margin, :] = False
            clear_mask[-edge_margin:, :] = False
            clear_mask[:, :edge_margin] = False
            clear_mask[:, -edge_margin:] = False

        clear_rows, clear_cols = np.where(clear_mask)
        n_available = len(clear_rows)
        if n_available == 0:
            raise ValueError(
                "No clear-sky pixels found far enough from cloud edges. "
                "Try reducing dilation_pixels."
            )

        rng = np.random.default_rng(self.config.seed)
        n_samples = min(self.config.n_samples, n_available)
        idx = rng.choice(n_available, n_samples, replace=False)
        return clear_rows[idx], clear_cols[idx], clear_fraction

    def _pixel_to_latlon(
        self,
        scene: Sentinel2Scene,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> tuple:
        """Convert pixel row/col to WGS84 lat/lon via the scene affine transform."""
        t = scene.transform
        px = t.c + cols * t.a + rows * t.b
        py = t.f + cols * t.d + rows * t.e

        transformer = pyproj.Transformer.from_crs(
            scene.crs, 'EPSG:4326', always_xy=True,
        )
        lons, lats = transformer.transform(px, py)
        return lats, lons

    def _extract_reflectances(
        self,
        scene: Sentinel2Scene,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> dict:
        """Extract TOA reflectance at sample locations for each configured band.

        When ``window_size_m > 0``, applies a windowed box-mean (only over
        valid pixels) to the entire band, then indexes the sample locations.
        When ``window_size_m == 0``, falls back to single-pixel extraction.
        """
        ref_res = BAND_RESOLUTIONS.get('B02', 10)
        window_m = self.config.window_size_m
        band_data = {}

        for band_name in self.config.bands:
            if band_name not in scene.bands:
                logger.warning(f"Band {band_name} not loaded, skipping")
                continue

            raw = scene.bands[band_name]
            if raw.ndim == 3:
                raw = raw[0]

            band_res = BAND_RESOLUTIONS.get(band_name, ref_res)
            scale = ref_res / band_res
            band_rows = np.clip(
                (rows * scale).astype(int), 0, raw.shape[0] - 1,
            )
            band_cols = np.clip(
                (cols * scale).astype(int), 0, raw.shape[1] - 1,
            )

            window_px = int(window_m / band_res) if window_m > 0 else 0
            valid_mask = raw != 0  # nodata pixels have DN=0
            smoothed = windowed_clear_mean(raw, valid_mask, window_px)
            dn = smoothed[band_rows, band_cols]

            offset = scene.radio_add_offset.get(band_name, 0.0)
            band_data[band_name] = (dn + offset) / scene.quantification_value

        return band_data
