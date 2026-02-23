"""Data-driven albedo estimation processor.

Predicts per-band surface albedo using a trained AlbedoNet MLP.  For each
output pixel the model receives engineered physical features (sun/view
geometry, wind, bathymetry, location, time) and returns TOA reflectance
estimates for all 13 Sentinel-2 bands.

Produces an ``AlbedoData`` object identical to the GP estimator so
it is a drop-in replacement in the processing chain.
"""
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pyproj
import rasterio as rio
import torch

from clouds_decoded.data import Sentinel2Scene, AlbedoData, AlbedoMetadata
from clouds_decoded.constants import BANDS

from .config import AlbedoModelConfig, NUM_INPUT_FEATURES, NUM_OUTPUT_BANDS
from .dataset import engineer_features
from .model import AlbedoNet, NormalizationWrapper

logger = logging.getLogger(__name__)


class DataDrivenAlbedoEstimator:
    """Estimates surface albedo using a trained MLP.

    Usage::

        config = AlbedoModelConfig(model_path="path/to/checkpoint.pth")
        estimator = DataDrivenAlbedoEstimator(config)
        albedo = estimator.process(scene)
    """

    def __init__(
        self,
        config: AlbedoModelConfig,
        device: Optional[str] = None,
    ):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        state = torch.load(model_path, map_location=self.device, weights_only=True)

        core = AlbedoNet(
            input_size=NUM_INPUT_FEATURES,
            output_size=NUM_OUTPUT_BANDS,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            dropout=0.0,
        )
        # Dummy stats overwritten by load_state_dict
        dummy_in = {"min": [0.0] * NUM_INPUT_FEATURES, "max": [1.0] * NUM_INPUT_FEATURES}
        dummy_out = {"min": [0.0] * NUM_OUTPUT_BANDS, "max": [1.0] * NUM_OUTPUT_BANDS}
        self.model = NormalizationWrapper(core, dummy_in, dummy_out)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Loaded data-driven albedo model from {model_path} "
            f"(device={self.device})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        scene: Sentinel2Scene,
        output_resolution: int = 300,
        batch_size: int = 32768,
        bathymetry: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> AlbedoData:
        """Estimate surface albedo for a full scene.

        Args:
            scene: Sentinel-2 scene (only metadata is needed, not bands).
            output_resolution: Resolution of the output albedo grid in metres.
            batch_size: Pixels per inference batch.
            bathymetry: Optional vectorised callable(lats, lons) -> depths.
                If None, bathymetry feature is set to 0.

        Returns:
            AlbedoData with shape ``(n_bands, H, W)`` at ``output_resolution``.
        """
        out_h, out_w, output_transform = self._compute_output_grid(
            scene, output_resolution,
        )

        # Build feature grid for every output pixel
        import pandas as pd
        feature_df = self._build_feature_grid(
            scene, out_h, out_w, output_transform, bathymetry,
        )
        features = engineer_features(feature_df)  # (N, 14)

        # Batch inference
        band_names = list(self.config.bands)
        n_pixels = len(features)
        predictions = np.empty((n_pixels, len(band_names)), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_pixels, batch_size):
                batch = torch.from_numpy(features[i:i + batch_size]).to(self.device)
                pred = self.model(batch)
                predictions[i:i + batch_size] = pred.cpu().numpy()

        # Reshape (N, n_bands) → (n_bands, H, W)
        albedo_array = predictions.reshape(out_h, out_w, len(band_names))
        albedo_array = np.clip(albedo_array, 0.0, None)
        albedo_array = albedo_array.transpose(2, 0, 1).astype(np.float32)

        metadata = AlbedoMetadata(
            band_names=band_names,
            method="datadriven",
            clear_fraction=1.0,
            fallback_used=False,
        )

        logger.info(
            f"Data-driven albedo estimation complete for {len(band_names)} bands "
            f"(grid={out_h}x{out_w}, resolution={output_resolution}m)"
        )

        return AlbedoData(
            data=albedo_array,
            transform=output_transform,
            crs=scene.crs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_output_grid(
        self,
        scene: Sentinel2Scene,
        output_resolution: int,
    ) -> tuple:
        """Compute output grid dimensions and transform from scene extent."""
        scene_res = abs(scene.transform.a)

        if "B02" in scene.bands:
            ref_band = scene.bands["B02"]
        else:
            ref_band = max(scene.bands.values(), key=lambda b: b.size)
        ref_h, ref_w = ref_band.shape[1:] if ref_band.ndim == 3 else ref_band.shape

        extent_h = ref_h * scene_res
        extent_w = ref_w * scene_res
        out_h = max(1, int(extent_h / output_resolution))
        out_w = max(1, int(extent_w / output_resolution))

        output_transform = rio.transform.Affine(
            output_resolution, 0, scene.transform.c,
            0, -output_resolution, scene.transform.f,
        )
        return out_h, out_w, output_transform

    def _build_feature_grid(
        self,
        scene: Sentinel2Scene,
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
        bathymetry: Optional[Callable],
    ) -> "pd.DataFrame":
        """Build a DataFrame with one row per output pixel, containing the raw
        physical columns that ``engineer_features`` expects."""
        import pandas as pd

        n_pixels = out_h * out_w
        rows, cols = np.mgrid[0:out_h, 0:out_w]
        rows = rows.ravel().astype(np.float64)
        cols = cols.ravel().astype(np.float64)

        # Pixel centres → map coordinates
        t = output_transform
        px = t.c + (cols + 0.5) * t.a
        py = t.f + (rows + 0.5) * t.e

        # Map → WGS84
        transformer = pyproj.Transformer.from_crs(
            scene.crs, "EPSG:4326", always_xy=True,
        )
        lons, lats = transformer.transform(px, py)

        # Sun/view angles at output pixel centres (in B02-pixel coordinates)
        scene_res = abs(scene.transform.a)
        out_res = abs(output_transform.a)
        scale = out_res / scene_res
        ref_rows = (rows * scale + scale / 2).astype(np.float64)
        ref_cols = (cols * scale + scale / 2).astype(np.float64)

        angles = scene.get_angles_at_pixels(
            ref_rows.astype(np.float64),
            ref_cols.astype(np.float64),
            resolution=scene_res,
        )

        # Wind
        try:
            wind_speed, wind_direction = scene.get_wind_data()
        except Exception as e:
            logger.warning(f"Could not read wind data: {e}. Using 0.")
            wind_speed, wind_direction = 0.0, 0.0

        # Day of year
        if scene.sensing_time is not None:
            day_of_year = scene.sensing_time.timetuple().tm_yday
        else:
            day_of_year = 0

        # Bathymetry
        if bathymetry is not None:
            bathy_values = bathymetry(lats, lons).astype(np.float32)
        else:
            bathy_values = np.full(n_pixels, -1.0, dtype=np.float32)

        # Detector index from B02 footprint
        footprint = scene.footprints.get("B02")
        if footprint is not None:
            fp = footprint[0] if footprint.ndim == 3 else footprint
            fp_rows = np.clip(ref_rows.astype(int), 0, fp.shape[0] - 1)
            fp_cols = np.clip(ref_cols.astype(int), 0, fp.shape[1] - 1)
            detector = fp[fp_rows, fp_cols].astype(np.int16)
        else:
            logger.warning("B02 footprint not available — detector set to 0")
            detector = np.zeros(n_pixels, dtype=np.int16)

        df = pd.DataFrame({
            "latitude": lats.astype(np.float64),
            "longitude": lons.astype(np.float64),
            "sun_zenith": angles["sun_zenith"],
            "sun_azimuth": angles["sun_azimuth"],
            "view_zenith": angles["view_zenith"],
            "view_azimuth": angles["view_azimuth"],
            "wind_speed": np.full(n_pixels, wind_speed, dtype=np.float32),
            "wind_direction": np.full(n_pixels, wind_direction, dtype=np.float32),
            "bathymetry": bathy_values,
            "day_of_year": np.full(n_pixels, day_of_year, dtype=np.int16),
            "detector": detector,
        })
        return df
