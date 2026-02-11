"""Albedo estimation processor.

Estimates per-band surface albedo by fitting a 2D polynomial to clear-sky pixels
identified by a cloud mask. Falls back to a simple percentile method when
insufficient clear pixels are available or no mask is provided.
"""
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple

import rasterio as rio
from skimage.transform import resize

from clouds_decoded.data import (
    Sentinel2Scene, AlbedoData, AlbedoMetadata, CloudMaskData,
)
from .config import AlbedoEstimatorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2D polynomial helpers
# ---------------------------------------------------------------------------

def _build_vandermonde_2d(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Build 2D Vandermonde matrix for polynomial of given order.

    For order=2 the columns are [1, x, y, x², xy, y²].
    Total columns = (order+1)(order+2)/2.
    """
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


def _eval_polynomial_2d(
    coeffs: np.ndarray, x: np.ndarray, y: np.ndarray, order: int
) -> np.ndarray:
    """Evaluate a 2D polynomial on a coordinate grid."""
    shape = x.shape
    V = _build_vandermonde_2d(x.ravel(), y.ravel(), order)
    return (V @ coeffs).reshape(shape)


def _n_poly_coeffs(order: int) -> int:
    """Number of coefficients for a 2D polynomial of given order."""
    return (order + 1) * (order + 2) // 2


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class AlbedoEstimator:
    """Estimates surface albedo from Sentinel-2 scenes.

    When a cloud mask is provided, fits a 2D polynomial to clear-sky pixels
    per band, producing a spatially varying albedo field at a configurable
    coarse resolution. When no mask is available (or too few clear pixels),
    falls back to a simple percentile-based constant per band.
    """

    def __init__(self, config: AlbedoEstimatorConfig):
        self.config = config
        logger.info(f"Initialized AlbedoEstimator (method={config.method}, "
                   f"order={config.polynomial_order})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        scene: Sentinel2Scene,
        cloud_mask: Optional[CloudMaskData] = None,
    ) -> AlbedoData:
        """Estimate surface albedo for all bands in the scene.

        Args:
            scene: Sentinel-2 scene with loaded bands.
            cloud_mask: Optional cloud mask. When provided and method is
                'polynomial', clear-sky pixels are used for fitting.

        Returns:
            AlbedoData with shape (n_bands, H_out, W_out) at
            ``config.output_resolution``.
        """
        band_names = sorted(scene.bands.keys())
        if not band_names:
            raise ValueError("Scene has no loaded bands. Call scene.read() first.")

        # Determine output grid from scene extent
        out_h, out_w, output_transform = self._compute_output_grid(scene)

        # Route to the appropriate method
        use_polynomial = (
            self.config.method == "polynomial"
            and cloud_mask is not None
        )

        if use_polynomial:
            return self._fit_polynomial(
                scene, cloud_mask, band_names, out_h, out_w, output_transform,
            )
        else:
            if self.config.method == "polynomial" and cloud_mask is None:
                logger.warning("Polynomial method requested but no cloud mask provided. "
                             "Falling back to percentile method.")
            return self._fit_percentile(
                scene, band_names, out_h, out_w, output_transform,
            )

    # ------------------------------------------------------------------
    # Polynomial fitting path
    # ------------------------------------------------------------------

    def _fit_polynomial(
        self,
        scene: Sentinel2Scene,
        cloud_mask: CloudMaskData,
        band_names: List[str],
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
    ) -> AlbedoData:
        """Fit 2D polynomial to clear-sky pixels per band."""
        order = self.config.polynomial_order
        n_bands = len(band_names)

        # 1. Build clear-sky boolean mask at reference resolution
        ref_band = scene.bands[band_names[0]]
        ref_h, ref_w = (ref_band.shape[1:] if ref_band.ndim == 3 else ref_band.shape)

        clear_mask = self._extract_clear_mask(cloud_mask, ref_h, ref_w)
        clear_fraction = clear_mask.sum() / clear_mask.size
        logger.info(f"Clear-sky fraction: {clear_fraction:.1%}")

        # 2. Check if we have enough clear pixels
        if clear_fraction < self.config.min_clear_fraction:
            logger.warning(f"Clear fraction {clear_fraction:.1%} < "
                         f"threshold {self.config.min_clear_fraction:.0%}. "
                         f"Falling back to percentile method.")
            result = self._fit_percentile(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        # 3. Get clear-sky pixel coordinates (normalized to [0, 1])
        rows, cols = np.where(clear_mask)
        y_norm = rows.astype(np.float64) / max(ref_h - 1, 1)
        x_norm = cols.astype(np.float64) / max(ref_w - 1, 1)

        # Subsample if too many clear pixels
        n_clear = len(rows)
        if n_clear > self.config.max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_clear, self.config.max_samples, replace=False)
            rows, cols = rows[idx], cols[idx]
            x_norm, y_norm = x_norm[idx], y_norm[idx]
            logger.info(f"Subsampled {n_clear} -> {self.config.max_samples} clear pixels")

        # Build Vandermonde matrix once (shared across bands)
        V = _build_vandermonde_2d(x_norm, y_norm, order)

        # 4. Build output coordinate grid (normalized)
        gy = np.linspace(0, 1, out_h)
        gx = np.linspace(0, 1, out_w)
        grid_x, grid_y = np.meshgrid(gx, gy)

        # 5. Fit per band
        albedo_array = np.zeros((n_bands, out_h, out_w), dtype=np.float32)
        poly_coefficients: Dict[str, List[float]] = {}

        for idx, band_name in enumerate(band_names):
            band_data = self._get_band_2d(scene.get_band(band_name))

            # Resize band to reference resolution if needed
            if band_data.shape != (ref_h, ref_w):
                band_data = resize(
                    band_data, (ref_h, ref_w), order=1, preserve_range=True,
                ).astype(np.float32)

            # Extract clear-sky reflectance values
            values = band_data[rows, cols].astype(np.float64)

            # Filter out NaN/inf
            valid = np.isfinite(values)
            if valid.sum() < _n_poly_coeffs(order):
                fallback = self.config.default_albedo.get(band_name, 0.05)
                logger.warning(f"Band {band_name}: too few valid clear pixels "
                             f"({valid.sum()}). Using default albedo {fallback}.")
                albedo_array[idx] = fallback
                poly_coefficients[band_name] = []
                continue

            V_valid = V[valid]
            values_valid = values[valid]

            # Least-squares fit
            coeffs, _, _, _ = np.linalg.lstsq(V_valid, values_valid, rcond=None)
            poly_coefficients[band_name] = coeffs.tolist()

            # Evaluate on output grid
            surface = _eval_polynomial_2d(coeffs, grid_x, grid_y, order)
            albedo_array[idx] = np.clip(surface, 0.0, None).astype(np.float32)

        metadata = AlbedoMetadata(
            band_names=band_names,
            method="polynomial",
            polynomial_order=order,
            polynomial_coefficients=poly_coefficients,
            clear_fraction=clear_fraction,
            fallback_used=False,
        )

        logger.info(f"Polynomial albedo estimation complete for {n_bands} bands "
                   f"(order={order}, grid={out_h}x{out_w})")

        return AlbedoData(
            data=albedo_array,
            transform=output_transform,
            crs=scene.crs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Percentile fallback path
    # ------------------------------------------------------------------

    def _fit_percentile(
        self,
        scene: Sentinel2Scene,
        band_names: List[str],
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
    ) -> AlbedoData:
        """Estimate constant albedo per band using a percentile of reflectance."""
        n_bands = len(band_names)
        albedo_array = np.zeros((n_bands, out_h, out_w), dtype=np.float32)
        fallback_values: Dict[str, float] = {}

        for idx, band_name in enumerate(band_names):
            band_data = self._get_band_2d(scene.get_band(band_name))
            valid = np.isfinite(band_data)

            if np.any(valid):
                val = float(np.percentile(band_data[valid], self.config.percentile))
            else:
                fallback = self.config.default_albedo.get(band_name, 0.05)
                logger.warning(f"Band {band_name} all NaN. Using default {fallback}")
                val = fallback

            albedo_array[idx] = val
            fallback_values[band_name] = val

        metadata = AlbedoMetadata(
            band_names=band_names,
            method="percentile",
            polynomial_order=0,
            clear_fraction=0.0,
            fallback_used=True,
            fallback_values=fallback_values,
        )

        logger.info(f"Percentile albedo estimation complete for {n_bands} bands")

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
        self, scene: Sentinel2Scene,
    ) -> Tuple[int, int, rio.transform.Affine]:
        """Compute output grid dimensions and transform from scene extent.

        Uses B02 (10m) as the reference band since scene.transform is always
        relative to B02. Falls back to the largest available band.
        """
        # scene.transform is always the B02 (10m) transform
        scene_res = abs(scene.transform.a)

        # Pick B02 as reference since transform matches it; fall back to largest band
        if 'B02' in scene.bands:
            ref_band = scene.bands['B02']
        else:
            ref_band = max(scene.bands.values(), key=lambda b: b.size)
        ref_h, ref_w = (ref_band.shape[1:] if ref_band.ndim == 3 else ref_band.shape)

        target_res = self.config.output_resolution

        # Scene extent in meters, then divide by target resolution
        extent_h = ref_h * scene_res
        extent_w = ref_w * scene_res
        out_h = max(1, int(extent_h / target_res))
        out_w = max(1, int(extent_w / target_res))

        output_transform = rio.transform.Affine(
            target_res, 0, scene.transform.c,
            0, -target_res, scene.transform.f,
        )

        return out_h, out_w, output_transform

    def _extract_clear_mask(
        self,
        cloud_mask: CloudMaskData,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Extract boolean clear-sky mask, resized to target dimensions.

        Expects a postprocessed binary mask (0=clear, 1=cloud).
        Falls back to categorical/probability handling for raw masks.
        """
        mask_data = cloud_mask.data
        if mask_data is None:
            raise ValueError("Cloud mask has no data")

        if mask_data.ndim == 3:
            mask_data = mask_data[0]

        if cloud_mask.metadata.categorical:
            # Binary postprocessed: 0=clear, 1=cloud
            # Raw categorical: 0=clear, 1/2/3=cloud classes
            # Both cases: clear where mask == 0
            clear = (mask_data == 0)
        else:
            clear = (mask_data[0] > self.config.confidence_threshold)

        if clear.shape != (target_h, target_w):
            clear = resize(
                clear.astype(np.float32), (target_h, target_w),
                order=0, preserve_range=True,
            ) > 0.5

        return clear.astype(bool)

    @staticmethod
    def _get_band_2d(band_data: np.ndarray) -> np.ndarray:
        """Ensure band array is 2D."""
        if band_data.ndim == 3 and band_data.shape[0] == 1:
            return band_data[0]
        return band_data
