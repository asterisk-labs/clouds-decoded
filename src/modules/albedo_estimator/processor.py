"""Albedo estimation processor.

Estimates per-band surface albedo by fitting a Gaussian Process to clear-sky
pixels identified by a cloud mask.  The GP uses an RBF kernel with a constant
mean, so predictions in data-sparse regions revert smoothly to the mean albedo
instead of diverging like polynomial extrapolation.

Falls back to a data-driven MLP or constant values when insufficient clear
pixels are available or no mask is provided.
"""
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple

import rasterio as rio
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from skimage.transform import resize

from clouds_decoded.data import (
    Sentinel2Scene, AlbedoData, AlbedoMetadata, CloudMaskData,
)
from clouds_decoded.constants import BAND_RESOLUTIONS
from .config import AlbedoEstimatorConfig
from .sampling import (
    clear_pixel_count,
    farthest_point_sample,
    sample_clear_locations,
    windowed_clear_mean,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GP helpers
# ---------------------------------------------------------------------------

_GP_LS_CANDIDATES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
_GP_NOISE_NORMALISED = 0.01  # observation noise in unit-variance space


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
    """Squared-exponential (RBF) kernel."""
    return np.exp(-cdist(X1, X2, "sqeuclidean") / (2.0 * length_scale ** 2))


def _gp_log_marginal_likelihood(
    X: np.ndarray, y_norm: np.ndarray, length_scale: float, noise_var: float,
) -> float:
    """Log marginal likelihood of a zero-mean GP (y already centred & scaled)."""
    K = _rbf_kernel(X, X, length_scale)
    K[np.diag_indices_from(K)] += noise_var
    try:
        L, low = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = cho_solve((L, low), y_norm)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * float(y_norm @ alpha) - 0.5 * log_det


def _select_length_scale(
    X: np.ndarray, y_norm: np.ndarray, noise_var: float,
) -> float:
    """Pick the RBF length scale that maximises the marginal likelihood."""
    best_lml = -np.inf
    best_ls = _GP_LS_CANDIDATES[len(_GP_LS_CANDIDATES) // 2]
    for ls in _GP_LS_CANDIDATES:
        lml = _gp_log_marginal_likelihood(X, y_norm, ls, noise_var)
        if lml > best_lml:
            best_lml, best_ls = lml, ls
    return best_ls


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class AlbedoEstimator:
    """Estimates surface albedo from Sentinel-2 scenes.

    When a cloud mask is provided, fits a Gaussian Process to clear-sky pixels
    per band, producing a spatially varying albedo field at a configurable
    coarse resolution.  The GP naturally reverts to the mean albedo in regions
    with few clear-sky observations, avoiding polynomial-extrapolation artefacts.
    """

    def __init__(self, config: AlbedoEstimatorConfig):
        self.config = config
        self._dd_estimator = None
        logger.info(f"Initialized AlbedoEstimator (method={config.method}, "
                   f"fallback={config.fallback})")

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
                'gp', clear-sky pixels are used for fitting.

        Returns:
            AlbedoData with shape (n_bands, H_out, W_out) at
            ``config.output_resolution``.
        """
        if self.config.method == "datadriven":
            result = self._fit_datadriven(scene)
        else:
            # method == "gp" or "idw"
            band_names = sorted(scene.bands.keys())
            if not band_names:
                raise ValueError("Scene has no loaded bands. Call scene.read() first.")

            out_h, out_w, output_transform = self._compute_output_grid(scene)

            if cloud_mask is not None:
                if self.config.method == "idw":
                    result = self._fit_idw(
                        scene, cloud_mask, band_names, out_h, out_w, output_transform,
                    )
                else:
                    result = self._fit_gp(
                        scene, cloud_mask, band_names, out_h, out_w, output_transform,
                    )
            else:
                logger.warning(f"{self.config.method.upper()} method requested but "
                             "no cloud mask provided. Running fallback.")
                result = self._run_fallback(
                    scene, band_names, out_h, out_w, output_transform,
                )

        self._apply_nodata_mask(scene, result)
        return result

    # ------------------------------------------------------------------
    # Gaussian-process fitting path
    # ------------------------------------------------------------------

    def _fit_gp(
        self,
        scene: Sentinel2Scene,
        cloud_mask: CloudMaskData,
        band_names: List[str],
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
    ) -> AlbedoData:
        """Fit a GP with RBF kernel to spatially-averaged clear-sky pixels.

        Training targets are windowed means (``gp_window_m``) rather than
        individual noisy pixels, giving the GP much cleaner observations.

        Optimisations
        -------------
        * Each band is smoothed at its **native resolution** (10 / 20 / 60 m)
          instead of being resized to the 10 m reference grid first.
          The ``uniform_filter`` denominator (``val_count``) is cached per
          resolution group, halving the smoothing cost per band.
        * The Cholesky factorisation of K_train and the K_star prediction
          matrix are computed **once** and reused for all bands (the GP
          geometry is identical; only the target vector changes).
        """
        n_bands = len(band_names)
        scene_res = abs(scene.transform.a)
        ref_res = BAND_RESOLUTIONS.get("B02", 10)

        # 1. Use B02 (or largest band) as reference — matches _compute_output_grid
        if "B02" in scene.bands:
            ref_band = scene.get_band("B02", reflectance=False)
        else:
            ref_band = max(scene.bands.values(), key=lambda b: b.size)
        ref_h, ref_w = (ref_band.shape[1:] if ref_band.ndim == 3 else ref_band.shape)

        clear_mask = self._extract_clear_mask(cloud_mask, ref_h, ref_w)

        # Exclude nodata pixels (DN=0 at scene edges / detector gaps)
        ref_2d = ref_band[0] if ref_band.ndim == 3 else ref_band
        valid_data = (ref_2d != 0)
        clear_mask &= valid_data

        clear_fraction = clear_mask.sum() / clear_mask.size
        logger.info(f"Clear-sky fraction: {clear_fraction:.1%}")

        # 2. Check if we have enough clear pixels
        if clear_fraction < self.config.min_clear_fraction:
            logger.warning(f"Clear fraction {clear_fraction:.1%} < "
                         f"threshold {self.config.min_clear_fraction:.0%}. "
                         f"Running fallback.")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        # 3. Sample clear-sky locations (B02 reference coordinates)
        window_px_ref = max(1, int(self.config.gp_window_m / scene_res))
        edge_margin = window_px_ref // 2

        rows, cols = sample_clear_locations(
            clear_mask, n_samples=self.config.max_samples,
            seed=42, edge_margin=edge_margin,
            cloud_dilation_px=self.config.gp_dilation_pixels,
        )
        n_train = len(rows)
        if n_train < 10:
            logger.warning(f"Only {n_train} clear samples after edge margin. "
                         "Running fallback.")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        X_train = np.column_stack([
            cols.astype(np.float64) / max(ref_w - 1, 1),
            rows.astype(np.float64) / max(ref_h - 1, 1),
        ])

        logger.info(f"GP training: {n_train} samples "
                   f"(window={window_px_ref}px / {self.config.gp_window_m:.0f}m)")

        # 4. Build output coordinate grid (normalised)
        gy = np.linspace(0, 1, out_h)
        gx = np.linspace(0, 1, out_w)
        grid_x, grid_y = np.meshgrid(gx, gy)
        X_pred = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # 5. Extract smoothed training values for ALL bands at native
        #    resolution.  The clear-pixel count (uniform_filter denominator)
        #    is cached per resolution group via clear_pixel_count() so it is
        #    computed only once per group.
        _res_cache: Dict[int, tuple] = {}
        band_values: Dict[str, np.ndarray] = {}

        prefetched = {
            name: self._get_band_2d(obj.data)
            for name, obj in zip(
                band_names,
                scene.get_bands(band_names, reflectance=True, n_workers=len(band_names)),
            )
        }

        for band_name in band_names:
            band_data = prefetched[band_name]
            band_res = BAND_RESOLUTIONS.get(band_name, ref_res)

            if band_res not in _res_cache:
                native_h, native_w = band_data.shape[:2]
                window_px = max(1, int(self.config.gp_window_m / band_res))

                if band_res == ref_res:
                    s_rows, s_cols = rows, cols
                    native_clear = clear_mask
                else:
                    scale = ref_res / band_res
                    s_rows = np.clip(
                        (rows * scale).astype(int), 0, native_h - 1,
                    )
                    s_cols = np.clip(
                        (cols * scale).astype(int), 0, native_w - 1,
                    )
                    native_clear = resize(
                        clear_mask.astype(np.float32), (native_h, native_w),
                        order=0, preserve_range=True,
                    ) > 0.5

                # Exclude nodata at native resolution (DN=0)
                native_clear = native_clear & (band_data != 0)

                val_count = clear_pixel_count(native_clear, window_px)
                _res_cache[band_res] = (
                    s_rows, s_cols, val_count, native_clear, window_px,
                )

            s_rows, s_cols, val_count, native_clear, window_px = _res_cache[band_res]

            smoothed = windowed_clear_mean(
                band_data, native_clear, window_px,
                precomputed_count=val_count,
            )
            band_values[band_name] = smoothed[s_rows, s_cols].astype(np.float64)

        # 6. Find globally valid samples (finite in ALL bands)
        all_valid = np.ones(n_train, dtype=bool)
        for values in band_values.values():
            all_valid &= np.isfinite(values)

        n_valid = int(all_valid.sum())
        if n_valid < 10:
            logger.warning(f"Only {n_valid} globally valid samples. "
                         "Running fallback.")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        X_train_valid = X_train[all_valid]

        # 7. Select GP length scale from first band
        first_vals = band_values[band_names[0]][all_valid]
        if self.config.gp_length_scale is not None:
            length_scale = self.config.gp_length_scale
            logger.info(f"Using configured GP length scale: {length_scale:.3f}")
        else:
            std = float(first_vals.std())
            if std > 1e-12:
                pilot_norm = (first_vals - first_vals.mean()) / std
                length_scale = _select_length_scale(
                    X_train_valid, pilot_norm, _GP_NOISE_NORMALISED,
                )
            else:
                length_scale = 0.3
            logger.info(f"Auto-selected GP length scale: {length_scale:.3f}")

        # 8. Pre-compute GP matrices ONCE for all bands
        K_train = _rbf_kernel(X_train_valid, X_train_valid, length_scale)
        K_train[np.diag_indices_from(K_train)] += _GP_NOISE_NORMALISED
        try:
            L, low = cho_factor(K_train, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("GP Cholesky failed — running fallback")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        K_star = _rbf_kernel(X_pred, X_train_valid, length_scale)
        logger.info(f"GP matrices pre-computed: K_train={n_valid}x{n_valid}, "
                   f"K_star={len(X_pred)}x{n_valid}")

        # 9. Predict per band (cheap: cho_solve + matmul only)
        albedo_array = np.zeros((n_bands, out_h, out_w), dtype=np.float32)

        for idx, band_name in enumerate(band_names):
            values = band_values[band_name][all_valid]
            mu = float(np.mean(values))
            std = float(np.std(values))

            if std < 1e-12:
                albedo_array[idx] = mu
                continue

            y_norm = (values - mu) / std
            alpha = cho_solve((L, low), y_norm)
            pred_norm = K_star @ alpha
            pred = (pred_norm * std + mu).astype(np.float32)
            albedo_array[idx] = np.clip(pred, 0.0, None).reshape(out_h, out_w)

        metadata = AlbedoMetadata(
            band_names=band_names,
            method="gp",
            length_scale=length_scale,
            n_training_samples=n_valid,
            clear_fraction=clear_fraction,
            fallback_used=False,
        )

        logger.info(f"GP albedo estimation complete for {n_bands} bands "
                   f"(length_scale={length_scale:.3f}, "
                   f"train={n_valid}, grid={out_h}x{out_w})")

        return AlbedoData(
            data=albedo_array,
            transform=output_transform,
            crs=scene.crs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # IDW fitting path
    # ------------------------------------------------------------------

    def _fit_idw(
        self,
        scene: Sentinel2Scene,
        cloud_mask: CloudMaskData,
        band_names: List[str],
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
    ) -> AlbedoData:
        """Inverse-distance weighted interpolation of clear-sky albedo.

        Uses farthest-point sampling for spatially uniform coverage,
        then interpolates to the output grid using 1/d weights limited
        to the k nearest neighbours per output pixel.
        """
        n_bands = len(band_names)
        scene_res = abs(scene.transform.a)
        ref_res = BAND_RESOLUTIONS.get("B02", 10)

        # 1. Reference band and clear mask (same logic as GP)
        if "B02" in scene.bands:
            ref_band = scene.bands["B02"]
        else:
            ref_band = max(scene.bands.values(), key=lambda b: b.size)
        ref_h, ref_w = (ref_band.shape[1:] if ref_band.ndim == 3 else ref_band.shape)

        clear_mask = self._extract_clear_mask(cloud_mask, ref_h, ref_w)

        ref_2d = ref_band[0] if ref_band.ndim == 3 else ref_band
        clear_mask &= (ref_2d != 0)

        clear_fraction = clear_mask.sum() / clear_mask.size
        logger.info(f"Clear-sky fraction: {clear_fraction:.1%}")

        # 2. Check minimum clear fraction
        if clear_fraction < self.config.min_clear_fraction:
            logger.warning(f"Clear fraction {clear_fraction:.1%} < "
                         f"threshold {self.config.min_clear_fraction:.0%}. "
                         f"Running fallback.")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        # 3. Farthest-point sample clear locations
        window_px_ref = max(1, int(self.config.gp_window_m / scene_res))
        edge_margin = window_px_ref // 2

        rows, cols = farthest_point_sample(
            clear_mask, n_samples=self.config.max_samples,
            seed=42, edge_margin=edge_margin,
            cloud_dilation_px=self.config.gp_dilation_pixels,
        )
        n_train = len(rows)
        if n_train < 10:
            logger.warning(f"Only {n_train} clear samples after filtering. "
                         "Running fallback.")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        logger.info(f"IDW: {n_train} farthest-point samples "
                   f"(window={window_px_ref}px / {self.config.gp_window_m:.0f}m)")

        # 4. Compute windowed-mean band values at sample locations only.
        #    Instead of running uniform_filter over the full image, we
        #    extract a small window around each sample point and average
        #    the clear pixels directly — O(n_samples * window²) per band.
        _res_cache: Dict[int, tuple] = {}
        band_values: Dict[str, np.ndarray] = {}

        prefetched_raw = {
            name: self._get_band_2d(obj.data)
            for name, obj in zip(
                band_names,
                scene.get_bands(band_names, reflectance=False, n_workers=len(band_names)),
            )
        }

        for band_name in band_names:
            # Use raw DN to avoid full-array float conversion; the
            # reflectance transform is linear so we convert the small
            # per-point means afterwards.
            band_raw = prefetched_raw[band_name]
            band_res = BAND_RESOLUTIONS.get(band_name, ref_res)

            native_h, native_w = band_raw.shape[:2]
            # Use actual dimensions to detect resolution — dummy scenes
            # have all bands at the same size regardless of band_res.
            actual_res = band_res if (native_h, native_w) != (ref_h, ref_w) else ref_res

            if actual_res not in _res_cache:
                half_w = max(1, int(self.config.gp_window_m / actual_res)) // 2

                if actual_res == ref_res:
                    s_rows, s_cols = rows, cols
                    native_clear = clear_mask
                else:
                    scale = ref_res / band_res
                    s_rows = np.clip(
                        (rows * scale).astype(int), 0, native_h - 1,
                    )
                    s_cols = np.clip(
                        (cols * scale).astype(int), 0, native_w - 1,
                    )
                    # Downsample clear mask via slicing — much faster
                    # than skimage.resize for nearest-neighbour on booleans.
                    step = int(round(band_res / ref_res))
                    native_clear = clear_mask[::step, ::step][:native_h, :native_w]

                # DN=0 is Sentinel-2 nodata
                native_clear = native_clear & (band_raw != 0)
                _res_cache[actual_res] = (
                    s_rows, s_cols, native_clear, half_w, native_h, native_w,
                )

            s_rows, s_cols, native_clear, half_w, native_h, native_w = (
                _res_cache[actual_res]
            )

            dn_means = self._windowed_mean_at_points(
                band_raw, native_clear, s_rows, s_cols, half_w,
                native_h, native_w,
            )
            # Convert DN means to reflectance
            offset = scene.radio_add_offset.get(band_name, 0.0)
            band_values[band_name] = (
                (dn_means + offset) / scene.quantification_value
            )

        # 5. Filter to globally valid samples
        all_valid = np.ones(n_train, dtype=bool)
        for values in band_values.values():
            all_valid &= np.isfinite(values)

        n_valid = int(all_valid.sum())
        if n_valid < 10:
            logger.warning(f"Only {n_valid} globally valid samples. "
                         "Running fallback.")
            result = self._run_fallback(
                scene, band_names, out_h, out_w, output_transform,
            )
            result.metadata.clear_fraction = clear_fraction
            return result

        # Sample coordinates in metres (pixel index × scene resolution)
        sample_coords = np.column_stack([
            cols[all_valid].astype(np.float64) * scene_res,
            rows[all_valid].astype(np.float64) * scene_res,
        ])

        # 6. IDW interpolation with k-nearest-neighbour sparsity
        extent_h = ref_h * scene_res
        extent_w = ref_w * scene_res
        gy = np.linspace(0, extent_h, out_h)
        gx = np.linspace(0, extent_w, out_w)
        grid_x, grid_y = np.meshgrid(gx, gy)
        X_pred = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        k = min(self.config.idw_k_neighbours, n_valid)
        n_pred = len(X_pred)
        d0 = self.config.idw_smoothing_m

        # Compute full distance matrix and find k nearest per output pixel
        dist = cdist(X_pred, sample_coords)  # (n_pred, n_valid)
        knn_idx = np.argpartition(dist, k, axis=1)[:, :k]  # (n_pred, k)
        # Gather distances for the k nearest
        knn_dist = np.take_along_axis(dist, knn_idx, axis=1)  # (n_pred, k)
        knn_weights = 1.0 / (knn_dist + d0)  # (n_pred, k)
        weight_sums = knn_weights.sum(axis=1, keepdims=True)  # (n_pred, 1)

        logger.info(f"IDW: interpolating {n_pred} output pixels from "
                   f"{n_valid} samples (k={k})")

        # 7. Predict per band
        albedo_array = np.zeros((n_bands, out_h, out_w), dtype=np.float32)

        for idx, band_name in enumerate(band_names):
            values = band_values[band_name][all_valid]
            # Gather values at k nearest neighbours
            knn_vals = values[knn_idx]  # (n_pred, k)
            pred = (knn_weights * knn_vals).sum(axis=1) / weight_sums.ravel()
            albedo_array[idx] = np.clip(pred, 0.0, None).reshape(out_h, out_w).astype(np.float32)

        metadata = AlbedoMetadata(
            band_names=band_names,
            method="idw",
            n_training_samples=n_valid,
            clear_fraction=clear_fraction,
            fallback_used=False,
        )

        logger.info(f"IDW albedo estimation complete for {n_bands} bands "
                   f"(samples={n_valid}, k={k}, grid={out_h}x{out_w})")

        return AlbedoData(
            data=albedo_array,
            transform=output_transform,
            crs=scene.crs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Data-driven path
    # ------------------------------------------------------------------

    def _fit_datadriven(self, scene: Sentinel2Scene) -> AlbedoData:
        """Estimate albedo using the trained data-driven MLP."""
        if self._dd_estimator is None:
            from .datadriven.config import AlbedoModelConfig
            from .datadriven.processor import DataDrivenAlbedoEstimator
            dd_config = AlbedoModelConfig(model_path=self.config.model_path)
            self._dd_estimator = DataDrivenAlbedoEstimator(dd_config)

        return self._dd_estimator.process(
            scene, output_resolution=self.config.output_resolution,
        )

    # ------------------------------------------------------------------
    # Constant fallback path
    # ------------------------------------------------------------------

    def _fit_constant(
        self,
        scene: Sentinel2Scene,
        band_names: List[str],
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
    ) -> AlbedoData:
        """Fill output grid with constant default albedo per band."""
        n_bands = len(band_names)
        albedo_array = np.zeros((n_bands, out_h, out_w), dtype=np.float32)
        fallback_values: Dict[str, float] = {}

        for idx, band_name in enumerate(band_names):
            val = self.config.default_albedo.get(band_name, 0.05)
            albedo_array[idx] = val
            fallback_values[band_name] = val

        metadata = AlbedoMetadata(
            band_names=band_names,
            method="constant",
            clear_fraction=0.0,
            fallback_used=True,
            fallback_values=fallback_values,
        )

        logger.info(f"Constant albedo fallback complete for {n_bands} bands")

        return AlbedoData(
            data=albedo_array,
            transform=output_transform,
            crs=scene.crs,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Fallback router
    # ------------------------------------------------------------------

    def _run_fallback(
        self,
        scene: Sentinel2Scene,
        band_names: List[str],
        out_h: int,
        out_w: int,
        output_transform: rio.transform.Affine,
    ) -> AlbedoData:
        """Route to the configured fallback method."""
        if self.config.fallback == "datadriven":
            logger.info("Using data-driven MLP fallback")
            result = self._fit_datadriven(scene)
            result.metadata.fallback_used = True
            return result
        else:
            logger.info("Using constant albedo fallback")
            return self._fit_constant(
                scene, band_names, out_h, out_w, output_transform,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_nodata_mask(
        self, scene: Sentinel2Scene, result: AlbedoData,
    ) -> None:
        """Set output pixels to NaN where the scene has no valid data (DN=0).

        Operates in-place on ``result.data``.  Uses B02 as the reference
        band to identify nodata regions, then resizes to the output grid.
        """
        if "B02" in scene.bands:
            ref_band = scene.get_band("B02", reflectance=False)
        else:
            ref_band = max(scene.bands.values(), key=lambda b: b.size)
        ref_2d = ref_band[0] if ref_band.ndim == 3 else ref_band
        nodata_ref = (ref_2d == 0)

        if not nodata_ref.any():
            return

        out_h, out_w = result.data.shape[1], result.data.shape[2]
        nodata_out = resize(
            nodata_ref.astype(np.float32), (out_h, out_w),
            order=0, preserve_range=True,
        ) > 0.5
        result.data[:, nodata_out] = np.nan

    def _compute_output_grid(
        self, scene: Sentinel2Scene,
    ) -> Tuple[int, int, rio.transform.Affine]:
        """Compute output grid dimensions and transform from scene extent.

        Uses B02 (10m) as the reference band since scene.transform is always
        relative to B02. Falls back to the largest available band.
        """
        scene_res = abs(scene.transform.a)

        if 'B02' in scene.bands:
            ref_band = scene.bands['B02']
        else:
            ref_band = max(scene.bands.values(), key=lambda b: b.size)
        ref_h, ref_w = (ref_band.shape[1:] if ref_band.ndim == 3 else ref_band.shape)

        target_res = self.config.output_resolution

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

    @staticmethod
    def _windowed_mean_at_points(
        band_data: np.ndarray,
        clear_mask: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        half_w: int,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """Compute windowed clear-sky mean at specific pixel locations.

        Instead of filtering the entire image, extracts a small window
        around each sample point and averages only the clear pixels.
        Cost is O(n_samples * window_px²) instead of O(H * W).

        Args:
            band_data: 2-D array (H, W) of reflectance / DN values.
            clear_mask: 2-D boolean array (H, W), True where clear.
            rows: Row indices of sample locations.
            cols: Column indices of sample locations.
            half_w: Half-width of the averaging window in pixels.
            img_h: Image height.
            img_w: Image width.

        Returns:
            1-D float64 array of length ``len(rows)``.  Entries are NaN
            where the window contains no clear pixels.
        """
        n = len(rows)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            r0 = max(rows[i] - half_w, 0)
            r1 = min(rows[i] + half_w + 1, img_h)
            c0 = max(cols[i] - half_w, 0)
            c1 = min(cols[i] + half_w + 1, img_w)
            patch_clear = clear_mask[r0:r1, c0:c1]
            if patch_clear.any():
                result[i] = band_data[r0:r1, c0:c1][patch_clear].mean()
        return result
