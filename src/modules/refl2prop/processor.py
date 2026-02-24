import torch
import numpy as np
import logging
import rasterio as rio
from typing import Dict, Union, Optional
from pathlib import Path
from skimage.transform import resize

# Standardized Imports
from clouds_decoded.data import (
    Sentinel2Scene, 
    CloudHeightGridData, 
    CloudPropertiesData, 
    AlbedoData,
    CloudMaskData
)

from .model import InversionNet, NormalizationWrapper
from .model_shading import ShadingAwareInversionNet, ShadingNormalizationWrapper
from .config import Refl2PropConfig, ShadingRefl2PropConfig, OutputFeature

logger = logging.getLogger(__name__)

class CloudPropertyInverter:
    def __init__(self, config: Refl2PropConfig, device: str = 'cuda'):
        """
        Initializes the CloudPropertyInverter.
        
        Args:
            config: Refl2PropConfig object.
            device: 'cuda' or 'cpu'.
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load state
        if not Path(config.model_path).exists():
            raise FileNotFoundError(
                f"Refl2prop model weights not found at {config.model_path}.\n"
                f"Run:  clouds-decoded download refl2prop\n"
                f"or set CLOUDS_DECODED_ASSETS_DIR to a directory containing "
                f"models/refl2prop/default.pth"
            )
             
        state = torch.load(config.model_path, map_location=self.device)
        
        # Init Model
        # TODO: Move input/output sizes to config or infer from checkpoint meta if available
        # Note: noise_output_size=6 matches the training configuration for model_ood.pth
        core_model = InversionNet(
             input_size=self.config.input_size, 
             output_size=self.config.output_size, 
             noise_output_size=self.config.noise_output_size
        )
        
        # Reconstruct Normalization Wrapper using dummy stats (overwritten by load_state_dict)
        dummy = {'min': [0]*self.config.input_size, 'max': [1]*self.config.input_size}
        out_dummy = {'min': [0]*self.config.output_size, 'max': [1]*self.config.output_size}
        
        self.model = NormalizationWrapper(core_model, dummy, out_dummy)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
    def process(
        self,
        scene: Sentinel2Scene,
        height_data: CloudHeightGridData,
        albedo_data: Optional[AlbedoData] = None,
    ) -> CloudPropertiesData:
        """
        Runs the inversion on a full scene.

        Args:
            scene: The standardized Sentinel2Scene object.
            height_data: The result from CloudHeightProcessor (Cloud Top Height).
            albedo_data: Optional pre-computed albedo. If None, albedo is estimated
                         internally using constant default values.

        Returns:
            CloudPropertiesData: The inferred cloud properties (COT, CER, etc).
        """
        input_transform = height_data.transform
        input_crs = height_data.crs

        # Handle case where single band (H, W) vs multi (C, H, W)
        if height_data.data.ndim == 2:
            input_shape = height_data.data.shape
            height_map = height_data.data
        else:
            input_shape = height_data.data.shape[1:]
            height_map = height_data.data[0]  # Assume first band is height

        # Determine target resolution and shape (derived from height data grid)
        input_resolution = abs(input_transform.a)  # Current pixel size in meters
        target_resolution = self.config.output_resolution
        scale = input_resolution / target_resolution
        target_shape = (int(input_shape[0] * scale), int(input_shape[1] * scale))

        # Create transform for target resolution
        output_transform = rio.transform.Affine(
            target_resolution, 0, input_transform.c,
            0, -target_resolution, input_transform.f
        )

        # Resize height map if needed
        if height_map.shape != target_shape:
            logger.info(f"Resampling height from {input_resolution}m to {target_resolution}m "
                       f"({input_shape} -> {target_shape})")
            height_map = resize(height_map, target_shape, order=0, preserve_range=True).astype(np.float32)

        # 1. Prepare Inputs
        # -----------------

        # A. Bands
        logger.info("Loading bands...")
        required_bands = self.config.bands
        band_objects = scene.get_bands(required_bands, reflectance=True, n_workers=len(required_bands))
        bands_map = {}
        for b, obj in zip(required_bands, band_objects):
            band_arr = obj.data.astype(np.float32)
            if band_arr.shape != target_shape:
                band_arr = resize(band_arr, target_shape, order=1, preserve_range=True).astype(np.float32)
            bands_map[b] = band_arr

        # B. Surface Albedo
        # -----------------
        albedo_maps = {}

        if albedo_data is None:
            logger.warning("No albedo provided. Using per-band constant defaults.")
        else:
            logger.info("Using pre-computed albedo data")

        if albedo_data is not None:
            band_to_index = {band: idx for idx, band in enumerate(albedo_data.metadata.band_names)}
        else:
            band_to_index = {}

        for b in required_bands:
            if albedo_data is not None and b in band_to_index:
                band_idx = band_to_index[b]
                alb_arr = albedo_data.data[band_idx]
            else:
                fallback = self.config.default_albedo.get(b, 0.05)
                alb_arr = np.full(target_shape, fallback, dtype=np.float32)

            if alb_arr.shape != target_shape:
                 alb_arr = resize(alb_arr, target_shape, order=0, preserve_range=True).astype(np.float32)
            else:
                 alb_arr = alb_arr.astype(np.float32)

            albedo_maps[b] = alb_arr

        # C. Geometry (Sun/View Angles)
        # -----------------------------
        # Sun Zenith (Incidence Angle)
        # Handle scalar or array
        def to_grid(val):
            return np.full(target_shape, val, dtype=np.float32)

        sza = to_grid(scene.sun_zenith)
        
        # Cosine of View Zenith (mu)
        vza_rad = np.radians(scene.view_zenith)
        mu = to_grid(np.cos(vza_rad))
        
        # Relative Azimuth (phi)
        rel_az = np.radians(np.abs(scene.sun_azimuth - scene.view_azimuth))
        phi = to_grid(rel_az)
        
        geometry = {
            'incidence_angle': sza,
            'mu': mu,
            'phi': phi
        }
        
        # 2. Run Inference
        # ----------------
        logger.info("Running Neural Inversion...")
        
        # Flatten helper
        def _flat(v):
            if np.isscalar(v):
                return np.full(target_shape[0]*target_shape[1], v, dtype=np.float32)
            return v.reshape(-1).astype(np.float32)
        
        def _norm_height(v):
            return (v - self.config.norm_height_offset) / self.config.norm_height_scale

        cols = []
        # Bands are already in reflectance via get_band()
        for b in required_bands:
             cols.append(_flat(bands_map[b]))

        # Albedos are already in reflectance
        for b in required_bands:
             cols.append(_flat(albedo_maps[b]))
        
        # Geometry
        cols.append(_flat(geometry['incidence_angle']))
        
        # Shading Ratio (Shadows)
        # Placeholder: 0.5 (Completely ambiguous)
        cols.append(np.full(target_shape[0]*target_shape[1], self.config.default_shading_ratio, dtype=np.float32))
        
        # Cloud Top Height
        cols.append(_norm_height(_flat(height_map)))
        
        # Correction: geometry['mu'] usage was duplicated in original code? No, let's check.
        # Original: cols.append(_flat(geometry['mu'])); cols.append(_flat(geometry['phi']))
        cols.append(_flat(geometry['mu']))
        cols.append(_flat(geometry['phi']))
        
        # Stack -> (N, 17)
        input_matrix = np.stack(cols, axis=1)
        
        # BATCH INFERENCE
        batch_size = self.config.batch_size
        n_pixels = input_matrix.shape[0]
        output_list = []
        uncertainty_list = []
        return_uncertainty = self.config.return_uncertainty
        
        steps = range(0, n_pixels, batch_size)
        
        with torch.no_grad():
            for i in steps:
                batch_np = input_matrix[i : i+batch_size]
                
                # Check for NaNs and replace with 0
                batch_np = np.nan_to_num(batch_np)
                
                batch_t = torch.from_numpy(batch_np).to(self.device)
                
                if return_uncertainty:
                    # Model returns (physics, uncertainty)
                    pred, unc = self.model(batch_t, return_uncertainty=True)
                    output_list.append(pred.cpu().numpy())
                    # uncertainty is (Batch,) -> reshape to (Batch, 1) for concatenation later if needed
                    uncertainty_list.append(unc.cpu().numpy())
                else:
                    pred = self.model(batch_t)
                    output_list.append(pred.cpu().numpy())
                
        # Stack outputs -> (N, 4)
        results = np.concatenate(output_list, axis=0)

        # Append uncertainty if requested
        if return_uncertainty:
            unc_results = np.concatenate(uncertainty_list, axis=0)
            # Expand dims to (N, 1)
            unc_results = unc_results[:, np.newaxis]
            results = np.concatenate([results, unc_results], axis=1)
        
        num_channels = results.shape[1]

        # Reshape to (C, H, W)
        # Note: PyTorch/Rasterio standard is (Channels, Height, Width)
        results = results.reshape((target_shape[0], target_shape[1], num_channels))
        results = np.moveaxis(results, -1, 0) # (H, W, C) -> (C, H, W)
        
        # Post-process Masking
        if self.config.mask_invalid_height:
             logger.info("Masking pixels where Cloud Height is invalid (<=0 or NaN)")
             invalid_mask = ~(height_map > 0)  # Catches NaN, zero, and negative
             # results shape (C, H, W)
             for i in range(results.shape[0]):
                 results[i][invalid_mask] = np.nan
        
        # Create CloudPropertiesData
        # Explicit data=results passed to base
        retrieval_obj = CloudPropertiesData()
        retrieval_obj.data = results
        retrieval_obj.transform = output_transform
        retrieval_obj.crs = input_crs
        
        if return_uncertainty:
            current_bands = list(retrieval_obj.metadata.band_names)
            if OutputFeature.UNCERTAINTY.value not in current_bands:
                 current_bands.append(OutputFeature.UNCERTAINTY.value)
                 retrieval_obj.metadata.band_names = current_bands

        return retrieval_obj


class ShadingPropertyInverter(CloudPropertyInverter):
    """
    Shading-aware cloud property inverter that processes pixels in windows.

    Uses a self-attention model to predict:
    - Per-pixel tau_effective_shading
    - Global (per-window) physical properties: tau, ice_liq_ratio, r_eff_liq, r_eff_ice

    Overlapping windows are averaged:
    - Log-space averaging for tau, r_eff_liq, r_eff_ice
    - Linear-space averaging for ice_liq_ratio
    """

    def __init__(
        self,
        config: Union[Refl2PropConfig, ShadingRefl2PropConfig],
        device: str = 'cuda',
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_attention_layers: Optional[int] = None,
    ):
        """
        Args:
            config: ShadingRefl2PropConfig (preferred) or Refl2PropConfig with model_path.
            device: 'cuda' or 'cpu'.
            window_size: Override config window_size. Default: 16.
            stride: Override config stride. Default: 8.
            hidden_dim: Override config hidden_dim. Default: 256.
            n_heads: Override config n_heads. Default: 4.
            n_attention_layers: Override config n_attention_layers. Default: 2.
        """
        # Don't call super().__init__() as we need different model initialization
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Use config values if ShadingRefl2PropConfig, else use defaults/overrides
        if isinstance(config, ShadingRefl2PropConfig):
            self.window_size = window_size if window_size is not None else config.window_size
            self.stride = stride if stride is not None else config.stride
            _hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
            _n_heads = n_heads if n_heads is not None else config.n_heads
            _n_attention_layers = n_attention_layers if n_attention_layers is not None else config.n_attention_layers
        else:
            # Fallback to explicit args or defaults
            self.window_size = window_size if window_size is not None else 16
            self.stride = stride if stride is not None else 8
            _hidden_dim = hidden_dim if hidden_dim is not None else 256
            _n_heads = n_heads if n_heads is not None else 4
            _n_attention_layers = n_attention_layers if n_attention_layers is not None else 2

        self.bag_size = self.window_size * self.window_size

        # Load state
        if not Path(config.model_path).exists():
            raise FileNotFoundError(
                f"Refl2prop model weights not found at {config.model_path}.\n"
                f"Run:  clouds-decoded download refl2prop\n"
                f"or set CLOUDS_DECODED_ASSETS_DIR to a directory containing "
                f"models/refl2prop/default.pth"
            )

        state = torch.load(config.model_path, map_location=self.device, weights_only=False)

        # Init Shading Model
        core_model = ShadingAwareInversionNet(
            input_size=config.input_size,
            hidden_dim=_hidden_dim,
            n_heads=_n_heads,
            n_attention_layers=_n_attention_layers,
            output_size=config.output_size,
        )

        # Reconstruct Normalization Wrapper with dummy stats (overwritten by load_state_dict)
        dummy_input = {'min': [0] * config.input_size, 'max': [1] * config.input_size}
        dummy_output = {'min': [0] * config.output_size, 'max': [1] * config.output_size}
        dummy_shading = {'min': 0.0, 'max': 100.0}

        self.model = ShadingNormalizationWrapper(
            core_model, dummy_input, dummy_output, dummy_shading
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"ShadingPropertyInverter initialized: window={self.window_size}x{self.window_size}, "
                   f"stride={self.stride}, bag_size={self.bag_size}")

    def _extract_windows(self, data: np.ndarray) -> tuple:
        """
        Extract overlapping windows from 2D or 3D array.

        Args:
            data: Array of shape (H, W) or (C, H, W)

        Returns:
            windows: Array of shape (n_windows, window_size, window_size) or
                     (n_windows, C, window_size, window_size)
            positions: List of (row, col) top-left corner positions
        """
        if data.ndim == 2:
            H, W = data.shape
            has_channels = False
        else:
            C, H, W = data.shape
            has_channels = True

        windows = []
        positions = []

        # Generate window positions
        for row in range(0, H - self.window_size + 1, self.stride):
            for col in range(0, W - self.window_size + 1, self.stride):
                if has_channels:
                    window = data[:, row:row+self.window_size, col:col+self.window_size]
                else:
                    window = data[row:row+self.window_size, col:col+self.window_size]
                windows.append(window)
                positions.append((row, col))

        return np.array(windows), positions

    def _aggregate_predictions(
        self,
        predictions: np.ndarray,
        shading_predictions: np.ndarray,
        positions: list,
        output_shape: tuple
    ) -> tuple:
        """
        Aggregate overlapping window predictions with appropriate averaging.

        Args:
            predictions: (n_windows, 4) global predictions per window
            shading_predictions: (n_windows, bag_size) per-pixel shading
            positions: List of (row, col) top-left positions
            output_shape: (H, W) target output shape

        Returns:
            global_output: (4, H, W) averaged global predictions
            shading_output: (H, W) averaged shading predictions
        """
        H, W = output_shape

        # Accumulators for global predictions
        # Indices: 0=tau, 1=ice_liq_ratio, 2=r_eff_liq, 3=r_eff_ice
        log_indices = [0, 2, 3]  # tau, r_eff_liq, r_eff_ice
        linear_indices = [1]     # ice_liq_ratio

        global_sum = np.zeros((4, H, W), dtype=np.float64)
        global_count = np.zeros((H, W), dtype=np.float64)

        # For log-space averaging: store sum of log values
        global_log_sum = np.zeros((4, H, W), dtype=np.float64)

        # Shading accumulators (always log-space)
        shading_log_sum = np.zeros((H, W), dtype=np.float64)
        shading_count = np.zeros((H, W), dtype=np.float64)

        eps = 1e-6

        for i, (row, col) in enumerate(positions):
            # Global predictions - broadcast to window
            for j in range(4):
                val = predictions[i, j]
                if j in log_indices:
                    # Log-space: accumulate log(val)
                    log_val = np.log(max(val, eps))
                    global_log_sum[j, row:row+self.window_size, col:col+self.window_size] += log_val
                else:
                    # Linear-space: accumulate directly
                    global_sum[j, row:row+self.window_size, col:col+self.window_size] += val

            global_count[row:row+self.window_size, col:col+self.window_size] += 1

            # Shading predictions - reshape from (bag_size,) to (window_size, window_size)
            shading_window = shading_predictions[i].reshape(self.window_size, self.window_size)
            shading_log = np.log(np.maximum(shading_window, eps))
            shading_log_sum[row:row+self.window_size, col:col+self.window_size] += shading_log
            shading_count[row:row+self.window_size, col:col+self.window_size] += 1

        # Compute averages
        global_output = np.zeros((4, H, W), dtype=np.float32)

        # Avoid division by zero
        valid_mask = global_count > 0

        for j in range(4):
            if j in log_indices:
                # Log-space: exp(mean(log(x)))
                avg_log = np.divide(
                    global_log_sum[j],
                    global_count,
                    out=np.zeros_like(global_log_sum[j]),
                    where=valid_mask
                )
                global_output[j] = np.exp(avg_log).astype(np.float32)
            else:
                # Linear-space: simple mean
                global_output[j] = np.divide(
                    global_sum[j],
                    global_count,
                    out=np.zeros_like(global_sum[j]),
                    where=valid_mask
                ).astype(np.float32)

        # Shading average (log-space)
        avg_shading_log = np.divide(
            shading_log_sum,
            shading_count,
            out=np.zeros_like(shading_log_sum),
            where=shading_count > 0
        )
        shading_output = np.exp(avg_shading_log).astype(np.float32)

        # Mark uncovered pixels as NaN
        uncovered = ~valid_mask
        if uncovered.any():
            global_output[:, uncovered] = np.nan
            shading_output[uncovered] = np.nan

        return global_output, shading_output

    def process(
        self,
        scene: Sentinel2Scene,
        height_data: CloudHeightGridData,
        albedo_data: Optional[AlbedoData] = None,
    ) -> CloudPropertiesData:
        """
        Runs shading-aware inversion on a full scene using windowed processing.

        Args:
            scene: The standardized Sentinel2Scene object.
            height_data: The result from CloudHeightProcessor (Cloud Top Height).
            albedo_data: Optional pre-computed albedo. If None, albedo is estimated
                         internally using constant default values.

        Returns:
            CloudPropertiesData: The inferred cloud properties including tau_shading.
        """
        input_transform = height_data.transform
        input_crs = height_data.crs

        # Handle shape
        if height_data.data.ndim == 2:
            input_shape = height_data.data.shape
            height_map = height_data.data
        else:
            input_shape = height_data.data.shape[1:]
            height_map = height_data.data[0]

        # Determine target resolution and shape (derived from height data grid)
        input_resolution = abs(input_transform.a)
        target_resolution = self.config.output_resolution
        scale = input_resolution / target_resolution
        target_shape = (int(input_shape[0] * scale), int(input_shape[1] * scale))

        output_transform = rio.transform.Affine(
            target_resolution, 0, input_transform.c,
            0, -target_resolution, input_transform.f
        )

        # Resize height map if needed
        if height_map.shape != target_shape:
            logger.info(f"Resampling height from {input_resolution}m to {target_resolution}m "
                       f"({input_shape} -> {target_shape})")
            height_map = resize(height_map, target_shape, order=0, preserve_range=True).astype(np.float32)

        # 1. Prepare Inputs
        logger.info("Loading bands...")
        required_bands = self.config.bands
        band_objects = scene.get_bands(required_bands, reflectance=True, n_workers=len(required_bands))
        bands_map = {}
        for b, obj in zip(required_bands, band_objects):
            band_arr = obj.data.astype(np.float32)
            if band_arr.shape != target_shape:
                band_arr = resize(band_arr, target_shape, order=1, preserve_range=True).astype(np.float32)
            bands_map[b] = band_arr

        # Surface Albedo
        albedo_maps = {}

        if albedo_data is None:
            logger.warning("No albedo provided. Using per-band constant defaults.")
        else:
            logger.info("Using pre-computed albedo data")

        if albedo_data is not None:
            band_to_index = {band: idx for idx, band in enumerate(albedo_data.metadata.band_names)}
        else:
            band_to_index = {}

        for b in required_bands:
            if albedo_data is not None and b in band_to_index:
                band_idx = band_to_index[b]
                alb_arr = albedo_data.data[band_idx]
            else:
                fallback = self.config.default_albedo.get(b, 0.05)
                alb_arr = np.full(target_shape, fallback, dtype=np.float32)

            if alb_arr.shape != target_shape:
                alb_arr = resize(alb_arr, target_shape, order=0, preserve_range=True).astype(np.float32)

            albedo_maps[b] = alb_arr

        # Geometry
        def to_grid(val):
            return np.full(target_shape, val, dtype=np.float32)

        sza = to_grid(scene.sun_zenith)
        vza_rad = np.radians(scene.view_zenith)
        mu = to_grid(np.cos(vza_rad))
        rel_az = np.radians(np.abs(scene.sun_azimuth - scene.view_azimuth))
        phi = to_grid(rel_az)

        # 2. Build input tensor
        # Stack all inputs: (C, H, W) where C = num_bands*2 + geometry
        def _norm_height(v):
            return (v - self.config.norm_height_offset) / self.config.norm_height_scale

        input_layers = []

        # Bands are already in reflectance via get_band()
        for b in required_bands:
            input_layers.append(bands_map[b])

        # Albedos are already in reflectance
        for b in required_bands:
            input_layers.append(albedo_maps[b])

        # Geometry
        input_layers.append(sza)
        input_layers.append(np.full(target_shape, self.config.default_shading_ratio, dtype=np.float32))
        input_layers.append(_norm_height(height_map))
        input_layers.append(mu)
        input_layers.append(phi)

        # Stack to (C, H, W)
        input_stack = np.stack(input_layers, axis=0)

        # 3. Extract windows
        logger.info(f"Extracting windows (size={self.window_size}, stride={self.stride})...")
        windows, positions = self._extract_windows(input_stack)
        n_windows = len(windows)
        logger.info(f"  Generated {n_windows} windows")

        # Reshape windows for model: (n_windows, bag_size, input_dim)
        # Currently (n_windows, C, window_size, window_size) -> (n_windows, window_size^2, C)
        C = input_stack.shape[0]
        windows = windows.reshape(n_windows, C, self.bag_size)
        windows = np.transpose(windows, (0, 2, 1))  # (n_windows, bag_size, C)

        # Replace NaNs
        windows = np.nan_to_num(windows)

        # 4. Batch inference
        logger.info("Running shading-aware inference...")
        batch_size = max(1, self.config.batch_size // self.bag_size)  # Adjust for bags

        all_global_preds = []
        all_shading_preds = []

        with torch.no_grad():
            for i in range(0, n_windows, batch_size):
                batch = windows[i:i+batch_size]
                batch_t = torch.from_numpy(batch).float().to(self.device)

                # Model returns (shading, global) both denormalized
                shading_pred, global_pred = self.model(batch_t, return_denormalized=True)

                all_shading_preds.append(shading_pred.cpu().numpy())
                all_global_preds.append(global_pred.cpu().numpy())

        # Concatenate
        global_preds = np.concatenate(all_global_preds, axis=0)  # (n_windows, 4)
        shading_preds = np.concatenate(all_shading_preds, axis=0)  # (n_windows, bag_size)

        # 5. Aggregate predictions
        logger.info("Aggregating overlapping window predictions...")
        global_output, shading_output = self._aggregate_predictions(
            global_preds, shading_preds, positions, target_shape
        )

        # 6. Post-process masking
        if self.config.mask_invalid_height:
            logger.info("Masking pixels where Cloud Height is invalid (<=0 or NaN)")
            invalid_mask = ~(height_map > 0)  # Catches NaN, zero, and negative
            global_output[:, invalid_mask] = np.nan
            shading_output[invalid_mask] = np.nan

        # 7. Build output
        # Stack global outputs + shading as (5, H, W)
        results = np.concatenate([global_output, shading_output[np.newaxis, :, :]], axis=0)

        retrieval_obj = CloudPropertiesData()
        retrieval_obj.data = results
        retrieval_obj.transform = output_transform
        retrieval_obj.crs = input_crs

        # Update band names to include tau_shading
        current_bands = list(retrieval_obj.metadata.band_names)
        if 'tau_shading' not in current_bands:
            current_bands.append('tau_shading')
            retrieval_obj.metadata.band_names = current_bands

        return retrieval_obj