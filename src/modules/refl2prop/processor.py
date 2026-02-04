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
# AlbedoEstimator import
from clouds_decoded.modules.albedo_estimator import AlbedoEstimator

from .model import InversionNet, NormalizationWrapper
from .config import InputFeature, Refl2PropConfig, OutputFeature

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
             raise FileNotFoundError(f"Model checkpoint not found at {config.model_path}")
             
        state = torch.load(config.model_path, map_location=self.device)
        
        # Init Model
        # TODO: Move input/output sizes to config or infer from checkpoint meta if available
        # Note: noise_output_size=6 matches the training configuration for model_ood.pth
        core_model = InversionNet(input_size=17, output_size=4, noise_output_size=6)
        
        # Reconstruct Normalization Wrapper using dummy stats (overwritten by load_state_dict)
        dummy = {'min': [0]*17, 'max': [1]*17}
        out_dummy = {'min': [0]*4, 'max': [1]*4}
        
        self.model = NormalizationWrapper(core_model, dummy, out_dummy)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize internal Albedo Estimator
        if AlbedoEstimator:
            self.albedo_estimator = AlbedoEstimator()
        else:
            self.albedo_estimator = None
            logger.warning("AlbedoEstimator not available. Using default 0.1 albedo.")

    def process(self, scene: Sentinel2Scene, height_data: CloudHeightGridData) -> CloudPropertiesData:
        """
        Runs the inversion on a full scene.
        
        Args:
            scene: The standardized Sentinel2Scene object.
            height_data: The result from CloudHeightProcessor (Cloud Top Height).
            
        Returns:
            CloudPropertiesData: The inferred cloud properties (COT, CER, etc).
        """
        start_transform = height_data.transform
        start_crs = height_data.crs
        # Handle case where single band (H, W) vs multi (C, H, W)
        if height_data.data.ndim == 2:
            start_shape = height_data.data.shape
            height_map = height_data.data
        else:
            start_shape = height_data.data.shape[1:]
            height_map = height_data.data[0] # Assume first band is height
        
        target_shape = start_shape
        
        # 1. Prepare Inputs
        # -----------------
        
        # A. Bands (B01, B02, B04, B08, B11, B12)
        logger.info("Loading bands and resizing to target resolution...")
        required_bands = self.config.required_bands
        bands_map = {}
        
        for b in required_bands:
             if b not in scene.bands:
                  raise ValueError(f"Band {b} not found in scene object. Ensure scene.read() was called with required bands.")
             
             band_arr = scene.bands[b]
             # Ensure band_arr is 2D
             if band_arr.ndim == 3:
                 band_arr = band_arr[0]

             if band_arr.shape != target_shape:
                  # Resize to match cloud height grid
                  band_arr = resize(band_arr, target_shape, order=1, preserve_range=True).astype(np.float32)
             else:
                  band_arr = band_arr.astype(np.float32)
             
             bands_map[b] = band_arr

        # B. Surface Albedo
        # -----------------
        logger.info("Estimating Surface Albedo (Geospatial)...")
        albedo_maps = {} 
        
        # We need to get albedo for each required band
        if self.albedo_estimator:
            try:
                # Returns Dict[str, AlbedoData]
                raw_albedo_dict = self.albedo_estimator.process(scene)
            except Exception as e:
                logger.error(f"Albedo estimation failed: {e}")
                raw_albedo_dict = {}
        else:
             raw_albedo_dict = {}

        for b in required_bands:
            # 1. Get raw albedo raster (or fallback constant)
            if b in raw_albedo_dict and raw_albedo_dict[b].data is not None:
                alb_arr = raw_albedo_dict[b].data
                # Ensure 2D
                if alb_arr.ndim == 3: alb_arr = alb_arr[0]
            else:
                logger.warning(f"Albedo for {b} missing. Using {self.config.default_albedo} fallback.")
                # Create constant array of target shape directly (optimization: skip resize step)
                alb_arr = np.full(target_shape, self.config.default_albedo, dtype=np.float32)

            # 2. Resize to match Cloud Height Grid if needed
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
        rel_az = np.abs(scene.sun_azimuth - scene.view_azimuth)
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
        
        def _norm_band(v):
            return (v - self.config.norm_bands_offset) / self.config.norm_bands_scale
            
        def _norm_height(v):
            return (v - self.config.norm_height_offset) / self.config.norm_height_scale

        cols = []
        # Band Order strictly enforced by model training
        for b in required_bands:
             cols.append(_norm_band(_flat(bands_map[b])))
        
        # Albedos
        for b in required_bands:
             cols.append(_norm_band(_flat(albedo_maps[b])))
        
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
             logger.info("Masking pixels where Cloud Height <= 0 with NaN")
             invalid_mask = (height_map <= 0)
             # results shape (C, H, W)
             for i in range(results.shape[0]):
                 results[i][invalid_mask] = np.nan
        
        # Create CloudPropertiesData
        # Explicit data=results passed to base
        retrieval_obj = CloudPropertiesData()
        retrieval_obj.data = results
        retrieval_obj.transform = start_transform
        retrieval_obj.crs = start_crs
        
        if return_uncertainty:
            current_bands = list(retrieval_obj.metadata.band_names)
            if OutputFeature.UNCERTAINTY.value not in current_bands:
                 current_bands.append(OutputFeature.UNCERTAINTY.value)
                 retrieval_obj.metadata.band_names = current_bands

        #Debugging prints, show histograms, min/max, etc. of all inputs and all outputs, in 17+4 png files and 17+4 lines of print
        for i in range(17):
            col_data = input_matrix[:,i]
            print(f"Input Feature {i}: min={np.nanmin(col_data)}, max={np.nanmax(col_data)}, mean={np.nanmean(col_data)}")
        for i in range(num_channels):
            out_data = results[i].reshape(-1)
            print(f"Output Property {i}: min={np.nanmin(out_data)}, max={np.nanmax(out_data)}, mean={np.nanmean(out_data)}")

        import pprint
        pprint.pprint(self.model.ranges)

        
        return retrieval_obj