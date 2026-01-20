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
    RetrievalsData, 
    AlbedoData,
    CloudMaskData
)
# We will import AlbedoEstimator properly once we fix that module.
# For now, we mock or try import
try:
    from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator
except ImportError:
    AlbedoEstimator = None

from .model import InversionNet, NormalizationWrapper
from .config import InputFeature

logger = logging.getLogger(__name__)

class CloudPropertyInverter:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initializes the CloudPropertyInverter.
        
        Args:
            checkpoint_path: Path to the .pth model checkpoint.
            device: 'cuda' or 'cpu'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load state
        if not Path(checkpoint_path).exists():
             raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
             
        state = torch.load(checkpoint_path, map_location=self.device)
        
        # Init Model
        # TODO: Move input/output sizes to config or infer from checkpoint meta if available
        core_model = InversionNet(input_size=17, output_size=4)
        
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

    def process(self, scene: Sentinel2Scene, height_data: CloudHeightGridData) -> RetrievalsData:
        """
        Runs the inversion on a full scene.
        
        Args:
            scene: The standardized Sentinel2Scene object.
            height_data: The result from CloudHeightProcessor (Cloud Top Height).
            
        Returns:
            RetrievalsData: The inferred cloud properties (COT, CER, etc).
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
        required_bands = ['B01', 'B02', 'B04', 'B08', 'B11', 'B12']
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
        logger.info("Estimating Surface Albedo...")
        if self.albedo_estimator:
            # Assuming albedo_estimator.process returns AlbedoData
            # For this immediate step, let's mock the return because we haven't fixed AlbedoEstimator signature yet
            # It currently calculates per-band scalar percentile.
            # We need to construct the dict expected by the flattener
            try:
                # We assume process returns a Dict[str, float] for now based on 'percentile' logic
                # If it currently returns AlbedoData object, we need to extract data.
                # Let's just use a stub for safety until AlbedoEstimator is confirmed standard.
                # surface_albedo = self.albedo_estimator.process(scene) 
                surface_albedo = {b: 0.1 for b in required_bands} # Fallback
            except Exception as e:
                logger.error(f"Albedo estimation failed: {e}. Using default.")
                surface_albedo = {b: 0.1 for b in required_bands}
        else:
             surface_albedo = {b: 0.1 for b in required_bands}

        
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

        cols = []
        # Band Order strictly enforced by model training
        cols.append(_flat(bands_map['B01']))
        cols.append(_flat(bands_map['B02']))
        cols.append(_flat(bands_map['B04']))
        cols.append(_flat(bands_map['B08']))
        cols.append(_flat(bands_map['B11']))
        cols.append(_flat(bands_map['B12']))
        
        # Albedos
        cols.append(_flat(surface_albedo['B01']))
        cols.append(_flat(surface_albedo['B02']))
        cols.append(_flat(surface_albedo['B04']))
        cols.append(_flat(surface_albedo['B08']))
        cols.append(_flat(surface_albedo['B11']))
        cols.append(_flat(surface_albedo['B12']))
        
        # Geometry
        cols.append(_flat(geometry['incidence_angle']))
        
        # Shading Ratio (Shadows)
        # Placeholder: 0.5 (Completely ambiguous)
        cols.append(np.full(target_shape[0]*target_shape[1], 0.5, dtype=np.float32))
        
        # Cloud Top Height
        cols.append(_flat(height_map))
        
        cols.append(_flat(geometry['mu']))
        cols.append(_flat(geometry['phi']))
        
        # Stack -> (N, 17)
        input_matrix = np.stack(cols, axis=1)
        
        # BATCH INFERENCE
        batch_size = 32768
        n_pixels = input_matrix.shape[0]
        output_list = []
        
        steps = range(0, n_pixels, batch_size)
        
        with torch.no_grad():
            for i in steps:
                batch_np = input_matrix[i : i+batch_size]
                
                # Check for NaNs and replace with 0
                batch_np = np.nan_to_num(batch_np)
                
                batch_t = torch.from_numpy(batch_np).to(self.device)
                pred = self.model(batch_t)
                output_list.append(pred.cpu().numpy())
                
        # Stack outputs -> (N, 4)
        results = np.concatenate(output_list, axis=0)
        
        # Reshape to (4, H, W)
        # Note: PyTorch/Rasterio standard is (Channels, Height, Width)
        results = results.reshape((target_shape[0], target_shape[1], 4))
        results = np.moveaxis(results, -1, 0) # (H, W, 4) -> (4, H, W)
        
        # Create RetrievalsData
        # Explicit data=results passed to base
        retrieval_obj = RetrievalsData()
        retrieval_obj.data = results
        retrieval_obj.transform = start_transform
        retrieval_obj.crs = start_crs
        retrieval_obj.metadata = {"description": "Refl2Prop Inversion Results (COT, CER, etc)"}
        
        return retrieval_obj
                
                # Model returns PHYSICAL values
                pred_t = self.model(batch_t)
                output_list.append(pred_t.cpu().numpy())
                
        results_flat = np.concatenate(output_list, axis=0)
        
        # 4. Reshape to Image
        # Output order: [Tau, IceLiq, ReffL, ReffI]
        res_img = results_flat.reshape(shape[0], shape[1], 4)
        
        return {
            'tau': res_img[..., 0],
            'ice_liq_ratio': res_img[..., 1],
            'r_eff_liq': res_img[..., 2],
            'r_eff_ice': res_img[..., 3]
        }

    def process(self, scene: Sentinel2Scene, cloud_height: CloudHeightGridData, albedo: Optional[AlbedoData] = None) -> RetrievalsData:
        """
        High-level processing method that accepts standardized data objects.
        """
        # 1. Prepare Bands
        req_bands = ['B01', 'B02', 'B04', 'B08', 'B11', 'B12']
        bands_input = {}
        
        # We assume scene bands logic needs to read the data
        ref_transform = None
        ref_crs = None
        
        for b in req_bands:
            if b not in scene.bands:
                raise ValueError(f"Missing required band {b} in scene.")
            
            # Use simple read for now.
            path = scene.bands[b]
            gd = GeoRasterData.from_file(path)
            
            # Handle [1, H, W] -> [H, W]
            data = gd.data
            if data.ndim == 3:
                data = data[0]
            
            bands_input[b] = data #[H, W]

            if b == 'B02': 
                ref_transform = gd.transform
                ref_crs = gd.crs
        
        # 2. Prepare Albedo
        surface_albedo = {}
        if albedo is None:
            # Default
            for b in req_bands:
                surface_albedo[b] = 0.1
        else:
             # Basic usage: assume albedo matches band order if mapped via name
             # For now, fallback to default as AlbedoData matching is complex without metadata standards
             for b in req_bands:
                surface_albedo[b] = 0.1

        # 3. Geometry
        sun_zen = scene.sun_zenith if scene.sun_zenith is not None else 30.0
        sun_azi = scene.sun_azimuth if scene.sun_azimuth is not None else 180.0
        
        # Use mean view angles if available, else defaults
        view_zen = scene.view_zenith if scene.view_zenith is not None else 0.0 # Nadir default
        view_azi = scene.view_azimuth if scene.view_azimuth is not None else 0.0

        # Mu is cosine of VIEW zenith (not sun)
        mu_val = np.cos(np.deg2rad(view_zen))
        
        # Phi is relative azimuth (Sun - View)
        phi_val = np.deg2rad(sun_azi - view_azi)
        
        geometry = {
            'incidence_angle': sun_zen, 
            'mu': mu_val,
            'phi': phi_val
        }
        
        # 4. Height
        input_height = cloud_height.data
        if input_height is None:
             raise ValueError("Cloud Height data is empty")
        
        # 5. Run Prediction
        results_dict = self.predict_scene(
            bands=bands_input,
            surface_albedo=surface_albedo,
            geometry=geometry,
            cloud_top_height=input_height,
            shading_ratio=0.5
        )
        
        # 6. Package Output
        keys = ['tau', 'ice_liq_ratio', 'r_eff_liq', 'r_eff_ice']
        out_arrays = []
        for k in keys:
            out_arrays.append(results_dict[k])
            
        stacked = np.stack(out_arrays, axis=0) # [4, H, W]
        
        output = RetrievalsData()
        output.data = stacked
        
        # Use height transform as preference, else band
        if cloud_height.transform:
            output.transform = cloud_height.transform
            output.crs = cloud_height.crs
        else:
            output.transform = ref_transform
            output.crs = ref_crs
            
        # Pydantic model dump will handle dict
        # output.metadata is a Metadata object with 'extra' allowed
        # output.metadata.extra = {'band_names': keys} # Not standard way to set extra
        # Use simple dict update if possible or just ignore for now
        
        return output