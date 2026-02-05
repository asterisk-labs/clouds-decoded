
import numpy as np
import logging
import torch
from typing import Dict, Union, Optional
from skimage.transform import resize
from skimage.morphology import disk, dilation
import rasterio

from clouds_decoded.data import Sentinel2Scene, CloudMaskData, CloudMaskMetadata
from senseiv2.inference import CloudMask
from senseiv2.utils import get_model_files
from senseiv2.constants import SENTINEL2_BANDS, SENTINEL2_DESCRIPTORS
from .config import CloudMaskConfig, PostProcessParams

logger = logging.getLogger(__name__)

class ThresholdCloudMaskProcessor:
    def __init__(self, config: Optional[CloudMaskConfig] = None):
        if config is None:
            config = CloudMaskConfig(method="threshold")
        self.config = config

    def process(self, scene: Sentinel2Scene) -> CloudMaskData:
        """
        Creates a basic cloud mask by thresholding a specific band.
        Values GREATER than the threshold are considered cloud (1), others not (0).
        """
        threshold_band = self.config.threshold_band
        threshold_value = self.config.threshold_value
        
        # Ensure bands are loaded
        if threshold_band not in scene.bands:
             raise ValueError(f"Required band {threshold_band} not loaded in scene.")
             
        # Get raw data
        band_data = scene.bands[threshold_band]
        
        # Ensure 2D
        if isinstance(band_data, np.ndarray) and band_data.ndim == 3:
             band_data = band_data[0]
             
        logger.info(f"Generating cloud mask using: {threshold_band} > {threshold_value}")
        
        mask = (band_data > threshold_value).astype(np.uint8)
        
        # Construct CloudMaskData using Sentinel2Scene georeferencing
        out = CloudMaskData(
             data=mask,
             transform=scene.transform,
             crs=scene.crs,
             metadata=CloudMaskMetadata(
                 categorical=True,
                 classes={0: 'Clear', 1: 'Cloud'},
                 method="simple_threshold",
                 threshold_band=threshold_band,
                 threshold_value=threshold_value
             )
        )
        return out

class CloudMaskProcessor:
    def __init__(self, config: Optional[CloudMaskConfig] = None):
        if config is None:
            config = CloudMaskConfig(method="senseiv2")
        self.config = config
        
        self.model_name = self.config.model_name
        self.output_style = self.config.output_style
        
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cm = None

    def _load_model(self):
        if self.cm is None:
            logger.info(f"Loading SEnSeIv2 model: {self.model_name} on {self.device}")
            config, weights = get_model_files(self.model_name)
            self.cm = CloudMask(
                config, 
                weights, 
                device=self.device, 
                output_style=self.output_style,
                verbose=False
            )

    def process(self, scene: Sentinel2Scene) -> CloudMaskData:
        """
        Generates a cloud mask using SEnSeIv2 deep learning model.
        The output is the raw model result (typically 4-class map) 
        at the configured input resolution ('resolution' in config).
        
        Args:
            scene: Sentinel2Scene object with loaded bands.
            
        Returns:
            CloudMaskData: Cloud mask (raw classification).
        """
        self._load_model()
        
        # 1. Prepare Data
        # SEnSeIv2 expects all bands normalized and stacked.
        data_list = []
        
        # Determine target shape based on config.resolution
        # Use B02 (10m) as the reference for scale calculation
        ref_band_10m = "B02"
        if ref_band_10m in scene.bands:
            base_arr = scene.bands[ref_band_10m]
        else:
            base_arr = next(iter(scene.bands.values()))
            
        if base_arr.ndim == 3: base_arr = base_arr[0]
        
        # Calculate scale factor relative to 10m (approx or whatever base_arr is)
        # Assuming base_arr is 10m. If config.resolution is 20m, scale is 0.5.
        # Ideally we should use the transform to know real resolution, but 
        # let's assume B02 is nominal 10m for simplicity or calculate from transform.
        
        current_res_x = scene.transform[0] # Pixel width (usually 10, 20, 60 or 0.0001 for lat/lon)
        # Handle negative y-scale for transform[4]
        current_res = abs(current_res_x)
        
        target_res = float(self.config.resolution)
        scale_factor = current_res / target_res
        
        target_h = int(base_arr.shape[0] * scale_factor)
        target_w = int(base_arr.shape[1] * scale_factor)
        target_shape = (target_h, target_w)
        
        logger.info(f"Input Resolution Processing: Native ~{current_res:.2f}m -> Target {target_res}m. Shape: {target_shape}")

        # Determine Processing Baseline for Normalization
        offset = 0.0
        if scene.scene_directory:
            try:
                name = scene.scene_directory.name
                import re
                match = re.search(r'_N(\d{4})_', name)
                if match:
                    pb = int(match.group(1))
                    if pb >= 400: # N0400
                        offset = 1000.0
            except Exception:
                pass

        for band_def in SENTINEL2_BANDS:
            bname = band_def['name']
            if bname not in scene.bands:
                raise ValueError(f"Missing required band for SEnSeIv2: {bname}")
            
            band_arr = scene.bands[bname]
            if band_arr.ndim == 3: band_arr = band_arr[0]
            
            # Resize using bilinear interpolation (order=1) to target resolution
            if band_arr.shape != target_shape:
                 band_arr = resize(band_arr, target_shape, preserve_range=True, order=1).astype(band_arr.dtype)
            
            # Normalize to 0-1
            arr_float = band_arr.astype(np.float32)
            if offset > 0:
                 arr_float = (arr_float - offset) / 10000.0
            else:
                 arr_float = arr_float / 10000.0
                 
            data_list.append(arr_float)
            
        # Stack to (N_bands, H, W)
        input_data = np.stack(data_list, axis=0)
        
        # 2. Run Inference
        logger.info("Running SEnSeIv2 inference...")
        mask_out = self.cm(input_data, descriptors=SENTINEL2_DESCRIPTORS)
        
        # 3. Process Output
        if mask_out.ndim == 3 and mask_out.shape[0] == 1: 
             mask_out = mask_out[0]
             
        # If float, we assume probabilities, but if output_style is 4-class, we expect ints?
        # SEnSeIv2 returns argmax if categorise=True (which is default inside CloudMask if not specified differently).
        # We want the raw classes (0,1,2,3) usually.
        # If mask_out is float, maybe it's probabilities? 
        # Let's ensure integer output for the "raw" mask if it looks like classes.
        if "class" in self.output_style and np.issubdtype(mask_out.dtype, np.floating):
            # If it's floating point but 4-class, it might be prob map?
            # Or it might be just cast to float. 
            # If max value <= 3, just cast.
            mask_out = mask_out.astype(np.uint8)

        # Update transform for new resolution
        # scale transform
        # Affine.scale(sx, sy) * transform ??
        # Or just adjust pixel size.
        # transform = [a, b, c, 
        #              d, e, f]
        # a = pixel width, e = pixel height (neg)
        new_transform = list(scene.transform)
        # Adjust pixel size
        # We need to preserve the corner, so c and f stay same (if top-left aligned)
        # We just change pixel size
        # scale_factor > 1 means finer res (smaller pixels? No, current_res / target_res)
        # if current is 10, target is 20, scale = 0.5. shape halves. pixel size doubles.
        # So new pixel size = old pixel size / scale_factor
        
        # Using rasterio.Affine
        from rasterio.transform import Affine
        if isinstance(scene.transform, Affine):
            # scale transform
            # To get from new grid to geocoords:
            # We stepped by (1/scale_factor) in pixel space relative to old.
            # So new pixel is (1/scale_factor) times larger?
            # Wait. Target 20m. Current 10m. Scale 0.5.
            # New pixel 0 covers Old pixels 0,1.
            # New pixel width should be 20. Old was 10.
            # So we multiply pixel size by (target_res / current_res) = 1/scale_factor
            
            s = 1.0 / scale_factor
            new_affine = scene.transform * Affine.scale(s, s)
        else:
            # Fallback for tuple
             new_affine = scene.transform 

        out = CloudMaskData(
             data=mask_out,
             transform=new_affine,
             crs=scene.crs,
             metadata=CloudMaskMetadata(
                 method="senseiv2",
                 model=self.model_name,
                 resolution=target_res
             )
        )
        return out

    def postprocess(self, mask_data: CloudMaskData, params: PostProcessParams) -> CloudMaskData:
        """
        Refines the cloud mask based on specific requirements (resolution, buffering, classes).
        """
        mask = mask_data.data.copy()
        current_transform = mask_data.transform
        
        # 1. Resize if needed
        if params.output_resolution is not None:
             # Calculate target shape
             # We can't trust mask_data.metadata['resolution'] to be present/accurate always? 
             # Use transform to guess current res.
             current_res_x = current_transform[0]
             current_res = abs(current_res_x)
             
             target_res = float(params.output_resolution)
             
             if abs(current_res - target_res) > 0.1: # simple float tolerance
                  scale = current_res / target_res
                  new_h = int(mask.shape[0] * scale)
                  new_w = int(mask.shape[1] * scale)
                  
                  logger.info(f"Post-processing: Resizing mask {mask.shape} -> {(new_h, new_w)}")
                  # Nearest neighbor for categorical/binary
                  mask = resize(mask, (new_h, new_w), order=0, preserve_range=True).astype(np.uint8)
                  
                  # Adjust transform
                  s = 1.0 / scale
                  from rasterio.transform import Affine
                  if isinstance(current_transform, Affine):
                      current_transform = current_transform * Affine.scale(s, s)
        
        # 2. Apply Class Selection & Binarization
        if params.binary_mask:
             # If mask is categorical (e.g. 0,1,2,3), select classes
             # Check if mask has values outside 0,1 first?
             # Assuming input is categorical
             
             # Create binary
             binary = np.isin(mask, params.classes_to_mask).astype(np.uint8)
             mask = binary
             
        # 3. Buffering (Dilation)
        if params.buffer_size > 0:
             # Need resolution to convert buffer_size (meters) to pixels
             if params.output_resolution:
                  res = params.output_resolution
             else:
                  res = abs(current_transform[0])
                  
             pixels = int(params.buffer_size / res)
             if pixels > 0:
                  logger.info(f"The mask will be buffered by {pixels} pixels ({params.buffer_size}m)")
                  # Use disk for isotropic buffering
                  selem = disk(pixels)
                  mask = dilation(mask, selem)
        
        # Create updated metadata with postprocessed flag
        updated_metadata = CloudMaskMetadata(
            **mask_data.metadata.model_dump(),
            postprocessed=True
        )

        return CloudMaskData(
             data=mask,
             transform=current_transform,
             crs=mask_data.crs,
             metadata=updated_metadata
        )

