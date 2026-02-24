import numpy as np
from tqdm import tqdm
import multiprocessing
import os
import tempfile
import pickle
import logging
from rasterio.transform import Affine


# Standardized Imports
from typing import Optional, Union
from pathlib import Path
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata, CloudMaskData
from clouds_decoded.base_processor import BaseProcessor
from .data import ColumnExtractor, ColumnIterator, RetrievalCube
from .physics import heightsToOffsets
from .config import CloudHeightConfig
from clouds_decoded.constants import BAND_RESOLUTIONS

# Module-level logger setup
logger = logging.getLogger(__name__)

class CloudHeightProcessor(BaseProcessor):
    def __init__(self, config: CloudHeightConfig):
        """
        Initializes the CloudHeightProcessor with a configuration object.
        """
        self.config = config
        self.gaussian_kernel = self._constructGaussianKernel()

    def _process(self, scene: Sentinel2Scene, cloud_mask: Optional[Union[CloudMaskData, np.ndarray, str, Path]] = None) -> CloudHeightGridData:
        """
        Main processing method.
        Args:
            scene: The standardized Sentinel2Scene object.
            cloud_mask: Optional cloud mask to restrict processing to cloudy pixels.
                        Can be CloudMaskData object, numpy array, or path to file.
                        If provided, only pixels where mask > 0 are processed.
        Returns:
            CloudHeightGridData: The result containing the height map.
        """
        logger.info(f"Processing scene: {scene.scene_directory}")
        
        # Resolve mask
        mask_array = None
        if cloud_mask is not None:
             logger.info("Using provided cloud mask.")
             if isinstance(cloud_mask, (str, Path)):
                  try:
                       # Attempt to load as CloudMaskData
                       cm_obj = CloudMaskData.from_file(str(cloud_mask))
                       mask_array = cm_obj.data
                  except (FileNotFoundError, OSError, ValueError, TypeError) as e:
                       logger.warning(f"Could not load {cloud_mask} as CloudMaskData ({e}). Ignoring mask.")
             elif isinstance(cloud_mask, CloudMaskData):
                  mask_array = cloud_mask.data
             elif isinstance(cloud_mask, np.ndarray):
                  mask_array = cloud_mask
             
             # Ensure mask is 2D
             if mask_array is not None:
                  if mask_array.ndim == 3:
                       mask_array = mask_array[0]
                  
                  # If we have a mask, ensure it matches scene dimensions? 
                  # ColumnExtractor's interpolator handles this somewhat, but resolution mismatch might be weird.
                  # Assuming 10m based mask for now.

        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
            logger.info("Using temporary directory: %s", temp_dir)

            

            width_m, height_m = scene.get_scene_size_meters()
            max_points = int((width_m * height_m) / (self.config.stride ** 2) * 2)
            logger.info(f"Initialized buffer for up to {max_points} points for scene size {width_m:.0f}m x {height_m:.0f}m")
            
            N_heights = len(self.config.heights)
            heights_buffer = np.zeros((max_points, N_heights), dtype=np.float32)

            coords_buffer = np.zeros((max_points, 2), dtype=np.float32)
            count = 0
            
            # Pass mask to extractor
            column_extractor = ColumnExtractor(scene, self.config, mask=mask_array)
            column_iterator = ColumnIterator(
                column_extractor, n_workers=self.config.n_workers, temp_dir=temp_dir)
            
            # Worker and Queue logic
            result_queue = multiprocessing.Queue()
            workers = []
            for _ in range(self.config.n_workers):
                p = multiprocessing.Process(target=self._worker_job, args=(column_iterator.queue, result_queue))
                workers.append(p)
                p.start()

            total_columns = len(column_iterator)
            with tqdm(total=total_columns, desc="Processing Columns") as pbar:
                for _ in range(total_columns):
                    result = result_queue.get()
                    if 'error' in result:
                        logger.error(result['error'])
                    else:
                        retrievals = result.get('retrievals')
                        if retrievals and len(retrievals) > 0:
                            num = len(retrievals)
                            if count + num > max_points:
                                logger.warning("Exceeded buffer space")
                            else:
                                heights_buffer[count:count+num] = retrievals
                                coords_buffer[count:count+num] = result.get('coords')
                                count += num
                                
                    pbar.update(1)

            for p in workers:
                p.join()
                
            if count == 0:
                 logger.warning("No valid points retrieved. Process aborting.")
                 # Return empty result with correct transform
                 ref_band = self.config.reference_band
                 ref_shape = scene.bands[ref_band].shape
                 actual_res = abs(scene.transform.a)
                 ref_res = BAND_RESOLUTIONS[ref_band]
                 geo_pixel_size = self.config.stride * actual_res / ref_res
                 t = Affine(geo_pixel_size, 0, scene.transform.c, 0, -geo_pixel_size, scene.transform.f)
                 return CloudHeightGridData(data=None, transform=t, crs=scene.crs)

            final_heights = heights_buffer[:count]
            final_coords = coords_buffer[:count]

        # Post-Processing
        final_gridded_heights = self._smooth_and_grid(final_heights, final_coords, scene, mask_array)

        # Construct Output Object
        
        # Prepare Metadata
        meta_dict = self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.dict()
        meta = CloudHeightMetadata(processing_config=meta_dict)

        # Calculate Transform and CRS
        if scene.transform is None:
             raise ValueError("Scene transform is missing. Ensure input scene is georeferenced.")

        # Construct output transform: map grid pixels to actual geographic extent
        # The grid is built in "internal meters" (BAND_RESOLUTIONS-based) but
        # the transform must map to actual geographic coordinates (scene.transform-based).
        ref_band = self.config.reference_band
        actual_res = abs(scene.transform.a)
        ref_res = BAND_RESOLUTIONS[ref_band]
        geo_pixel_size = self.config.stride * actual_res / ref_res
        transform = Affine(geo_pixel_size, 0, scene.transform.c, 0, -geo_pixel_size, scene.transform.f)
        crs = scene.crs
        
        return CloudHeightGridData(
            data=final_gridded_heights,
            metadata=meta,
            transform=transform,
            crs=crs
        )

    def _worker_job(self, data_queue, result_queue):
        while True:
            path = data_queue.get()
            if path is None: break
            if path == "EMPTY_COLUMN":
                result_queue.put({'retrievals': [], 'coords': []})
                continue
            try:
                with open(path, 'rb') as f:
                    col = pickle.load(f)
                retrievals, retrieved_coords = self._processColumn(col)
                result_queue.put({'retrievals': retrievals, 'coords': retrieved_coords})
            except Exception as e:
                result_queue.put({'error': str(e)})
            finally:
                if os.path.exists(path): os.remove(path)

    def _processColumn(self, column, brightness_mask=True):
        """
        Processes a single column of pixels to retrieve cloud heights.
        """
        # Unpack configuration for clarity
        along_track_stride = self.config.stride // self.config.along_track_resolution
        along_track_size = self.config.convolved_size_along_track // self.config.along_track_resolution

        # Use the interpolated mask from the column if available
        mask = column.mask if (brightness_mask and column.mask is not None) else None

        target_features = column.bands

        # Determine the range of centers to process based on max parallax offset
        max_offset_val = heightsToOffsets([self.config.max_height] * len(target_features), target_features.keys(), self.config.along_track_resolution)
        max_offset = int(np.ceil(max_offset_val.max()))
        
        shape_0 = column.bands[self.config.reference_band].shape[0]
        if column.direction == 'up':
            centres = np.arange(along_track_size // 2, shape_0 - max_offset - along_track_size // 2, along_track_stride)
        else:
            centres = np.arange(along_track_size // 2 + max_offset, shape_0 - along_track_size // 2, along_track_stride)

        # Extract coordinates of the centers
        centre_x = target_features[self.config.reference_band].shape[1] // 2
        extracted_coords = column.points[centres, centre_x, :]
        
        retrievals = []
        retrieved_coords = []
        for centre, coord in zip(centres, extracted_coords):
            if mask is not None and mask[centre, centre_x] == 0:
                continue
            
            # Call to _correlateAtHeights is now cleaner
            scores = self._correlateAtHeights(target_features, centre, column.direction)
            
            retrievals.append(scores)
            retrieved_coords.append(coord)
            
        return retrievals, retrieved_coords
    
    def _correlateAtHeights(self, footprint_bands, centre, direction):
        along_track_size = self.config.convolved_size_along_track // self.config.along_track_resolution
        scores = [self._correlateAtHeight(footprint_bands, centre, along_track_size, h, direction) for h in self.config.heights]
        return np.array(scores)

    def _correlateAtHeight(self, footprint_bands, centre, along_track_size, height, direction):
        pixel_size = self.config.along_track_resolution
        offsets = heightsToOffsets([height] * len(footprint_bands), footprint_bands.keys(), pixel_size=pixel_size)
        if direction == 'up': offsets = -offsets
        
        patches = []
        for i, (name, data) in enumerate(footprint_bands.items()):
            offset = offsets[i]
            if np.isnan(offset): continue
            
            start = int(centre - along_track_size/2 - offset)
            end = start + along_track_size
            if start < 0 or end > data.shape[0]: 
                return 0 # Boundary check
            
            patch = data[start:end, :]
            # Normalize
            # Inline normalization logic check
            if np.std(patch) > 0:
                 patch = (patch - np.mean(patch)) / np.std(patch)
            else:
                 patch = patch - np.mean(patch)
                 
            patches.append(patch)
            
        # Gaussian-weighted cross-correlation
        corr = 0
        n = 0
        for i, patch in enumerate(patches):
            for other_patch in patches[i+1:]:
                corr += np.mean(patch * other_patch * self.gaussian_kernel)
                n += 1
        return corr / n if n > 0 else 0

    def _constructGaussianKernel(self):
        """
        Constructs a Gaussian kernel based on the configuration parameters.
        """
        along_size = self.config.convolved_size_along_track // self.config.along_track_resolution
        across_size = self.config.convolved_size_across_track // self.config.across_track_resolution

        logger.info(f"Constructing Gaussian kernel of size {along_size} x {across_size}")

        x = np.linspace(-across_size // 2, across_size // 2, across_size)
        y = np.linspace(-along_size // 2, along_size // 2, along_size)
        X, Y = np.meshgrid(x, y)

        # Set sigma as half the kernel size
        sigma_x = across_size / 2
        sigma_y = along_size / 2

        kernel = np.exp(-((X**2) / (2 * sigma_x**2) + (Y**2) / (2 * sigma_y**2)))
        kernel /= np.mean(kernel) # Normalize to have mean of 1, so pearson correlation is unaffected
        return kernel

    def _smooth_and_grid(self, heights, coords, scene, mask_array=None):      
        """
        Applies spatial smoothing to the retrieved height correlation scores and grids them.
        """
        logger.info("Smoothing and gridding points...")

        retrievalcube = RetrievalCube(heights, coords, self.config)
        retrievalcube.createRtree()

        grid_stride = self.config.stride
        smoothing_sigma = self.config.spatial_smoothing_sigma

        # Use actual scene dimensions
        ref_band = self.config.reference_band
        ref_res = BAND_RESOLUTIONS[ref_band]
        height_m = scene.bands[ref_band].shape[0] * ref_res
        width_m = scene.bands[ref_band].shape[1] * ref_res

        grid_x, grid_y = np.meshgrid(np.arange(0, width_m, grid_stride), np.arange(0, height_m, grid_stride))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        smoothed_scores_list = []

        # Determine valid grid points based on mask
        valid_mask_indices = np.ones(len(grid_points), dtype=bool)
        if mask_array is not None:
             mask_h, mask_w = mask_array.shape
             res_y = height_m / mask_h
             res_x = width_m / mask_w
             
             idx_x = (grid_points[:, 0] / res_x).astype(int)
             idx_y = (grid_points[:, 1] / res_y).astype(int)
             
             idx_x = np.clip(idx_x, 0, mask_w - 1)
             idx_y = np.clip(idx_y, 0, mask_h - 1)
             
             valid_mask_indices = mask_array[idx_y, idx_x] > 0
        
        with tqdm(total=len(grid_points), desc="Smoothing points") as pbar:
            for i, point in enumerate(grid_points):
                # Skip if masked out (clear sky)
                if not valid_mask_indices[i]:
                    smoothed_scores_list.append(np.nan * np.ones_like(self.config.heights))
                    pbar.update(1)
                    continue

                coords_found, scores = retrievalcube.queryRadius(point, smoothing_sigma * 2)
                
                if len(coords_found) == 0:
                    smoothed_scores_list.append(np.nan * np.ones_like(self.config.heights))
                    pbar.update(1)
                    continue

                dists = np.linalg.norm(coords_found - point, axis=1)

                if np.min(dists) < smoothing_sigma * 1.5:
                    confs = np.max(scores, axis=1)**2
                    weights = np.exp(-0.5 * (dists / smoothing_sigma)**2) * confs
                    
                    if np.sum(weights) > 0:
                        weights /= np.sum(weights)
                        smoothed_value = np.sum(scores * weights[:, np.newaxis], axis=0)
                        smoothed_scores_list.append(smoothed_value)
                    else:
                        smoothed_scores_list.append(np.nan * np.ones_like(self.config.heights))
                else:
                    smoothed_scores_list.append(np.nan * np.ones_like(self.config.heights))
                pbar.update(1)

        smoothed_scores = np.array(smoothed_scores_list)
        
        # Mask where all correlations were NaN (to avoid ValueError in nanargmax)
        all_nan_mask = np.all(np.isnan(smoothed_scores), axis=1)
        
        # Temporarily fill all-NaN rows with 0 so nanargmax doesn't fail
        # We don't care what index is chosen for these rows, as we'll set the height to NaN later
        if np.any(all_nan_mask):
            smoothed_scores[all_nan_mask] = 0.0
        
        # Final Height Selection (argmax of smoothed correlation)
        final_height_indices = np.nanargmax(smoothed_scores, axis=1)
        final_gridded_heights = self.config.heights[final_height_indices]
        
        final_gridded_heights = final_gridded_heights.astype(np.float32)
        final_gridded_heights[all_nan_mask] = np.nan

        # Mark zero-height pixels as invalid — height=0 is the search lower
        # bound, not a physically meaningful cloud height.
        final_gridded_heights[final_gridded_heights <= 0] = np.nan

        # Reshape to grid
        return final_gridded_heights.reshape(grid_y.shape)