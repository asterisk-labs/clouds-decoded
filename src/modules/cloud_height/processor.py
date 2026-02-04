import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import os
import tempfile
import pickle
import logging
from rasterio.transform import Affine


# Standardized Imports
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata
from .data import ColumnExtractor, ColumnIterator, RetrievalCube
from .physics import heightsToOffsets
from .config import CloudHeightConfig
from clouds_decoded.constants import BAND_RESOLUTIONS

# Module-level logger setup
logger = logging.getLogger(__name__)

class CloudHeightProcessor:
    def __init__(self, config: CloudHeightConfig):
        """
        Initializes the CloudHeightProcessor with a configuration object.
        """
        self.config = config
        self.final_heights = None
        self.final_coords = None
        self.final_gridded_heights = None
        self.gaussian_kernel = self._constructGaussianKernel()

    def process(self, scene: Sentinel2Scene) -> CloudHeightGridData:
        """
        Main processing method.
        Args:
            scene: The standardized Sentinel2Scene object.
        Returns:
            CloudHeightGridData: The result containing the height map.
        """
        logger.info(f"Processing scene: {scene.scene_directory}")
        self.scene = scene # Helper for postprocess
        
        # Override config scene_dir if needed, or trust the passed scene object?
        # Ideally we trust the scene object. The locally stored config might have a path, but the object is source of truth.
        
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
            logger.info("Using temporary directory: %s", temp_dir)

            
            # Using specific value for max points buffer
            max_points = int(109800**2 / self.config.stride**2)*2 
            
            N_heights = len(self.config.heights)
            heights_buffer = np.zeros((max_points, N_heights), dtype=np.float32)

            coords_buffer = np.zeros((max_points, 2), dtype=np.float32)
            count = 0
            
            column_extractor = ColumnExtractor(scene, self.config)
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
                 # Return empty result
                 scale_factor = self.config.stride / BAND_RESOLUTIONS['B02']
                 t = self.scene.transform * Affine.scale(scale_factor, scale_factor)
                 return CloudHeightGridData(data=None, transform=t, crs=self.scene.crs)

            self.final_heights = heights_buffer[:count]
            self.final_coords = coords_buffer[:count]

        # Post-Processing
        self.postprocess()

        # Construct Output Object
        # Note: final_gridded_heights is populated by postprocess
        # We need to attach CRS and Transform. 
        # Sentinel2Scene likely knows its native CRS/Transform, but we might have regridded?
        # The grid inside postprocess uses 'grid_stride'. 
        # If grid_stride != 1, resolution changed.
        
        # Prepare Metadata
        meta_dict = self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.dict()
        meta = CloudHeightMetadata(processing_config=meta_dict)

        # Calculate Transform and CRS
        if self.scene.transform is None:
             raise ValueError("Scene transform is missing. Ensure input scene is georeferenced.")

        # Calculate scaling factor based on ratio of output stride to reference band resolution
        scale_factor = self.config.stride / BAND_RESOLUTIONS['B02']
        transform = self.scene.transform * Affine.scale(scale_factor, scale_factor)
        crs = self.scene.crs
        
        result_data = CloudHeightGridData(
            data=self.final_gridded_heights,
            metadata=meta,
            transform=transform,
            crs=crs
        )
        return result_data

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

        # Create a brightness mask if required
        mask = column.getMask(self.config.threshold_band, self.config.cloudy_thresh) if brightness_mask else None

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
            if brightness_mask and mask[centre, centre_x] == 0:
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
            
        # Correlate
        corr = 0
        n = 0
        for i, patch in enumerate(patches):
            for other_patch in patches[i+1:]:
                corr += np.mean(patch * other_patch)
                n += 1
        return corr / n if n > 0 else 0

    def _constructGaussianKernel(self):
        """
        Constructs a Gaussian kernel based on the configuration parameters.
        """
        along_size = self.config.convolved_size_along_track // self.config.along_track_resolution
        across_size = self.config.convolved_size_across_track // self.config.across_track_resolution

        print(f"Constructing Gaussian kernel of size {along_size} x {across_size}...")

        x = np.linspace(-across_size // 2, across_size // 2, across_size)
        y = np.linspace(-along_size // 2, along_size // 2, along_size)
        X, Y = np.meshgrid(x, y)

        # Set sigma as half the kernel size
        sigma_x = across_size / 2
        sigma_y = along_size / 2

        kernel = np.exp(-((X**2) / (2 * sigma_x**2) + (Y**2) / (2 * sigma_y**2)))
        kernel /= np.mean(kernel) # Normalize to have mean of 1, so pearson correlation is unaffected
        return kernel

    def postprocess(self):         
        """
        Applies spatial smoothing to the retrieved height correlation scores.
        """
        logger.info("Post-processing...")

        retrievalcube = RetrievalCube(self.final_heights, self.final_coords, self.config)
        retrievalcube.createRtree()

        grid_stride = self.config.stride
        smoothing_sigma = self.config.spatial_smoothing_sigma

        # Use actual scene dimensions
        ref_band = self.config.reference_band
        ref_res = BAND_RESOLUTIONS[ref_band]
        height_m = self.scene.bands[ref_band].shape[0] * ref_res
        width_m = self.scene.bands[ref_band].shape[1] * ref_res

        grid_x, grid_y = np.meshgrid(np.arange(0, width_m, grid_stride), np.arange(0, height_m, grid_stride))
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        smoothed_scores_list = []
        
        with tqdm(total=len(grid_points), desc="Smoothing points") as pbar:
            for point in grid_points:
                coords, scores = retrievalcube.queryRadius(point, smoothing_sigma * 2)
                
                if len(coords) == 0:
                    smoothed_scores_list.append(np.nan * np.ones_like(self.config.heights))
                    pbar.update(1)
                    continue

                dists = np.linalg.norm(coords - point, axis=1)

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
        final_heights = self.config.heights[np.argmax(smoothed_scores, axis=1)]
        
        # Remove NaNs that may have resulted from empty neighborhoods
        valid_mask = ~np.isnan(final_heights) & (final_heights > 0)
        self.final_heights = final_heights[valid_mask]
        self.final_coords = grid_points[valid_mask]

        final_gridded_heights = np.nan*np.ones_like(grid_x)
        final_gridded_coords_stride = grid_points[valid_mask] // grid_stride
        

        final_gridded_heights[final_gridded_coords_stride[:, 1], final_gridded_coords_stride[:, 0]] = self.final_heights
        self.final_gridded_heights = final_gridded_heights

if __name__ == '__main__':
    # Barebones CLI for backward compatibility
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = CloudHeightConfig(args.config, args.scene_dir)
    processor = CloudHeightProcessor(config)
    
    # Create Scene manually here
    scene = Sentinel2Scene.from_file(args.scene_dir)
    result = processor.process(scene)
    
    print(f"Processed scene. Max Height: {np.nanmax(result.data)}")
