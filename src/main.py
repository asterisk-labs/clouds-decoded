import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import os
from typing import Callable
import tempfile
import pickle

from data import heightsToOffsets, Sentinel2Scene, ColumnExtractor, ColumnIterator, getBands, RetrievalCube
from config import CloudHeightConfig
from constants import BAND_RESOLUTIONS

# import warnings
# warnings.filterwarnings("error")


class CloudHeightProcessor:
    def __init__(self, config: CloudHeightConfig):
        """
        Initializes the CloudHeightProcessor with a configuration object.

        Parameters:
            config: CloudHeightConfig, the configuration for the processing run.
        """
        self.config = config
        self.scene = Sentinel2Scene(config.scene_dir, bands=config.bands)
        self.final_heights = None
        self.final_coords = None


    def _normalizePatch(self, patch, remove_mean=True, remove_var=True):
        """
        Normalizes a patch by removing its mean and scaling by its standard deviation.
        """

        # quick and dirty nan-check: look at the four corners of the patch
        if np.any(np.isnan([patch[0, 0], patch[0, -1], patch[-1, 0], patch[-1, -1]])):
            nan_check = True
        else:
            nan_check = False

        # take subsample because mean/std are not important to get perfect
        sample = patch[::4,::2]

        try:
            if nan_check:
                if remove_mean:
                    sample = sample - np.nanmean(sample)
                if remove_var:
                    sample = sample / np.nanstd(sample)
            else:
                if remove_mean:
                    sample = sample - np.mean(sample)
                if remove_var:
                    sample = sample / np.std(sample)
        except Warning as w:
            print(f"Warning during normalization: {w}.")
            import pdb; pdb.set_trace()

        return sample

    def _correlateAtHeight(self, footprint_bands, centre, along_track_size, height, direction, remove_mean=True, remove_var=True):
        """
        Calculates the correlation of all bands for a given height and center point.
        Pulls pixel_size and reference_band from the instance's config.
        """
        # Get pixel_size directly from config
        pixel_size = self.config.along_track_resolution
        offsets = heightsToOffsets([height] * len(footprint_bands), footprint_bands.keys(), pixel_size=pixel_size)

        if direction == 'up':
            offsets = -offsets

        patches = []
        ref_n = -1


        # Get reference_band directly from config
        for i, (name, data) in enumerate(footprint_bands.items()):
            if name == self.config.reference_band:
                ref_n = i
            offset = offsets[i]
            along_track_start = int(centre - along_track_size / 2 - offset)
            along_track_end = along_track_start + along_track_size

            patch = data[along_track_start:along_track_end, :]


            patch = self._normalizePatch(patch, remove_mean=remove_mean, remove_var=remove_var)

            patches.append(patch)
        
        # # Sum all non-reference patches and correlate with the reference patch
        # corr = 0
        # patch = patches[ref_n]
        # other_sum = np.zeros_like(patch)
        # for i, other_patch in enumerate(patches):
        #     if i != ref_n:
        #         other_sum += other_patch
    
        # # Normalize for Pearson correlation
        # other_sum /= (len(patches) - 1)

        # # Dot product and mean to get correlation
        # corr = np.mean(patch * other_sum)

        # Instead of above block, do all pairwise correlations and sum them
        corr = 0
        n = 0
        for i, patch in enumerate(patches):
            other_sum = np.zeros_like(patch)
            for other_patch in patches[i+1:]:
                other_sum += other_patch
                n += 1
            corr += np.mean(patch * other_sum)
        corr /= n


        return corr

    def _correlateAtHeights(self, footprint_bands, centre, direction):
        """
        Calculates correlation scores for a range of heights.
        Pulls height steps and resolution info from the instance's config.
        """
        # Calculate along_track_size directly from config
        along_track_size = self.config.convolved_size_along_track // self.config.along_track_resolution

        scores = [
            self._correlateAtHeight(
                footprint_bands,
                centre,
                along_track_size,
                height,
                direction,
            ) for height in self.config.heights
        ]
        return np.array(scores)

    def _processColumn(self, column, brightness_mask=True):
        """
        Processes a single column of pixels to retrieve cloud heights.
        """
        # Unpack configuration for clarity
        along_track_stride = self.config.stride // self.config.along_track_resolution
        along_track_size = self.config.convolved_size_along_track // self.config.along_track_resolution
        gradient_power_ratio = self.config.along_track_resolution / self.config.across_track_resolution

        # Create a brightness mask if required
        mask = column.getMask(self.config.threshold_band, self.config.cloudy_thresh) if brightness_mask else None

        # Select target features (reflectance or gradient)
        if self.config.target_features == "reflectance":
            target_features = column.bands
        elif self.config.target_features == "gradient":
            target_features = column.getGradients(gradient_power_ratio)
        else:
            raise ValueError(f"Invalid target_features: {self.config.target_features}")

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
            
            if self.config.smoothing_mode == 'independent':
                height = self.config.heights[np.argmax(scores)]
                retrievals.append(height)
            elif self.config.smoothing_mode == 'spatial':
                retrievals.append(scores)
            
            retrieved_coords.append(coord)
            
        return retrievals, retrieved_coords

    def _worker_job(self, data_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
        """
        Worker process function that gets a column file path, loads it, processes it, and cleans up.
        """
        while True:
            path = data_queue.get()

            if path is None:  # Sentinel value received
                break
            
            if path == "EMPTY_COLUMN":
                result_queue.put({'retrievals': [], 'coords': []})
                continue

            try:
                with open(path, 'rb') as f:
                    col = pickle.load(f)
                
                retrievals, retrieved_coords = self._processColumn(col, brightness_mask=True)
                result_queue.put({'retrievals': retrievals, 'coords': retrieved_coords})
            except Exception as e:
                result_queue.put({'error': f"Worker failed on {os.path.basename(path)}: {e}"})
            finally:
                if os.path.exists(path):
                    os.remove(path)

    def process(self):
        """
        Main processing method to run the entire cloud height retrieval workflow for the scene.
        """
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
            print(f"Using temporary directory for this run: {temp_dir}")
            
            column_extractor = ColumnExtractor(self.scene, self.config)
            column_iterator = ColumnIterator(column_extractor, n_workers=self.config.n_workers, temp_dir=temp_dir)

            max_points = int(109800**2 / self.config.stride**2)
            if self.config.smoothing_mode == 'spatial':
                N_heights = len(self.config.heights)
                heights_buffer = np.zeros((max_points, N_heights), dtype=np.float32)
            elif self.config.smoothing_mode == 'independent':
                heights_buffer = np.zeros(max_points, dtype=np.float32)
            else:
                raise ValueError(f"Invalid smoothing mode: {self.config.smoothing_mode}. Choose 'spatial' or 'independent'.")
            
            coords_buffer = np.zeros((max_points, 2), dtype=np.float32)
            count = 0
            
            result_queue = multiprocessing.Queue()
            total_columns = len(column_iterator)

            workers = []
            print(f"Spawning {self.config.n_workers} worker processes...")
            for _ in range(self.config.n_workers):
                p = multiprocessing.Process(target=self._worker_job, args=(column_iterator.queue, result_queue))
                workers.append(p)
                p.start()

            with tqdm(total=total_columns, desc="Processing Columns", smoothing=0.05) as pbar:
                for _ in range(total_columns):
                    result = result_queue.get()
                    
                    if 'error' in result:
                        print(f"Error from worker: {result['error']}")
                        pbar.update(1)
                        continue

                    retrievals = result.get('retrievals')
                    if retrievals and len(retrievals) > 0:
                        num_retrieved = len(retrievals)
                        if count + num_retrieved > max_points:
                            raise ValueError("Exceeded preallocated space for results.")
                        heights_buffer[count:count + num_retrieved] = retrievals
                        coords_buffer[count:count + num_retrieved] = result.get('coords')
                        count += num_retrieved
                    pbar.update(1)
            
            for p in workers:
                p.join()

            self.final_heights = heights_buffer[:count]
            self.final_coords = coords_buffer[:count]

        if self.config.smoothing_mode == 'spatial':
            self.postprocess()

    def postprocess(self):
        """
        Applies spatial smoothing to the retrieved height correlation scores.
        This method corresponds to the original `regridAndSmooth` function.
        """
        print("Starting post-processing: Regridding and Smoothing...")
        retrievalcube = RetrievalCube(self.final_heights, self.final_coords, self.config)
        retrievalcube.createRtree()

        grid_stride = self.config.stride
        smoothing_sigma = self.config.spatial_smoothing_sigma

        grid_x, grid_y = np.meshgrid(np.arange(0, 109800, grid_stride), np.arange(0, 109800, grid_stride))
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
        print("Post-processing complete.")

    def plot_results(self):
        """
        Generates and displays a plot of the retrieved cloud heights overlaid on an RGB image of the scene.
        """
        print("Generating plot...")
        rgb_bands = getBands(self.config.scene_dir, ['B04', 'B03', 'B02'])
        rgb = np.stack([rgb_bands['B04'], rgb_bands['B03'], rgb_bands['B02']], axis=-1)
        # Basic atmospheric correction and gamma correction for visualization
        rgb = np.clip(((rgb - 1000) / 10000)**0.7, 0, 1)

        resolution = BAND_RESOLUTIONS[self.config.reference_band]
        point_size = 6 * (self.config.stride / 500)
        
        fig, ax = plt.subplots(1, 1, figsize=(25, 25))
        ax.imshow(rgb, extent=[0, rgb.shape[1] * resolution, 0, rgb.shape[0] * resolution])

        # mask = self.final_heights > 0
        # heights = self.final_heights[mask]
        # coords = self.final_coords[mask]

        # heights_log = np.log10(heights)
        # scatter = ax.scatter(coords[:, 0], rgb.shape[0] * resolution - coords[:, 1], c=heights_log, cmap='jet', s=point_size, alpha=0.3)
        
        # use gridded heights if available
        heights_log = np.log10(np.clip(self.final_gridded_heights,500,16000))
        height_map = ax.imshow(heights_log, cmap='gist_ncar', alpha=0.33, extent=[0, rgb.shape[1] * resolution, 0, rgb.shape[0] * resolution])
        cbar = plt.colorbar(height_map, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log10(Height) (m)')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        if self.config.plot_writeto:
            print(f"Saving plot to {self.config.plot_writeto}")
            fig.savefig(self.config.plot_writeto)
        
        plt.show()


        fig, ax = plt.subplots(1, 1, figsize=(25, 25))
        ax.imshow(rgb, extent=[0, rgb.shape[1] * resolution, 0, rgb.shape[0] * resolution])

        # mask = self.final_heights > 0
        # heights = self.final_heights[mask]
        # coords = self.final_coords[mask]

        # heights_log = np.log10(heights)
        # scatter = ax.scatter(coords[:, 0], rgb.shape[0] * resolution - coords[:, 1], c=heights_log, cmap='jet', s=point_size, alpha=0.3)
        
        # use gridded heights if available
        heights_log = self.final_gridded_heights
        height_map = ax.imshow(heights_log, cmap='prism', alpha=0.33, extent=[0, rgb.shape[1] * resolution, 0, rgb.shape[0] * resolution], vmin=0, vmax=50000)
        cbar = plt.colorbar(height_map, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Height (m)')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        if self.config.plot_writeto:
            print(f"Saving plot to {self.config.plot_writeto.replace('.png','_linear.png')}")
            fig.savefig(self.config.plot_writeto.replace('.png','_linear.png'))

        plt.show()

    def save_results(self, output_path: str):
        """
        Saves the final retrieved heights and coordinates to a compressed NumPy file (.npz).
        
        Parameters:
            output_path: The path to save the output file.
        """
        if self.final_heights is not None and self.final_coords is not None:
            np.savez_compressed(output_path, heights=self.final_heights, coords=self.final_coords, grid=self.final_gridded_heights)
            print(f"Results saved to {output_path}")
        else:
            print("No results to save.")

def correlateAtHeight(footprint_bands,centre,along_track_size,height,direction,pixel_size,remove_mean=True,remove_var=True,reference_band='B02'):
    # Find the correlation of all the data at a given height around a given centre point along a column
    # The centre is in the rotated footprint pixel coordinates, so that everything here is simple and efficient
    offsets = heightsToOffsets([height]*len(footprint_bands),footprint_bands.keys(),pixel_size=pixel_size)

    if direction == 'up':
        offsets = -offsets

    patches = []
    for i,band in enumerate(footprint_bands.items()):
        name,data = band
        if name == reference_band:
            ref_n = i
        offset = offsets[i]
        along_track_start = int(centre - along_track_size/2 - offset)
        along_track_end = along_track_start + along_track_size

        patch = data[along_track_start:along_track_end,:]

        if remove_mean:
            patch = ( patch - np.nanmean(patch) )
        if remove_var:
            patch = patch / np.nanstd(patch)

        patches.append(patch)

    # if not np.any(np.isnan(np.array(patches))):
    #     np.save('scratch/debug_patches.npy',np.array(patches))
    #     input(' debug patches saved, press enter to continue ')

    # Now we have the patches, we can correlate them. We want all pairwise correlations at each pixel summed
    corr = 0
    patch = patches[ref_n]

    # Instead of doing N-1 correlations, we can just sum all the other patches and do one correlation with that
    other_sum = np.zeros_like(patch)
    for i,other_patch in enumerate(patches):
        if i != ref_n:
            other_sum += other_patch

    # Normalize so that Pearson correlation is between -1 and 1
    other_sum /= (len(patches)-1)

    # Dot product and mean to get correlation
    corr = np.mean(patch * other_sum)

    return corr

def correlateAtHeights(footprint_bands,centre,along_track_size,heights,direction,pixel_size,reference_band='B02'):
    # Find the correlation of all the data at a given height around a given centre along a column
    # The centre point is in the rotated footprint pixel coordinates, so that everything here is simple and efficient

    scores = []
    for height in heights:
        scores.append(
            correlateAtHeight(
                footprint_bands,
                centre,along_track_size,
                height,
                direction,
                pixel_size,
                reference_band=reference_band
            )
        )
    return np.array(scores)

def processColumn(
        column,
        config,
        brightness_mask=True,
    ):
    """
    Takes the column of pixels, assumed to be fairly thin across.

    Parameters:
        column: Column object
        config: CloudHeightConfig, Configuration object
        brightness_mask: bool, Whether to use a mask for brightness

    Returns:
        retrievals: np.array, Array of retrieved heights, or scores at each height if spatial smoothing is used
        retrieved_coords: np.array, Array of coordinates of the retrieved heights
    """

    # Unpack the configuration
    reference_band = config.reference_band
    convolved_size_along_track = config.convolved_size_along_track
    along_track_resolution = config.along_track_resolution
    across_track_resolution = config.across_track_resolution
    along_track_stride = config.stride
    heights = config.heights
    max_height = config.max_height
    cloudy_thresh = config.cloudy_thresh
    threshold_band = config.threshold_band
    
    along_track_size = convolved_size_along_track // along_track_resolution 
    along_track_stride = along_track_stride // along_track_resolution
    gradient_power_ratio = along_track_resolution / across_track_resolution

    # Create mask for brightness
    if brightness_mask:
        mask = column.getMask(threshold_band,cloudy_thresh)
        
    # Take gradients of the bands
    if config.target_features == "reflectance":
        target_features = column.bands
    elif config.target_features == "gradient":
        target_features = column.getGradients(gradient_power_ratio)
    # TODO: Check if block is actually finding correct range that heights should be calculated for...

    # Find the max offset that we need to consider
    max_offset = heightsToOffsets(
        [max_height]*len(target_features),
        target_features.keys(),
        along_track_resolution
    )
    max_offset = int(np.ceil(max_offset.max()))
    if column.direction == 'up':
        centres = np.arange(
            along_track_size//2,
            column.bands[reference_band].shape[0] - along_track_size//2 - max_offset,
            along_track_stride
            )
    else:
        centres = np.arange(
            along_track_size//2 + max_offset,
            column.bands[reference_band].shape[0]- along_track_size//2,
            along_track_stride
            )

    # Extract the coordinates of the centres
    width = target_features[reference_band].shape[1]
    centre_x = width//2
    extracted_coords = column.points[centres,centre_x,:]
    retrievals = []
    retrieved_coords = []
    for centre,coord in zip(centres,extracted_coords):
        # Check if reflectance is > threshold in the reference band
        if brightness_mask:
            if mask[centre,centre_x] == 0:
                continue
        scores = correlateAtHeights(
            target_features,
            centre,
            along_track_size,
            heights,
            column.direction,
            along_track_resolution,
            reference_band=reference_band
        )
        if config.smoothing_mode == 'independent':
            height = heights[np.argmax(scores)]
            retrievals.append(height)
        elif config.smoothing_mode == 'spatial':
            retrievals.append(scores)
        retrieved_coords.append(coord)
    return retrievals,retrieved_coords

def ProcessScene(config_file=None):
    """
    Same as ProcessScene but the columns are put into a threadpoolexecutor
    """

    """
    Process a scene, extracting the heights at each point

    Parameters:
        config_file: str or CloudHeightConfig, Path to the configuration file or configuration object

    Returns:
        final_heights: np.array, Array of retrieved heights
        final_coords: np.array, Array of coordinates of the retrieved heights
    """
    if isinstance(config_file,str) or config_file is None:
        conf = CloudHeightConfig(config_file=config_file)
    elif isinstance(config_file,CloudHeightConfig):
        conf = config_file
    else:
        raise ValueError("config_file must be a string or CloudHeightConfig object")

    # Use a context manager for a temporary directory in the RAM disk
    with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
        print(f"Using temporary directory for this run: {temp_dir}")
        scene = Sentinel2Scene(conf.scene_dir)
        # The ColumnExtractor now handles prefetching in a background thread.
        column_extractor = ColumnExtractor(scene,conf)
        column_iterator = ColumnIterator(column_extractor, n_workers=conf.n_workers, temp_dir=temp_dir)

        # The 'extraction' time is now happening in the background, so we only time processing.
        
        max_points = int(109800**2 / conf.stride**2)
        if conf.smoothing_mode == 'spatial':
            N_heights = len(conf.heights)
            final_heights = np.zeros((max_points, N_heights), dtype=np.float32)
        elif conf.smoothing_mode == 'independent':
            final_heights = np.zeros(max_points, dtype=np.float32)
        else:
            raise ValueError(f"Invalid smoothing mode: {conf.smoothing_mode}. Choose 'spatial' or 'independent'.")
        final_coords = np.zeros((max_points, 2), dtype=np.float32)
        count = 0

        def _worker_job(
            worker_id: int, 
            data_queue: multiprocessing.Queue,
            result_queue: multiprocessing.Queue, 
            conf: CloudHeightConfig, 
            processColumn: Callable
        ):
            """Worker gets a path, loads the file, processes it, and cleans up."""
            while True:
                path = data_queue.get()

                if path is None: # Sentinel value received
                    break
                
                if path == "EMPTY_COLUMN":
                    result_queue.put({'retrievals': [], 'coords': []})
                    continue

                try:
                    # Load the object from the RAM disk
                    with open(path, 'rb') as f:
                        # 'col' is now a complete Column object.
                        col = pickle.load(f) 
                    
                    retrievals, retrieved_coords = processColumn(
                        col,
                        conf,
                        brightness_mask=True
                    )
                    result_queue.put({'retrievals': retrievals, 'coords': retrieved_coords})
                except Exception as e:
                    result_queue.put({'error': f"Worker {worker_id} failed on {os.path.basename(path)}: {e}"})
                finally:
                    # CRITICAL: Clean up the file immediately after use
                    if os.path.exists(path):
                        os.remove(path)

        # Create the result queue for inter-process communication
        result_queue = multiprocessing.Queue()
        total_columns = len(column_iterator)

        # Create and start the worker processes
        workers = []
        print(f"Spawning {conf.n_workers} worker processes...")
        for i in range(conf.n_workers):
            p = multiprocessing.Process(
                target=_worker_job,
                args=(i, column_iterator.queue, result_queue, conf, processColumn)
            )
            workers.append(p)
            p.start()

        # Process results from the queue as they come in
        with tqdm(total=total_columns, desc="Processing Columns",smoothing=0.05) as pbar:
            processed_count = 0
            while processed_count < total_columns:
                result = result_queue.get()
                
                if 'error' in result:
                    # Handle errors from the worker processes
                    print(f"Error from worker: {result['error']}")
                    pbar.update(1)
                    processed_count += 1
                    continue

                retrievals = result.get('retrievals')
                retrieved_coords = result.get('coords')

                if retrievals is not None and len(retrievals) > 0:
                    num_retrieved = len(retrievals)
                    if count + num_retrieved > max_points:
                        raise ValueError("Exceeded preallocated space for heights and coordinates.")
                    final_heights[count:count + num_retrieved] = retrievals
                    final_coords[count:count + num_retrieved] = retrieved_coords
                    count += num_retrieved
                pbar.update(1)
                processed_count += 1

        # Wait for all worker processes to finish
        for p in workers:
            p.join()

    final_heights = final_heights[:count]
    final_coords = final_coords[:count]
    
    final_heights,final_coords = np.array(final_heights),np.array(final_coords)

    # # # Debug save
    # np.save('debug_heights_refl_colmeanvar_s100.npy',final_heights)
    # np.save('debug_coords_refl_colmeanvar_s100.npy',final_coords)

    # # Load from debug files
    # final_heights,final_coords = np.load('debug_heights_refl_colmeanvar_s100.npy'),np.load('debug_coords_refl_colmeanvar_s100.npy')

    print(final_heights.shape, final_coords.shape)

    if conf.smoothing_mode == 'spatial':
        final_heights, final_coords = regridAndSmooth(final_heights, final_coords, conf)

    # return final_heights,final_coords,times

def regridAndSmooth(height_scores, coords, config):
    """
    Smooth the correlation scores spatially to get final heights. Can use gaussian or median smoothing.

    Tricky thing is that we're not using a regular grid, so we have to do a nearest neighbour search to find points within a certain radius.

    First we build a KDTree of the points, then build a regular grid of sampling points. For each sampling point, we find all points within the smoothing radius, and average their scores by using a weighted average (gaussian) or median.
    """
    retrievalcube = RetrievalCube(height_scores, coords, config)
    retrievalcube.createRtree()

    grid_stride = config.stride // 2

    grid_x, grid_y = np.meshgrid(
        np.arange(0, 109800, grid_stride),
        np.arange(0, 109800, grid_stride)
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    print(f"Number of grid points: {len(grid_points)}")

    point_sets = list(map(lambda point: retrievalcube.queryRadius(point, config.spatial_smoothing_sigma * 2), grid_points))
    print(f"Number of point sets: {len(point_sets)}")
    print(point_sets[0])

    print(point_sets[100])
    # Now check minimum distance to any point in the set is less than the spatial smoothing sigma
    smoothed_points = []
    point_confidences = []
    for origin,point_set in zip(grid_points, point_sets):
        # if its an empty set, just return nans
        if len(point_set[0]) == 0:
            smoothed_points.append((origin, np.nan * np.ones_like(config.heights)))
            point_confidences.append(np.nan)
            continue
        dists = np.linalg.norm(point_set[0] - origin, axis=1)
        if np.min(dists) < config.spatial_smoothing_sigma * 1.5:

            # Compute confidences based on difference between min and max of correlation retrievals
            confs = np.max(point_set[1], axis=1) - np.min(point_set[1], axis=1)
            confs = confs**2

            # compute gaussian weighted average of the scores
            weights = np.exp(-0.5 * (dists / config.spatial_smoothing_sigma)**2)
            weights *= confs
            point_confidences.append(np.sum(weights))
            weights /= np.sum(weights)


            # This is the weighted gaussian average of the scores
            smoothed_value = np.sum(point_set[1] * weights[:, np.newaxis], axis=0)
            smoothed_points.append((origin, smoothed_value))
        else:
            smoothed_points.append((origin, np.nan * np.ones_like(config.heights)))
            point_confidences.append(np.nan)

    # Now make 3D array of smoothed scores
    smoothed_scores = np.array([sp[1] for sp in smoothed_points])
    smoothed_coords = np.array([sp[0] for sp in smoothed_points])
    point_confidences = np.array(point_confidences)

    # Plot some random smoothed scores vs a few of the original scores that were closest to them
    ids = np.random.choice(len(smoothed_points), size=3, replace=False)
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    colors = ['r','g','b'] # We'll use dashed lines for the original scores, solid for the smoothed
    for c,id in zip(colors,ids):
        original_point_set = point_sets[id]
        smoothed_point = smoothed_points[id]
        for i in range(original_point_set[1].shape[0]):
            plt.plot(config.heights, original_point_set[1][i], linestyle='--', color=c, alpha=0.3)
        plt.plot(config.heights, smoothed_point[1], linestyle='-', color=c, linewidth=2)
    plt.xlabel('Height (m)')
    plt.ylabel('Correlation Score')
    plt.title('Original Scores (dashed) and Smoothed Score (solid)')
    plt.savefig(f'debug_smoothed_scores_example_colmeanvar_12B_nostd.png')
    plt.close()

    fig,ax = plt.subplots(1,1,figsize=(15,15))
    # Histogram of the number of points used in each smoothed score
    num_points = np.array([len(ps[0]) for ps in point_sets])
    plt.hist(num_points, np.arange(-0.5, 50.5, 1), density=True)
    plt.xlabel('Number of Points Used in Smoothing')
    plt.ylabel('Number of Grid Points')
    plt.title('Histogram of Number of Points Used in Smoothing')
    plt.grid()
    plt.savefig('debug_smoothed_num_points_hist_colmeanvar_12B_nostd.png')
    plt.close(fig)

    print(smoothed_scores.shape, point_confidences.shape)
    smoothed_scores = smoothed_scores.reshape((int(109800/grid_stride), int(109800/grid_stride), smoothed_scores.shape[-1]))
    point_confidences = point_confidences.reshape((int(109800/grid_stride), int(109800/grid_stride)))
    point_confidences = (point_confidences - np.nanmin(point_confidences)) / (np.nanmax(point_confidences) - np.nanmin(point_confidences))

    fig,ax = plt.subplots(1,1,figsize=(15,15))
    heights = np.argmax(smoothed_scores[...,4:], axis=-1) * config.height_step
    
    ax.imshow(np.clip(heights,0,16000), extent=[0, 109800, 0, 109800], origin='lower', cmap='gist_ncar', vmin=0, vmax=16000)
    ax.set_xticks([])
    ax.set_yticks([])
    # add a colorbar
    cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)')
    fig.tight_layout()
    fig.savefig('debug_smoothed_heights_colmeanvar_12B_nostd.png')
    plt.close(fig)

    fig,ax = plt.subplots(1,1,figsize=(15,15))
    ax.imshow(point_confidences, extent=[0, 109800, 0, 109800], origin='lower', cmap='jet')
    ax.set_xticks([])
    ax.set_yticks([])
    # add a colorbar
    cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Point Confidence')
    fig.tight_layout()
    fig.savefig('debug_smoothed_confidences_colmeanvar_12B_nostd.png')
    plt.close(fig)


    fig,ax = plt.subplots(1,1,figsize=(15,15))
    # Plot original points
    original_heights = np.argmax(height_scores, axis=-1) * config.height_step
    ax.scatter(coords[:,0], coords[:,1], c=original_heights, cmap='jet', s=1, alpha=0.5)
    ax.set_xlim([0, 109800])
    ax.set_ylim([0, 109800])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('debug_original_points_colmeanvar_12B_nostd.png')
    plt.close(fig)


def debug_plot(extracted,extracted_points,footprint_id):
    """
    Debug plot of the extracted data
    """
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    c = extracted['B02'].ravel()
    print(extracted_points.shape)
    print(extracted['B02'].shape)
    ax.scatter(extracted_points[...,0].ravel(),109800-extracted_points[...,1].ravel(),c=c,cmap='plasma',s=3,alpha=0.5)
    ax.set_xlim([0,109800])
    ax.set_ylim([0,109800])
    fig.savefig(f'debug_{footprint_id}.png')
    plt.close(fig)
    return None

def plot_height(config,final_heights,final_coords):
    """
    Simple display of the heights found in the scene

    Parameters:
        config: CloudHeightConfig, Configuration object
        final_heights: np.array, Array of retrieved heights
        final_coords: np.array, Array of coordinates of the retrieved heights

    Returns:
        None
    """
    rgb = getBands(config.scene_dir, ['B04','B03','B02'])
    rgb = (np.stack([rgb['B04'],rgb['B03'],rgb['B02']],axis=-1)-1000)/10_000
    rgb = np.clip((rgb**0.7),0,1) # Gamma correction
    resolution = BAND_RESOLUTIONS[config.reference_band]
    point_size = 6*(config.stride/500)
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    ax.imshow(rgb,extent=[0,rgb.shape[1]*resolution,0,rgb.shape[0]*resolution])

    mask = final_heights > 0
    final_heights = final_heights[mask]
    final_coords = final_coords[mask]

    final_heights_log = np.log10(final_heights)
    ax.scatter(final_coords[:,0],rgb.shape[1]*resolution-final_coords[:,1],c=final_heights_log,cmap='gist_ncar',s=point_size,alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if config.plot_writeto is not None:
        fig.savefig(config.plot_writeto)
    plt.show()
    

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process a scene to retrieve cloud heights')
    parser.add_argument('--config','-c',type=str,help='Path to the configuration file')
    parser.add_argument('--output','-o',type=str,help='Path to the output file')
    parser.add_argument('--plot','-p',action='store_true',help='Plot the heights')
    args = parser.parse_args()

    config = CloudHeightConfig(args.config)

    # final_heights,final_coords,times = ProcessScene(config)
    # print(times)
    # # final_heights,final_coords = np.load('output.npz')['heights'],np.load('output.npz')['coords']
    # if args.plot:
    #     plot_height(config,final_heights,final_coords)
    # if args.output is not None:
    #     np.savez(args.output,heights=final_heights,coords=final_coords)
        
    processor = CloudHeightProcessor(config)


    processor.process()
    
    # # load from output
    # if args.output is not None and os.path.exists(args.output):
    #     data = np.load(args.output)
    #     processor.final_heights = data['heights']
    #     processor.final_coords = data['coords']
    # else:
    #     processor.process()

    
    if args.plot:
        processor.plot_results()
    if args.output is not None:
        processor.save_results(args.output) 