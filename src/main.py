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
        
        # For each patch, add up all the other patches and then do correlation 
        # Summed patches correlated much quicker than all correlations summed
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

            # DEBUG: Save out intermediate results from here

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

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process a scene to retrieve cloud heights')
    parser.add_argument('--config','-c',type=str,help='Path to the configuration file')
    parser.add_argument('--output','-o',type=str,help='Path to the output file')
    parser.add_argument('--plot','-p',action='store_true',help='Plot the heights')
    args = parser.parse_args()

    config = CloudHeightConfig(args.config)
        
    processor = CloudHeightProcessor(config)

    processor.process()
    
    if args.plot:
        processor.plot_results()
    if args.output is not None:
        processor.save_results(args.output) 