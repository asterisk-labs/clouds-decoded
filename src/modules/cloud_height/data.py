import os
import numpy as np
import rasterio as rio
from scipy.interpolate import RegularGridInterpolator
import time
from multiprocessing import Queue, Process
import uuid
import pickle
import rtree

from clouds_decoded.data import Sentinel2Scene
from clouds_decoded.constants import BAND_RESOLUTIONS
from .physics import RotationTransform

class Column:
    def __init__(self,bands,points,footprint_id,mask=None):
        self.bands = bands
        self.points = points
        self.footprint_id = footprint_id
        self.direction = 'up' if footprint_id % 2 == 0 else 'down'
        self.mask = mask

    
class RetrievalCube:
    """
    Object to store height retrievals for a scene. The cube is a 3D array of X-Y-Z where Z is the number of height steps
    """

    def __init__(self,retrievals,coords,conf):
        self.conf = conf
        self.retrievals = retrievals
        self.coords = coords
        self.ids = np.arange(len(retrievals))
        self.clean_nans()

    def clean_nans(self):
        """
        Remove any retrievals that are NaN
        """
        mask = ~np.any(np.isnan(self.retrievals),axis=1)
        self.retrievals = self.retrievals[mask]
        self.coords = self.coords[mask]
        self.ids = self.ids[mask]
        
    def createRtree(self):
        """
        Create an R-tree (in X and Y) for the retrieval cube
        """
        def generate_points():
            for id in range(len(self.ids)):
                coord = self.coords[id]
                retrieval = self.retrievals[id]
                yield id, (coord[0], coord[1], coord[0], coord[1]), retrieval
        
        self.idx = rtree.index.Index(generate_points())

    def getNearestRetrievals(self,point,N=1):
        """
        Get the nearest retrieval to a point
        """
        ids = list(self.idx.nearest(point,N))
        return self.coords[ids], self.retrievals[ids]

    
    def queryRadius(self,point,radius):
        """
        Query the R-tree for all retrievals within a radius of a point
        """
        x_min = point[0] - radius
        x_max = point[0] + radius
        y_min = point[1] - radius
        y_max = point[1] + radius
        ids = list(self.idx.intersection((x_min, y_min, x_max, y_max)))
        return self.coords[ids], self.retrievals[ids]

class ColumnExtractor:
    def __init__(self, scene: Sentinel2Scene, conf, mask=None):
        self.bands = scene.bands
        self.footprints = scene.footprints
        self.angle = scene.image_azimuth
        self.conf = conf
        self.mask = mask
        self.band_interpolators = self.getBandInterpolators(self.bands) 
        self.footprint_interpolators = self.getFootprintInterpolators(self.footprints)
        
        if self.mask is not None:
             self.mask_interpolator = self.getMaskInterpolator(self.mask)
        else:
             self.mask_interpolator = None
             
        self.original_resolution = BAND_RESOLUTIONS[conf.reference_band]
        self.along_track_resolution = conf.along_track_resolution
        self.across_track_resolution = conf.across_track_resolution
        self.width = conf.convolved_size_across_track
        self.stride = conf.stride
        self.reference_band = conf.reference_band
        self.unrotated_origin_column, self.col_shape = self.getUnrotatedOriginColumn()
        self.rotation = RotationTransform(self.angle)

    def getBandInterpolators(self,bands):
        """
        Get interpolators for the bands
        """
        assert isinstance(bands, dict), "Bands must be a dictionary"
        interpolators = {}
        for band in bands.keys():
            interpolators[band] = RegularGridInterpolator((
                    np.arange(bands[band].shape[0]) * BAND_RESOLUTIONS[band],
                    np.arange(bands[band].shape[1]) * BAND_RESOLUTIONS[band]
                ), 
                bands[band], 
                fill_value=np.nan,
                bounds_error=False, 
                method='linear'
            )
        return interpolators
    
    def getFootprintInterpolators(self,footprints):
        """
        Get interpolators for the footprints
        """
        assert isinstance(footprints, dict), "Footprints must be a dictionary"
        interpolators = {}
        for band in footprints.keys():
            interpolators[band] = RegularGridInterpolator((
                    np.arange(footprints[band].shape[0]) * BAND_RESOLUTIONS[band],
                    np.arange(footprints[band].shape[1]) * BAND_RESOLUTIONS[band]
                ), 
                footprints[band], 
                fill_value=np.nan,
                bounds_error=False, 
                method='nearest'
            )
        return interpolators
    
    def getMaskInterpolator(self, mask):
        """
        Get interpolator for the binary cloud mask.
        Infers the mask resolution from the scene extent and mask shape,
        so it works correctly regardless of the mask's native resolution.
        """
        # Compute scene extent from the reference band
        ref_band = getattr(self, 'reference_band', self.conf.reference_band)
        ref_res = getattr(self, 'original_resolution', BAND_RESOLUTIONS[ref_band])
        ref_shape = self.bands[ref_band].shape
        scene_extent_y = ref_shape[0] * ref_res
        scene_extent_x = ref_shape[1] * ref_res

        # Derive mask pixel size from the scene extent and mask dimensions
        mask_res_y = scene_extent_y / mask.shape[0]
        mask_res_x = scene_extent_x / mask.shape[1]

        return RegularGridInterpolator((
                np.arange(mask.shape[0]) * mask_res_y,
                np.arange(mask.shape[1]) * mask_res_x
            ),
            mask,
            fill_value=0, # Assume clear if out of bounds
            bounds_error=False,
            method='nearest' # Binary mask, so nearest neighbor
        )

    def getUnrotatedOriginColumn(self):
        """
        Get the unrotated column of points 
        """
        # Find length of column in metres, based on height of image, angle, and along track resolution
        col_length = self.bands[self.reference_band].shape[0] * self.original_resolution / np.abs(np.cos(self.angle))

        # Get the width of the column in metres
        row_length = self.width
        assert row_length % self.across_track_resolution == 0, "CONVOLVED_SIZE_ACROSS_TRACK must be a multiple of ACROSS_TRACK_RESOLUTION"

        # Add a buffer to the length, equal to the 2x width of the column (to allow for rotation)
        col_length += 2 * row_length

        # Get coordinates of unrotated column (in metres)
        xs = np.arange(0, row_length, self.across_track_resolution)
        ys = np.arange(-row_length, col_length - row_length, self.along_track_resolution)
        points = np.meshgrid(xs, ys)
        points = np.array([points[0].flatten(), points[1].flatten()])
        return points, (len(ys), len(xs))
    
    def extractRotatedColumn(self, col_start):
        """
        Extract a column of pixels from the bands, rotate it by the given angle, and return the rotated bands
        """
        points = self.rotation(self.unrotated_origin_column) # Rotate the points
        points += np.array([[col_start],[0]]) # Final, translated points

        footprint_id = self.getFootprintID(points)
        if footprint_id is None:
            return None
        
        rotated_bands = {}
        for band in self.bands.keys():
            rotated_band = self.band_interpolators[band]((points[1].flatten(),points[0].flatten()))
            rotated_band = rotated_band.reshape(self.col_shape)
            rotated_bands[band] = rotated_band

        rotated_mask = None
        if self.mask_interpolator is not None:
             rotated_mask = self.mask_interpolator((points[1].flatten(), points[0].flatten()))
             rotated_mask = rotated_mask.reshape(self.col_shape)
             # Binarize again just in case interpolation did something weird
             rotated_mask = (rotated_mask > 0.5).astype(bool)

        points = self.reshapePointsToArr(points,self.col_shape)
        return Column(rotated_bands,points,footprint_id, mask=rotated_mask)
    
    def reshapePointsToArr(self,points,shape):
        points = np.reshape(points,(2,shape[0],shape[1]))
        points = np.moveaxis(points,0,-1)
        return points
    
    def getFootprintID(self,points):
        sparse_points = points
        id = None
        for band in self.footprints.keys():
            interpolator = self.footprint_interpolators[band]
            col_footprint_ids = interpolator((sparse_points[1].flatten(),sparse_points[0].flatten()))
            present_ids = np.unique(col_footprint_ids[~np.isnan(col_footprint_ids)])

            if present_ids.size != 1:
                return None
            
            if id is None:
                id = present_ids[0]
            elif id != present_ids[0]:
                return None
        return id


    def __getitem__(self,idx):
        col_start = int(idx * self.stride)
        column = self.extractRotatedColumn(col_start)
        if column is None:
            return None
        if column.bands is None:
            return None
        return column
        
    def __len__(self):
        base_length = self.bands[self.reference_band].shape[1] * self.original_resolution * (1 + np.abs(np.sin(self.angle)))
        return int(base_length / self.stride)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ColumnIterator:
    def __init__(self, extractor, n_workers, temp_dir, max_queue_size=None):
        if max_queue_size is None:
            max_queue_size = n_workers * 2
        self.length = len(extractor)
        self.extractor = extractor
        self.temp_dir = temp_dir 
        self.queue = Queue(maxsize=max_queue_size)
        self.process = Process(target=self._worker, args=(self.queue, n_workers))
        self.process.start()

    def _worker(self, queue, n_workers):
        for i in range(self.length):
            column = self.extractor[i]

            while queue.full():
                time.sleep(0.1)  # Wait for space in the queue
                
            if column is not None:
                filename = f"column_{uuid.uuid4()}.pkl" 
                column_path = os.path.join(self.temp_dir, filename)
                
                with open(column_path, 'wb') as f:
                    pickle.dump(column, f)
                
                queue.put(column_path)
            else:
                queue.put("EMPTY_COLUMN")

        for _ in range(n_workers):
            queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        column = self.queue.get()
        if column is None:
            self.process.join() 
            raise StopIteration
        return column

    def close(self):
        """Terminate the producer process if still running."""
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)

    def __len__(self):
        return self.length
