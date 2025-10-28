import os
import xml.etree.ElementTree as ET
import numpy as np
import rasterio as rio
import pyproj
from scipy.interpolate import RegularGridInterpolator, interpn
import time
from multiprocessing import Queue, Process
import uuid
import pickle
import rtree

import defaults
from constants import BAND_TIME_DELAYS, ORBITAL_VELOCITY, SPACECRAFT_ALTITUDE, BAND_RESOLUTIONS

def offsetsToHeights(offsets,bands,pixel_size):
    """
    Convert the offset to a height in meters
    """
    assert len(offsets) == len(bands) or len(bands) == 1, "Length of offsets must be the same as the number of bands, or the number of bands must be 1"

    if isinstance(offsets,list):
        offsets = np.array(offsets)

    # Calculate the time delay between the bands
    if len(bands) == 1:
        bands = [bands[0]] * len(offsets)
    delays =  np.array([BAND_TIME_DELAYS[band] for band in bands])
    # The height can be found with the velocity of the spacecraft, and it's orbital altitude
    # We can work out what a given offset translates to in terms of altitude for misaligned bands, given that the bands are aligned perfectly at the ground
    motion_in_space = ORBITAL_VELOCITY * delays # m
    motion_at_h = offsets * pixel_size # m
    
    # Trigonometry to retrieve height. Approximately linear with height for small angles
    heights = SPACECRAFT_ALTITUDE * motion_at_h/motion_in_space

    return heights

def heightsToOffsets(heights,bands,pixel_size):
    """
    Convert the height to an offset in meters
    """

    if isinstance(heights,list):
        heights = np.array(heights)
    

    # Calculate the time delay between the bands
    delays = [BAND_TIME_DELAYS[band] for band in bands]
    delays = np.array(delays)

    # The height can be found with the velocity of the spacecraft, and it's orbital altitude
    # We can work out what a given offset translates to in terms of altitude for misaligned bands, given that the bands are aligned perfectly at the ground
    motion_in_space = ORBITAL_VELOCITY * delays # m
    
    # Trigonometry to retrieve height. Approximately linear with height for small angles
    if len(delays) > 1:
        motion_at_h = heights * motion_in_space / SPACECRAFT_ALTITUDE
    else:
        motion_at_h = heights * motion_in_space / SPACECRAFT_ALTITUDE
    offsets = motion_at_h / pixel_size

    return offsets


def getGranuleDirectory(sceneDirectory):
    """
    Get the directory of the granule within the scene directory
    
    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        granule: str, path to the granule directory
    """

    granule = os.path.join(sceneDirectory, "GRANULE")
    # Not including DS_Store
    ls = os.listdir(granule)
    if '.DS_Store' in ls:
        ls.remove('.DS_Store')
    assert len(ls) == 1, "Multiple granules found"
    granule = os.path.join(granule, ls[0])
    return granule

def getBandPaths(sceneDirectory, bands):
    """
    Get the paths to the bands in the scene directory
    
    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        paths: dict, paths to the bands
    """

    granule = getGranuleDirectory(sceneDirectory)
    paths = {}
    files = os.listdir(os.path.join(granule, "IMG_DATA"))
    for band in bands:
        path = [f for f in files if f.endswith(f"{band}.jp2")]
        assert len(path) == 1, f"Band {band} not found"
        paths[band] = os.path.join(granule, "IMG_DATA", path[0])
    return paths

def getFootprintPaths(sceneDirectory, bands):
    """
    Get the paths to the footprint files in the scene directory
    
    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        paths: dict, paths to the footprint
    """

    granule = getGranuleDirectory(sceneDirectory)
    paths = {}
    files = os.listdir(os.path.join(granule, "QI_DATA"))
    for band in bands:
        path = [f for f in files if f.endswith(f"DETFOO_{band}.jp2")]
        assert len(path) == 1, f"Band {band} not found"
        paths[band] = os.path.join(granule, "QI_DATA", path[0])
    if len(paths) == 1:
        return paths[0]
    return paths

def getSunAngle(sceneDirectory):
    """
    Get the sun angle from the metadata file of a Sentinel-2 scene

    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        zenith: float, sun zenith angle in degrees
        azimuth: float, sun azimuth angle in degrees (from north)
    """

    # TODO: rewrite for more complex pixel-wise sun angle calculation later

    # Get metadata file
    granule = getGranuleDirectory(sceneDirectory)
    path = os.path.join(granule, "MTD_TL.xml")
    assert os.path.exists(path), "Metadata file not found"
    mtd = ET.parse(path)
    root = mtd.getroot()

    # Look in tree for "Mean_Sun_Angle"
    sunAngle = root.find(".//Mean_Sun_Angle")

    # Get zenith and azimuth angles
    zenith = float(sunAngle.find("ZENITH_ANGLE").text)
    azimuth = float(sunAngle.find("AZIMUTH_ANGLE").text)

    return zenith, azimuth

def getOrbitImageAngle(sceneDirectory):
    """
    Previous function doesn't handle no data well, because it messes up the assumptions about the footprints.

    Instead, we can iterate through columns in the footprint raster, and wait until we find a column
    with >1 footprint IDs (where no data is treated as ID=0)

    Then we can look at the ensuing columns until a new unique set of footprint IDs is found. We can then 
    calculate the angle based on the distance between the first and last columns with the same footprint IDs.
    """

    footprint_path = getFootprintPaths(sceneDirectory, bands=['B02','B03'])['B02']
    with rio.open(footprint_path) as src:
        footprint = src.read(1)

    for col_idx in range(footprint.shape[1]):
        column = footprint[:,col_idx]
        
        diffs = np.diff(column)

        if np.any(diffs == 1):
            # Look for first  diff
            lower_point_y = np.where(diffs == 1)[0][0]
            lower_id = column[lower_point_y]
            upper_id = column[lower_point_y + 1]

            assert lower_id != upper_id, "Lower and upper IDs are the same"

            break

    # Now we have the lower and upper IDs, we can look for the next column with the same IDs
    for next_col_idx in range(col_idx + 1, footprint.shape[1]):
        next_column = footprint[:,next_col_idx]
    
        # If both lower and upper IDs are present, continue, otherwise we've found our last column
        if lower_id in next_column and upper_id in next_column:
            continue
        else:
            last_col_idx = next_col_idx - 1
            break

    horizontal_dist = last_col_idx - col_idx
    
    # Vertical distance is the distance between lower_point_y and the same transition in the new column
    next_column = footprint[:,last_col_idx]
    
    upper_point_y = np.argmax(next_column == lower_id)

    vertical_dist =  lower_point_y - upper_point_y
    angle = -np.arctan2(horizontal_dist, vertical_dist)
    print(f"Image azimuth: {angle}")

    # # Debug plot
    # import matplotlib.pyplot as plt
    # plt.imshow(footprint==lower_id,cmap='gray')
    
    # centre_point = (footprint.shape[0]//2, footprint.shape[1]//2)
    # magnitude = footprint.shape[1]//8
    # # plot an arrow from the centre point at the angle of the image azimuth
    # plt.arrow(centre_point[1], centre_point[0],
    #     magnitude * np.sin(angle), 
    #     magnitude * np.cos(angle), 
    #     color='red', width=2)
    # plt.scatter([col_idx, last_col_idx], [lower_point_y, upper_point_y], color='blue')

    # plt.savefig("debug_image_azimuth_new.png")
    # plt.close()


    return angle

def getLatitude(sceneDirectory):
    """
    Use rasterio to get the bbox of the scene and calculate the centre latitude

    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        lat: float, latitude in radians
    """

    band_path = getBandPaths(sceneDirectory, bands=['B02'])['B02']
    with rio.open(band_path) as src:
        bounds = src.bounds
    transform = pyproj.transformer.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
    lat = (
        transform.transform(bounds.left, bounds.top)[1] + 
        transform.transform(bounds.right, bounds.bottom)[1]
    ) / 2
    # lat = np.deg2rad(lat)
    return lat

def getLongitude(sceneDirectory):
    """
    Use rasterio to get the bbox of the scene and calculate the centre longitude

    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        lon: float, longitude in radians
    """

    band_path = getBandPaths(sceneDirectory, bands=['B02'])['B02']
    with rio.open(band_path) as src:
        bounds = src.bounds
    transform = pyproj.transformer.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
    lon = (
        transform.transform(bounds.left, bounds.top)[0] + 
        transform.transform(bounds.right, bounds.bottom)[0]
    ) / 2
    # lon = np.deg2rad(lon)
    return lon

def getOrbitType(sceneDirectory):
    """
    Get the orbit type of the scene

    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        orbit_type: str, 'ASCENDING' or 'DESCENDING'
    """

    mtd = ET.parse(os.path.join(sceneDirectory, "MTD_MSIL1C.xml"))
    root = mtd.getroot()
    # SENSING_ORBIT_DIRECTION
    sensing_orbit = root.find(".//SENSING_ORBIT_DIRECTION").text
    return sensing_orbit

def getSceneOrientation(sceneDirectory):
    """
    Get the orientation of the scene in degrees from the north direction (clockwise)

    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        orientation: float, orientation in radians    
    """

    band_path = getBandPaths(sceneDirectory, bands=['B02'])['B02']
    with rio.open(band_path) as src:
        topLeftUTM = src.transform * (0, 0)
        topRightUTM = src.transform * (src.width, 0) 
        transform = pyproj.transformer.Transformer.from_crs(src.crs, 'EPSG:4326')
        topLeft = transform.transform(topLeftUTM[0], topLeftUTM[1])
        topRight = transform.transform(topRightUTM[0], topRightUTM[1])
        orientation = np.pi/2 - np.arctan2(
            topRight[1] - topLeft[1], 
            topRight[0] - topLeft[0]
        )
    return orientation

def getBands(sceneDirectory, bands):
    """
    Get the bands from the scene directory
    
    Parameters:
        sceneDirectory: str, path to the scene directory
        bands: list, bands to get

    Returns:
        bands: dict, bands
    """

    paths = getBandPaths(sceneDirectory, bands)
    bands = {}
    for band,path in paths.items():
        with rio.open(path) as src:
            bands[band] = src.read(1)
    return bands

def getFootprints(sceneDirectory, bands):
    """
    Get the valid footprint shape from the DETFOO file
    
    Parameters:
        sceneDirectory: str, path to the scene directory
        bands: list, bands to get
    
    Returns:
        footprints: dict, footprints
    """

    footprint_paths = getFootprintPaths(sceneDirectory, bands=bands)
    footprints = {}
    for band,path in footprint_paths.items():
        with rio.open(path) as src:
            footprints[band] = src.read(1)
    return footprints

class RotationTransform:

    def __init__(self, angle, centre=(0,0)):
        """
        Create a rotation transform that rotates points around a given centre by a given angle

        Parameters:
            angle: float, angle in radians
            centre: tuple, centre of rotation

        Returns:
            None
        """
        self.angle = angle
        self.centre = centre

    def __call__(self,points,inverse=False):
        """
        Rotate a set of points around a given centre

        Parameters:
            points: np.array, points to rotate
            inverse: bool, if True, rotate in the opposite direction

        Returns:
            rotated_points: np.array, rotated points
        """
        if inverse:
            angle = -self.angle
        else:
            angle = self.angle

        vectors = np.array([points[0] - self.centre[0], points[1] - self.centre[1]])
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        rotated_vectors = rotation_matrix @ vectors
        rotated_points = rotated_vectors + np.array([[self.centre[0]],[self.centre[1]]])
        return rotated_points


class Sentinel2Scene:
    def __init__(self,scene_directory,bands=defaults.BANDS):
        self.scene_directory = scene_directory
        self.bands = getBands(scene_directory, bands)
        self.footprints = getFootprints(scene_directory, bands)
        self.sun_zenith, self.sun_azimuth = getSunAngle(scene_directory)
        self.image_azimuth = getOrbitImageAngle(scene_directory)
        self.latitude = getLatitude(scene_directory)
        self.orientation = getSceneOrientation(scene_directory)
        self.orbit_type = getOrbitType(scene_directory)
      

class Column:
    def __init__(self,bands,points,footprint_id):
        self.bands = bands
        self.points = points
        self.footprint_id = footprint_id
        self.direction = 'up' if footprint_id % 2 == 0 else 'down'

    def getGradients(self, gradient_power_ratio=1):
        """
        Calculate the gradients of the bands
        """
        # Take gradients of the bands
        gradients = {
            name: np.stack((
                np.gradient(band,axis=0), 
                np.gradient(band,axis=1) * gradient_power_ratio # TODO: Check if these gradients have same relative power along and across track
            ), axis=-1) for name,band in self.bands.items()
        }
        return gradients
    
    def getMask(self,threshold_band,threshold):
        """
        Get the mask of the column based on the threshold band
        """
        mask = self.bands[threshold_band] > threshold
        return mask
    
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

        # Create an R-tree for the retrieval cube

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
    def __init__(self,scene,conf):
        self.bands = scene.bands
        self.footprints = scene.footprints
        self.angle = scene.image_azimuth
        self.conf = conf
        self.band_interpolators = self.getBandInterpolators(self.bands) 
        self.footprint_interpolators = self.getFootprintInterpolators(self.footprints)
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
        
        Parameters:
            bands: dict, bands

        Returns:
            interpolators: list, interpolators
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
        
        Parameters:
            footprints: dict, footprints

        Returns:
            interpolators: list, interpolators
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
    
    def getUnrotatedOriginColumn(self):
        """
        Get the unrotated column of points that will be rotated and translated for a given column -- much faster
        than recalculating the points each time
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

        Parameters:
            col_start: int, starting position (in metres) of the column
            angle: float, angle in radians

        Returns:
            column: Column, rotated column
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

        points = self.reshapePointsToArr(points,self.col_shape)
        # rotated_bands, points = self.depad(rotated_bands,points)
        return Column(rotated_bands,points,footprint_id)
    
    def reshapePointsToArr(self,points,shape):
        """
        Reshape the points to the shape of the rotated bands, YxXx2
        """
        points = np.reshape(points,(2,shape[0],shape[1]))
        points = np.moveaxis(points,0,-1)
        return points
    
    def depad(self,data,points):
        """
        Checks for nan values and removes them from the rotated bands and points
        """
        delete_cols = np.where(np.all(np.isnan(data[self.reference_band]),axis=0))[0]
        delete_rows = np.where(np.all(np.isnan(data[self.reference_band]),axis=1))[0]
        for name,band in data.items():
            data[name] = np.delete(band,delete_cols,axis=1)
            data[name] = np.delete(data[name],delete_rows,axis=0)
        points = np.delete(points,delete_cols,axis=1)
        points = np.delete(points,delete_rows,axis=0)
        if np.size(points) == 0:
            return None, None
        return data, points

    def getFootprintID(self,points):
        """
        Make sure that all points are within the valid footprint, and return the ID of the footprint
        """
        # Check that the points are within the valid footprint
        # Only check first and last of each row, along the whole column
        sparse_points = points

        id = None
        for band in self.footprints.keys():
            interpolator = self.footprint_interpolators[band]
            col_footprint_ids = interpolator((sparse_points[1].flatten(),sparse_points[0].flatten()))
            present_ids = np.unique(col_footprint_ids[~np.isnan(col_footprint_ids)])

            # If there are no valid IDs, or more than one, return False
            if present_ids.size != 1:
                return None
            
            # Set the ID if it is not set
            if id is None:
                id = present_ids[0]

            # Check to see if the ID is the same as the previous band
            elif id != present_ids[0]:
                return None
        return id


    def __getitem__(self,idx):
        """
        Get the rotated column at the given index

        Parameters:
            idx: int, index of the column

        Returns:
            rotated_bands: dict, rotated bands
        """
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
        self.temp_dir = temp_dir # + Store the unique temp directory path
        self.queue = Queue(maxsize=max_queue_size)
        self.process = Process(target=self._worker, args=(self.queue, n_workers))
        self.process.start()

    def _worker(self, queue, n_workers):
        for i in range(self.length):
            column = self.extractor[i]

            while queue.full():
                time.sleep(0.1)  # Wait for space in the queue
            
            if column is not None:
                # Change the file extension to .pkl for clarity
                filename = f"column_{uuid.uuid4()}.pkl" # <-- Changed extension
                column_path = os.path.join(self.temp_dir, filename)
                
                # Save out column object to temp_dir
                with open(column_path, 'wb') as f:
                    pickle.dump(column, f)
                
                queue.put(column_path)
            else:
                queue.put("EMPTY_COLUMN")

        # Add a sentinel value for EACH worker process
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

    def __len__(self):
        return self.length
