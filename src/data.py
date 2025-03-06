import os
import xml.etree.ElementTree as ET
import numpy as np
import rasterio as rio
import pyproj
import skimage.transform
from scipy.interpolate import RegularGridInterpolator

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
    delays = [BAND_TIME_DELAYS[band] for band in bands]
    delays = np.array(delays)
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
    For an orbit: cos(inclination)=cos(lat)*sin(azimuth)
    Therefore, azimuth = arcsin(cos(inclination)/cos(lat))

    Azimuth is from north in the clockwise direction

    Parameters:
        sceneDirectory: str, path to the scene directory

    Returns:
        image_azimuth: float, image azimuth angle in degrees (from north)
    """

    # lat = getLatitude(sceneDirectory)
    # image_orientation = getSceneOrientation(sceneDirectory)
    # inclination = ORBIT_INCLINATION
    # descending_or_acsending = getOrbitType(sceneDirectory)
    # if descending_or_acsending == 'DESCENDING':
    #     print("Descending orbit")
    #     inclination = np.pi - inclination
    # print(f"Inclination: {np.rad2deg(inclination)}")
    # azimuth = np.arcsin(np.cos(inclination)/np.cos(lat))
    # print(f"Azimuth: {np.rad2deg(azimuth)}")
    # image_azimuth = azimuth - image_orientation
    # image_azimuth = np.rad2deg(image_azimuth)
    # return image_azimuth

    # TODO: Fix this so that it gives correct angle. Currently, there is a disagreement between this function
    # and what I measure by looking at the angle of the boundaries in the DETFOO.jp2 file.
    # Not sure which one is correct but I'm inclined to think I got it wrong. For now we will hardcode the value

    return defaults.HACK_IMAGE_AZIMUTH

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
    lat = np.deg2rad(lat)
    return lat

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
    print(f"Image orientation: {np.rad2deg(orientation)}")
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

def getBandInterpolators(bands):
    """
    Get interpolators for the bands
    
    Parameters:
        bands: dict, bands

    Returns:
        interpolators: list, interpolators
    """
    if not isinstance(bands, list):
        bands = [bands]

    interpolators = []
    for band in bands:
        interpolators.append(RegularGridInterpolator((
            np.arange(band.shape[0]), 
            np.arange(band.shape[1])
        ), band))
    if len(interpolators) == 1:
        return interpolators[0]
    return interpolators

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
    
def getValidFootprintIDs(footprints,check_all=False, reference_band='B02'):
    """
    Get the valid footprint IDs from the DETFOO files. The valid footprint IDs are the unique 
    IDs in the reference band that are valid for all bands

    Parameters:
        footprints: dict, footprints

    Returns:
        IDs: list, valid footprint IDs
    """

    assert reference_band in list(footprints.keys()), "Reference band not found in footprints"

    reference_footprint = footprints[reference_band]
    IDs = np.unique(reference_footprint)
    if not check_all:
        return IDs
    valid_IDs = []
    for b,fp in footprints.items():
        IDs = np.unique(fp)
        valid_IDs.append(IDs)
    valid_IDs = np.unique(np.concatenate(valid_IDs))
    return valid_IDs

def getValidFootprintShape(footprints, footprint_ID, pooling='intersection', reference_band='B02'):

    """
    Get the valid footprint shape from the DETFOO files.
    
    Each detfoo file corresponds to a band. The values in the DETFOO files are the detector
    from which the pixel was taken, for that band.

    The valid footprint shape is the intersection of the detector footprints for all bands 
    for a given footprint_ID (1->12)

    The footprints are always stripes along the along-track direction. We can find check the 
    edge pixels of the footprints to determine the valid footprint shape.

    Parameters:
        footprints: dict, footprints

    Returns:
        valid_footprint: np.array, valid footprint
    """

    assert reference_band in list(footprints.keys()), "Reference band not found in footprints"

    reference_footprint = footprints[reference_band]
    reference_shape = reference_footprint.shape
    
    for b,footprint in footprints.items():
        if footprint.shape != reference_shape:
            footprints[b] = skimage.transform.resize(
                footprint, 
                reference_shape, 
                order=0, 
                preserve_range=True, 
                anti_aliasing=False
            )
    
    footprint_arr = np.stack([footprints[b]==footprint_ID for b in footprints.keys()], axis=0)

    if pooling=='intersection':
        valid_footprint = np.sum(footprint_arr, axis=0) == len(footprints.keys())
    elif pooling=='union':
        valid_footprint = np.sum(footprint_arr, axis=0) > 0
    else:
        raise ValueError("Pooling must be either 'intersection' or 'union'")

    return valid_footprint

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


def getFootprintBounds(footprint):
    """
    Get the bounds of the footprint and the centre of the footprint, in pixels, not metres

    Parameters:
        footprint: np.array, footprint

    Returns:
        x_min: int, minimum x value
        x_max: int, maximum x value
        y_min: int, minimum y value
        y_max: int, maximum y value
    """

    x_test = np.where(np.any(footprint,axis=0))
    y_test = np.where(np.any(footprint,axis=1))
    x_min,x_max = np.min(x_test),np.max(x_test)
    y_min,y_max = np.min(y_test),np.max(y_test)
    return x_min,x_max,y_min,y_max

def extractAndRotateFootprint(bands,footprint,angle,resolution,reference_band='B02'):
    """
    This function will take a binary footprint, and then extract a (padded) region from each band 
    that is sampled at a higher rate than the original image. 

    The region will be rotated by the given angle, and then returned as a single tensor.

    Assumes all bands are the same size and that the footprint is the same size as the bands.

    Parameters:
        bands: list, bands
        footprint: np.array, footprint
        angle: float, angle in radians
        resolution: float, resolution of the new grid
        reference_band: str, reference

    Returns:
        rotated_bands: dict, rotated bands
        rotated_points: np.array, rotated points
    """
    assert reference_band in bands.keys(), "Reference band not found in bands"
    
    original_resolution = BAND_RESOLUTIONS[reference_band]
    along_track_resolution = resolution
    across_track_resolution = resolution
    
    # Get grid of all points in original coordinate system that completely encloses the footprint
    x_min,x_max,y_min,y_max = getFootprintBounds(footprint) # In pixels, not metres
    centre_x,centre_y = (x_min+x_max)/2,(y_min+y_max)/2
    y_buffer = (x_max-x_min) * np.abs(np.sin(angle))
    y_min -= y_buffer
    y_max += y_buffer

    # Find max width of all rows, use as new width
    x_width = np.max(np.sum(footprint,axis=1))
    x_min = centre_x - x_width/2
    x_max = centre_x + x_width/2

    # Convert to metres
    x_min *= original_resolution
    x_max *= original_resolution
    y_min *= original_resolution
    y_max *= original_resolution
    centre_x *= original_resolution
    centre_y *= original_resolution

    points = np.meshgrid(
        np.arange(x_min,x_max,across_track_resolution),
        np.arange(y_min,y_max,along_track_resolution)
    )
    # Turn points into a 2xN array
    points = np.array([points[0].flatten(),points[1].flatten()])
    size_x = len(np.arange(x_min,x_max,across_track_resolution))
    size_y = len(np.arange(y_min,y_max,along_track_resolution))

    # Rotate the mesh
    if angle != 0:
        rotation = RotationTransform(angle,centre=(centre_x,centre_y))
        rotated_points = rotation(points)
    else:
        rotated_points = points
    # Interpolate the bands at the rotated points
    rotated_bands = {}
    for name,band in bands.items():

        band_resolution = BAND_RESOLUTIONS[name]
        if band_resolution == original_resolution:
            band_footprint = footprint
        else:
            footprint_sampling_rate = band_resolution / original_resolution

            assert footprint_sampling_rate.is_integer(), \
                "Band resolution must be a multiple of the reference band resolution"
            
            footprint_sampling_rate = int(footprint_sampling_rate)
            band_footprint = footprint[::footprint_sampling_rate,::footprint_sampling_rate]

        # Mask with nan where footprint is not valid
        masked_band = np.where(band_footprint,band,np.nan)
        interpolator = RegularGridInterpolator(
            (
                band_resolution*np.arange(band.shape[0]),  # Original data set in grid with metre coordinates
                band_resolution*np.arange(band.shape[1]),
            ), 
            masked_band, 
            fill_value=np.nan,
            bounds_error=False,
            method='linear'
        )
        if angle == 0:
            rotated_band = interpolator((points[1].flatten(),points[0].flatten()))
        else:
            rotated_band = interpolator((rotated_points[1],rotated_points[0]))
        rotated_band = rotated_band.reshape(size_y,size_x)
        rotated_bands[name] = rotated_band
    
    # Turn rotated_points into a YxXx2 array
    rotated_points = rotated_points.reshape(2,size_y,size_x)
    rotated_points = np.moveaxis(rotated_points,0,-1)
    
    # De-pad the rotated bands
    delete_cols = np.where(np.all(np.isnan(rotated_bands[reference_band]),axis=0))[0]
    delete_rows = np.where(np.all(np.isnan(rotated_bands[reference_band]),axis=1))[0]
    for name,band in rotated_bands.items():
        rotated_bands[name] = np.delete(band,delete_cols,axis=1)
        rotated_bands[name] = np.delete(rotated_bands[name],delete_rows,axis=0)
    rotated_points = np.delete(rotated_points,delete_cols,axis=1)
    rotated_points = np.delete(rotated_points,delete_rows,axis=0)
    return rotated_bands, rotated_points


def extractAndRotateColumn(bands,col_start,angle,conf,reference_band='B02'):
    """
    Similar to extractAndRotateFootprint, but extracts a column of pixels from the bands
    directly, without using a footprint. This is useful for extracting a column of pixels

    Parameters:
        bands: list, bands
        col_start: int, starting position (in metres) of the column
        angle: float, angle in radians
        resolution: float, resolution of the new grid
        reference_band: str, reference

    Returns:
        rotated_bands: dict, rotated bands
        rotated_points: np.array, rotated points
    """
    assert reference_band in bands.keys(), "Reference band not found in bands"
    
    original_resolution = BAND_RESOLUTIONS[reference_band]
    along_track_resolution = conf.along_track_resolution
    across_track_resolution = conf.across_track_resolution


    # Find length of column in metres, based on height of image, angle, and along track resolution
    col_length = bands[reference_band].shape[0] * original_resolution / np.abs(np.cos(angle))

    
    # Find the width of the column in metres
    row_length = conf.convolved_size_across_track
    assert row_length % across_track_resolution == 0, "CONVOLVED_SIZE_ACROSS_TRACK must be a multiple of ACROSS_TRACK_RESOLUTION"

    # Add a buffer to the length, equal to the 2x width of the column (to allow for rotation)
    col_length += 2 * row_length

    # Get coordinates of unrotated column (in metres)
    xs = np.arange(col_start, col_start + row_length, across_track_resolution)
    ys = np.arange(-row_length, col_length - row_length, along_track_resolution)
    points = np.meshgrid(xs, ys)
    points = np.array([points[0].flatten(), points[1].flatten()])

    # Rotate the mesh
    if angle != 0:
        rotation = RotationTransform(angle, centre=(col_start, 0))
        rotated_points = rotation(points)
    else:
        rotated_points = points

    # Interpolate the bands at the rotated points
    rotated_bands = {}
    for name, band in bands.items():
        band_resolution = BAND_RESOLUTIONS[name]

        interpolator = RegularGridInterpolator(
            (
                band_resolution * np.arange(band.shape[0]),  # Original data set in grid with metre coordinates
                band_resolution * np.arange(band.shape[1]),
            ),
            band,
            fill_value=np.nan,
            bounds_error=False,
            method='linear'
        )
        if angle == 0:
            rotated_band = interpolator((points[1].flatten(), points[0].flatten()))
        else:
            rotated_band = interpolator((rotated_points[1], rotated_points[0]))
        rotated_band = rotated_band.reshape(len(ys), len(xs))
        rotated_bands[name] = rotated_band

class Sentinel2Scene:
    def __init__(self,scene_directory):
        self.scene_directory = scene_directory
        self.bands = getBands(scene_directory, defaults.BANDS)
        self.footprints = getFootprints(scene_directory, defaults.BANDS)
        self.sun_zenith, self.sun_azimuth = getSunAngle(scene_directory)
        self.image_azimuth = getOrbitImageAngle(scene_directory)
        self.latitude = getLatitude(scene_directory)
        self.orientation = getSceneOrientation(scene_directory)
        self.orbit_type = getOrbitType(scene_directory)

class ColumnExtractor:
    def __init__(self,scene,conf,angle):
        self.bands = scene.bands
        self.footprints = scene.footprints
        self.angle = angle
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
        self.rotation = RotationTransform(angle)

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
            rotated_bands: dict, rotated bands
        """
        rotated_points = self.rotation(self.unrotated_origin_column) # Rotate the points
        points = rotated_points + np.array([[col_start],[0]]) # Final, translated points

        footprint_id = self.getFootprintID(points)

        if footprint_id is None:
            print('invalid footprint')
            return None, None, None
        
        rotated_bands = {}
        for band in self.bands.keys():
            rotated_band = self.band_interpolators[band]((points[1].flatten(),points[0].flatten()))
            rotated_band = rotated_band.reshape(self.col_shape)
            rotated_bands[band] = rotated_band

        rotated_bands, points = self.depad(rotated_bands,points)
        return rotated_bands, points, footprint_id
    
    def depad(self,data,points):
        """
        Checks for nan values and removes them from the rotated bands and points
        """

        delete_cols = np.where(np.all(np.isnan(data[self.reference_band]),axis=0))[0]
        delete_rows = np.where(np.all(np.isnan(data[self.reference_band]),axis=1))[0]
        for name,band in data.items():
            data[name] = np.delete(band,delete_cols,axis=1)
            data[name] = np.delete(data[name],delete_rows,axis=0)
        points = np.reshape(points,(2,self.col_shape[0],self.col_shape[1]))
        points = np.delete(points,delete_cols,axis=2)
        points = np.delete(points,delete_rows,axis=1)
        if np.size(points) == 0:
            print('no points')
            return None, None
        return data, points.reshape(2,-1)

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
        col_start = idx * self.stride
        data, points,footprint_id = self.extractRotatedColumn(col_start)
        return data, points, footprint_id
        
    def __len__(self):
        return self.bands[self.reference_band].shape[1] // self.stride
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
