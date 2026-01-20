import numpy as np
from .constants import BAND_TIME_DELAYS, ORBITAL_VELOCITY, SPACECRAFT_ALTITUDE

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
