import numpy as np
from clouds_decoded.constants import BAND_TIME_DELAYS, ORBITAL_VELOCITY, SPACECRAFT_ALTITUDE

class RotationTransform:
    def __init__(self, angle, centre=(0,0)):
        """Create a rotation transform around a given centre.

        Args:
            angle: Rotation angle in radians.
            centre: Centre of rotation as (x, y) tuple.
        """
        self.angle = angle
        self.centre = centre

    def __call__(self,points,inverse=False):
        """Rotate a set of points around the configured centre.

        Args:
            points: Array of shape (2, N) with x and y coordinates.
            inverse: If True, rotate in the opposite direction.

        Returns:
            Rotated points as an array of shape (2, N).
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
    """Convert along-track pixel offsets to cloud heights in metres.

    Uses the spacecraft orbital velocity and altitude to relate the observed
    parallax shift (in pixels) to the physical height of the target.

    Args:
        offsets: Per-band pixel offsets (list or array).
        bands: Band names corresponding to each offset. If a single-element
            list, it is broadcast to match the length of offsets.
        pixel_size: Pixel size in metres along-track.

    Returns:
        Array of heights in metres, one per offset.
    """
    if not (len(offsets) == len(bands) or len(bands) == 1):
        raise ValueError(
            f"Length of offsets ({len(offsets)}) must equal number of bands "
            f"({len(bands)}), or bands must be a single-element list."
        )

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
    """Convert cloud heights in metres to expected along-track pixel offsets.

    Inverse of ``offsetsToHeights``. Computes the parallax shift each band
    would exhibit for a target at the given height.

    Args:
        heights: Per-band heights in metres (list or array).
        bands: Band names corresponding to each height.
        pixel_size: Pixel size in metres along-track.

    Returns:
        Array of pixel offsets, one per band.
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
    motion_at_h = heights * motion_in_space / SPACECRAFT_ALTITUDE
    offsets = motion_at_h / pixel_size

    return offsets
