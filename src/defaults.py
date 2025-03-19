import numpy as np

##### USER PARAMETERS TO SET #####
 # degrees, TODO: Find a way to calculate this automatically. Currently manual from looking at the detfoo_*.jp2 files
HACK_IMAGE_AZIMUTH = np.deg2rad(-12.5) # degrees

REFERENCE_BAND = 'B02' # Band that is fixed whilst others move. Naturally easiest to be B02 as it has zero time delay
N_WORKERS = 16 # Number of workers to use in parallel processing

CLOUDY_THRESH = 2000 # Basic thresholding for cloudiness in DN of Sentinel-2. TOO: Implement way to skip cloudy points using this
THRESHOLD_BAND = 'B08' # Band to use for thresholdings
ALONG_TRACK_RESOLUTION = 2 # pixel size that is used during convolution, in metres
ACROSS_TRACK_RESOLUTION = 10 # pixel size that is used during convolution, in metres
STRIDE = 200 # Stride between points in metres
CONVOLVED_SIZE_ALONG_TRACK = 300 # Size that is used in correlation along track. In metres
CONVOLVED_SIZE_ACROSS_TRACK = 300 # Size that is used in correlation across track. In metres
MAX_HEIGHT = 15_000
HEIGHT_STEP = 200
BANDS = ['B01','B02','B03','B04','B05','B07','B06','B08','B8A'] # Don't use B10 as it is cirrus band. B11, B12 are risky as they are SWIR so may not work well
TARGET_FEATURES = 'gradient' # 'gradient' or 'reflectance', whether to correlate the gradient or the reflectance of the bands
##################################