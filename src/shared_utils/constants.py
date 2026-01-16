import numpy as np


########### CONSTANTS ############

BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12"
]

BAND_TIME_DELAYS = { # In seconds, taken from Heiselberg and Heiselberg (2021)
    "B02": 0,     # inc. (reference band)
    "B03": 0.527, # inc.
    "B04": 1.005, # inc.
    "B08": 0.263, # inc.
    "B05": 1.269, # inc.
    "B06": 1.525, # inc.
    "B07": 1.790, # inc.
    "B8A": 2.055, # inc.
    "B11": 1.468,
    "B12": 2.085,
    "B01": 2.314, # inc.
    "B09": 2.586,
    "B10": 0.851
}

BAND_RESOLUTIONS = { # In metres
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B08": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B8A": 20,
    "B11": 20,
    "B12": 20,
    "B01": 60,
    "B09": 60,
    "B10": 60
}

ORBIT_INCLINATION = np.deg2rad(98.62) # degrees
SPACECRAFT_ALTITUDE = 790_000 # meters
EARTH_RADIUS = 6_371_000 # meters
GRAV_CONST = 6.67430e-11 # m^3 kg^-1 s^-2
EARTH_MASS = 5.972e24 # kg
ORBITAL_VELOCITY = np.sqrt(GRAV_CONST * EARTH_MASS / (SPACECRAFT_ALTITUDE + EARTH_RADIUS)) # m/s