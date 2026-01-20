import os

from . import defaults
import yaml
import numpy as np
from pprint import pprint

import uuid


class CloudHeightConfig:

    def __init__(self,config_file,scene_dir):
        """
        Load the configuration from a YAML file, using defaults for any missing values

        Parameters:
            config_file: str, Path to the configuration file
            scene_dir: str, Path to the scene directory
    
        Returns:
            None
        """

        if config_file is not None:
            try:
                with open(config_file,'r') as f:
                    config = yaml.load(f,Loader=yaml.FullLoader)
            except FileNotFoundError:
                print(f"Configuration file {config_file} not found. Using default parameters.")
        else:
            config = {}


        self.scene_dir = scene_dir
    
        self.n_workers = config.get('N_WORKERS',defaults.N_WORKERS)
        self.cloudy_thresh = config.get('CLOUDY_THRESH',defaults.CLOUDY_THRESH)
        self.threshold_band = config.get('THRESHOLD_BAND',defaults.THRESHOLD_BAND)
        self.stride = config.get('STRIDE',defaults.STRIDE)
        self.along_track_resolution = config.get('ALONG_TRACK_RESOLUTION',defaults.ALONG_TRACK_RESOLUTION)
        self.across_track_resolution = config.get('ACROSS_TRACK_RESOLUTION',defaults.ACROSS_TRACK_RESOLUTION)
        self.convolved_size_along_track = config.get('CONVOLVED_SIZE_ALONG_TRACK',defaults.CONVOLVED_SIZE_ALONG_TRACK)
        self.convolved_size_across_track = config.get('CONVOLVED_SIZE_ACROSS_TRACK',defaults.CONVOLVED_SIZE_ACROSS_TRACK)
        self.correlation_weighting = config.get('CORRELATION_WEIGHTING',defaults.CORRELATION_WEIGHTING)
        self.max_height = config.get('MAX_HEIGHT',defaults.MAX_HEIGHT)
        self.height_step = config.get('HEIGHT_STEP',defaults.HEIGHT_STEP)
        self.heights = np.arange(0,self.max_height,self.height_step)
        if self.heights[-1] != self.max_height:
            self.heights = np.append(self.heights,self.max_height)
        self.bands = config.get('BANDS',defaults.BANDS)
        self.spatial_smoothing_sigma = config.get('SPATIAL_SMOOTHING_SIGMA',defaults.SPATIAL_SMOOTHING_SIGMA)    

        self.temp_dir = config.get('TEMP_DIR',f"/dev/shm/cloudheight_temp_{uuid.uuid4()}")
        self.output_dir = config.get('OUTPUT_DIR',defaults.OUTPUT_DIR)

        if self.scene_dir[-1]=="/":
            product_id = os.path.splitext(self.scene_dir.split("/")[-2])[0]
        else:
            product_id =  os.path.splitext(os.path.basename(self.scene_dir))[0]
        self.plot_writeto = os.path.join(self.output_dir,"plots",f"{product_id}.png")
        self.log_writeto = os.path.join(self.output_dir,"log",product_id)
        self.pcloud_writeto = os.path.join(self.output_dir,"pcloud",f"{product_id}.npz")


        # Probably shouldn't change this from B02!
        self.reference_band = config.get('reference_band',defaults.REFERENCE_BAND)
