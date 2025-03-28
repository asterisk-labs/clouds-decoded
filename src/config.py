
import defaults
import yaml as yaml
import numpy as np
from pprint import pprint


class CloudHeightConfig:

    def __init__(self,config_file):
        """
        Load the configuration from a YAML file, using defaults for any missing values

        Parameters:
            config_file: str, Path to the configuration file
    
        Returns:
            None
        """

        if config_file is not None:
            try:
                with open(config_file,'r') as f:
                    config = yaml.load(f,Loader=yaml.FullLoader)
            except FileNotFoundError:
                print(f"Configuration file not found: {config_file}")
        else:
            config = {}
        try:
            self.scene_dir = config['SCENE_DIR']
        except KeyError:
            raise KeyError("SCENE_DIR not found in configuration file, this is a required parameter")

        self.hack_image_azimuth = config.get('HACK_IMAGE_AZIMUTH',defaults.HACK_IMAGE_AZIMUTH)
        self.n_workers = config.get('N_WORKERS',defaults.N_WORKERS)
        self.cloudy_thresh = config.get('CLOUDY_THRESH',defaults.CLOUDY_THRESH)
        self.threshold_band = config.get('THRESHOLD_BAND',defaults.THRESHOLD_BAND)
        self.stride = config.get('STRIDE',defaults.STRIDE)
        self.along_track_resolution = config.get('ALONG_TRACK_RESOLUTION',defaults.ALONG_TRACK_RESOLUTION)
        self.across_track_resolution = config.get('ACROSS_TRACK_RESOLUTION',defaults.ACROSS_TRACK_RESOLUTION)
        self.convolved_size_along_track = config.get('CONVOLVED_SIZE_ALONG_TRACK',defaults.CONVOLVED_SIZE_ALONG_TRACK)
        self.convolved_size_across_track = config.get('CONVOLVED_SIZE_ACROSS_TRACK',defaults.CONVOLVED_SIZE_ACROSS_TRACK)
        self.max_height = config.get('MAX_HEIGHT',defaults.MAX_HEIGHT)
        self.height_step = config.get('HEIGHT_STEP',defaults.HEIGHT_STEP)
        self.heights = np.arange(0,self.max_height,self.height_step)
        if self.heights[-1] != self.max_height:
            self.heights = np.append(self.heights,self.max_height)
        self.bands = config.get('BANDS',defaults.BANDS)
        self.target_features = config.get('TARGET_FEATURES',defaults.TARGET_FEATURES)

        self.plot_writeto = config.get('PLOT_WRITETO',None)

        # Probably shouldn't change this from B02!
        self.reference_band = config.get('reference_band',defaults.REFERENCE_BAND)

        # print out the configuration
        print('Starting cloud height algorithm with the following configuration:')    
        pprint(vars(self))




