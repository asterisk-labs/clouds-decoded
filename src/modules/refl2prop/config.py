# refl2prop/config.py
from enum import Enum
from typing import List

# Bands used for the inversion (matching your notebook)
INVERSION_BANDS = ['B01', 'B02', 'B04', 'B08', 'B11', 'B12']

class InputFeature(str, Enum):
    """
    Strict definition of the input vector order for the Neural Network.
    Total inputs = 17 (6 bands + 6 albedos + 5 geometry/priors)
    """
    # 1. Reflectances (TOA)
    B01 = 'B01'
    B02 = 'B02'
    B04 = 'B04'
    B08 = 'B08'
    B11 = 'B11'
    B12 = 'B12'
    
    # 2. Surface Albedos (Priors)
    ALBEDO_B01 = 'B01_surface_albedo'
    ALBEDO_B02 = 'B02_surface_albedo'
    ALBEDO_B04 = 'B04_surface_albedo'
    ALBEDO_B08 = 'B08_surface_albedo'
    ALBEDO_B11 = 'B11_surface_albedo'
    ALBEDO_B12 = 'B12_surface_albedo'

    # 3. Geometry & Scene Priors
    INCIDENCE_ANGLE = 'incidence_angle' # Solar Zenith Angle
    SHADING_RATIO = 'shading_ratio'
    CLOUD_TOP_HEIGHT = 'cloud_top_height'
    MU = 'mu'   # Cosine of View Zenith
    PHI = 'phi' # Relative Azimuth

    @classmethod
    def list(cls) -> List[str]:
        return [param.value for param in cls]

class OutputFeature(str, Enum):
    """
    Strict definition of the output vector order (Physical Properties).
    """
    TAU = 'tau'
    ICE_LIQ_RATIO = 'ice_liq_ratio'
    R_EFF_LIQ = 'r_eff_liq'
    R_EFF_ICE = 'r_eff_ice'
    
    @classmethod
    def list(cls) -> List[str]:
        return [param.value for param in cls]


# --- New Configuration Implementation ---
from pydantic import Field
from clouds_decoded.shared_utils.config import BaseProcessorConfig

class Refl2PropConfig(BaseProcessorConfig):
    """
    Configuration for the Cloud Property Inversion (Refl2Prop) module.
    """
    # Resources
    model_path: str = Field(..., description="Path to the .pth model checkpoint")
    
    # Input Specification
    required_bands: List[str] = ["B01", "B02", "B04", "B08", "B11", "B12"]
    
    # Processing Parameters
    mask_invalid_height: bool = Field(True, description="If True, masks pixels with Cloud Height <= 0 as NaN")
    batch_size: int = Field(32768, description="Inference batch size (pixels)")
    output_resolution: Optional[int] = Field(None, description="Target spatial resolution in meters. If None, uses input resolution.")
    
    # Normalization Parameters (Raw -> Model Input)
    # Model expects: (value - offset) / scale
    # Default training used: (v - 1000) / 10000 for bands and albedo
    norm_bands_offset: float = 1000.0
    norm_bands_scale: float = 10000.0
    
    # Height was divided by 1000 in legacy code.
    norm_height_offset: float = 0.0
    norm_height_scale: float = 1000.0 
    
    # Default Fallbacks
    default_albedo: float = 0.1
    default_shading_ratio: float = 0.5
    
    # Output Features (Logic control)
    output_feature_names: List[str] = ['tau', 'ice_liq_ratio', 'r_eff_liq', 'r_eff_ice']
