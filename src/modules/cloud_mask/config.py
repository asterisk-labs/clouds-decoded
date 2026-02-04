from typing import Optional, Literal
from pydantic import Field
from clouds_decoded.config import BaseProcessorConfig


class PostProcessParams(BaseProcessorConfig):
    """
    Configuration parameters for post-processing the cloud mask.
    This allows different downstream applications (e.g. Albedo vs Cloud Height)
    to request different mask properties specific to their needs.
    """
    # Output Resolution
    output_resolution: Optional[int] = Field(None, description="Output resolution in meters. If None, uses the mask's native resolution.")
    
    # Class Selection & Confidence
    classes_to_mask: list[int] = Field([1, 2, 3], description="List of class indices to consider as clouds/mask. Defaults: 1=thick, 2=thin, 3=shadow.")
    threshold_confidence: float = Field(0.5, description="Confidence threshold for categorisation.")
    
    # Morphological Operations
    buffer_size: int = Field(0, description="Buffer size in meters to dilate cloud edges (morphological dilation).")
    
    # Categorisation
    binary_mask: bool = Field(True, description="If True, returns a binary mask (cloud/no-cloud). If False, returns the categorical mask.")


class CloudMaskConfig(BaseProcessorConfig):
    """
    Configuration for Cloud Mask Processor.
    """
    method: Literal["senseiv2", "threshold"] = Field("senseiv2", description="Method to use for cloud masking.")

    # SEnSeIv2 Parameters
    model_name: str = Field("SegFormerB2-S2-unambiguous", description="HuggingFace model name for SEnSeIv2.")
    output_style: str = Field("4-class", description="Output style: 'cloud-noncloud', '4-class', etc.")
    device: Optional[str] = Field(None, description="Device to run inference on. If None, auto-detects cuda/cpu.")
    batch_size: int = Field(4, description="Batch size for model inference.")
    resolution: int = Field(10, description="Target spatial resolution in meters for the INPUT to the model. Bands will be resampled to this resolution before inference.")
    
    # Threshold Parameters
    threshold_band: str = Field("B08", description="Band to use for simple thresholding.")
    threshold_value: float = Field(1600.0, description="Pixel value threshold for cloud detection.")

