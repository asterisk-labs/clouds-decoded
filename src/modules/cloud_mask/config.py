from typing import Optional, Literal, List
from pydantic import Field, field_validator
from clouds_decoded.config import BaseProcessorConfig


class PostProcessParams(BaseProcessorConfig):
    """Post-processing parameters for cloud masks.

    Allows different downstream applications (e.g., Albedo vs Cloud Height)
    to request mask properties specific to their needs.
    """
    output_resolution: Optional[int] = Field(
        default=None,
        ge=10,
        le=60,
        description="Output resolution (meters). None=native resolution"
    )

    classes_to_mask: List[int] = Field(
        default=[1, 2, 3],
        description="Class indices to mask (1=thick, 2=thin, 3=shadow)"
    )

    threshold_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for classification [0-1]"
    )

    buffer_size: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Dilation buffer around clouds (meters)"
    )

    binary_mask: bool = Field(
        default=True,
        description="True=binary output, False=categorical classes"
    )


class CloudMaskConfig(BaseProcessorConfig):
    """Configuration for Cloud Mask Processor.

    Supports two methods:
    - 'senseiv2': Deep learning model (SEnSeI-v2)
    - 'threshold': Simple reflectance thresholding
    """
    method: Literal["senseiv2", "threshold"] = Field(
        default="senseiv2",
        description="Detection method: 'senseiv2' (ML) or 'threshold' (simple)"
    )

    # SEnSeIv2 Parameters
    model_name: str = Field(
        default="SegFormerB2-S2-unambiguous",
        description="HuggingFace model name for SEnSeIv2"
    )
    output_style: str = Field(
        default="4-class",
        description="Output style: 'cloud-noncloud', '4-class', etc."
    )
    device: Optional[str] = Field(
        default=None,
        description="Compute device ('cuda', 'cpu', or None=auto)"
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Batch size for model inference"
    )
    resolution: int = Field(
        default=10,
        ge=10,
        le=60,
        description="Input resolution for model (meters)"
    )

    # Threshold Parameters
    threshold_band: str = Field(
        default="B08",
        description="Band for threshold method"
    )
    threshold_value: float = Field(
        default=1600.0,
        ge=0,
        le=10000,
        description="Reflectance threshold (DN, 0-10000)"
    )

    @field_validator('threshold_band')
    @classmethod
    def validate_threshold_band(cls, v):
        """Validate threshold band is a valid Sentinel-2 band."""
        valid_bands = {'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                       'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'}
        if v not in valid_bands:
            raise ValueError(f"Invalid band: {v}. Must be one of {valid_bands}")
        return v

