from typing import Optional, Literal, List
from pydantic import Field, field_validator, model_validator
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import BANDS


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
    model_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to the cloud mask model weights (.pt). "
            "Defaults to the managed assets directory; run "
            "'clouds-decoded download cloud_mask' to fetch weights."
        ),
    )
    device: Optional[str] = Field(
        default=None,
        description="Compute device ('cuda', 'cpu', or None=auto)"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for model inference"
    )
    working_resolution: int = Field(
        default=15,
        ge=10,
        le=60,
        description="Resolution in metres at which inference is performed."
    )

    stride: int = Field(
        default=170,
        ge=1,
        le=256,
        description="Stride for tiling inputs to model (pixels at model resolution)"
    )

    # Shadow reclassification — fix self-shading artefacts inside clouds
    reclassify_embedded_shadow: bool = Field(
        default=True,
        description="Reclassify shadow pixels (class 3) surrounded by cloud as thick cloud. "
                    "Fixes self-shading artefacts predicted inside optically thick clouds.",
    )
    shadow_reclassify_radius: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Radius (pixels at working resolution) of the neighbourhood window "
                    "used to compare cloud vs clear fractions around shadow pixels.",
    )

    # Binarization parameters (applied after inference to produce the output mask)
    cloud_mask_classes: List[int] = Field(
        default=[1, 2],
        description="Class indices to treat as cloud for binarization (1=thick, 2=thin, 3=shadow).",
    )
    cloud_mask_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Probability threshold for the summed positive-class confidence. "
                    "Lower values are more permissive (catch more cloud).",
    )

    # Threshold Parameters
    threshold_band: str = Field(
        default="B08",
        description="Band for threshold method"
    )
    threshold_value: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Reflectance threshold for cloud detection (0-1)"
    )

    @model_validator(mode='after')
    def _resolve_model_path(self) -> 'CloudMaskConfig':
        """If no explicit path is given, point at the managed asset location."""
        if self.model_path is None:
            from clouds_decoded.assets import get_asset
            object.__setattr__(
                self, "model_path", str(get_asset("models/cloud_mask/default.pt"))
            )
        return self

    @field_validator('threshold_band')
    @classmethod
    def validate_threshold_band(cls, v):
        """Validate threshold band is a valid Sentinel-2 band."""
        if v not in set(BANDS):
            raise ValueError(f"Invalid band: {v}. Must be one of {BANDS}")
        return v

