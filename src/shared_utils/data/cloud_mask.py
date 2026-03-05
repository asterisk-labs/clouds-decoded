from typing import Dict, List, Optional
import logging
from pydantic import Field
import numpy as np
from .base import GeoRasterData, Metadata

logger = logging.getLogger(__name__)

class CloudMaskMetadata(Metadata):
    """Metadata for cloud mask outputs."""
    categorical: bool = True
    classes: Dict[int, str] = {
        0: 'Clear',
        1: 'Thick Cloud',
        2: 'Thin Cloud',
        3: 'Cloud Shadow'
    }
    # Processing info
    method: Optional[str] = Field(
        default=None,
        description="Detection method used ('senseiv2', 'simple_threshold', etc.)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name (for ML-based methods)"
    )
    resolution: Optional[float] = Field(
        default=None,
        description="Processing resolution in meters"
    )
    # Threshold-specific
    threshold_band: Optional[str] = Field(
        default=None,
        description="Band used for thresholding"
    )
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold value used"
    )
    # Post-processing
    postprocessed: bool = Field(
        default=False,
        description="Whether post-processing was applied"
    )

class CloudMaskData(GeoRasterData):
    """
    Data model for cloud masks using the SEnSeI-v2 variants.
    Supports either discrete 4-class classification or 4-channel probability maps.
    """
    nodata: Optional[float] = Field(default=255, description="Nodata sentinel for uint8 cloud masks")
    metadata: CloudMaskMetadata = Field(default_factory=CloudMaskMetadata)

    def validate(self) -> bool:
        """Validate that data conforms to the categorical strictness."""
        if self.data is None:
            return True

        if self.metadata.categorical:
            # Discrete values check
            # Expecting 2D array of integers
            if self.data.ndim != 2:
                 # It's possible to have (1, H, W) for single band categorical
                 if self.data.ndim == 3 and self.data.shape[0] == 1:
                     pass
                 else:
                     return False

            allowed = set(self.metadata.classes.keys())
            if self.nodata is not None:
                allowed.add(int(self.nodata))
            unique_vals = np.unique(self.data)
            for val in unique_vals:
                if val not in allowed:
                    return False
        else:
            # Probabilities check
            # Expecting 3D array (4, H, W) with floats 0-1
            if self.data.ndim != 3:
                return False
            if self.data.shape[0] != 4:
                return False
            if not np.issubdtype(self.data.dtype, np.floating):
                 return False
            if self.data.min() < 0 or self.data.max() > 1:
                 return False

        return True

    def to_binary(
        self,
        positive_classes: Optional[List[int]] = None,
        dilation_pixels: int = 0
    ) -> 'CloudMaskData':
        """Convert mask to binary (cloud/no-cloud) for downstream use.

        Args:
            positive_classes: Class indices to treat as "cloud". Default [1, 2] (thick + thin cloud).
            dilation_pixels: Dilate mask by this many pixels (useful for buffer zones).

        Returns:
            New CloudMaskData with binary mask (0=clear, 1=cloud).
        """
        if self.data is None:
            raise ValueError("No data to transform")

        if positive_classes is None:
            positive_classes = [1, 2]  # Default: thick + thin cloud

        # Handle categorical vs probability
        if self.metadata.categorical:
            # Multi-class: extract specified classes
            binary = np.isin(self.data, positive_classes).astype(np.uint8)
        else:
            # Probability map: sum probabilities of positive classes
            if self.data.ndim == 3:
                prob_sum = sum(self.data[c] for c in positive_classes if c < self.data.shape[0])
                binary = (prob_sum > 0.5).astype(np.uint8)
            else:
                binary = (self.data > 0.5).astype(np.uint8)

        # Ensure 2D output
        if binary.ndim == 3 and binary.shape[0] == 1:
            binary = binary[0]

        # Apply dilation if requested
        if dilation_pixels > 0:
            from scipy.ndimage import binary_dilation
            struct = np.ones((dilation_pixels * 2 + 1, dilation_pixels * 2 + 1))
            binary = binary_dilation(binary, structure=struct).astype(np.uint8)

        # Create new CloudMaskData with binary metadata
        return CloudMaskData(
            data=binary,
            transform=self.transform,
            crs=self.crs,
            metadata=CloudMaskMetadata(
                categorical=True,
                classes={0: 'Clear', 1: 'Cloud'}
            )
        )
