from typing import List, Literal, Optional
import numpy as np
from pydantic import Field, field_validator
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import BANDS

class CloudHeightConfig(BaseProcessorConfig):
    """Configuration for Cloud Height Processor.

    Retrieves cloud top height from Sentinel-2 parallax using multi-band correlation.
    """
    # Core Parameters
    reference_band: str = Field(
        default='B02',
        description="Reference band (fixed while others shift for parallax)"
    )
    bands: List[str] = Field(
        default=['B01','B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08'],
        description="Bands to use for correlation (minimum 2 required)"
    )

    # Thresholding
    cloudy_thresh: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Reflectance threshold for cloud detection (0-1)"
    )
    threshold_band: str = Field(
        default='B08',
        description="Band to use for cloud thresholding"
    )

    # Spatial / Convolution
    along_track_resolution: int = Field(
        default=3,
        ge=1,
        le=60,
        description="Pixel size along track during convolution (meters)"
    )
    across_track_resolution: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Pixel size across track during convolution (meters)"
    )
    stride: int = Field(
        default=180,
        ge=10,
        le=5000,
        description="Stride between retrieval points (meters). Controls how "
                    "densely parallax correlations are sampled. Independent of "
                    "the output grid resolution."
    )
    grid_resolution: Optional[int] = Field(
        default=None,
        ge=10,
        le=5000,
        description="Output grid pixel size in metres. When None, defaults to "
                    "the retrieval stride. Set lower than stride to get a "
                    "smoother output without increasing retrieval density."
    )
    convolved_size_along_track: int = Field(
        default=200,
        ge=50,
        le=2000,
        description="Correlation window size along track (meters)"
    )
    convolved_size_across_track: int = Field(
        default=200,
        ge=50,
        le=2000,
        description="Correlation window size across track (meters)"
    )

    # Method
    correlation_weighting: bool = Field(
        default=True,
        description="Weight height estimates by correlation strength"
    )
    spatial_smoothing_sigma: float = Field(
        default=180.0,
        ge=0,
        le=5000,
        description="Gaussian smoothing kernel sigma (meters, 0=no smoothing)"
    )

    # Height Search Space
    min_height: int = Field(
        default=0,
        ge=-10000,
        le=0,
        description=(
            "Minimum cloud height to search (meters). 0 is the physical floor "
            "and the production default. Negative values are a DIAGNOSTIC: a "
            "detector whose retrievals come back systematically negative has "
            "its parallax direction inverted, which a 0-floored grid hides by "
            "pinning those columns at the floor instead."
        )
    )
    max_height: int = Field(
        default=18000,
        ge=1000,
        le=25000,
        description="Maximum cloud height to search (meters, troposphere limit ~18km)"
    )
    height_step: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Height search step size (meters)"
    )
    offset_rounding: Literal["truncate", "nearest"] = Field(
        default="nearest",
        description=(
            "How the per-band patch offset is rounded to a whole pixel. "
            "'truncate' is the historical behaviour (int(), i.e. floor for "
            "positive indices). Because the offset changes sign with detector "
            "parity, a floor bias of ~0.5 px maps to +0.5 px of height in one "
            "parity and -0.5 px in the other -- a full-pixel differential, "
            "worth ~600 m (B03) to ~1200 m (B08). 'nearest' removes the "
            "systematic part of that and is the default: truncation is a bug, "
            "not a modelling choice. Measured over 23 scenes, the B02-B03 seam "
            "step goes from +0.299 px under 'truncate' to 0.000 px under "
            "'nearest', matching an independent implementation. Set 'truncate' "
            "only to reproduce a product built before this default changed."
        )
    )
    tie_break: Literal["floor", "centre"] = Field(
        default="centre",
        description=(
            "Which height to report when several candidates share the winning "
            "score. Because the patch offset is rounded to a whole cell of the "
            "along-track grid, every height inside one cell-crossing produces "
            "BYTE-IDENTICAL patches and therefore an identical score: the "
            "score-vs-height curve is a staircase. All that a tied tread tells "
            "you is that the height lies somewhere inside it. "
            "'floor' is the historical behaviour: "
            "np.nanargmax returns the FIRST maximum and the height grid ascends "
            "from min_height, so the lowest height of the tread is reported. "
            "That biases every retrieval low by (W - height_step)/2, "
            "which is why two-band pairs read below ground over low cloud. "
            "'centre' reports the midpoint of the tied tread instead, which is "
            "unbiased and halves the worst-case error, and is the default: "
            "reporting the floor of a tie is a bias, not a choice. It does NOT "
            "improve resolution: the tread width is still the uncertainty. "
            "Set 'floor' only to reproduce a pre-existing product. "
            "Mixed band sets are barely affected: with 13 bands the score changes "
            "whenever ANY band crosses a cell, so treads are ~18 m, narrower "
            "than a typical height_step, and there are few ties to break."
        )
    )
    quality_bands: bool = Field(
        default=False,
        description=(
            "Append per-cell quality metrics to the output raster. Band 0 stays "
            "the height, so existing consumers are unaffected; the extra bands "
            "are named in the output metadata. They are:\n"
            "  peak_correlation -- the winning score. Signal strength.\n"
            "  fwhm -- width of the correlation peak at half its amplitude, in "
            "metres. The retrieval's PRECISION.\n"
            "  tied_span -- width of the exactly-tied run, in metres. The hard "
            "resolution floor imposed by rounding the patch offset to a whole "
            "cell: heights inside it are indistinguishable by construction.\n"
            "All three are computed from the retrieval's own score curve and so "
            "all three measure PRECISION, not accuracy."
        )
    )

    # System
    use_emulator: bool = Field(
        default=False,
        description="Use deep learning emulator for cloud height retrieval"
    )
    n_workers: int = Field(
        default=96,
        ge=1,
        description="Number of parallel workers for processing"
    )
    temp_dir: Optional[str] = Field(
        default=None,
        description="Temporary directory for intermediate files (default: /dev/shm)"
    )
    stall_timeout: float = Field(
        default=900.0,
        gt=0,
        description=(
            "Seconds to wait for the next column result before declaring the "
            "scene stalled and aborting it. A process that dies holding the "
            "work queue's internal lock blocks every other worker without "
            "raising anything, so an unbounded wait never returns. Must exceed "
            "the slowest single column; columns typically take 1-2 s."
        )
    )

    @field_validator('bands')
    @classmethod
    def validate_bands(cls, v):
        """Ensure at least 2 bands for correlation."""
        if len(v) < 2:
            raise ValueError("At least 2 bands required for parallax correlation")
        return v

    @field_validator('reference_band')
    @classmethod
    def validate_reference_band(cls, v):
        """Validate reference band is a valid Sentinel-2 band."""
        if v not in set(BANDS):
            raise ValueError(f"Invalid band: {v}. Must be one of {BANDS}")
        return v

    @property
    def heights(self) -> np.ndarray:
        """Derived property: Array of heights to search."""
        hs = np.arange(self.min_height, self.max_height, self.height_step)
        if hs[-1] != self.max_height:
            hs = np.append(hs, self.max_height)
        return hs
