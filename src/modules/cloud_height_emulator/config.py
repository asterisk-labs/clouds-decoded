from __future__ import annotations

from typing import Optional, Tuple, List

from pydantic import Field, ConfigDict, model_validator

from clouds_decoded.config import BaseProcessorConfig


class CloudHeightEmulatorConfig(BaseProcessorConfig):
    """Configuration for Cloud Height Emulator Processor."""

    model_config = ConfigDict(extra='forbid')

    pth_path: Optional[str] = Field(
        default=None,
        description="Path to the model weights file (.pth)."
    )
    bands: List[str] = Field(
        default=["B02", "B03", "B04", "B08", "B11", "B12","B09","B10"],
        description="Bands to use for inference."
    )
    window_size: Tuple[int, int] = Field(
        default=(1024, 1024),
        description="Size of the sliding window for inference (height, width)."
    )
    overlap: int = Field(
        default=512,
        ge=0,
        description="Overlap between windows in pixels."
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for inference."
    )
    in_channels: int = Field(
        default=8,
        ge=1,
        description="Number of input channels expected by the model."
    )
    device: Optional[str] = Field(
        default="cuda",
        description="Device to run inference on (e.g., 'cuda', 'cpu'). If None, auto-detects."
    )

    @model_validator(mode='after')
    def check_overlap_window_size(self) -> CloudHeightEmulatorConfig:
        """Ensure overlap is smaller than window size."""
        height, width = self.window_size
        if self.overlap >= height or self.overlap >= width:
            raise ValueError(f"Overlap ({self.overlap}) must be smaller than window dimensions {self.window_size}")
        return self
