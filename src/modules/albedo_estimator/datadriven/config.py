"""Configuration for the data-driven albedo estimation model."""
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field, model_validator

from clouds_decoded.constants import BANDS

OUTPUT_BANDS: Tuple[str, ...] = tuple(BANDS)

INPUT_FEATURE_NAMES: Tuple[str, ...] = (
    "sin_lat", "cos_lat",
    "sin_lon", "cos_lon",
    "cos_sun_zenith",
    "cos_view_zenith",
    "sin_rel_azimuth", "cos_rel_azimuth",
    "wind_speed",
    "sin_wind_direction", "cos_wind_direction",
    "log_depth",
    "sin_doy", "cos_doy",
    "detector_mod2",
)

NUM_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)
NUM_OUTPUT_BANDS = len(OUTPUT_BANDS)


class AlbedoModelConfig(BaseModel):
    """Configuration for training and inference of the data-driven albedo model."""

    # --- Data ---
    dataset_path: str = Field(
        default="scratch/data/albedo_training_dataset.parquet",
        description="Path to the parquet training dataset",
    )
    bands: List[str] = Field(
        default_factory=lambda: list(OUTPUT_BANDS),
        description="Output band names (order matches model output columns)",
    )

    # --- Model architecture ---
    hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Width of the first hidden layer; later layers halve progressively",
    )
    num_hidden_layers: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of hidden blocks in the MLP",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout rate (0 = no dropout)",
    )

    # --- Training ---
    batch_size: int = Field(default=4096, ge=64)
    epochs: int = Field(default=100, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    lr_decay_step: int = Field(default=30, ge=1, description="Halve LR every N epochs")
    lr_decay_factor: float = Field(default=0.5, gt=0, le=1.0)
    weight_decay: float = Field(default=1e-5, ge=0)
    early_stopping_patience: int = Field(default=15, ge=1)
    num_workers: int = Field(default=4, ge=0)

    # --- Paths ---
    model_path: Optional[str] = Field(
        default=None,
        description="Path to save/load the model checkpoint. "
                    "When None, resolved from the managed assets directory.",
    )
    plot_dir: str = Field(
        default="plots/albedo",
        description="Directory for validation scatter plots",
    )

    # --- Computed fields ---
    @computed_field
    @property
    def input_size(self) -> int:
        return NUM_INPUT_FEATURES

    @computed_field
    @property
    def output_size(self) -> int:
        return len(self.bands)

    @computed_field
    @property
    def input_feature_names(self) -> List[str]:
        return list(INPUT_FEATURE_NAMES)

    @model_validator(mode="after")
    def _resolve_model_path(self) -> "AlbedoModelConfig":
        """Resolve model_path from managed assets when not explicitly set."""
        if self.model_path is None:
            from clouds_decoded.assets import get_asset
            object.__setattr__(
                self, "model_path",
                str(get_asset("models/albedo_datadriven/default.pth")),
            )
        return self
