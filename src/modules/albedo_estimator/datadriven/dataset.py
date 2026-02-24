"""PyTorch Dataset for the data-driven albedo model."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import INPUT_FEATURE_NAMES, OUTPUT_BANDS

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> np.ndarray:
    """Transform raw sampler columns into engineered feature matrix.

    Applies physically motivated transforms so the model sees clean,
    bounded inputs. This function is the single source of truth for the
    transform — it is called both during training (here) and at inference
    time (in the processor).

    Args:
        df: DataFrame with raw columns from the albedo sampler:
            latitude, longitude, sun_zenith, sun_azimuth,
            view_zenith, view_azimuth, wind_speed, wind_direction,
            bathymetry, day_of_year, detector.

    Returns:
        float32 array of shape (N, 14), one column per entry in
        INPUT_FEATURE_NAMES.
    """
    n = len(df)
    features = np.empty((n, len(INPUT_FEATURE_NAMES)), dtype=np.float32)

    lat_rad = np.radians(df["latitude"].values)
    features[:, 0] = np.sin(lat_rad)
    features[:, 1] = np.cos(lat_rad)

    lon_rad = np.radians(df["longitude"].values)
    features[:, 2] = np.sin(lon_rad)
    features[:, 3] = np.cos(lon_rad)

    features[:, 4] = np.cos(np.radians(df["sun_zenith"].values))
    features[:, 5] = np.cos(np.radians(df["view_zenith"].values))

    rel_az_rad = np.radians(df["sun_azimuth"].values - df["view_azimuth"].values)
    features[:, 6] = np.sin(rel_az_rad)
    features[:, 7] = np.cos(rel_az_rad)

    features[:, 8] = df["wind_speed"].values

    wind_dir_rad = np.radians(df["wind_direction"].values)
    features[:, 9] = np.sin(wind_dir_rad)
    features[:, 10] = np.cos(wind_dir_rad)

    depth_positive = np.maximum(-df["bathymetry"].values, 1.0)
    features[:, 11] = np.log10(depth_positive)

    doy_rad = 2.0 * np.pi * df["day_of_year"].values / 365.25
    features[:, 12] = np.sin(doy_rad)
    features[:, 13] = np.cos(doy_rad)

    detector = df["detector"].values if "detector" in df.columns else np.zeros(n)
    features[:, 14] = detector.astype(np.float32) % 2

    # Replace any NaN (e.g. missing wind data) with 0
    np.nan_to_num(features, nan=0.0, copy=False)

    return features


class AlbedoDataset(Dataset):
    """PyTorch Dataset for albedo training from parquet.

    Loads the full split into memory (~100 MB as float32). Feature
    engineering is applied once at init.

    Args:
        parquet_path: Path to the combined parquet with a ``split`` column.
        split: One of ``'train'``, ``'val'``, ``'test'``.
        bands: Output band names. Defaults to all 13 Sentinel-2 bands.
    """

    def __init__(
        self,
        parquet_path: str,
        split: str = "train",
        bands: Optional[list] = None,
    ):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if bands is None:
            bands = list(OUTPUT_BANDS)

        # Prefer per-split file (albedo_train.parquet etc.) for fast loading
        split_path = path.parent / f"albedo_{split}.parquet"
        if split_path.exists():
            logger.info(f"Loading {split} split from {split_path}")
            df = pd.read_parquet(split_path)
        else:
            logger.info(f"Loading {split} split from {path}")
            df = pd.read_parquet(path)
            df = df[df["split"] == split]
        df = df.reset_index(drop=True)
        logger.info(f"  {len(df)} samples loaded")

        self.features = engineer_features(df)
        self.targets = df[bands].values.astype(np.float32)

        self.input_stats: Dict[str, list] = {
            "min": self.features.min(axis=0).tolist(),
            "max": self.features.max(axis=0).tolist(),
        }
        self.output_stats: Dict[str, list] = {
            "min": self.targets.min(axis=0).tolist(),
            "max": self.targets.max(axis=0).tolist(),
        }

        self.bands = bands
        self.split = split

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
        )
