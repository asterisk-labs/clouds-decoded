"""Generic stats functions usable for any single- or multi-band raster step."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from clouds_decoded.stats import StatsCaller


def _band_name(band_names: Optional[list], i: int, n_bands: int) -> Optional[str]:
    """Return the display name for band index *i*, or None for unnamed single-band data."""
    if band_names and i < len(band_names):
        return band_names[i]
    return f"b{i:02d}" if n_bands > 1 else None


def _valid_mask(band: "np.ndarray", nodata: object = np.nan) -> "np.ndarray":
    """Return a boolean mask of valid (non-nodata) pixels.

    Uses ``isnan`` for NaN nodata, equality check otherwise.
    """
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        return np.isfinite(band)
    return band != nodata


def mean(caller: "StatsCaller", step_name: str) -> Dict[str, Union[float, int]]:
    """Compute band-wise mean over valid pixels (excluding nodata).

    Args:
        caller: The :class:`~clouds_decoded.stats.StatsCaller` for this run.
        step_name: The step whose output to analyse.

    Returns:
        Dict with keys ``{band}__mean`` and ``{band}__n_pixels`` for each band
        (prefix omitted for single-band data with no band names).
        Returns an empty dict if the step output is unavailable.
    """
    data = caller.load(step_name)
    if data is None or data.data is None:
        return {}
    arr = data.data
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    nodata = getattr(data, "nodata", np.nan)
    band_names = getattr(data.metadata, "band_names", None)
    result: Dict[str, Union[float, int]] = {}
    for i, band in enumerate(arr):
        name = _band_name(band_names, i, arr.shape[0])
        valid = band[_valid_mask(band, nodata)]
        if valid.size == 0:
            continue
        prefix = f"{name}__" if name else ""
        result[f"{prefix}mean"] = float(valid.mean())
        result[f"{prefix}n_pixels"] = int(valid.size)
    return result


def median(caller: "StatsCaller", step_name: str) -> Dict[str, Union[float, int]]:
    """Compute band-wise median over valid pixels (excluding nodata).

    Args:
        caller: The :class:`~clouds_decoded.stats.StatsCaller` for this run.
        step_name: The step whose output to analyse.

    Returns:
        Dict with keys ``{band}__median`` and ``{band}__n_pixels`` for each band
        (prefix omitted for single-band data with no band names).
        Returns an empty dict if the step output is unavailable.
    """
    data = caller.load(step_name)
    if data is None or data.data is None:
        return {}
    arr = data.data
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    nodata = getattr(data, "nodata", np.nan)
    band_names = getattr(data.metadata, "band_names", None)
    result: Dict[str, Union[float, int]] = {}
    for i, band in enumerate(arr):
        name = _band_name(band_names, i, arr.shape[0])
        valid = band[_valid_mask(band, nodata)]
        if valid.size == 0:
            continue
        prefix = f"{name}__" if name else ""
        result[f"{prefix}median"] = float(np.median(valid))
        result[f"{prefix}n_pixels"] = int(valid.size)
    return result


_DEFAULT_PERCENTILES = [0, 5, 25, 50, 75, 95, 100]


def percentiles(
    caller: "StatsCaller",
    step_name: str,
    percentiles: list[int] = _DEFAULT_PERCENTILES,
) -> Dict[str, Union[float, int]]:
    """Compute per-band percentiles and pixel count over valid pixels (excluding nodata).

    Args:
        caller: The :class:`~clouds_decoded.stats.StatsCaller` for this run.
        step_name: The step whose output to analyse.
        percentiles: Integer percentile values to compute (default: 0,5,25,50,75,95,100).

    Returns:
        Dict with keys ``{band}__p{pct:03d}`` and ``{band}__n_pixels`` for each band.
        Returns an empty dict if the step output is unavailable.
    """
    data = caller.load(step_name)
    if data is None or data.data is None:
        return {}
    arr = data.data
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    nodata = getattr(data, "nodata", np.nan)
    band_names = getattr(data.metadata, "band_names", None)
    result: Dict[str, Union[float, int]] = {}
    for i, band in enumerate(arr):
        name = _band_name(band_names, i, arr.shape[0])
        valid = band[_valid_mask(band, nodata)]
        if valid.size == 0:
            continue
        pcts = np.percentile(valid, percentiles)
        prefix = f"{name}__" if name else ""
        for j, v in zip(percentiles, pcts):
            result[f"{prefix}p{j:03d}"] = float(v)
        result[f"{prefix}mean"] = float(valid.mean())
        result[f"{prefix}n_pixels"] = int(valid.size)
    return result
