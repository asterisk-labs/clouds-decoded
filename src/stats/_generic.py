"""Generic stats functions usable for any single- or multi-band raster step."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union

import numpy as np

if TYPE_CHECKING:
    from clouds_decoded.stats import StatsCaller


def mean(caller: "StatsCaller", step_name: str) -> Dict[str, Union[float, int]]:
    """Compute band-wise mean over valid (non-zero) pixels.

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
    band_names = getattr(data.metadata, "band_names", None)
    result: Dict[str, Union[float, int]] = {}
    for i, band in enumerate(arr):
        name = (band_names[i] if band_names and i < len(band_names)
                else (f"b{i:02d}" if arr.shape[0] > 1 else None))
        valid = band[band != 0]
        if valid.size == 0:
            continue
        prefix = f"{name}__" if name else ""
        result[f"{prefix}mean"] = float(valid.mean())
    return result

def median(caller: "StatsCaller", step_name: str) -> Dict[str, Union[float, int]]:
    """Compute band-wise median over valid (non-zero) pixels.

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
    band_names = getattr(data.metadata, "band_names", None)
    result: Dict[str, Union[float, int]] = {}
    for i, band in enumerate(arr):
        name = (band_names[i] if band_names and i < len(band_names)
                else (f"b{i:02d}" if arr.shape[0] > 1 else None))
        valid = band[band != 0]
        if valid.size == 0:
            continue
        prefix = f"{name}__" if name else ""
        result[f"{prefix}median"] = float(np.median(valid))
    return result


def percentiles(caller: "StatsCaller", step_name: str, percentiles: list[int] = list(range(101))) -> Dict[str, Union[float, int]]:
    """Compute per-band 0–100th percentiles and mean over valid (non-zero) pixels.

    Args:
        caller: The :class:`~clouds_decoded.stats.StatsCaller` for this run.
        step_name: The step whose output to analyse.

    Returns:
        Dict with keys ``{band}__p{pct:03d}``, ``{band}__mean``,
        ``{band}__n_pixels`` for each band.
        Returns an empty dict if the step output is unavailable.
    """
    data = caller.load(step_name)
    if data is None or data.data is None:
        return {}
    arr = data.data
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    band_names = getattr(data.metadata, "band_names", None)
    result: Dict[str, Union[float, int]] = {}
    for i, band in enumerate(arr):
        name = (band_names[i] if band_names and i < len(band_names)
                else (f"b{i:02d}" if arr.shape[0] > 1 else None))
        valid = band[band != 0]
        if valid.size == 0:
            continue
        pcts = np.percentile(valid, percentiles)
        prefix = f"{name}__" if name else ""
        for j, v in zip(percentiles, pcts):
            result[f"{prefix}p{j:03d}"] = float(v)
    return result
