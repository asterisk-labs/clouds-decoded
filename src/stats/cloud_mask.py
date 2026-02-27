"""Step-specific stats for the cloud_mask step."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union

import numpy as np

if TYPE_CHECKING:
    from clouds_decoded.stats import StatsCaller


def class_fractions(caller: "StatsCaller", step_name: str) -> Dict[str, Union[float, int]]:
    """Compute fraction of each class: clear, thin, thick, shadow.

    Args:
        caller: The :class:`~clouds_decoded.stats.StatsCaller` for this run.
        step_name: The step whose output to analyse (should be ``"cloud_mask"``).

    Returns:
        Dict with keys ``clear_frac``, ``thin_frac``, ``thick_frac``,
        ``shadow_frac``, and ``n_pixels``. Returns empty dict if the step
        output is unavailable or all pixels are nodata (255).
    """
    data = caller.load(step_name)
    if data is None or data.data is None:
        return {}
    arr = data.data
    if arr.ndim == 3:
        arr = arr[0]
    classes = data.metadata.classes  # Dict[int, str] from CloudMaskMetadata
    n = int(np.isin(arr, list(classes.keys())).sum())
    if n == 0:
        return {}
    result: Dict[str, Union[float, int]] = {}
    for class_idx, class_name in classes.items():
        key = class_name.lower().replace(" ", "_") + "_frac"
        result[key] = float((arr == class_idx).sum() / n)
    result["n_pixels"] = n
    return result
