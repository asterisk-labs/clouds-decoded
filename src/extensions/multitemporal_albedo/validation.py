"""Time-series admissibility checks for the multitemporal albedo extension.

Cheap checks run before any heavy compute: they need only the staged scene
list (tile / date parsed from product names), not the imagery.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import MultitemporalAlbedoParams
from .stack import scene_datetime, scene_tile

logger = logging.getLogger(__name__)


class TimeSeriesValidationError(ValueError):
    """The staged scenes do not form a usable multitemporal series."""


def validate_time_series(
    scene_rows: List[Tuple[str, Optional[str]]],
    params: MultitemporalAlbedoParams,
) -> Dict:
    """Validate that staged scenes form a fit-worthy time series.

    Args:
        scene_rows: ``(scene_path, crop_window)`` for every staged scene.
        params: Extension parameters (thresholds).

    Returns:
        Summary dict: ``tile``, ``n_scenes``, ``years``, ``span_days``.

    Raises:
        TimeSeriesValidationError: With an actionable message on any failure.
    """
    from pathlib import Path

    if not scene_rows:
        raise TimeSeriesValidationError(
            "No scenes are staged. Stage the tile's scenes before running "
            "the multitemporal albedo stage.")

    crops = {cw for _, cw in scene_rows}
    if len(crops) > 1:
        raise TimeSeriesValidationError(
            "Scenes are staged with mixed crop windows "
            f"({sorted(str(c) for c in crops)}). The multitemporal fit "
            "needs one shared grid — stage every scene full-tile or with "
            "the same crop window.")
    crop_window = next(iter(crops))
    if crop_window is not None:
        from .stack import crop_grid
        try:
            crop_grid(crop_window, params.grid_res)
        except ValueError as exc:
            raise TimeSeriesValidationError(str(exc)) from exc

    sids = [Path(p).stem for p, _ in scene_rows]
    bad = [s for s in sids if len(s.split("_")) < 6]
    if bad:
        raise TimeSeriesValidationError(
            "Scene names not parseable as Sentinel-2 products: "
            + ", ".join(bad[:5]))

    tiles = Counter(scene_tile(s) for s in sids)
    if len(tiles) > 1:
        raise TimeSeriesValidationError(
            "Multitemporal albedo requires a single-tile project, got: "
            + ", ".join(f"{t} ({n} scenes)" for t, n in tiles.most_common())
            + ". Split the project per tile.")

    if len(sids) < params.min_scenes:
        raise TimeSeriesValidationError(
            f"Only {len(sids)} scene(s) staged; the multitemporal fit needs "
            f"at least {params.min_scenes} (min_scenes). Stage more of the "
            "tile's archive or lower the threshold.")

    dts = sorted(scene_datetime(s) for s in sids)
    years = sorted({d.year for d in dts})
    span = (dts[-1] - dts[0]).days
    if len(years) < params.min_years:
        raise TimeSeriesValidationError(
            f"Series covers {len(years)} year(s) ({years}); need at least "
            f"{params.min_years} for the cross-year kernel to help.")
    if span < params.min_date_span_days:
        raise TimeSeriesValidationError(
            f"Series spans only {span} days; need at least "
            f"{params.min_date_span_days}.")

    summary = {"tile": next(iter(tiles)), "n_scenes": len(sids),
               "years": years, "span_days": span,
               "crop_window": crop_window}
    logger.info("time series OK: tile %s, %d scenes, %d years, %d days%s",
                summary["tile"], summary["n_scenes"], len(years), span,
                f", crop {crop_window}" if crop_window else "")
    return summary
