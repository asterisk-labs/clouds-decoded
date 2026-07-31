"""Per-scene albedo pre-population.

Evaluates the fitted multitemporal model at each scene's sensing time and
writes a fully-provenanced ``albedo.tif`` plus manifest entry, so the
per-scene project run sees the albedo step as legitimately complete.

The provenance, config hash, and manifest updates are produced by the same
``Project`` methods the orchestrator later validates them with — nothing is
hand-rolled.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def predict_scene_albedo(model: Dict, scene_id: str,
                         default_albedo: Dict[str, float],
                         footprint: Optional[np.ndarray] = None):
    """Model prediction for one scene, ready to wrap in AlbedoData.

    Args:
        model: Fitted model dict (from :mod:`.fit`).
        scene_id: Sentinel-2 product name (provides the sensing time).
        default_albedo: Per-band constant fallback for pixels with too few
            clear observations to join the fit.
        footprint: Optional ``(H, W)`` bool — scene swath; pixels outside
            are set to NaN (nodata).

    Returns:
        ``(data, fallback_frac)`` — ``(B, H, W)`` float32 and the fraction
        of in-footprint pixels that used the constant fallback.
    """
    from .stack import scene_datetime

    dt = scene_datetime(scene_id)
    t_star = dt.toordinal() + dt.hour / 24 - model["ord0"]
    from .model import predict_reflectance

    pred = predict_reflectance(model, float(t_star),
                               float(dt.timetuple().tm_yday), dt.year)

    bands = model["bands"]
    invalid = ~np.isfinite(pred)
    fallback_mask = invalid.any(0)
    for bi, band in enumerate(bands):
        pred[bi][invalid[bi]] = default_albedo.get(band, 0.05)
    if footprint is not None:
        pred[:, ~footprint] = np.nan
        denom = max(int(footprint.sum()), 1)
        fallback_frac = float((fallback_mask & footprint).sum() / denom)
    else:
        fallback_frac = float(fallback_mask.mean())
    return pred, fallback_frac


def prepopulate_scene(project, model: Dict, scene_path: str,
                      albedo_config, git_hash: Optional[str] = None,
                      footprint: Optional[np.ndarray] = None) -> bool:
    """Write ``albedo.tif`` + manifest entry for one scene.

    Returns True if written, False if the step was already complete.
    """
    from rasterio.transform import Affine

    from clouds_decoded.config import BaseProcessorConfig
    from clouds_decoded.data.base import AlbedoData, AlbedoMetadata
    from clouds_decoded.project import StepResult

    scene_id = project._scene_id(scene_path)
    config_hash = project._config_hash("albedo")
    manifest = project._load_manifest(scene_id, scene_path)
    if manifest.is_step_complete("albedo", config_hash):
        return False

    started = datetime.now()
    data, fallback_frac = predict_scene_albedo(
        model, scene_id, albedo_config.default_albedo, footprint)

    a, b, c, d, e, f = (float(x) for x in model["transform"])
    result = AlbedoData(
        data=data,
        transform=Affine(a, b, c, d, e, f),
        crs=model["crs"],
        metadata=AlbedoMetadata(
            band_names=list(model["bands"]),
            method="multitemporal",
            n_training_samples=int(model["valid_mask"].sum()),
            clear_fraction=1.0 - fallback_frac,
            fallback_used=fallback_frac > 0,
            fallback_values=dict(albedo_config.default_albedo)
            if fallback_frac > 0 else {},
        ),
    )
    result = result.resample(albedo_config.output_resolution)

    config_dict = BaseProcessorConfig._semantic_dump(albedo_config)
    product_id = Path(scene_path).name.removesuffix(".SAFE")
    provenance = project._build_provenance(
        scene_path, product_id, "albedo", config_dict,
        crop_window=None, git_hash=git_hash,
    )
    result.metadata.provenance = provenance.model_dump()

    scene_out = project._scene_output_dir(scene_id)
    scene_out.mkdir(parents=True, exist_ok=True)
    output_path = scene_out / "albedo.tif"
    result.write(str(output_path))

    completed = datetime.now()
    manifest.steps["albedo"] = StepResult(
        status="completed",
        output_file=str(output_path),
        config_hash=config_hash,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        duration_seconds=(completed - started).total_seconds(),
    )
    project._save_manifest(scene_id, manifest)
    return True
