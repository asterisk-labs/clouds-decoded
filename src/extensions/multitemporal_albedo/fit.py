"""Fit driver: stack -> cached model artifact.

The fitted model is stored as ``<project>/multitemporal/model.npz`` keyed by
the stack signature and the semantic dump of the fit parameters; re-running
with unchanged data and config is a no-op load.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import MultitemporalAlbedoParams
from .model import fit_cluster_model, predict_within_year, time_kernel

logger = logging.getLogger(__name__)

_MODEL_KEYS = ("assign", "valid_mask", "offset_v", "signal_num", "signal_den",
               "rw", "times", "doy", "year", "H", "W", "K", "B", "T",
               "sigma_t", "sigma_doy", "lam", "d0")


def params_hash(params: MultitemporalAlbedoParams) -> str:
    payload = json.dumps(params.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _model_key(stack_sig: str, p_hash: str) -> str:
    return f"{stack_sig}_{p_hash}"


def load_model(model_path: Path, expected_key: str) -> Optional[Dict]:
    """Load a cached model if it matches the expected data+params key."""
    if not model_path.exists():
        return None
    z = np.load(model_path, allow_pickle=True)
    if str(z.get("key", "")) != expected_key:
        logger.info("cached model stale (data or params changed) — refitting")
        return None
    model = {k: z[k] for k in _MODEL_KEYS}
    model["ord0"] = float(z["ord0"])
    model["transform"] = z["transform"]
    model["crs"] = str(z["crs"])
    model["bands"] = [str(b) for b in z["bands"]]
    model["key"] = str(z["key"])
    return model


def fit_and_save(stack: Dict, stack_sig: str,
                 params: MultitemporalAlbedoParams,
                 model_path: Path) -> Dict:
    """Fit the cluster model on a stack and persist it.

    Returns the model dict (including grid geometry carried over from the
    stack, which prediction consumers need).
    """
    key = _model_key(stack_sig, params_hash(params))
    cached = load_model(model_path, key)
    if cached is not None:
        logger.info("using cached multitemporal model (%s)", model_path.name)
        return cached

    if params.qc_holdout:
        _run_qc_holdout(stack, params)

    model = fit_cluster_model(
        stack["refl"], stack["clear"], stack["times"], stack["doy"],
        stack["year"], params,
    )
    model["ord0"] = stack["ord0"]
    model["transform"] = stack["transform"]
    model["crs"] = stack["crs"]
    model["bands"] = [str(b) for b in stack["bands"]]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, key=key, **model)
    model["key"] = key
    logger.info("fitted multitemporal model saved -> %s", model_path)
    return model


def _run_qc_holdout(stack: Dict, params: MultitemporalAlbedoParams,
                    roi_size: int = 100, band_idx: int = 3) -> None:
    """Fit with a held-out block on a small central ROI and log MAE of the
    cluster model vs the per-pixel within-year baseline (B04 by default)."""
    year = stack["year"]
    years = np.unique(year)
    hold_year = params.qc_holdout_year or int(years[-2] if len(years) > 1
                                              else years[-1])
    lo, hi = params.qc_holdout_doy_window
    in_block = ((year == hold_year) & (stack["doy"] >= lo)
                & (stack["doy"] <= hi))
    if not in_block.any():
        logger.warning("QC holdout window contains no scenes — skipping QC")
        return

    T, B, H, W = stack["refl"].shape
    r0 = max(0, H // 2 - roi_size // 2)
    c0 = max(0, W // 2 - roi_size // 2)
    sl = np.s_[r0:r0 + roi_size, c0:c0 + roi_size]
    refl = stack["refl"][:, :, sl[0], sl[1]]
    clear = stack["clear"][:, sl[0], sl[1]]
    held = clear & in_block[:, None, None]
    train = clear & ~in_block[:, None, None]

    qc_params = params.model_copy(update={"qc_holdout": False})
    model = fit_cluster_model(refl, train, stack["times"], stack["doy"],
                              stack["year"], qc_params)

    h, w = refl.shape[2], refl.shape[3]
    held_idx = np.where(in_block)[0]
    err_model, err_base, n = 0.0, 0.0, 0
    Rf = np.nan_to_num(refl.astype(np.float32)).reshape(T, B, h * w)
    Kt = time_kernel(stack["times"], params.sigma_t)
    base = predict_within_year(Rf, train.reshape(T, h * w), Kt,
                               params.robust_iters)
    from .model import predict_reflectance
    for t in held_idx:
        pred = predict_reflectance(model, float(stack["times"][t]),
                                   float(stack["doy"][t]),
                                   int(stack["year"][t]))
        obs = refl[t, band_idx].astype(np.float32)
        m = held[t] & np.isfinite(pred[band_idx]) & np.isfinite(obs)
        if not m.any():
            continue
        err_model += np.abs(pred[band_idx] - obs)[m].sum()
        err_base += np.abs(
            base[t, band_idx].reshape(h, w) - obs)[m].sum()
        n += int(m.sum())
    if n:
        logger.warning(
            "QC holdout (%d cells, year %d doy %d-%d): cluster-model MAE "
            "%.4f vs within-year baseline %.4f",
            n, hold_year, lo, hi, err_model / n, err_base / n)
