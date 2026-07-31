"""Full-tile multitemporal stack assembly.

Builds a compact ``(T, B, G, G)`` float16 reflectance stack on a fixed
full-tile analysis grid (default 180 m -> 610x610) from a project's scenes
and their completed ``cloud_mask.tif`` outputs. Each scene fills only its
swath; per pixel, clear observations accumulate from whichever orbits cover
it.

Reflectance is read via fast decimated JP2 reads and calibrated manually:
``refl = (DN + radio_add_offset) / quantification_value``. Nodata is taken
from raw B02 DN==0 at native resolution (NOT from reflectance — the -1000
offset maps nodata to -0.1). ``clear = (cloud_mask == 0) & valid``.

Per-scene extractions are cached under ``<project>/multitemporal/scene_cache``
so the stack is cheap to rebuild as more masks complete.
"""
from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from clouds_decoded.constants import BANDS

logger = logging.getLogger(__name__)

# Sentinel-2 tile is 10980 px at 10 m -> 109800 m across.
TILE_EXTENT_M = 109_800

_SCENE_FIELDS = ("refl", "clear", "valid", "sid", "date", "doy", "year",
                 "orbit", "ord")


def grid_size(grid_res: int) -> int:
    """Analysis grid dimension for a resolution in metres (180 -> 610)."""
    return TILE_EXTENT_M // grid_res


def scene_datetime(scene_id: str) -> datetime:
    """Acquisition datetime parsed from a Sentinel-2 product name."""
    return datetime.strptime(scene_id.split("_")[2], "%Y%m%dT%H%M%S")


def scene_tile(scene_id: str) -> str:
    """MGRS tile token (e.g. ``T37VCC``) from a Sentinel-2 product name."""
    return scene_id.split("_")[5]


def _find_band_jp2(safe: Path, band: str) -> Optional[str]:
    hits = glob.glob(str(safe / "GRANULE" / "*" / "IMG_DATA" / f"*_{band}.jp2"))
    return hits[0] if hits else None


def _b02_valid_fraction(safe: Path, grid: int) -> Optional[np.ndarray]:
    """Per-cell fraction of valid (DN != 0) native B02 pixels.

    Thresholding must happen at native resolution (before averaging), so the
    full-res B02 is read once and block-reduced.
    """
    import rasterio

    jp2 = _find_band_jp2(safe, "B02")
    if jp2 is None:
        return None
    with rasterio.open(jp2) as src:
        full = src.read(1)
    n = full.shape[0] // grid
    if n < 1:
        return None
    full = full[: n * grid, : n * grid]
    return (full > 0).reshape(grid, n, grid, n).mean((1, 3)).astype(np.float32)


def read_grid_transform(safe: Path, grid: int):
    """Affine transform + CRS of the analysis grid, from the scene's B02."""
    import rasterio
    from rasterio.transform import Affine

    jp2 = _find_band_jp2(safe, "B02")
    if jp2 is None:
        raise FileNotFoundError(f"No B02 JP2 under {safe}")
    with rasterio.open(jp2) as src:
        scale = src.width / grid
        transform = src.transform * Affine.scale(scale)
        crs = str(src.crs)
    return transform, crs


def extract_scene(scene_path: str, mask_path: Path,
                  grid: int) -> Optional[Dict]:
    """Extract one scene's reflectance + clear mask on the analysis grid."""
    import rasterio
    from rasterio.enums import Resampling

    from clouds_decoded.data import Sentinel2Scene

    safe = Path(scene_path)
    sid = safe.stem

    # Calibration metadata only (quantification value + per-band offsets).
    sc = Sentinel2Scene()
    sc.read(str(safe), bands=["B02"])
    quant = float(sc.quantification_value)
    offsets = sc.radio_add_offset or {}

    vf = _b02_valid_fraction(safe, grid)
    if vf is None:
        return None
    # Keep cells the swath majority-covers; thin edges get unbiased values.
    valid = vf >= 0.5

    refl = np.full((len(BANDS), grid, grid), np.nan, np.float16)
    for bi, band in enumerate(BANDS):
        jp2 = _find_band_jp2(safe, band)
        if jp2 is None:
            continue
        with rasterio.open(jp2) as src:
            # Average decimation INCLUDES nodata (DN=0); unbias by /valid_frac.
            dn = src.read(1, out_shape=(grid, grid),
                          resampling=Resampling.average)
        off = float(offsets.get(band, 0.0) or 0.0)
        dn_valid = dn.astype(np.float32) / np.maximum(vf, 1e-6)
        r = (dn_valid + off) / quant
        r[~valid] = np.nan
        refl[bi] = r.astype(np.float16)

    with rasterio.open(mask_path) as src:
        m = src.read(
            1, out_shape=(grid, grid), resampling=Resampling.nearest)
    clear = (m == 0) & valid

    dt = scene_datetime(sid)
    return {
        "refl": refl, "clear": clear, "valid": valid, "sid": sid,
        "date": sid.split("_")[2][:8], "doy": dt.timetuple().tm_yday,
        "year": dt.year, "orbit": sid.split("_")[4],
        "ord": dt.toordinal() + dt.hour / 24,
    }


def build_stack(scene_rows: List[Tuple[str, Path]], cache_dir: Path,
                grid_res: int) -> Dict:
    """Assemble the stack from ``(scene_path, cloud_mask_path)`` rows.

    Per-scene extractions are cached as npz in ``cache_dir`` and reused.

    Returns:
        Stack dict: ``refl (T,B,G,G) float16``, ``clear/valid (T,G,G) bool``,
        ``times (T,) float64`` days since first obs, ``ord0`` first-obs
        ordinal, ``doy``, ``year``, ``orbit``, ``dates``, ``sids``,
        ``transform`` (6-tuple), ``crs``, ``bands``.
    """
    grid = grid_size(grid_res)
    cache_dir.mkdir(parents=True, exist_ok=True)

    recs: List[Dict] = []
    n_new = 0
    transform = crs = None
    for scene_path, mask_path in scene_rows:
        sid = Path(scene_path).stem
        cf = cache_dir / f"{sid}.npz"
        if cf.exists():
            d = np.load(cf, allow_pickle=True)
            recs.append({f: (d[f].item() if d[f].ndim == 0 else d[f])
                         for f in _SCENE_FIELDS})
            continue
        if not mask_path.exists():
            logger.warning("skip %s: no cloud mask at %s", sid, mask_path)
            continue
        try:
            rec = extract_scene(scene_path, mask_path, grid)
        except Exception as exc:
            logger.warning("skip %s: %s", sid, exc)
            continue
        if rec is None:
            logger.warning("skip %s: could not read B02", sid)
            continue
        np.savez_compressed(cf, **{f: rec[f] for f in _SCENE_FIELDS})
        recs.append(rec)
        n_new += 1
        if n_new % 20 == 0:
            logger.info("extracted %d new scenes", n_new)

    if not recs:
        raise RuntimeError(
            "No scenes could be added to the multitemporal stack — are the "
            "cloud masks in place?")

    # Grid geometry from the first available scene (all scenes share a tile).
    for scene_path, _ in scene_rows:
        if Path(scene_path).exists():
            transform, crs = read_grid_transform(Path(scene_path), grid)
            break
    if transform is None:
        raise RuntimeError("Could not derive grid transform from any scene.")

    logger.info("assembled %d scenes (%d newly extracted, %d cached)",
                len(recs), n_new, len(recs) - n_new)

    recs.sort(key=lambda r: r["ord"])
    ords = np.array([r["ord"] for r in recs], np.float64)
    stack = {
        "refl": np.stack([r["refl"] for r in recs]),
        "clear": np.stack([r["clear"] for r in recs]),
        "valid": np.stack([r["valid"] for r in recs]),
        "times": (ords - ords.min()),
        "ord0": float(ords.min()),
        "doy": np.array([r["doy"] for r in recs], np.float64),
        "year": np.array([r["year"] for r in recs], np.int16),
        "orbit": np.array([str(r["orbit"]) for r in recs]),
        "dates": np.array([str(r["date"]) for r in recs]),
        "sids": np.array([str(r["sid"]) for r in recs]),
        "transform": np.array(
            [transform.a, transform.b, transform.c,
             transform.d, transform.e, transform.f], np.float64),
        "crs": str(crs),
        "bands": np.array(BANDS),
    }
    valid = stack["valid"]
    clear = stack["clear"]
    logger.info(
        "stack T=%d grid=%dx%d  mean coverage %.0f%%  mean clear-of-valid %.0f%%",
        len(recs), grid, grid, 100 * valid.mean(),
        100 * (clear[valid].mean() if valid.any() else 0.0))
    return stack


def stack_signature(stack: Dict) -> str:
    """Cheap content signature: scene ids + grid shape."""
    import hashlib

    h = hashlib.sha256()
    for sid in stack["sids"]:
        h.update(str(sid).encode())
    h.update(str(stack["refl"].shape).encode())
    return h.hexdigest()[:16]
