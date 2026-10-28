"""georvq S2 stores as a band source.

The manifest points at a ``.georvq.zarr`` store, by a path relative to the
product directory (normally a copy in its ``IMG_DATA``) or an absolute
one. A store holds every quality rung, and a read at rung ``8-k`` fetches
only what that rung needs, so the rungs are the product's levels:

    {"reader": "georvq", "store": "GRANULE/<granule>/IMG_DATA/<id>.georvq.zarr",
     "bf16": false, "device": null, "default": "8-8",
     "levels": {"8-8": {"quality": "8-8"}, "8-4": {"quality": "8-4"}, ...}}

georvq codes all 13 bands jointly, so the whole product is one group:
any band read decodes every band of the window. ``device`` ``null``
picks CUDA when available; the ``CLOUDS_DECODED_GEORVQ_DEVICE``
environment variable overrides both.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rasterio.windows import Window

from ..band_source import BandGeoref, BandSource, require_optional
from ...constants import BANDS
from .shell import make_shell, write_manifest

_INSTALL = ('pip install "georvq @ git+https://github.com/earth-compress/georvq" (model '
            "weights are fetched from huggingface.co/earth-compress/georvq-s2-v1 on first read)")


def _georvq():
    return require_optional("georvq", "read georvq-compressed Sentinel-2 products", _INSTALL)


def _device(requested: Optional[str]) -> str:
    env = os.environ.get("CLOUDS_DECODED_GEORVQ_DEVICE")
    if env:
        return env
    if requested:
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class GeorvqBandSource(BandSource):
    """The bands of a georvq S2 store, decoded at one quality rung."""

    name = "georvq"

    def __init__(self, scene_directory: Path, manifest: Dict[str, Any]):
        georvq = _georvq()
        store = Path(manifest["store"])
        if not store.is_absolute():
            store = Path(scene_directory) / store
        self.store = store
        self.quality = manifest.get("quality")
        self.level = manifest.get("level", self.quality)
        self.device = _device(manifest.get("device"))
        self._ds = georvq.open(str(store), device=self.device,
                               bf16=bool(manifest.get("bf16", False)))
        self._grid_of: Dict[str, Tuple[str, int]] = {}
        self._shapes: Dict[str, Tuple[int, int]] = {}
        for grid in self._ds.grids:
            g = self._ds.grid(grid)
            for i, band in enumerate(g.descriptions):
                self._grid_of[band] = (grid, i)
                self._shapes[band] = (g.height, g.width)

    @property
    def groups(self) -> Dict[str, List[str]]:
        return {"all": [b for b in BANDS if b in self._grid_of]}

    def georef(self) -> BandGeoref:
        ds = self._ds
        return BandGeoref(ds.crs, ds.transform, ds.width, ds.height)

    def shape(self, band: str) -> Tuple[int, int]:
        return self._shapes[band]

    def _read_group(self, group: str, windows: Dict[str, Optional[Window]]) -> Dict[str, np.ndarray]:
        w = windows.get("B02")
        region = None if w is None else (
            (int(w.row_off), int(w.row_off + w.height)),
            (int(w.col_off), int(w.col_off + w.width)))
        grids = self._ds.read_grids(region, quality=self.quality)
        out = {}
        for band in self.groups[group]:
            grid, i = self._grid_of[band]
            a = grids[grid][i]
            bw = windows.get(band)
            if bw is not None:
                # a grid covers every pixel the window touches; keep the
                # pixels clouds-decoded's crop scaling assigns to it
                h, wd = int(bw.height), int(bw.width)
                if a.shape[0] < h or a.shape[1] < wd:
                    raise ValueError(f"georvq returned {a.shape} for {band}, expected ({h}, {wd})")
                a = a[:h, :wd]
            out[band] = np.ascontiguousarray(a)
        return out

    def describe(self) -> Dict[str, Any]:
        return {"reader": self.name, "codec": "georvq", "level": self.level,
                "quality": self.quality, "store": str(self.store)}


def _rung_key(rung: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in rung.split("-"))


def write_georvq_safe(store: Union[str, Path], out_safe: Union[str, Path],
                      levels: Optional[List[str]] = None, embed: bool = True,
                      bf16: bool = False) -> Dict[str, Any]:
    """Make a compressed product backed by a georvq store.

    The skeleton comes from the store's own copy of the product metadata
    (``assets["safe_meta"]``). One store serves every rung, so the product
    lists each rung as a level, reading the same store; the finest is the
    default.

    Args:
        store: The ``.georvq.zarr`` store.
        out_safe: The ``.SAFE`` directory to create.
        levels: Rungs to offer, e.g. ``["8-8", "8-4", "8-0"]`` (``None``: all
            of the store's rungs).
        embed: Copy the store into the product's ``IMG_DATA``, referenced by a
            relative path, so the directory is self-contained; otherwise
            reference the store where it is, by absolute path.
        bf16: Decode in bfloat16 (faster on GPU; the published quality figures are fp32).

    Returns:
        The manifest written.
    """
    import shutil

    georvq = _georvq()
    store = Path(store)
    with georvq.open(str(store), device="cpu") as ds:
        meta = ds.assets["safe_meta"]
        meta = meta if isinstance(meta, (bytes, bytearray)) else bytes(meta)
        rungs = ["-".join(map(str, r)) if not isinstance(r, str) else r for r in ds.rungs]
    if levels is None:
        levels = rungs
    missing = [q for q in levels if q not in rungs]
    if missing:
        raise ValueError(f"rungs {missing} are not in {store.name}; it has {rungs}")
    out_safe = Path(out_safe)
    granule = make_shell(meta, out_safe)
    if embed:
        dst = granule / "IMG_DATA" / store.name
        shutil.copytree(store, dst)
        ref = str(dst.relative_to(out_safe))
    else:
        ref = str(store.absolute())
    return write_manifest(
        out_safe, "georvq", store=ref, bf16=bf16, device=None,
        levels={q: {"quality": q} for q in levels},
        default=max(levels, key=_rung_key))
