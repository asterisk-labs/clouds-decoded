"""georvq S2 stores as a band source.

The manifest points at a ``.georvq.zarr`` store and fixes the quality
rung, so one store can back a product directory per rung:

    {"reader": "georvq", "store": "/data/.../<id>.georvq.zarr",
     "quality": "8-4", "bf16": false, "device": null}

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

_INSTALL = ("pip install georvq (the georvq S2 v1 release; model weights are "
            "fetched from huggingface.co/earth-compress/georvq-s2-v1 on first read)")


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
        return {"reader": self.name, "codec": "georvq", "quality": self.quality,
                "store": str(self.store)}


def write_georvq_safe(store: Union[str, Path], out_safe: Union[str, Path],
                      quality: Optional[str], bf16: bool = False) -> Path:
    """Make a compressed product backed by a georvq store.

    The skeleton comes from the store's own copy of the product metadata
    (``assets["safe_meta"]``); the band data stays in the store.

    Args:
        store: The ``.georvq.zarr`` store.
        out_safe: The ``.SAFE`` directory to create.
        quality: Rung the product decodes at, e.g. ``"8-4"`` (``None``: full depth).
        bf16: Decode in bfloat16 (faster on GPU; the published quality figures are fp32).

    Returns:
        ``out_safe``.
    """
    georvq = _georvq()
    with georvq.open(str(store), device="cpu") as ds:
        meta = ds.assets["safe_meta"]
        meta = meta if isinstance(meta, (bytes, bytearray)) else bytes(meta)
    make_shell(meta, out_safe)
    write_manifest(out_safe, "georvq", store=str(Path(store).absolute()), quality=quality,
                   bf16=bf16, device=None)
    return Path(out_safe)
