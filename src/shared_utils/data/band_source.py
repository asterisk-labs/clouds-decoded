"""Where a scene's band images come from.

A ``Sentinel2Scene`` reads its metadata (``MTD_MSIL1C.xml``, ``MTD_TL.xml``,
``QI_DATA``, ``AUX_DATA``) from the ``.SAFE`` directory and its band images
through a ``BandSource``. For a standard product that is ``Jp2BandSource``,
the 13 JPEG 2000 images in ``IMG_DATA``. A *compressed* product is a
``.SAFE`` directory with the same metadata but with its band images
replaced by another representation, described by a ``COMPRESSED.json``
manifest at the product root; the manifest's ``reader`` names the
``BandSource`` that decodes it.

Bands are decoded in *groups*. A JPEG 2000 band is its own group, but a
representation that codes several bands jointly (georvq codes all 13
together; SZ3 can code each resolution's bands as one stack) has to be
decoded a group at a time, so asking for one band of a group decodes the
whole group, and the others come with it.

The readers for compressed products need packages that are not
clouds-decoded dependencies (``georvq``, ``imagecodecs``). They are
imported only when a product that needs them is read, and a missing one
raises an ``ImportError`` that says what to install.
"""
from __future__ import annotations

import importlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.windows import Window

from ..constants import BANDS

logger = logging.getLogger(__name__)

MANIFEST_NAME = "COMPRESSED.json"
MANIFEST_FORMAT = "clouds-decoded-compressed"
MANIFEST_VERSION = 1

# reader name in a manifest -> "module:class", imported on first use
READERS: Dict[str, str] = {
    "georvq": "clouds_decoded.data.compressed.georvq:GeorvqBandSource",
    "sz3": "clouds_decoded.data.compressed.tiled:TiledBandSource",
    "jpegxl": "clouds_decoded.data.compressed.tiled:TiledBandSource",
}


@dataclass(frozen=True)
class BandGeoref:
    """Georeferencing of the B02 (10 m) reference grid of a full product."""

    crs: Any
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` in CRS units."""
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)
        return left, bottom, right, top


def scale_crop_window(
    crop_window: tuple,
    b02_w: int,
    b02_h: int,
    src_w: int,
    src_h: int,
) -> Window:
    """Scale a B02-pixel crop window to the coordinate space of a band raster.

    Args:
        crop_window: ``(col_off, row_off, width, height)`` in B02 (10 m) pixels.
        b02_w: Full-scene B02 pixel width (from the actual file, not BAND_RESOLUTIONS).
        b02_h: Full-scene B02 pixel height.
        src_w: Pixel width of the target band raster.
        src_h: Pixel height of the target band raster.

    Returns:
        A rasterio ``Window`` in the target band's pixel coordinates.
    """
    col_off, row_off, width, height = crop_window
    scale_x = src_w / b02_w
    scale_y = src_h / b02_h
    return Window(
        col_off=int(col_off * scale_x),
        row_off=int(row_off * scale_y),
        width=int(width * scale_x),
        height=int(height * scale_y),
    )


def require_optional(module: str, purpose: str, install: str):
    """Import an optional dependency, or explain what is missing.

    Args:
        module: Module to import, e.g. ``"imagecodecs"``.
        purpose: What it is needed for, completing "... is needed to <purpose>".
        install: How to install it.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"The '{module}' package is needed to {purpose}, but it could not be "
            f"imported ({e}). It is not a clouds-decoded dependency; install it "
            f"with: {install}"
        ) from e


class BandSource(ABC):
    """The band images of one product, decoded a group at a time.

    Subclasses define ``groups`` (decode units, group name -> band names),
    ``georef()``, ``shape()`` and ``_read_group()``.
    """

    #: Short name of the representation, e.g. ``"jp2"`` or ``"sz3"``.
    name: str = "unknown"

    @property
    @abstractmethod
    def groups(self) -> Dict[str, List[str]]:
        """Decode units: group name -> the bands decoded together."""

    @abstractmethod
    def georef(self) -> BandGeoref:
        """CRS, transform and size of the full product's B02 grid."""

    @abstractmethod
    def shape(self, band: str) -> Tuple[int, int]:
        """``(height, width)`` of a band's full native grid."""

    @abstractmethod
    def _read_group(self, group: str, windows: Dict[str, Optional[Window]]) -> Dict[str, np.ndarray]:
        """Decode one group.

        Args:
            group: A key of ``groups``.
            windows: Every band of the group -> its native-grid window, or
                ``None`` for the full band. All bands of a group cover the
                same area.

        Returns:
            Band name -> uint16 DN array, for every band of the group.
        """

    @property
    def bands(self) -> List[str]:
        """All bands the source holds, in canonical order."""
        held = {b for members in self.groups.values() for b in members}
        return [b for b in BANDS if b in held]

    def native_resolution(self, band: str) -> int:
        """Actual pixel size of a band in metres, from its grid size."""
        ref = self.georef()
        return round(abs(ref.transform.a) * ref.width / self.shape(band)[1])

    def window(self, band: str, crop_window: Optional[tuple]) -> Optional[Window]:
        """A B02-pixel crop window on a band's native grid (``None`` for none)."""
        if not crop_window:
            return None
        ref = self.georef()
        h, w = self.shape(band)
        return scale_crop_window(crop_window, ref.width, ref.height, w, h)

    def read(self, bands: List[str], crop_window: Optional[tuple] = None) -> Dict[str, np.ndarray]:
        """Read bands, decoding each group that holds one of them once.

        Args:
            bands: Band names to return.
            crop_window: Optional ``(col_off, row_off, width, height)`` in B02 pixels.

        Returns:
            Band name -> DN array on the band's native grid, for ``bands``.

        Raises:
            KeyError: If a band is not held by this source.
        """
        group_of = {b: g for g, members in self.groups.items() for b in members}
        missing = [b for b in bands if b not in group_of]
        if missing:
            raise KeyError(f"{self.name} source has no band(s) {missing}; it holds {self.bands}")
        out: Dict[str, np.ndarray] = {}
        for group in dict.fromkeys(group_of[b] for b in bands):
            windows = {b: self.window(b, crop_window) for b in self.groups[group]}
            decoded = self._read_group(group, windows)
            out.update({b: a for b, a in decoded.items() if b in bands})
        return {b: out[b] for b in bands}

    def describe(self) -> Dict[str, Any]:
        """Provenance of the representation, e.g. codec and level."""
        return {"reader": self.name}


class Jp2BandSource(BandSource):
    """The JPEG 2000 band images of a standard ``.SAFE`` product."""

    name = "jp2"

    def __init__(self, scene_directory: Path, locate):
        """
        Args:
            scene_directory: The ``.SAFE`` directory.
            locate: Callable ``(bands) -> {band: path}`` finding band images;
                images are located only when first needed, so a product
                missing some bands can still be read for the others.
        """
        self.scene_directory = Path(scene_directory)
        self._locate = locate
        self._paths: Dict[str, Path] = {}
        self._georef: Optional[BandGeoref] = None
        self._shapes: Dict[str, Tuple[int, int]] = {}
        self._res: Dict[str, int] = {}

    @property
    def groups(self) -> Dict[str, List[str]]:
        return {b: [b] for b in BANDS}

    def path(self, band: str) -> Path:
        if band not in self._paths:
            self._paths.update(self._locate([band]))
        return self._paths[band]

    def _open_meta(self, band: str) -> None:
        with rio.open(self.path(band)) as src:
            self._shapes[band] = (src.height, src.width)
            # the actual pixel size of the jp2, not the nominal one
            self._res[band] = round(abs(src.transform.a))
            if band == "B02":
                self._georef = BandGeoref(src.crs, src.transform, src.width, src.height)

    def georef(self) -> BandGeoref:
        if self._georef is None:
            self._open_meta("B02")
        return self._georef

    def shape(self, band: str) -> Tuple[int, int]:
        if band not in self._shapes:
            self._open_meta(band)
        return self._shapes[band]

    def native_resolution(self, band: str) -> int:
        if band not in self._res:
            self._open_meta(band)
        return self._res[band]

    def _read_group(self, group: str, windows: Dict[str, Optional[Window]]) -> Dict[str, np.ndarray]:
        (band,) = self.groups[group]
        with rio.open(self.path(band)) as src:
            self._shapes[band] = (src.height, src.width)
            self._res[band] = round(abs(src.transform.a))
            return {band: src.read(1, window=windows[band])}


def read_manifest(scene_directory: Path) -> Optional[Dict[str, Any]]:
    """The product's ``COMPRESSED.json``, or ``None`` for a standard product.

    Raises:
        ValueError: If the manifest is not a clouds-decoded compressed manifest
            or is of a newer version than this code reads.
    """
    path = Path(scene_directory) / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not a {MANIFEST_FORMAT} manifest")
    if int(manifest.get("version", 0)) > MANIFEST_VERSION:
        raise ValueError(
            f"{path} is manifest version {manifest['version']}; this clouds-decoded "
            f"reads up to version {MANIFEST_VERSION}")
    return manifest


def open_band_source(scene_directory: Path, band_paths=None) -> BandSource:
    """The band source of a product directory.

    Args:
        scene_directory: The ``.SAFE`` directory.
        band_paths: Callable ``(bands) -> {band: path}`` locating the JPEG 2000
            images of a standard product.

    Returns:
        A ``Jp2BandSource`` for a standard product, otherwise the reader the
        product's manifest names.

    Raises:
        ValueError: If the manifest names an unknown reader.
        ImportError: If the reader needs a package that is not installed.
    """
    scene_directory = Path(scene_directory)
    manifest = read_manifest(scene_directory)
    if manifest is None:
        return Jp2BandSource(scene_directory, band_paths)
    reader = manifest.get("reader")
    if reader not in READERS:
        raise ValueError(
            f"{scene_directory / MANIFEST_NAME} names reader '{reader}'; "
            f"known readers are {sorted(READERS)}")
    module, cls = READERS[reader].split(":")
    source = getattr(importlib.import_module(module), cls)(scene_directory, manifest)
    logger.info("Reading %s band images through the %s reader (%s)",
                scene_directory.name, reader, source.describe())
    return source
