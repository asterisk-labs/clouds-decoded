"""The ``.SAFE`` skeleton of a compressed product: the original product's
files without its band images, and the manifest."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Union

from ..band_source import MANIFEST_FORMAT, MANIFEST_NAME, MANIFEST_VERSION

# band images (and the true-colour image derived from them) that a
# compressed product replaces
_BAND_IMAGE = re.compile(r"(^|/)IMG_DATA/[^/]*_(B0[1-9]|B1[0-2]|B8A|TCI)\.jp2$")


def is_band_image(relpath: str) -> bool:
    """Whether a path inside a product is one of its band images."""
    return bool(_BAND_IMAGE.search(relpath.replace(os.sep, "/")))


def make_shell(source: Union[str, Path, bytes], out_safe: Union[str, Path],
               link: bool = False) -> Path:
    """Write a product's skeleton: every file except the band images.

    Args:
        source: The original ``.SAFE`` directory, or the bytes of a zip of
            its files (as a georvq store holds in ``assets["safe_meta"]``).
        out_safe: The ``.SAFE`` directory to create.
        link: Symlink to the source's files instead of copying them (only
            for a directory source).

    Returns:
        The granule directory of the skeleton, where band data goes in
        ``IMG_DATA``.

    Raises:
        FileExistsError: If ``out_safe`` exists.
        ValueError: If the source has no single granule directory.
    """
    out_safe = Path(out_safe)
    if out_safe.exists():
        raise FileExistsError(out_safe)
    out_safe.mkdir(parents=True)
    if isinstance(source, (bytes, bytearray)):
        with zipfile.ZipFile(io.BytesIO(source)) as z:
            for info in z.infolist():
                if info.is_dir() or is_band_image(info.filename):
                    continue
                dst = out_safe / info.filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as f, open(dst, "wb") as g:
                    shutil.copyfileobj(f, g)
    else:
        source = Path(source)
        for root, _, files in os.walk(source):
            rel = Path(root).relative_to(source)
            for name in files:
                if is_band_image(str(rel / name)):
                    continue
                dst = out_safe / rel / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if link:
                    os.symlink(os.path.abspath(Path(root) / name), dst)
                else:
                    shutil.copy2(Path(root) / name, dst)
    granules = [p for p in (out_safe / "GRANULE").iterdir() if p.is_dir()]
    if len(granules) != 1:
        raise ValueError(f"expected one granule in {source}, found {len(granules)}")
    (granules[0] / "IMG_DATA").mkdir(exist_ok=True)
    return granules[0]


def write_manifest(out_safe: Union[str, Path], reader: str, **fields: Any) -> Dict[str, Any]:
    """Write ``COMPRESSED.json`` into a skeleton.

    Args:
        out_safe: The product directory.
        reader: Reader name (a key of ``band_source.READERS``).
        **fields: The reader's own fields.

    Returns:
        The manifest written.
    """
    manifest = dict(format=MANIFEST_FORMAT, version=MANIFEST_VERSION, reader=reader, **fields)
    path = Path(out_safe) / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, path)
    return manifest
