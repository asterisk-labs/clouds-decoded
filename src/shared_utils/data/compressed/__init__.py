"""Readers and writers for compressed Sentinel-2 products.

A compressed product is a ``.SAFE`` directory holding the original
product's metadata but not its band images, plus a ``COMPRESSED.json``
manifest naming the reader of its band data (see ``band_source.py``):

- ``georvq``: a georvq S2 store, decoded at the manifest's quality rung
  (``georvq.py``); needs the ``georvq`` package.
- ``sz3``, ``jpegxl``: tiled band data coded with SZ3 or JPEG XL
  (``tiled.py``); needs ``imagecodecs``.

Neither package is a clouds-decoded dependency; each is imported only
when a product that needs it is read or written.
"""
from .shell import make_shell, write_manifest

__all__ = ["make_shell", "write_manifest"]
