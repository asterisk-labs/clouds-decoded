from __future__ import annotations

import importlib.util as _ilu
import os as _os

from clouds_decoded.assets import get_asset, get_assets_dir, require_asset, download_asset

__all__ = ["get_asset", "get_assets_dir", "require_asset", "download_asset"]

# Extend __path__ so that static analysis tools (griffe/mkdocstrings) can
# discover subpackages that live in separate directories due to the custom
# package-dir layout in pyproject.toml.
for _subpkg in ("modules", "cli", "stats"):
    _spec = _ilu.find_spec(f"clouds_decoded.{_subpkg}")
    if _spec and _spec.submodule_search_locations:
        for _p in _spec.submodule_search_locations:
            _parent = _os.path.dirname(_p)
            if _parent not in __path__:
                __path__.append(_parent)
