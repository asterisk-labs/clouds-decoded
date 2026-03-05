"""Stats plugin module for clouds-decoded projects.

Provides a resolution mechanism for per-step statistics functions.
Loading is centralised in :class:`StatsCaller`, which caches Data objects
so each step output file is opened at most once per manifest.

Stats function signature::

    def fn(caller: StatsCaller, step_name: str) -> Dict[str, Union[float, int]]:
        ...

*step_name* is the primary step associated with the identifier
(e.g. ``"cloud_mask"`` for ``"cloud_mask::class_fractions"``).
Generic functions use it to know which step to load.  Step-specific
functions that need multiple inputs call ``caller.load("other_step")``
directly — the cache ensures each file is opened only once.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from clouds_decoded.data.base import GeoRasterData
    from clouds_decoded.project import SceneManifest

logger = logging.getLogger(__name__)


# Map step_name → (module_path, class_name) for typed loading.
# Steps not listed here fall back to GeoRasterData.
_STEP_DATA_CLASSES: Dict[str, Tuple[str, str]] = {
    "cloud_mask":            ("clouds_decoded.data.cloud_mask",  "CloudMaskData"),
    "cloud_height":          ("clouds_decoded.data.cloud_height", "CloudHeightGridData"),
    "cloud_height_emulator": ("clouds_decoded.data.cloud_height", "CloudHeightGridData"),
    "cloud_properties":      ("clouds_decoded.data.refl2prop",   "CloudPropertiesData"),
    "albedo":                ("clouds_decoded.data.base",         "AlbedoData"),
}


class StatsCaller:
    """Loads step outputs once per manifest and dispatches stats functions.

    :meth:`load` returns the cached Data object for a given step, loading it
    on first access.  Stats functions receive this caller so they can load
    whatever steps they need — including data from multiple steps — without
    any extra file I/O.

    Args:
        manifest: The scene manifest whose step outputs will be loaded.
    """

    _warned = False

    def __init__(self, manifest: "SceneManifest") -> None:
        if not StatsCaller._warned:
            logger.warning(
                "Stats module is a work in progress — outputs may change in future releases."
            )
            StatsCaller._warned = True
        self._manifest = manifest
        self._cache: Dict[str, Optional[Any]] = {}

    def _data_class(self, step_name: str) -> type:
        """Return the appropriate Data class for *step_name*."""
        spec = _STEP_DATA_CLASSES.get(step_name)
        if spec:
            mod = importlib.import_module(spec[0])
            return getattr(mod, spec[1])
        from clouds_decoded.data.base import GeoRasterData
        return GeoRasterData

    def load(self, step_name: str) -> Optional["GeoRasterData"]:
        """Load and cache the Data object for *step_name*.

        Returns ``None`` if the step is not complete, has no output file, or
        the file does not exist.  Subsequent calls for the same *step_name*
        return the cached object without re-reading the file.

        Args:
            step_name: The pipeline step whose output to load.

        Returns:
            A loaded Data instance, or ``None`` if unavailable.
        """
        if step_name in self._cache:
            return self._cache[step_name]

        from ._utils import _get_output_path
        path = _get_output_path(self._manifest, step_name)
        if path is None:
            self._cache[step_name] = None
            return None

        data_cls = self._data_class(step_name)
        try:
            obj = data_cls.from_file(str(path))
        except Exception as exc:
            logger.warning("Failed to load %s output from %s: %s", step_name, path, exc)
            obj = None
        self._cache[step_name] = obj
        return obj

    def call(self, fn: Callable, step_name: str) -> Dict[str, Any]:
        """Call *fn(self, step_name)* and return its result.

        Passes this caller so the function can load any step it needs,
        plus *step_name* so generic functions know their primary step.
        Exceptions propagate to the caller (typically :meth:`~clouds_decoded.project.Project.run_stats`
        which logs them and continues).

        Args:
            fn: A stats function with signature
                ``fn(caller, step_name) -> dict``.
            step_name: Passed through to *fn* as the primary step name.

        Returns:
            The dict returned by *fn*.
        """
        return fn(self, step_name)


def resolve_stats_fn(identifier: str) -> Tuple[Callable, str]:
    """Resolve 'step_name::fn_name' to (callable, step_name).

    Looks in ``clouds_decoded.stats.{step_name}`` first, then falls back to
    ``clouds_decoded.stats._generic``.

    Args:
        identifier: String of the form ``'step_name::fn_name'``.

    Returns:
        A ``(callable, step_name)`` tuple.

    Raises:
        ValueError: If *identifier* does not contain ``::``.
        AttributeError: If the function is not found in either module.
    """
    if "::" not in identifier:
        raise ValueError(
            f"Invalid stats identifier {identifier!r}: expected 'step_name::fn_name'"
        )
    step_name, fn_name = identifier.split("::", maxsplit=1)
    for module_path in (
        f"clouds_decoded.stats.{step_name}",
        "clouds_decoded.stats._generic",
    ):
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                return fn, step_name
        except ModuleNotFoundError:
            continue
    raise AttributeError(
        f"Stats function '{identifier}' not found in "
        f"clouds_decoded.stats.{step_name} or _generic"
    )
