"""Base processor class providing automatic output resampling."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseProcessor:
    """Base class for all processors.

    Wraps each processor's ``_process()`` implementation so that the final
    output is automatically resampled to ``config.output_resolution`` before
    being returned to the caller.  Subclasses implement ``_process()`` instead
    of ``process()``.

    The only exception is :class:`~clouds_decoded.modules.refocus.processor.RefocusProcessor`,
    which returns a ``Sentinel2Scene`` (not a ``GeoRasterData``) and therefore
    overrides ``process()`` directly to bypass the resampling step.
    """

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Run the processor and resample the output to ``config.output_resolution``.

        Calls ``_process(*args, **kwargs)`` then invokes
        ``result.resample(self.config.output_resolution)`` on the returned
        object.  Subclasses should implement ``_process()`` rather than
        overriding this method.
        """
        result = self._process(*args, **kwargs)
        return result.resample(self.config.output_resolution)

    def _process(self, *args: Any, **kwargs: Any) -> Any:
        """Core processing logic.  Subclasses must implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _process()"
        )
