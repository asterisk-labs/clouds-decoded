"""Sentinel-2 band object with lazy evaluation and caching."""
from __future__ import annotations

import enum
import logging
from typing import Any, Callable, List, Optional

import numpy as np

from ..constants import BAND_RESOLUTIONS

logger = logging.getLogger(__name__)


class BandUnit(enum.Enum):
    """Unit of a Sentinel-2 band's pixel values."""
    DN = "dn"
    REFLECTANCE = "reflectance"


class Sentinel2Band:
    """A Sentinel-2 band with lazy evaluation and caching.

    Wraps a numpy array and supports lazy derivation from a parent band
    (e.g. reflectance conversion, resolution scaling).  Implements the
    numpy array protocol so it can be used transparently wherever a plain
    ndarray is expected.

    Args:
        name: Band identifier (e.g. ``'B02'``, ``'B8A'``).
        data: Pre-computed array, or ``None`` for lazy evaluation.
        native_resolution: Pixel size in metres (10, 20, or 60).
        unit: Whether values are raw DN or TOA reflectance.
        parent: Source band for lazy derivation.
        derive_fn: Callable that transforms ``parent.data`` into this band's data.
        interpolation: Interpolation method hint for downstream resampling.
    """

    __slots__ = (
        "name",
        "_data",
        "native_resolution",
        "unit",
        "parent",
        "_derive_fn",
        "_interpolation",
    )

    def __init__(
        self,
        name: str,
        data: Optional[np.ndarray] = None,
        native_resolution: Optional[int] = None,
        unit: BandUnit = BandUnit.DN,
        parent: Optional[Sentinel2Band] = None,
        derive_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        interpolation: str = "cubic",
    ) -> None:
        self.name = name
        self._data = data
        self.native_resolution = native_resolution
        self.unit = unit
        self.parent = parent
        self._derive_fn = derive_fn
        self._interpolation = interpolation

    # ------------------------------------------------------------------ #
    # Core data access
    # ------------------------------------------------------------------ #

    @property
    def data(self) -> np.ndarray:
        """The underlying array, computed lazily from *parent* if needed."""
        if self._data is None:
            if self.parent is not None and self._derive_fn is not None:
                self._data = self._derive_fn(self.parent.data)
            else:
                raise ValueError(
                    f"Band '{self.name}' has no data and no parent to derive from"
                )
        return self._data

    @property
    def is_cached(self) -> bool:
        """True if the array is currently held in memory."""
        return self._data is not None

    def release(self) -> None:
        """Free cached data.  Re-derivable bands will recompute on next access."""
        if self.parent is not None and self._derive_fn is not None:
            self._data = None
        else:
            logger.warning(
                "Cannot release root band '%s' — no parent to re-derive from.",
                self.name,
            )

    # ------------------------------------------------------------------ #
    # Numpy array protocol
    # ------------------------------------------------------------------ #

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self.data
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def flags(self) -> np.flagsobj:
        return self.data.flags

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)

    # Comparison operators — return plain ndarrays
    def __eq__(self, other: Any) -> np.ndarray:  # type: ignore[override]
        return self.data == (other.data if isinstance(other, Sentinel2Band) else other)

    def __ne__(self, other: Any) -> np.ndarray:  # type: ignore[override]
        return self.data != (other.data if isinstance(other, Sentinel2Band) else other)

    def __gt__(self, other: Any) -> np.ndarray:
        return self.data > (other.data if isinstance(other, Sentinel2Band) else other)

    def __lt__(self, other: Any) -> np.ndarray:
        return self.data < (other.data if isinstance(other, Sentinel2Band) else other)

    def __ge__(self, other: Any) -> np.ndarray:
        return self.data >= (other.data if isinstance(other, Sentinel2Band) else other)

    def __le__(self, other: Any) -> np.ndarray:
        return self.data <= (other.data if isinstance(other, Sentinel2Band) else other)

    def astype(self, dtype: Any, **kwargs: Any) -> np.ndarray:
        """Return a plain ndarray cast to *dtype*."""
        return self.data.astype(dtype, **kwargs)

    def __repr__(self) -> str:
        cached = "cached" if self.is_cached else "lazy"
        shape_str = str(self._data.shape) if self._data is not None else "?"
        return (
            f"Sentinel2Band(name='{self.name}', shape={shape_str}, "
            f"res={self.native_resolution}m, unit={self.unit.value}, {cached})"
        )

    # ------------------------------------------------------------------ #
    # Derivation factories
    # ------------------------------------------------------------------ #

    def to_reflectance(
        self, offset: float, quantification_value: float
    ) -> Sentinel2Band:
        """Create a child band with TOA reflectance values.

        Args:
            offset: Radiometric additive offset (typically -1000 or 0).
            quantification_value: Divisor for DN to reflectance (typically 10000).

        Returns:
            New ``Sentinel2Band`` whose data is lazily computed as
            ``(parent + offset) / quantification_value``.
        """
        def _derive(parent_data: np.ndarray) -> np.ndarray:
            return (parent_data.astype(np.float32) + offset) / quantification_value

        return Sentinel2Band(
            name=self.name,
            native_resolution=self.native_resolution,
            unit=BandUnit.REFLECTANCE,
            parent=self,
            derive_fn=_derive,
            interpolation=self._interpolation,
        )

    def to_resolution(
        self,
        target_resolution: int,
        target_shape: Optional[tuple] = None,
    ) -> Sentinel2Band:
        """Create a child band resampled to *target_resolution* metres.

        Args:
            target_resolution: Desired pixel size in metres.
            target_shape: Explicit ``(H, W)`` for the output.  If ``None``,
                computed from *native_resolution* and the current shape.

        Returns:
            New ``Sentinel2Band`` with lazily resampled data.
        """
        if self.native_resolution == target_resolution and target_shape is None:
            return self

        if target_shape is None:
            if self.native_resolution is None:
                raise ValueError(
                    "Cannot compute target shape without native_resolution. "
                    "Pass target_shape explicitly."
                )
            scale = self.native_resolution / target_resolution
            src_shape = self.shape  # may trigger parent eval
            target_shape = (round(src_shape[0] * scale), round(src_shape[1] * scale))

        _target_shape = target_shape

        def _derive(parent_data: np.ndarray) -> np.ndarray:
            from skimage.transform import resize
            return resize(
                parent_data,
                _target_shape,
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(parent_data.dtype)

        return Sentinel2Band(
            name=self.name,
            native_resolution=target_resolution,
            unit=self.unit,
            parent=self,
            derive_fn=_derive,
            interpolation=self._interpolation,
        )


class BandDict(dict):
    """A ``dict`` subclass that auto-wraps ``numpy`` arrays as ``Sentinel2Band``.

    Behaves identically to a plain dict for iteration, membership tests,
    ``keys()``, ``values()``, ``items()``, ``len()``, and ``sorted()`` — so
    existing processor code that treats ``scene.bands`` as ``Dict[str, Any]``
    continues to work.
    """

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            resolution = BAND_RESOLUTIONS.get(key)
            value = Sentinel2Band(
                name=key,
                data=value,
                native_resolution=resolution,
            )
        super().__setitem__(key, value)

    def update(self, other: Any = (), **kwargs: Any) -> None:
        """Override to route through ``__setitem__`` for auto-wrapping."""
        if isinstance(other, dict):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v
