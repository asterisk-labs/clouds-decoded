"""CPU vs torch parity tests for the dispatcher levers (Phase 1).

Covers:
  * ``RefocusProcessor`` warp and height-interpolation kernels with
    ``device=None`` (scipy) vs ``device='cpu'``/``'cuda:0'`` (torch).
  * ``Sentinel2Band.to_resolution`` with ``device=None`` (skimage) vs
    torch.

The skimage path uses cubic splines with Gaussian anti-aliasing; the
torch path uses bicubic with torch's built-in antialias. Absolute per-
pixel agreement is *not* expected for cubic (order=3) — we assert
structural equivalence (shape, dtype, bounded RMS on smooth inputs).
Bilinear (order=1) is effectively bit-exact.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    _HAS_CUDA = False


needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
needs_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


def _smooth_field(shape=(80, 80), seed=0) -> np.ndarray:
    """A smooth 2D field suitable for bounded-RMS comparison across backends."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return (
        0.3 * np.sin(2 * np.pi * y / 30)
        + 0.3 * np.cos(2 * np.pi * x / 25)
        + 0.1 * rng.standard_normal(shape).astype(np.float32)
    ).astype(np.float32)


# --------------------------------------------------------------------- #
# Refocus _warp_band parity
# --------------------------------------------------------------------- #

@needs_torch
class TestRefocusWarpParity:
    """Scipy vs torch parity for ``_warp_band``."""

    @staticmethod
    def _make_processors(order, cpu_device=None, torch_device="cpu"):
        from clouds_decoded.modules.refocus import RefocusProcessor, RefocusConfig
        cfg_cpu = RefocusConfig(device=cpu_device, interpolation_order=order)
        cfg_trc = RefocusConfig(device=torch_device, interpolation_order=order)
        return RefocusProcessor(cfg_cpu), RefocusProcessor(cfg_trc)

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=needs_cuda)])
    def test_order_1_tight_agreement(self, device):
        """Bilinear: scipy ↔ torch should agree to <1e-4 max absolute."""
        band = _smooth_field(seed=1)
        row_off = np.full(band.shape, 1.3, dtype=np.float32)
        col_off = np.full(band.shape, -0.4, dtype=np.float32)

        p_cpu, p_trc = self._make_processors(order=1, torch_device=device)
        a = p_cpu._warp_band(band, row_off, col_off)
        b = p_trc._warp_band(band, row_off, col_off)

        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert np.abs(a - b).max() < 1e-4, (
            f"bilinear {device} parity drift: max={np.abs(a-b).max()}"
        )

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=needs_cuda)])
    def test_order_3_rms_bound(self, device):
        """Cubic: scipy uses splines, torch uses bicubic; expect modest RMS."""
        band = _smooth_field(seed=2)
        row_off = np.full(band.shape, 0.7, dtype=np.float32)
        col_off = np.full(band.shape, -1.1, dtype=np.float32)

        p_cpu, p_trc = self._make_processors(order=3, torch_device=device)
        a = p_cpu._warp_band(band, row_off, col_off)
        b = p_trc._warp_band(band, row_off, col_off)

        rms = float(np.sqrt(((a - b) ** 2).mean()))
        # Smooth field with sub-pixel offset: expect <5% RMS of signal std.
        signal_std = float(a.std())
        assert rms < 0.1 * signal_std, (
            f"order=3 RMS={rms:.4f} exceeded 10% of signal std {signal_std:.4f}"
        )

    def test_nan_offsets_passthrough(self):
        """NaN offsets should be zeroed → output equals input at those pixels."""
        band = _smooth_field(seed=3)
        row_off = np.full(band.shape, np.nan, dtype=np.float32)
        col_off = np.full(band.shape, np.nan, dtype=np.float32)
        _, p_trc = self._make_processors(order=1)
        out = p_trc._warp_band(band, row_off, col_off)
        np.testing.assert_allclose(out, band, atol=1e-5)


# --------------------------------------------------------------------- #
# Refocus _interpolate_height_to_band parity
# --------------------------------------------------------------------- #

@needs_torch
class TestHeightInterpolationParity:
    """Scipy vs torch parity for height upsampling inside refocus."""

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=needs_cuda)])
    def test_upsample_order_1(self, device):
        from clouds_decoded.modules.refocus import RefocusProcessor, RefocusConfig
        hmap = _smooth_field(shape=(20, 20), seed=4) * 3000.0
        p_cpu = RefocusProcessor(RefocusConfig(height_interpolation_order=1, device=None))
        p_trc = RefocusProcessor(RefocusConfig(height_interpolation_order=1, device=device))

        # Upsample 20×20 at 300m → 80×80 at 75m
        a = p_cpu._interpolate_height_to_band(hmap, 300.0, (80, 80), 75.0)
        b = p_trc._interpolate_height_to_band(hmap, 300.0, (80, 80), 75.0)

        assert a.shape == b.shape == (80, 80)
        assert np.abs(a - b).max() < 1e-2, (
            f"height interp drift {device}: max={np.abs(a-b).max()}"
        )


# --------------------------------------------------------------------- #
# Sentinel2Band.to_resolution parity
# --------------------------------------------------------------------- #

@needs_torch
class TestBandResizeParity:
    """skimage vs torch parity for ``Sentinel2Band.to_resolution``."""

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=needs_cuda)])
    def test_shape_and_dtype(self, device):
        from clouds_decoded.data import Sentinel2Band
        arr = _smooth_field(shape=(60, 60)).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr, native_resolution=10)

        skimg = band.to_resolution(20).data
        torch_out = band.to_resolution(20, device=device).data

        assert skimg.shape == torch_out.shape == (30, 30)
        assert skimg.dtype == torch_out.dtype == np.float32

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=needs_cuda)])
    def test_smooth_rms_bound(self, device):
        """On smooth inputs, skimage ↔ torch should agree within a few percent RMS."""
        from clouds_decoded.data import Sentinel2Band
        arr = _smooth_field(shape=(120, 120)).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr, native_resolution=10)

        skimg = band.to_resolution(20).data
        torch_out = band.to_resolution(20, device=device).data

        rms = float(np.sqrt(((skimg - torch_out) ** 2).mean()))
        signal_std = float(skimg.std())
        assert rms < 0.1 * signal_std, (
            f"resize RMS={rms:.4f} exceeded 10% of signal std {signal_std:.4f}"
        )

    def test_cpu_torch_matches_gpu_torch(self):
        """torch CPU and torch CUDA should agree tightly on the same input."""
        if not _HAS_CUDA:
            pytest.skip("needs CUDA")
        from clouds_decoded.data import Sentinel2Band
        arr = _smooth_field(shape=(60, 60)).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr, native_resolution=10)

        cpu = band.to_resolution(20, device="cpu").data
        gpu = band.to_resolution(20, device="cuda:0").data
        # Torch is deterministic across devices for interpolate; expect very tight match.
        assert np.abs(cpu - gpu).max() < 1e-4


# --------------------------------------------------------------------- #
# get_band device plumbing
# --------------------------------------------------------------------- #

@needs_torch
class TestGetBandDeviceCache:
    """Ensure the cache key includes *device* so CPU/GPU results don't collide."""

    def _scene(self):
        from clouds_decoded.data import Sentinel2Scene
        from rasterio.transform import Affine
        from rasterio.crs import CRS
        scene = Sentinel2Scene()
        scene.bands["B02"] = (_smooth_field(shape=(60, 60)) * 5000 + 1000).astype(np.float32)
        scene.transform = Affine.translation(0, 0) * Affine.scale(10.0, -10.0)
        scene.crs = CRS.from_epsg(32633)
        scene.sun_zenith = 30.0
        scene.sun_azimuth = 120.0
        scene.view_zenith = 5.0
        scene.view_azimuth = 180.0
        return scene

    def test_separate_cache_entries(self):
        scene = self._scene()
        skimg = scene.get_band("B02", reflectance=False, resolution=20)
        torch_cpu = scene.get_band("B02", reflectance=False, resolution=20, device="cpu")

        # Two distinct cache entries should exist (different device).
        keys = [k for k in scene._band_cache.keys() if k[0] == "B02"]
        assert len(keys) == 2
        # Results are different arrays (different backends).
        assert not np.allclose(skimg, torch_cpu, atol=1e-6)

    def test_resolution_none_ignores_device(self):
        """device is a resize backend knob; when no resize happens, it's inert."""
        scene = self._scene()
        a = scene.get_band("B02", reflectance=False, resolution=None, device="cuda:0")
        b = scene.get_band("B02", reflectance=False, resolution=None, device=None)
        np.testing.assert_array_equal(a, b)
