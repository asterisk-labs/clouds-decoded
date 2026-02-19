"""Tests for Sentinel2Band, BandDict, and get_band() caching."""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from rasterio.transform import Affine
from rasterio.crs import CRS

from clouds_decoded.data import Sentinel2Band, BandDict, BandUnit, Sentinel2Scene
from clouds_decoded.constants import BAND_RESOLUTIONS


def _create_dummy_scene():
    """Reproduce the create_dummy_scene() pattern from test_smoke.py."""
    scene = Sentinel2Scene()
    h, w = 100, 100
    np.random.seed(42)
    for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                 'B08', 'B8A', 'B09', 'B11', 'B12']:
        band_data = np.random.uniform(500, 3000, (h, w)).astype(np.float32)
        band_data[30:50, 40:70] = np.random.uniform(5000, 8000, (20, 30))
        scene.bands[band] = band_data
    scene.transform = Affine.translation(0.0, 0.0) * Affine.scale(10.0, -10.0)
    scene.crs = CRS.from_epsg(32633)
    scene.sun_zenith = 30.0
    scene.sun_azimuth = 120.0
    scene.view_zenith = 5.0
    scene.view_azimuth = 180.0
    scene.image_azimuth = np.radians(10.0)
    return scene


# ------------------------------------------------------------------ #
# Sentinel2Band basics
# ------------------------------------------------------------------ #

class TestSentinel2BandBasics:
    """Construction and property delegation."""

    def test_construction_with_data(self):
        arr = np.ones((50, 50), dtype=np.uint16)
        band = Sentinel2Band(name="B02", data=arr, native_resolution=10)
        assert band.name == "B02"
        assert band.native_resolution == 10
        assert band.unit == BandUnit.DN
        assert band.is_cached

    def test_shape(self):
        band = Sentinel2Band(name="B02", data=np.zeros((100, 80)))
        assert band.shape == (100, 80)

    def test_ndim(self):
        band = Sentinel2Band(name="B02", data=np.zeros((100, 80)))
        assert band.ndim == 2

    def test_dtype(self):
        band = Sentinel2Band(name="B02", data=np.zeros((5, 5), dtype=np.float32))
        assert band.dtype == np.float32

    def test_size(self):
        band = Sentinel2Band(name="B02", data=np.zeros((10, 20)))
        assert band.size == 200

    def test_len(self):
        band = Sentinel2Band(name="B02", data=np.zeros((10, 20)))
        assert len(band) == 10  # first dimension

    def test_getitem(self):
        arr = np.arange(20).reshape(4, 5)
        band = Sentinel2Band(name="B02", data=arr)
        assert band[0, 0] == 0
        assert band[3, 4] == 19
        assert_array_equal(band[1], arr[1])
        assert_array_equal(band[:, 2], arr[:, 2])

    def test_setitem(self):
        arr = np.zeros((4, 5))
        band = Sentinel2Band(name="B02", data=arr)
        band[0, 0] = 42.0
        assert band[0, 0] == 42.0

    def test_astype(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        band = Sentinel2Band(name="B02", data=arr)
        result = band.astype(np.float32)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_repr_cached(self):
        band = Sentinel2Band(name="B02", data=np.zeros((3, 3)), native_resolution=10)
        r = repr(band)
        assert "B02" in r
        assert "cached" in r

    def test_repr_lazy(self):
        parent = Sentinel2Band(name="B02", data=np.zeros((3, 3)), native_resolution=10)
        child = parent.to_reflectance(0.0, 10000.0)
        r = repr(child)
        assert "lazy" in r


# ------------------------------------------------------------------ #
# Comparison operators
# ------------------------------------------------------------------ #

class TestSentinel2BandComparisons:

    def test_eq(self):
        band = Sentinel2Band(name="B02", data=np.array([0, 1, 2]))
        result = band == 0
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, [True, False, False])

    def test_ne(self):
        band = Sentinel2Band(name="B02", data=np.array([0, 1, 2]))
        result = band != 0
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, [False, True, True])

    def test_gt(self):
        band = Sentinel2Band(name="B02", data=np.array([0, 1, 2]))
        assert_array_equal(band > 1, [False, False, True])

    def test_lt(self):
        band = Sentinel2Band(name="B02", data=np.array([0, 1, 2]))
        assert_array_equal(band < 1, [True, False, False])

    def test_ge(self):
        band = Sentinel2Band(name="B02", data=np.array([0, 1, 2]))
        assert_array_equal(band >= 1, [False, True, True])

    def test_le(self):
        band = Sentinel2Band(name="B02", data=np.array([0, 1, 2]))
        assert_array_equal(band <= 1, [True, True, False])

    def test_eq_band_vs_band(self):
        a = Sentinel2Band(name="B02", data=np.array([1, 2, 3]))
        b = Sentinel2Band(name="B03", data=np.array([1, 0, 3]))
        assert_array_equal(a == b, [True, False, True])


# ------------------------------------------------------------------ #
# Numpy protocol
# ------------------------------------------------------------------ #

class TestNumpyProtocol:

    def test_np_asarray(self):
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr)
        result = np.asarray(band)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, arr)

    def test_np_asarray_dtype(self):
        arr = np.array([1, 2, 3], dtype=np.uint16)
        band = Sentinel2Band(name="B02", data=arr)
        result = np.asarray(band, dtype=np.float64)
        assert result.dtype == np.float64

    def test_np_array(self):
        arr = np.ones((3, 3))
        band = Sentinel2Band(name="B02", data=arr)
        result = np.array(band)
        assert isinstance(result, np.ndarray)

    def test_scipy_uniform_filter(self):
        from scipy.ndimage import uniform_filter
        arr = np.random.rand(20, 20).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr)
        result = uniform_filter(band, size=3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 20)

    def test_scipy_regular_grid_interpolator(self):
        """RegularGridInterpolator works when band is converted via np.asarray().

        Note: scipy's Cython fast path requires a true ndarray, so passing
        a Sentinel2Band directly to RegularGridInterpolator won't work for
        the __call__ evaluation.  In practice get_band() returns ndarray,
        so the workaround is np.asarray(band).
        """
        from scipy.interpolate import RegularGridInterpolator
        arr = np.random.rand(10, 10).astype(np.float64)
        band = Sentinel2Band(name="B02", data=arr)
        y = np.arange(10)
        x = np.arange(10)
        interp = RegularGridInterpolator((y, x), np.asarray(band))
        val = interp(np.array([[4.5, 4.5]]))
        assert val.shape == (1,)

    def test_np_mean(self):
        arr = np.array([2.0, 4.0, 6.0])
        band = Sentinel2Band(name="B02", data=arr)
        assert np.mean(band) == pytest.approx(4.0)

    def test_np_where(self):
        arr = np.array([0, 5, 0, 10])
        band = Sentinel2Band(name="B02", data=arr)
        result = np.where(band != 0, band, -1)
        assert_array_equal(result, [-1, 5, -1, 10])


# ------------------------------------------------------------------ #
# Derivation: to_reflectance, to_resolution
# ------------------------------------------------------------------ #

class TestDerivation:

    def test_to_reflectance_values(self):
        dn = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        band = Sentinel2Band(name="B02", data=dn, native_resolution=10)
        refl_band = band.to_reflectance(offset=-1000.0, quantification_value=10000.0)

        assert refl_band.unit == BandUnit.REFLECTANCE
        assert refl_band.native_resolution == 10
        expected = (dn.astype(np.float32) - 1000.0) / 10000.0
        assert_allclose(refl_band.data, expected)

    def test_to_reflectance_zero_offset(self):
        dn = np.array([[500, 1500]], dtype=np.float32)
        band = Sentinel2Band(name="B04", data=dn, native_resolution=10)
        refl_band = band.to_reflectance(offset=0.0, quantification_value=10000.0)
        assert_allclose(refl_band.data, dn / 10000.0)

    def test_to_reflectance_is_lazy(self):
        dn = np.ones((3, 3), dtype=np.uint16)
        band = Sentinel2Band(name="B02", data=dn, native_resolution=10)
        refl_band = band.to_reflectance(0.0, 10000.0)
        assert not refl_band.is_cached  # not yet evaluated
        _ = refl_band.data  # trigger evaluation
        assert refl_band.is_cached

    def test_to_resolution_shape(self):
        arr = np.random.rand(50, 50).astype(np.float32)
        band = Sentinel2Band(name="B05", data=arr, native_resolution=20)
        resized = band.to_resolution(10)
        assert resized.shape == (100, 100)
        assert resized.native_resolution == 10

    def test_to_resolution_same_noop(self):
        arr = np.random.rand(50, 50).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr, native_resolution=10)
        same = band.to_resolution(10)
        assert same is band  # no new object

    def test_to_resolution_explicit_shape(self):
        arr = np.random.rand(100, 100).astype(np.float32)
        band = Sentinel2Band(name="B02", data=arr, native_resolution=10)
        resized = band.to_resolution(20, target_shape=(55, 55))
        assert resized.shape == (55, 55)

    def test_to_resolution_no_native_raises(self):
        arr = np.random.rand(10, 10).astype(np.float32)
        band = Sentinel2Band(name="unknown", data=arr, native_resolution=None)
        with pytest.raises(ValueError, match="native_resolution"):
            band.to_resolution(10)

    def test_chain_reflectance_then_resolution(self):
        dn = np.random.randint(500, 3000, (50, 50)).astype(np.uint16)
        band = Sentinel2Band(name="B05", data=dn, native_resolution=20)
        refl = band.to_reflectance(0.0, 10000.0)
        resized = refl.to_resolution(10)
        assert resized.shape == (100, 100)
        assert resized.unit == BandUnit.REFLECTANCE
        assert resized.dtype == np.float32


# ------------------------------------------------------------------ #
# Parent-child caching and release
# ------------------------------------------------------------------ #

class TestParentChildCaching:

    def test_child_derives_from_parent(self):
        dn = np.array([[1000, 2000]], dtype=np.uint16)
        parent = Sentinel2Band(name="B02", data=dn, native_resolution=10)
        child = parent.to_reflectance(0.0, 10000.0)
        assert child.parent is parent
        assert not child.is_cached
        _ = child.data
        assert child.is_cached

    def test_release_and_recompute(self):
        dn = np.array([[1000, 2000]], dtype=np.uint16)
        parent = Sentinel2Band(name="B02", data=dn, native_resolution=10)
        child = parent.to_reflectance(0.0, 10000.0)

        first = child.data.copy()
        child.release()
        assert not child.is_cached

        second = child.data
        assert child.is_cached
        assert_allclose(first, second)

    def test_release_root_warns(self, caplog):
        root = Sentinel2Band(name="B02", data=np.zeros((3, 3)))
        with caplog.at_level("WARNING"):
            root.release()
        assert "Cannot release root" in caplog.text
        assert root.is_cached  # data not freed

    def test_grandchild_chain(self):
        dn = np.random.randint(500, 3000, (50, 50)).astype(np.uint16)
        root = Sentinel2Band(name="B05", data=dn, native_resolution=20)
        refl = root.to_reflectance(-1000.0, 10000.0)
        resized = refl.to_resolution(10)

        # Evaluate grandchild
        data = resized.data
        assert data.shape == (100, 100)

        # Release middle layer
        refl.release()
        assert not refl.is_cached
        # Root still has data
        assert root.is_cached

        # Re-evaluate grandchild after release
        resized.release()
        data2 = resized.data  # recomputes via refl, which recomputes from root
        assert_allclose(data, data2)


# ------------------------------------------------------------------ #
# BandDict
# ------------------------------------------------------------------ #

class TestBandDict:

    def test_auto_wrap_ndarray(self):
        bd = BandDict()
        bd["B02"] = np.zeros((100, 100), dtype=np.uint16)
        assert isinstance(bd["B02"], Sentinel2Band)
        assert bd["B02"].name == "B02"
        assert bd["B02"].native_resolution == 10

    def test_auto_wrap_unknown_band(self):
        bd = BandDict()
        bd["CUSTOM"] = np.zeros((10, 10))
        assert isinstance(bd["CUSTOM"], Sentinel2Band)
        assert bd["CUSTOM"].native_resolution is None

    def test_no_wrap_sentinel2band(self):
        bd = BandDict()
        band = Sentinel2Band(name="B02", data=np.zeros((5, 5)), native_resolution=10)
        bd["B02"] = band
        assert bd["B02"] is band

    def test_dict_keys(self):
        bd = BandDict()
        bd["B02"] = np.zeros((5, 5))
        bd["B03"] = np.zeros((5, 5))
        assert set(bd.keys()) == {"B02", "B03"}

    def test_dict_values(self):
        bd = BandDict()
        bd["B02"] = np.zeros((5, 5))
        vals = list(bd.values())
        assert len(vals) == 1
        assert isinstance(vals[0], Sentinel2Band)

    def test_dict_items(self):
        bd = BandDict()
        bd["B04"] = np.ones((3, 3))
        items = list(bd.items())
        assert len(items) == 1
        k, v = items[0]
        assert k == "B04"
        assert isinstance(v, Sentinel2Band)

    def test_in_operator(self):
        bd = BandDict()
        bd["B02"] = np.zeros((5, 5))
        assert "B02" in bd
        assert "B99" not in bd

    def test_len(self):
        bd = BandDict()
        bd["B02"] = np.zeros((5, 5))
        bd["B03"] = np.zeros((5, 5))
        assert len(bd) == 2

    def test_sorted_keys(self):
        bd = BandDict()
        for name in ["B04", "B02", "B03"]:
            bd[name] = np.zeros((5, 5))
        assert sorted(bd.keys()) == ["B02", "B03", "B04"]

    def test_max_values_by_size(self):
        bd = BandDict()
        bd["B02"] = np.zeros((100, 100))
        bd["B01"] = np.zeros((17, 17))  # smaller
        biggest = max(bd.values(), key=lambda b: b.size)
        assert biggest.name == "B02"

    def test_isinstance_dict(self):
        bd = BandDict()
        assert isinstance(bd, dict)

    def test_update(self):
        bd = BandDict()
        bd.update({"B02": np.zeros((5, 5)), "B03": np.ones((5, 5))})
        assert isinstance(bd["B02"], Sentinel2Band)
        assert isinstance(bd["B03"], Sentinel2Band)


# ------------------------------------------------------------------ #
# get_band() caching on Sentinel2Scene
# ------------------------------------------------------------------ #

class TestGetBandCaching:

    @pytest.fixture
    def scene(self):
        """Minimal scene for caching tests."""
        scene = Sentinel2Scene()
        np.random.seed(0)
        scene.bands["B02"] = np.random.uniform(500, 3000, (100, 100)).astype(np.float32)
        scene.bands["B04"] = np.random.uniform(500, 3000, (100, 100)).astype(np.float32)
        return scene

    def test_get_band_raw(self, scene):
        raw = scene.get_band("B02", reflectance=False)
        assert isinstance(raw, np.ndarray)
        assert_array_equal(raw, scene.bands["B02"].data)

    def test_get_band_reflectance(self, scene):
        refl = scene.get_band("B02", reflectance=True)
        assert refl.dtype == np.float32
        raw = scene.bands["B02"].data
        expected = (raw.astype(np.float32) + 0.0) / 10000.0
        assert_allclose(refl, expected)

    def test_get_band_caching_same_result(self, scene):
        r1 = scene.get_band("B02", reflectance=True)
        r2 = scene.get_band("B02", reflectance=True)
        # Should return the exact same array (from cache)
        assert r1 is r2

    def test_get_band_cache_invalidated_by_offset_change(self, scene):
        r1 = scene.get_band("B02", reflectance=True)
        scene.radio_add_offset = {"B02": -1000.0}
        r2 = scene.get_band("B02", reflectance=True)
        # Different offset → different cache key → different array
        assert r1 is not r2
        raw = scene.bands["B02"].data
        expected = (raw.astype(np.float32) - 1000.0) / 10000.0
        assert_allclose(r2, expected)

    def test_get_band_missing_raises(self, scene):
        with pytest.raises(KeyError):
            scene.get_band("B99")

    def test_get_bands_multiple(self, scene):
        bands = scene.get_bands(["B02", "B04"], reflectance=True)
        assert len(bands) == 2
        assert all(isinstance(b, Sentinel2Band) for b in bands)
        assert bands[0].name == "B02"
        assert bands[1].name == "B04"

    def test_get_bands_default_all(self, scene):
        bands = scene.get_bands()
        assert len(bands) == len(scene.bands)

    def test_get_band_cache_false_not_stored(self, scene):
        r1 = scene.get_band("B02", reflectance=True, cache=False)
        # Result is correct
        raw = scene.bands["B02"].data
        assert_allclose(r1, (raw.astype(np.float32) + 0.0) / 10000.0)
        # But nothing was stored in the cache
        assert len(scene._band_cache) == 0

    def test_get_band_cache_false_recomputes(self, scene):
        r1 = scene.get_band("B02", reflectance=True, cache=False)
        r2 = scene.get_band("B02", reflectance=True, cache=False)
        # Same values, but not the same object (independently computed)
        assert_allclose(r1, r2)
        assert r1 is not r2

    def test_get_band_cache_false_returns_existing_cached(self, scene):
        # Prime the cache with cache=True
        r_cached = scene.get_band("B02", reflectance=True, cache=True)
        assert len(scene._band_cache) == 1
        # cache=False still returns the already-cached result (no point recomputing)
        r_again = scene.get_band("B02", reflectance=True, cache=False)
        assert r_cached is r_again

    def test_get_bands_cache_false(self, scene):
        bands = scene.get_bands(["B02", "B04"], reflectance=True, cache=False)
        assert len(bands) == 2
        assert all(isinstance(b, Sentinel2Band) for b in bands)
        assert len(scene._band_cache) == 0

    def test_get_bands_parallel_matches_sequential(self, scene):
        seq = scene.get_bands(["B02", "B04"], reflectance=True, cache=False)
        par = scene.get_bands(["B02", "B04"], reflectance=True, cache=False, n_workers=2)
        assert len(par) == len(seq)
        for s, p in zip(seq, par):
            assert_allclose(np.asarray(s), np.asarray(p))

    def test_get_bands_parallel_caches(self, scene):
        bands = scene.get_bands(["B02", "B04"], reflectance=True, n_workers=2)
        assert len(scene._band_cache) == 2
        # Second call returns cached objects
        bands2 = scene.get_bands(["B02", "B04"], reflectance=True, n_workers=2)
        for b1, b2 in zip(bands, bands2):
            assert b1 is b2

    def test_get_bands_parallel_order_preserved(self):
        """Band order in the result matches the requested order."""
        scene = Sentinel2Scene()
        for name in ["B02", "B03", "B04", "B08"]:
            scene.bands[name] = np.random.rand(50, 50).astype(np.float32)
        names = ["B08", "B03", "B02", "B04"]
        bands = scene.get_bands(names, reflectance=True, n_workers=4)
        assert [b.name for b in bands] == names

    def test_get_bands_n_workers_auto(self, scene):
        """n_workers=-1 sizes the pool to the number of uncached bands."""
        bands = scene.get_bands(["B02", "B04"], reflectance=True, n_workers=-1)
        assert len(bands) == 2
        # Values match sequential
        seq = scene.get_bands(["B02", "B04"], reflectance=True, cache=False)
        for s, p in zip(seq, bands):
            assert_allclose(np.asarray(s), np.asarray(p))

    def test_get_bands_n_workers_auto_with_partial_cache(self, scene):
        """n_workers=-1 only counts uncached bands, not cache hits."""
        # Prime B02 into cache
        scene.get_band("B02", reflectance=True, cache=True)
        assert len(scene._band_cache) == 1
        # -1 should only spin up 1 thread (for B04), not 2
        bands = scene.get_bands(["B02", "B04"], reflectance=True, n_workers=-1)
        assert len(bands) == 2
        assert len(scene._band_cache) == 2


# ------------------------------------------------------------------ #
# Regression: create_dummy_scene still works
# ------------------------------------------------------------------ #

class TestRegression:

    def test_create_dummy_scene(self):
        """Ensure the standard create_dummy_scene pattern works with BandDict."""
        create_dummy_scene = _create_dummy_scene
        scene = create_dummy_scene()

        assert len(scene.bands) == 12
        assert isinstance(scene.bands, BandDict)

        # All values are Sentinel2Band
        for name, band in scene.bands.items():
            assert isinstance(band, Sentinel2Band), f"{name} is not Sentinel2Band"
            assert band.shape == (100, 100)

    def test_dummy_scene_get_band(self):
        """get_band produces correct reflectance on dummy scene."""
        create_dummy_scene = _create_dummy_scene
        scene = create_dummy_scene()

        raw = scene.get_band("B02", reflectance=False)
        refl = scene.get_band("B02", reflectance=True)

        assert raw.shape == refl.shape
        assert refl.dtype == np.float32
        assert_allclose(refl, raw / 10000.0, rtol=1e-5)

    def test_dummy_scene_band_ne_zero(self):
        """scene.bands['B02'] != 0 still returns ndarray mask."""
        create_dummy_scene = _create_dummy_scene
        scene = create_dummy_scene()
        mask = scene.bands["B02"] != 0
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_dummy_scene_sorted_keys(self):
        create_dummy_scene = _create_dummy_scene
        scene = create_dummy_scene()
        keys = sorted(scene.bands.keys())
        assert isinstance(keys, list)
        assert keys[0] == "B01"
