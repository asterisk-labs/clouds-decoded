"""Tests for band sources: the JPEG 2000 source, manifest dispatch, optional
dependency errors, and the tiled SZ3 / JPEG XL readers and writer."""
import json
import sys

import numpy as np
import pytest
import rasterio
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from rasterio.transform import Affine

from clouds_decoded.constants import BANDS, BAND_RESOLUTIONS
from clouds_decoded.data import Sentinel2Scene
from clouds_decoded.data.band_source import (
    MANIFEST_FORMAT, MANIFEST_NAME, Jp2BandSource, open_band_source, read_manifest,
    require_optional, resolve_level, scale_crop_window)

B02 = 120  # synthetic product: 120 px at 10 m, 60 at 20 m, 20 at 60 m
ORIGIN = Affine.translation(300000.0, 5000000.0)


def _band(name, rng):
    res = BAND_RESOLUTIONS[name]
    n = B02 * 10 // res
    a = rng.normal(3000, 400, (n, n)).clip(1, 20000).astype(np.uint16)
    a[: n // 4, : n // 3] = 0  # nodata corner
    return a, res


@pytest.fixture
def safe(tmp_path):
    """A minimal product: the 13 band images (GeoTIFF content, .jp2 names),
    plus a metadata file that a compressed product must keep."""
    rng = np.random.default_rng(0)
    root = tmp_path / "S2X_MSIL1C_TEST.SAFE"
    img = root / "GRANULE" / "L1C_TEST" / "IMG_DATA"
    img.mkdir(parents=True)
    (root / "GRANULE" / "L1C_TEST" / "QI_DATA").mkdir()
    (root / "MTD_MSIL1C.xml").write_text("<metadata/>")
    (root / "GRANULE" / "L1C_TEST" / "MTD_TL.xml").write_text("<tile/>")
    arrays = {}
    for b in BANDS:
        a, res = _band(b, rng)
        arrays[b] = a
        with rasterio.open(img / f"T00TST_20250101T000000_{b}.jp2", "w", driver="GTiff",
                           width=a.shape[1], height=a.shape[0], count=1, dtype="uint16",
                           crs=CRS.from_epsg(32633), transform=ORIGIN * Affine.scale(res, -res)) as dst:
            dst.write(a, 1)
    with rasterio.open(img / "T00TST_20250101T000000_TCI.jp2", "w", driver="GTiff",
                       width=B02, height=B02, count=1, dtype="uint8", crs=CRS.from_epsg(32633),
                       transform=ORIGIN * Affine.scale(10, -10)) as dst:
        dst.write(np.zeros((B02, B02), np.uint8), 1)
    return root, arrays


def _jp2_source(root, level=None):
    return open_band_source(root, lambda names: Sentinel2Scene()._get_band_paths(root, names),
                            level=level)


class TestScaleCropWindow:
    def test_matches_scene_helper(self):
        for src in [(10980, 10980), (5490, 5490), (1830, 1830)]:
            a = scale_crop_window((600, 1200, 1284, 1284), 10980, 10980, *src)
            b = Sentinel2Scene._scale_crop_window((600, 1200, 1284, 1284), 10980, 10980, *src)
            assert a == b


class TestJp2BandSource:
    def test_standard_product_is_jp2(self, safe):
        root, _ = safe
        src = _jp2_source(root)
        assert isinstance(src, Jp2BandSource)
        assert src.groups["B02"] == ["B02"]
        assert src.georef().width == B02
        assert src.georef().crs == CRS.from_epsg(32633)

    def test_read_full_and_crop(self, safe):
        root, arrays = safe
        src = _jp2_source(root)
        full = src.read(["B02", "B11", "B09"])
        for b in full:
            assert_array_equal(full[b], arrays[b])
        crop = src.read(["B02", "B11", "B09"], crop_window=(30, 60, 60, 30))
        assert_array_equal(crop["B02"], arrays["B02"][60:90, 30:90])
        assert_array_equal(crop["B11"], arrays["B11"][30:45, 15:45])
        assert_array_equal(crop["B09"], arrays["B09"][10:15, 5:15])

    def test_native_resolution_from_file(self, safe):
        root, _ = safe
        src = _jp2_source(root)
        assert src.native_resolution("B02") == 10
        assert src.native_resolution("B12") == 20
        assert src.native_resolution("B10") == 60

    def test_scene_get_bands_uses_source(self, safe):
        root, arrays = safe
        scene = Sentinel2Scene()
        scene._band_source = _jp2_source(root)
        bands = scene._get_bands(root, ["B03", "B8A"], crop_window=(0, 0, 60, 60))
        assert_array_equal(bands["B03"].data, arrays["B03"][:60, :60])
        assert bands["B8A"].native_resolution == 20


class TestManifest:
    def _write(self, root, **m):
        (root / MANIFEST_NAME).write_text(json.dumps(m))

    def test_no_manifest(self, safe):
        assert read_manifest(safe[0]) is None

    def test_wrong_format(self, safe):
        self._write(safe[0], format="something-else", version=1, reader="sz3")
        with pytest.raises(ValueError, match="not a"):
            read_manifest(safe[0])

    def test_newer_version(self, safe):
        self._write(safe[0], format=MANIFEST_FORMAT, version=99, reader="sz3")
        with pytest.raises(ValueError, match="version 99"):
            read_manifest(safe[0])

    def test_unknown_reader(self, safe):
        self._write(safe[0], format=MANIFEST_FORMAT, version=1, reader="nope")
        with pytest.raises(ValueError, match="known readers"):
            _jp2_source(safe[0])


class TestLevels:
    MANIFEST = dict(format=MANIFEST_FORMAT, version=2, reader="georvq", store="s.zarr",
                    default="8-8", levels={"8-8": {"quality": "8-8"}, "8-0": {"quality": "8-0"}})

    def test_default_level(self):
        m = resolve_level(self.MANIFEST)
        assert m["level"] == "8-8" and m["quality"] == "8-8" and "levels" not in m

    def test_chosen_level(self):
        m = resolve_level(self.MANIFEST, "8-0")
        assert m["quality"] == "8-0" and m["store"] == "s.zarr"

    def test_unknown_level(self):
        with pytest.raises(ValueError, match="no level '8-4'"):
            resolve_level(self.MANIFEST, "8-4")

    def test_single_level_manifest(self):
        m = dict(format=MANIFEST_FORMAT, version=1, reader="sz3", codec={"name": "sz3"})
        assert resolve_level(m) is m
        with pytest.raises(ValueError, match="single level"):
            resolve_level(m, "abs-10")

    def test_level_of_standard_product(self, safe):
        with pytest.raises(ValueError, match="not a compressed product"):
            _jp2_source(safe[0], level="8-4")


class TestOptionalDependencies:
    def test_require_optional_explains(self):
        with pytest.raises(ImportError, match="not a clouds-decoded dependency.*pip install x"):
            require_optional("no_such_module_xyz", "do things", "pip install x")

    def test_missing_georvq(self, safe, monkeypatch):
        monkeypatch.setitem(sys.modules, "georvq", None)
        (safe[0] / MANIFEST_NAME).write_text(json.dumps(
            dict(format=MANIFEST_FORMAT, version=1, reader="georvq", store="x.georvq.zarr")))
        with pytest.raises(ImportError, match="'georvq' package is needed"):
            _jp2_source(safe[0])

    def test_missing_imagecodecs(self, safe, monkeypatch):
        monkeypatch.setitem(sys.modules, "imagecodecs", None)
        from clouds_decoded.data.compressed.tiled import write_tiled_safe
        with pytest.raises(ImportError, match="'imagecodecs' package is needed"):
            write_tiled_safe(safe[0], safe[0].parent / "out.SAFE", "sz3", {"abs": 10})


def _skip_without(codec):
    ic = pytest.importorskip("imagecodecs")
    if not {"sz3": ic.SZ3, "jpegxl": ic.JPEGXL}[codec].available:
        pytest.skip(f"imagecodecs has no {codec}")


class TestTiled:
    def _make(self, safe, codec, params, layout, level=None):
        from clouds_decoded.data.compressed.tiled import write_tiled_safe
        root, _ = safe
        out = root.parent / f"{codec}-{layout}" / root.name
        manifest = write_tiled_safe(root, out, codec, params, level=level, layout=layout,
                                    tile=60, n_workers=2)
        return out, manifest, _jp2_source(out, level=level)

    def test_levels_in_one_product(self, safe):
        _skip_without("sz3")
        _, arrays = safe
        out, fine, _ = self._make(safe, "sz3", {"abs": 5}, "separate", level="fine")
        _, coarse, _ = self._make(safe, "sz3", {"abs": 200}, "separate", level="coarse")
        assert fine["bits_per_pixel"] > coarse["bits_per_pixel"]
        m = read_manifest(out)
        assert set(m["levels"]) == {"fine", "coarse"} and m["default"] == "fine"
        for level, bound in (("fine", 5), ("coarse", 200), (None, 5)):
            b02 = _jp2_source(out, level=level).read(["B02"])["B02"]
            v = arrays["B02"] > 0
            assert np.abs(b02[v].astype(int) - arrays["B02"][v]).max() <= bound
        assert (out / "GRANULE" / "L1C_TEST" / "IMG_DATA" / "coarse" / "10m.bin").exists()
        with pytest.raises(FileExistsError):
            self._make(safe, "sz3", {"abs": 7}, "separate", level="fine")

    def test_level_of_other_codec_rejected(self, safe):
        _skip_without("sz3")
        _skip_without("jpegxl")
        self._make(safe, "sz3", {"abs": 25}, "separate")
        from clouds_decoded.data.compressed.tiled import write_tiled_safe
        root = safe[0]
        with pytest.raises(ValueError, match="sz3 product, not jpegxl"):
            write_tiled_safe(root, root.parent / "sz3-separate" / root.name, "jpegxl",
                             {"distance": 1.0}, tile=60)

    @pytest.mark.parametrize("layout", ["separate", "grouped"])
    def test_sz3_round_trip(self, safe, layout):
        _skip_without("sz3")
        _, arrays = safe
        out, manifest, src = self._make(safe, "sz3", {"abs": 25}, layout)
        assert src.name == "sz3"
        assert manifest["bits_per_pixel"] > 0
        full = src.read(list(BANDS))
        for b in BANDS:
            a, d = arrays[b], full[b]
            assert d.shape == a.shape and d.dtype == np.uint16
            assert_array_equal(d == 0, a == 0)  # nodata survives exactly
            v = a > 0
            assert np.abs(d[v].astype(int) - a[v]).max() <= 25
        crop = src.read(["B02", "B05", "B01"], crop_window=(30, 60, 60, 54))
        assert_array_equal(crop["B02"], full["B02"][60:114, 30:90])
        assert_array_equal(crop["B05"], full["B05"][30:57, 15:45])
        assert_array_equal(crop["B01"], full["B01"][10:19, 5:15])

    def test_grouped_groups_by_resolution(self, safe):
        _skip_without("sz3")
        _, _, src = self._make(safe, "sz3", {"abs": 25}, "grouped")
        assert src.groups["10m"] == ["B02", "B03", "B04", "B08"]
        assert set(src.groups["60m"]) == {"B01", "B09", "B10"}

    def test_jpegxl_lossless_round_trip(self, safe):
        _skip_without("jpegxl")
        _, arrays = safe
        _, _, src = self._make(safe, "jpegxl", {"distance": 0.0}, "separate")
        full = src.read(["B02", "B11", "B10"])
        for b in full:
            assert_array_equal(full[b], arrays[b])

    def test_shell_drops_band_images_keeps_metadata(self, safe):
        _skip_without("sz3")
        out, _, _ = self._make(safe, "sz3", {"abs": 25}, "separate")
        img = out / "GRANULE" / "L1C_TEST" / "IMG_DATA"
        assert not list(img.rglob("*.jp2"))
        assert (out / "MTD_MSIL1C.xml").exists()
        assert (out / "GRANULE" / "L1C_TEST" / "MTD_TL.xml").exists()

    def test_grouped_jpegxl_rejected(self, safe):
        _skip_without("jpegxl")
        from clouds_decoded.data.compressed.tiled import write_tiled_safe
        with pytest.raises(ValueError, match="grouped"):
            write_tiled_safe(safe[0], safe[0].parent / "x.SAFE", "jpegxl",
                             {"distance": 1.0}, layout="grouped")
