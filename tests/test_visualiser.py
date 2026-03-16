"""Tests for the Visualiser and ProjectVisualiser classes."""
from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio as rio
from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.transform import Affine

from clouds_decoded.visualisation.layers import (
    RGBConfig,
    _PROPERTIES_RENDER,
    _apply_rgb_config,
)
from clouds_decoded.visualisation.visualiser import Visualiser


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------

_TRANSFORM = Affine(10.0, 0, 500000.0, 0, -10.0, 6000000.0)
_CRS = CRS.from_epsg(32632)
_METADATA_TAG = "clouds_decoded"
_H, _W = 64, 64


def _write_geotiff(path, data: np.ndarray, metadata: dict):
    """Write a minimal GeoTIFF with clouds_decoded metadata tag."""
    if data.ndim == 2:
        count, h, w = 1, data.shape[0], data.shape[1]
        data = data[np.newaxis]
    else:
        count, h, w = data.shape[0], data.shape[1], data.shape[2]

    with rio.open(
        str(path), "w", driver="GTiff",
        height=h, width=w, count=count,
        dtype=data.dtype,
        crs=_CRS,
        transform=_TRANSFORM,
    ) as dst:
        dst.update_tags(**{_METADATA_TAG: json.dumps(metadata)})
        for i in range(count):
            dst.write(data[i], i + 1)


def _make_cloud_mask(tmp_path):
    """Write a synthetic cloud_mask.tif."""
    data = np.random.randint(0, 4, size=(_H, _W), dtype=np.uint8)
    meta = {
        "categorical": True,
        "classes": {"0": "Clear", "1": "Thick Cloud", "2": "Thin Cloud", "3": "Cloud Shadow"},
        "method": "senseiv2",
    }
    path = tmp_path / "cloud_mask.tif"
    _write_geotiff(path, data, meta)
    return path


def _make_cloud_height(tmp_path):
    """Write a synthetic cloud_height.tif."""
    data = np.random.uniform(0, 12000, size=(_H, _W)).astype(np.float32)
    meta = {"processing_config": {"method": "parallax"}}
    path = tmp_path / "cloud_height.tif"
    _write_geotiff(path, data, meta)
    return path


def _make_cloud_properties(tmp_path, n_bands=5):
    """Write a synthetic properties.tif with n_bands."""
    data = np.random.uniform(0, 1, size=(n_bands, _H, _W)).astype(np.float32)
    # Band 0: tau (0.1–200)
    data[0] = np.random.uniform(0.1, 200, size=(_H, _W)).astype(np.float32)
    # Band 1: ice_liq_ratio (0–1)
    data[1] = np.random.uniform(0, 1, size=(_H, _W)).astype(np.float32)
    # Band 2: r_eff_liq
    data[2] = np.random.uniform(1, 50, size=(_H, _W)).astype(np.float32)
    # Band 3: r_eff_ice
    data[3] = np.random.uniform(3, 370, size=(_H, _W)).astype(np.float32)

    band_names = ["tau", "ice_liq_ratio", "r_eff_liq", "r_eff_ice"]
    if n_bands >= 5:
        data[4] = np.random.uniform(0, 1, size=(_H, _W)).astype(np.float32)
        band_names.append("uncertainty")

    meta = {
        "description": "Cloud Properties Inversion Results",
        "band_names": band_names,
    }
    path = tmp_path / "properties.tif"
    _write_geotiff(path, data, meta)
    return path


# ---------------------------------------------------------------------------
# Render preset tests
# ---------------------------------------------------------------------------

class TestRenderPresets:
    """Verify updated render settings match requirements."""

    def test_rgb_defaults(self):
        cfg = RGBConfig()
        assert cfg.gamma == 0.65
        assert cfg.gain == 1.0
        assert cfg.offset == 0.0

    def test_cloud_height_cmap(self):
        assert _PROPERTIES_RENDER.get("tau") is not None
        # Cloud height is set in the factory, not _PROPERTIES_RENDER.
        # We'll test via layer creation below.

    def test_r_eff_plasma(self):
        assert _PROPERTIES_RENDER["r_eff_liq"].cmap == "plasma"
        assert _PROPERTIES_RENDER["r_eff_ice"].cmap == "plasma"

    def test_uncertainty_jet(self):
        assert _PROPERTIES_RENDER["uncertainty"].cmap == "jet"

    def test_ice_liq_ratio_rdbu(self):
        assert _PROPERTIES_RENDER["ice_liq_ratio"].cmap == "RdBu"

    def test_rgb_config_from_data(self):
        from clouds_decoded.visualisation.layers import rgb_config_from_data

        rgb = np.random.uniform(0.02, 0.5, size=(32, 32, 3)).astype(np.float32)
        cfg = rgb_config_from_data(rgb)
        out = _apply_rgb_config(rgb, cfg)
        assert out.shape == rgb.shape
        assert out.max() <= 1.0
        assert out.min() >= 0.0
        # Most pixels should be well within [0, 1] (not blown out)
        assert np.median(out) < 0.95


# ---------------------------------------------------------------------------
# Visualiser tests
# ---------------------------------------------------------------------------

class TestVisualiser:
    """Tests for the Visualiser class."""

    def test_empty(self):
        vis = Visualiser()
        assert vis.layers == []
        assert vis.layer_names == []

    def test_add_cloud_mask(self, tmp_path):
        path = _make_cloud_mask(tmp_path)
        vis = Visualiser().add(path)
        assert "Cloud Mask" in vis.layer_names
        layer = vis._layers["Cloud Mask"]
        assert layer.render.categorical is True

    def test_add_cloud_height(self, tmp_path):
        path = _make_cloud_height(tmp_path)
        vis = Visualiser().add(path)
        assert "Cloud Height" in vis.layer_names
        layer = vis._layers["Cloud Height"]
        assert layer.render.cmap == "turbo"

    def test_add_cloud_properties_5_bands(self, tmp_path):
        path = _make_cloud_properties(tmp_path, n_bands=5)
        vis = Visualiser().add(path)

        expected = [
            "Properties: tau",
            "Properties: ice_liq_ratio",
            "Properties: r_eff_liq",
            "Properties: r_eff_ice",
            "Properties: uncertainty",
        ]
        for name in expected:
            assert name in vis.layer_names, f"{name} not found in {vis.layer_names}"

    def test_phase_masking(self, tmp_path):
        """r_eff_liq should be NaN where ice_liq_ratio > 0.5."""
        path = _make_cloud_properties(tmp_path, n_bands=5)

        # Read back to know the phase values
        from clouds_decoded.data import CloudPropertiesData
        data = CloudPropertiesData.from_file(str(path))
        phase = data.data[1]

        # Use native resolution so mask shape matches layer shape.
        vis = Visualiser(display_resolution_m=None).add(path)

        r_eff_liq = vis._layers["Properties: r_eff_liq"].data
        r_eff_ice = vis._layers["Properties: r_eff_ice"].data

        # Where ice-dominated, r_eff_liq should be NaN
        ice_mask = phase > 0.5
        if ice_mask.any():
            assert np.all(np.isnan(r_eff_liq[ice_mask]))

        # Where liquid-dominated, r_eff_ice should be NaN
        liq_mask = phase < 0.5
        if liq_mask.any():
            assert np.all(np.isnan(r_eff_ice[liq_mask]))

    def test_add_chaining(self, tmp_path):
        mask_path = _make_cloud_mask(tmp_path)
        height_path = _make_cloud_height(tmp_path)
        vis = Visualiser().add(mask_path).add(height_path)
        assert len(vis.layers) == 2

    def test_overview_returns_figure(self, tmp_path):
        import matplotlib
        matplotlib.use("agg")

        path = _make_cloud_mask(tmp_path)
        vis = Visualiser().add(path)
        fig = vis.overview()
        assert isinstance(fig, Figure)

    def test_overview_empty_raises(self):
        vis = Visualiser()
        with pytest.raises(ValueError, match="No layers"):
            vis.overview()

    def test_plot_unknown_layer_raises(self, tmp_path):
        path = _make_cloud_mask(tmp_path)
        vis = Visualiser().add(path)
        with pytest.raises(KeyError, match="nonexistent"):
            vis.plot("nonexistent")

    def test_composite_returns_figure(self, tmp_path):
        import matplotlib
        matplotlib.use("agg")

        mask_path = _make_cloud_mask(tmp_path)
        height_path = _make_cloud_height(tmp_path)
        vis = Visualiser().add(mask_path).add(height_path)
        fig = vis.composite("Cloud Mask", "Cloud Height", alpha=0.5)
        assert isinstance(fig, Figure)

    def test_from_directory(self, tmp_path):
        _make_cloud_mask(tmp_path)
        _make_cloud_height(tmp_path)
        _make_cloud_properties(tmp_path, n_bands=5)

        vis = Visualiser.from_directory(str(tmp_path))
        # Should have cloud mask + cloud height + 5 properties layers
        assert len(vis.layers) >= 7

    def test_save(self, tmp_path):
        import matplotlib
        matplotlib.use("agg")

        _make_cloud_mask(tmp_path)
        vis = Visualiser.from_directory(str(tmp_path))
        out = tmp_path / "overview.png"
        vis.save(str(out))
        assert out.exists()
