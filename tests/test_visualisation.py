"""Tests for the visualisation package."""
import matplotlib
matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pytest
from pathlib import Path
from rasterio.transform import Affine
from rasterio.crs import CRS


# ---------------------------------------------------------------------------
# Layer / RenderConfig basics
# ---------------------------------------------------------------------------

class TestLayerModels:
    """Test Layer and RenderConfig construction."""

    def test_render_config_defaults(self):
        from clouds_decoded.visualisation.layers import RenderConfig
        rc = RenderConfig()
        assert rc.cmap == "viridis"
        assert rc.log_scale is False
        assert rc.categorical is False

    def test_rgb_config_defaults(self):
        from clouds_decoded.visualisation.layers import RGBConfig
        rc = RGBConfig()
        assert rc.gamma == 0.7
        assert rc.gain == 1.0
        assert rc.offset == 0.0

    def test_layer_with_2d_array(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig
        data = np.random.rand(50, 50).astype(np.float32)
        layer = Layer(name="test", data=data, render=RenderConfig())
        assert layer.data.shape == (50, 50)
        assert layer.is_rgb is False

    def test_layer_with_rgb_array(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, RGBConfig
        data = np.random.rand(50, 50, 3).astype(np.float32)
        layer = Layer(name="rgb", data=data, render=RenderConfig(), is_rgb=True, rgb_config=RGBConfig())
        assert layer.is_rgb is True
        assert layer.rgb_config.gamma == 0.7

    def test_extent_from_transform(self):
        from clouds_decoded.visualisation.layers import _extent_from_transform
        transform = Affine(10, 0, 500000, 0, -10, 6000000)
        extent = _extent_from_transform(transform, height=100, width=200)
        assert extent == (500000, 502000, 5999000, 6000000)

    def test_apply_rgb_config(self):
        from clouds_decoded.visualisation.layers import _apply_rgb_config, RGBConfig
        rgb = np.full((10, 10, 3), 0.5, dtype=np.float32)
        cfg = RGBConfig(gamma=1.0, gain=1.0, offset=0.0)
        out = _apply_rgb_config(rgb, cfg)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_apply_rgb_config_gamma(self):
        from clouds_decoded.visualisation.layers import _apply_rgb_config, RGBConfig
        rgb = np.full((10, 10, 3), 0.25, dtype=np.float32)
        cfg = RGBConfig(gamma=0.5, gain=1.0, offset=0.0)
        out = _apply_rgb_config(rgb, cfg)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:
    """Test layer factory functions with synthetic data."""

    def _make_georaster(self, data, meta_cls=None, meta_kwargs=None):
        """Helper to create a GeoRasterData-like object."""
        from clouds_decoded.data.base import GeoRasterData, Metadata
        transform = Affine(10, 0, 500000, 0, -10, 6000000)
        crs = CRS.from_epsg(32633)

        if meta_cls and meta_kwargs:
            meta = meta_cls(**meta_kwargs)
        else:
            meta = Metadata()

        return GeoRasterData(data=data, transform=transform, crs=crs, metadata=meta)

    def test_layer_from_cloud_mask(self):
        from clouds_decoded.visualisation.layers import layer_from_cloud_mask
        from clouds_decoded.data import CloudMaskData

        data = np.random.randint(0, 4, (100, 100), dtype=np.uint8)
        mask = CloudMaskData(
            data=data,
            transform=Affine(10, 0, 500000, 0, -10, 6000000),
            crs=CRS.from_epsg(32633),
        )
        layer = layer_from_cloud_mask(mask)
        assert layer.name == "Cloud Mask"
        assert layer.render.categorical is True
        assert layer.render.category_labels is not None
        assert len(layer.render.category_labels) == 4
        assert layer.extent is not None

    def test_layer_from_cloud_height(self):
        from clouds_decoded.visualisation.layers import layer_from_cloud_height
        from clouds_decoded.data import CloudHeightGridData

        data = np.random.rand(1, 50, 50).astype(np.float32) * 10000
        height = CloudHeightGridData(
            data=data,
            transform=Affine(200, 0, 500000, 0, -200, 6000000),
            crs=CRS.from_epsg(32633),
        )
        layer = layer_from_cloud_height(height)
        assert layer.name == "Cloud Height"
        assert layer.render.cmap == "inferno"
        assert layer.data.ndim == 2

    def test_layers_from_cloud_properties(self):
        from clouds_decoded.visualisation.layers import layers_from_cloud_properties
        from clouds_decoded.data import CloudPropertiesData

        data = np.random.rand(4, 100, 100).astype(np.float32)
        props = CloudPropertiesData(
            data=data,
            transform=Affine(20, 0, 500000, 0, -20, 6000000),
            crs=CRS.from_epsg(32633),
        )
        layers = layers_from_cloud_properties(props)
        assert len(layers) == 4
        names = [l.name for l in layers]
        assert "Properties: tau" in names
        assert "Properties: ice_liq_ratio" in names
        assert "Properties: r_eff_liq" in names
        assert "Properties: r_eff_ice" in names

        # Check tau is log scale
        tau_layer = next(l for l in layers if "tau" in l.name)
        assert tau_layer.render.log_scale is True

        # Check r_eff ranges
        liq = next(l for l in layers if "r_eff_liq" in l.name)
        assert liq.render.log_scale is True
        assert liq.render.vmin == 1
        assert liq.render.vmax == 50

        ice = next(l for l in layers if "r_eff_ice" in l.name)
        assert ice.render.log_scale is True
        assert ice.render.vmin == 3
        assert ice.render.vmax == 370

    def test_layers_from_albedo(self):
        from clouds_decoded.visualisation.layers import layers_from_albedo
        from clouds_decoded.data import AlbedoData, AlbedoMetadata

        data = np.random.rand(3, 20, 20).astype(np.float32)
        albedo = AlbedoData(
            data=data,
            transform=Affine(3000, 0, 500000, 0, -3000, 6000000),
            crs=CRS.from_epsg(32633),
            metadata=AlbedoMetadata(band_names=["B02", "B03", "B04"]),
        )
        layers = layers_from_albedo(albedo)
        assert len(layers) == 3
        assert layers[0].name == "Albedo: B02"
        assert layers[0].render.cmap == "gray"


# ---------------------------------------------------------------------------
# Rendering utilities
# ---------------------------------------------------------------------------

class TestRenderLayerToPng:
    """Test PNG rendering of layers."""

    def test_scalar_layer_png(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, render_layer_to_png

        data = np.random.rand(50, 50).astype(np.float32)
        layer = Layer(name="test", data=data, render=RenderConfig(label="Test"))
        png = render_layer_to_png(layer)
        assert isinstance(png, bytes)
        assert len(png) > 0
        # Check PNG magic bytes
        assert png[:8] == b'\x89PNG\r\n\x1a\n'

    def test_rgb_layer_png(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, RGBConfig, render_layer_to_png

        data = np.random.rand(50, 50, 3).astype(np.float32)
        layer = Layer(name="rgb", data=data, render=RenderConfig(), is_rgb=True, rgb_config=RGBConfig())
        png = render_layer_to_png(layer)
        assert isinstance(png, bytes)
        assert png[:8] == b'\x89PNG\r\n\x1a\n'

    def test_rgb_layer_with_custom_config(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, RGBConfig, render_layer_to_png

        data = np.random.rand(30, 30, 3).astype(np.float32)
        layer = Layer(name="rgb", data=data, render=RenderConfig(), is_rgb=True, rgb_config=RGBConfig())
        # Re-render with different gamma
        png = render_layer_to_png(layer, rgb_config=RGBConfig(gamma=1.5, gain=1.2, offset=0.05))
        assert isinstance(png, bytes)
        assert png[:8] == b'\x89PNG\r\n\x1a\n'

    def test_categorical_layer_png(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, render_layer_to_png

        data = np.random.randint(0, 4, (50, 50)).astype(np.uint8)
        layer = Layer(
            name="mask",
            data=data,
            render=RenderConfig(
                cmap="cloud_mask",
                categorical=True,
                category_labels={0: "Clear", 1: "Thick", 2: "Thin", 3: "Shadow"},
                label="Cloud Mask",
            ),
        )
        png = render_layer_to_png(layer)
        assert isinstance(png, bytes)
        assert png[:8] == b'\x89PNG\r\n\x1a\n'

    def test_log_scale_layer_png(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, render_layer_to_png

        data = np.random.rand(50, 50).astype(np.float32) * 10 + 0.1
        layer = Layer(
            name="tau",
            data=data,
            render=RenderConfig(log_scale=True, vmin=0.1, vmax=30, label="tau"),
        )
        png = render_layer_to_png(layer)
        assert isinstance(png, bytes)
        assert png[:8] == b'\x89PNG\r\n\x1a\n'

    def test_layer_with_nan_data(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, render_layer_to_png

        data = np.random.rand(50, 50).astype(np.float32)
        data[10:20, 10:20] = np.nan
        layer = Layer(name="with_nan", data=data, render=RenderConfig())
        png = render_layer_to_png(layer)
        assert isinstance(png, bytes)
        assert png[:8] == b'\x89PNG\r\n\x1a\n'


# ---------------------------------------------------------------------------
# Colormap / norm utilities
# ---------------------------------------------------------------------------

class TestColormapUtilities:
    """Test _build_cmap and _build_norm."""

    def test_build_cmap_standard(self):
        from clouds_decoded.visualisation.layers import _build_cmap, RenderConfig
        cmap = _build_cmap(RenderConfig(cmap="inferno"))
        assert cmap.name == "inferno"

    def test_build_cmap_cloud_mask(self):
        from clouds_decoded.visualisation.layers import _build_cmap, RenderConfig
        cmap = _build_cmap(RenderConfig(cmap="cloud_mask"))
        assert cmap.name == "cloud_mask"
        assert cmap.N == 4

    def test_build_norm_categorical(self):
        from clouds_decoded.visualisation.layers import _build_norm, RenderConfig
        from matplotlib.colors import BoundaryNorm
        rc = RenderConfig(categorical=True, category_labels={0: "A", 1: "B", 2: "C"})
        norm = _build_norm(rc)
        assert isinstance(norm, BoundaryNorm)

    def test_build_norm_log_scale(self):
        from clouds_decoded.visualisation.layers import _build_norm, RenderConfig
        from matplotlib.colors import LogNorm
        rc = RenderConfig(log_scale=True, vmin=0.1, vmax=100)
        norm = _build_norm(rc)
        assert isinstance(norm, LogNorm)

    def test_build_norm_default(self):
        from clouds_decoded.visualisation.layers import _build_norm, RenderConfig
        rc = RenderConfig()
        norm = _build_norm(rc)
        assert norm is None


# ---------------------------------------------------------------------------
# CRS reprojection
# ---------------------------------------------------------------------------

class TestReprojectExtent:
    """Test extent reprojection to EPSG:4326."""

    def test_utm_to_4326(self):
        from clouds_decoded.visualisation.layers import reproject_extent_to_4326

        # UTM zone 33N extent (roughly central Europe)
        extent = (500000, 600000, 5500000, 5600000)
        crs = CRS.from_epsg(32633)
        west, south, east, north = reproject_extent_to_4326(extent, crs)

        # Should be reasonable lon/lat values
        assert -180 <= west <= 180
        assert -180 <= east <= 180
        assert -90 <= south <= 90
        assert -90 <= north <= 90
        assert west < east
        assert south < north


# ---------------------------------------------------------------------------
# MapLibreViewer init (no serve)
# ---------------------------------------------------------------------------

class TestMapLibreViewer:
    """Test MapLibreViewer construction (no server started)."""

    def test_viewer_init(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig
        from clouds_decoded.visualisation.viewer import MapLibreViewer

        layers = [
            Layer(name="A", data=np.random.rand(30, 30).astype(np.float32), render=RenderConfig()),
            Layer(name="B", data=np.random.rand(30, 30).astype(np.float32), render=RenderConfig()),
        ]
        viewer = MapLibreViewer(layers)
        assert len(viewer.layers) == 2
        # Cache is lazy — empty until first request
        assert len(viewer._png_cache) == 0

    def test_viewer_rejects_empty(self):
        from clouds_decoded.visualisation.viewer import MapLibreViewer

        with pytest.raises(ValueError, match="No layers"):
            MapLibreViewer([])

    def test_viewer_builds_metadata(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig
        from clouds_decoded.visualisation.viewer import MapLibreViewer

        layers = [
            Layer(
                name="Test Layer",
                data=np.random.rand(30, 30).astype(np.float32),
                render=RenderConfig(label="Test", units="m"),
            ),
        ]
        viewer = MapLibreViewer(layers)
        meta = viewer._build_layer_metadata()
        assert len(meta) == 1
        assert meta[0]["name"] == "Test Layer"
        assert meta[0]["label"] == "Test"
        assert meta[0]["units"] == "m"
        assert meta[0]["is_rgb"] is False

    def test_viewer_metadata_rgb(self):
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, RGBConfig
        from clouds_decoded.visualisation.viewer import MapLibreViewer

        layers = [
            Layer(
                name="RGB",
                data=np.random.rand(30, 30, 3).astype(np.float32),
                render=RenderConfig(),
                is_rgb=True,
                rgb_config=RGBConfig(gamma=0.8),
            ),
        ]
        viewer = MapLibreViewer(layers)
        meta = viewer._build_layer_metadata()
        assert meta[0]["is_rgb"] is True
        assert meta[0]["rgb_config"]["gamma"] == 0.8

    def test_png_cache_lazy(self):
        """PNG cache starts empty and is populated on demand via render_layer_to_png."""
        from clouds_decoded.visualisation.layers import Layer, RenderConfig, render_layer_to_png
        from clouds_decoded.visualisation.viewer import MapLibreViewer

        layers = [
            Layer(name="X", data=np.random.rand(20, 20).astype(np.float32), render=RenderConfig()),
        ]
        viewer = MapLibreViewer(layers)
        assert len(viewer._png_cache) == 0
        # Simulate what the HTTP handler does
        png = render_layer_to_png(layers[0])
        viewer._png_cache["X"] = png
        assert isinstance(viewer._png_cache["X"], bytes)
        assert viewer._png_cache["X"][:8] == b'\x89PNG\r\n\x1a\n'


# ---------------------------------------------------------------------------
# Loader (real data, skipped if not available)
# ---------------------------------------------------------------------------

_REAL_SCENE_DIR = Path("/home/ali/projects/clouds-decoded/mynewproject/scenes/S2A_MSIL1C_20241116T120401_N0511_R066_T29VNH_20241116T135149")

@pytest.mark.skipif(not _REAL_SCENE_DIR.exists(), reason="Real scene data not available")
class TestLoaderRealData:
    """Test load_scene_layers against real project outputs."""

    def test_load_outputs_only(self):
        from clouds_decoded.visualisation.loader import load_scene_layers

        layers = load_scene_layers(str(_REAL_SCENE_DIR), scene_path=None)
        # Should have at least: cloud_mask, cloud_height, properties (4 bands)
        assert len(layers) >= 6
        names = [l.name for l in layers]
        assert "Cloud Mask" in names
        assert "Cloud Height" in names
        assert any("tau" in n for n in names)
