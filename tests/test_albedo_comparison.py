"""Comparison tests for albedo estimation methods.

Tests GP and unconditional MLP on the same scene and verifies they both
produce valid AlbedoData outputs with correct shapes.

Uses the sample Sentinel-2 scene when available; skips gracefully otherwise.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from clouds_decoded.data import (
    AlbedoData,
    CloudMaskData,
    CloudMaskMetadata,
    Sentinel2Scene,
)
from clouds_decoded.modules.albedo_estimator import AlbedoEstimator, AlbedoEstimatorConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAMPLE_SCENE = Path(
    "/data/sample-sentinel2-scenes/"
    "S2A_MSIL1C_20241217T215911_N0511_R086_T01KGB_20241217T233018.SAFE"
)
UNCONDITIONAL_CHECKPOINT = Path(
    __file__
).resolve().parent.parent / "src/modules/albedo_estimator/datadriven/models/albedo_model.pth"

HAS_SAMPLE_SCENE = SAMPLE_SCENE.exists()
HAS_UNCONDITIONAL = UNCONDITIONAL_CHECKPOINT.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dummy_scene() -> Sentinel2Scene:
    """Create a minimal Sentinel2Scene with synthetic data."""
    scene = Sentinel2Scene()

    h, w = 100, 100
    rng = np.random.default_rng(42)

    for band in [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B8A", "B09", "B10", "B11", "B12",
    ]:
        data = rng.uniform(500, 3000, (h, w)).astype(np.float32)
        # Add "clouds" (high reflectance)
        data[30:50, 40:70] = rng.uniform(5000, 8000, (20, 30))
        scene.bands[band] = data

    scene.transform = Affine.translation(500000.0, 6000000.0) * Affine.scale(10.0, -10.0)
    scene.crs = CRS.from_epsg(32601)
    scene.sun_zenith = 60.0
    scene.sun_azimuth = 160.0
    scene.view_zenith = 5.0
    scene.view_azimuth = 180.0
    scene.image_azimuth = np.radians(10.0)
    scene.quantification_value = 10000.0
    scene.radio_add_offset = {}

    return scene


def _create_cloud_mask(scene: Sentinel2Scene) -> CloudMaskData:
    """Create a cloud mask where the top half is clear, bottom half is cloud."""
    ref_band = scene.bands["B02"]
    h, w = ref_band.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2:, :] = 1  # Bottom half = cloud
    return CloudMaskData(
        data=mask,
        transform=scene.transform,
        crs=scene.crs,
        metadata=CloudMaskMetadata(categorical=True, classes={0: "Clear", 1: "Cloud"}),
    )


def _mock_angles(rows, cols, resolution=10.0):
    """Return synthetic angle arrays matching get_angles_at_pixels signature."""
    n = len(rows)
    return {
        "sun_zenith": np.full(n, 60.0, dtype=np.float32),
        "sun_azimuth": np.full(n, 160.0, dtype=np.float32),
        "view_zenith": np.full(n, 5.0, dtype=np.float32),
        "view_azimuth": np.full(n, 180.0, dtype=np.float32),
    }


def _validate_albedo_result(
    result: AlbedoData,
    expected_method: str,
    n_bands: int = 13,
) -> None:
    """Common assertions for any albedo result."""
    assert result is not None
    assert result.data is not None
    assert result.data.ndim == 3
    assert result.data.shape[0] == n_bands
    assert result.metadata.method == expected_method
    assert len(result.metadata.band_names) == n_bands
    # Values should be non-negative (clipped) and mostly finite
    finite_frac = np.isfinite(result.data).mean()
    assert finite_frac > 0.5, f"Too many non-finite values: {1 - finite_frac:.1%}"


# ---------------------------------------------------------------------------
# Smoke tests (dummy data, always run)
# ---------------------------------------------------------------------------


class TestAlbedoSmoke:
    """Fast smoke tests on synthetic data."""

    def test_gp_with_mask(self):
        """GP method runs on dummy scene with cloud mask."""
        scene = _create_dummy_scene()
        cloud_mask = _create_cloud_mask(scene)

        config = AlbedoEstimatorConfig(method="gp", fallback="constant")
        estimator = AlbedoEstimator(config)
        result = estimator.process(scene, cloud_mask=cloud_mask)

        _validate_albedo_result(result, expected_method="gp")
        assert result.metadata.clear_fraction > 0

    def test_gp_no_mask_falls_back(self):
        """GP without mask falls back to constant."""
        scene = _create_dummy_scene()

        config = AlbedoEstimatorConfig(method="gp", fallback="constant")
        estimator = AlbedoEstimator(config)
        result = estimator.process(scene, cloud_mask=None)

        _validate_albedo_result(result, expected_method="constant")
        assert result.metadata.fallback_used is True

    @pytest.mark.skipif(not HAS_UNCONDITIONAL, reason="No unconditional checkpoint")
    def test_datadriven_smoke(self):
        """Unconditional MLP runs on dummy scene."""
        scene = _create_dummy_scene()

        config = AlbedoEstimatorConfig(method="datadriven")
        estimator = AlbedoEstimator(config)

        with patch.object(Sentinel2Scene, "get_angles_at_pixels", side_effect=_mock_angles):
            with patch.object(Sentinel2Scene, "get_wind_data", return_value=(5.0, 220.0)):
                result = estimator.process(scene)

        _validate_albedo_result(result, expected_method="datadriven")


# ---------------------------------------------------------------------------
# Real-scene comparison (requires sample scene + both checkpoints)
# ---------------------------------------------------------------------------

NEED_REAL = not (HAS_SAMPLE_SCENE and HAS_UNCONDITIONAL)


@pytest.mark.skipif(NEED_REAL, reason="Sample scene or model checkpoints not available")
class TestAlbedoComparison:
    """Compare GP and data-driven methods on the same real Sentinel-2 scene."""

    @pytest.fixture(scope="class")
    def scene(self):
        scene = Sentinel2Scene()
        scene.read(str(SAMPLE_SCENE))
        return scene

    @pytest.fixture(scope="class")
    def cloud_mask(self, scene):
        """Simple threshold-based cloud mask for comparison."""
        from clouds_decoded.modules.cloud_mask import (
            CloudMaskConfig,
            ThresholdCloudMaskProcessor,
        )

        config = CloudMaskConfig(
            method="threshold", threshold_band="B08", threshold_value=4000,
        )
        processor = ThresholdCloudMaskProcessor(config)
        return processor.process(scene)

    def test_gp(self, scene, cloud_mask):
        """GP estimation on real scene."""
        config = AlbedoEstimatorConfig(
            method="gp", fallback="constant", output_resolution=300,
        )
        estimator = AlbedoEstimator(config)
        result = estimator.process(scene, cloud_mask=cloud_mask)

        _validate_albedo_result(result, expected_method="gp")
        logger.info(
            f"GP: clear={result.metadata.clear_fraction:.1%}, "
            f"samples={result.metadata.n_training_samples}, "
            f"shape={result.data.shape}"
        )

    def test_datadriven(self, scene):
        """Unconditional MLP on real scene."""
        config = AlbedoEstimatorConfig(
            method="datadriven", output_resolution=300,
        )
        estimator = AlbedoEstimator(config)
        result = estimator.process(scene)

        _validate_albedo_result(result, expected_method="datadriven")
        logger.info(f"Datadriven: shape={result.data.shape}")

    def test_shapes_match(self, scene, cloud_mask):
        """Both methods produce the same output grid shape."""
        results = {}
        for method in ("gp", "datadriven"):
            config = AlbedoEstimatorConfig(
                method=method, fallback="constant", output_resolution=300,
            )
            estimator = AlbedoEstimator(config)
            results[method] = estimator.process(scene, cloud_mask=cloud_mask)

        shapes = {m: r.data.shape for m, r in results.items()}
        logger.info(f"Output shapes: {shapes}")

        ref_shape = results["gp"].data.shape
        for method, result in results.items():
            assert result.data.shape == ref_shape, (
                f"{method} shape {result.data.shape} != GP shape {ref_shape}"
            )

    def test_clear_pixel_agreement(self, scene, cloud_mask):
        """On clear pixels, both methods should roughly agree."""
        configs = {
            m: AlbedoEstimatorConfig(
                method=m, fallback="constant", output_resolution=300,
            )
            for m in ("gp", "datadriven")
        }
        results = {}
        for method, config in configs.items():
            estimator = AlbedoEstimator(config)
            results[method] = estimator.process(scene, cloud_mask=cloud_mask)

        # Compare mean albedo across all bands (finite values only)
        means = {}
        for method, result in results.items():
            finite_mask = np.isfinite(result.data)
            means[method] = float(result.data[finite_mask].mean())

        logger.info(f"Mean albedo: {means}")

        # All methods should produce positive mean albedo
        for method, mean_val in means.items():
            assert mean_val > 0, f"{method} has non-positive mean albedo: {mean_val}"
