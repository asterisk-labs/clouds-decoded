"""
Smoke tests to verify basic processor functionality after refactoring.

These are NOT comprehensive tests - just quick checks that processors
can instantiate and run without crashing on dummy data.
"""
import pytest
import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS

from clouds_decoded.data import Sentinel2Scene


def create_dummy_scene():
    """Create a minimal Sentinel2Scene with synthetic data."""
    scene = Sentinel2Scene()

    # Create dummy bands (100x100 pixels)
    h, w = 100, 100
    np.random.seed(42)

    # Simulate realistic S2 reflectance values (0-10000 range)
    for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                 'B08', 'B8A', 'B09', 'B11', 'B12']:
        # Random reflectance with some clouds (higher values)
        band_data = np.random.uniform(500, 3000, (h, w)).astype(np.float32)
        # Add some "clouds" (high reflectance)
        band_data[30:50, 40:70] = np.random.uniform(5000, 8000, (20, 30))
        scene.bands[band] = band_data

    # Set required metadata
    scene.transform = Affine.translation(0.0, 0.0) * Affine.scale(10.0, -10.0)
    scene.crs = CRS.from_epsg(32633)  # UTM 33N
    scene.sun_zenith = 30.0
    scene.sun_azimuth = 120.0
    scene.view_zenith = 5.0
    scene.view_azimuth = 180.0
    scene.image_azimuth = np.radians(10.0)  # Small rotation angle in radians

    return scene


@pytest.fixture
def dummy_scene():
    """Pytest fixture wrapper."""
    return create_dummy_scene()


def test_cloud_height_basic(dummy_scene):
    """Smoke test: CloudHeightProcessor can run without crashing."""
    from clouds_decoded.modules.cloud_height import CloudHeightProcessor, CloudHeightConfig

    # Fast config for testing
    config = CloudHeightConfig(
        stride=500,  # Very coarse
        max_height=5000,
        n_workers=1
    )

    processor = CloudHeightProcessor(config)

    # This should not crash - may return None data if no cloud parallax found
    result = processor.process(dummy_scene)

    # Processor ran without exception
    assert result is not None
    # Data may be None if no valid parallax detected (expected with random noise)
    if result.data is not None:
        print(f"✓ CloudHeightProcessor: output shape {result.data.shape}")
    else:
        print(f"✓ CloudHeightProcessor: ran successfully (no parallax in random data)")


def test_cloud_mask_threshold(dummy_scene):
    """Smoke test: ThresholdCloudMaskProcessor can run."""
    from clouds_decoded.modules.cloud_mask import ThresholdCloudMaskProcessor, CloudMaskConfig

    config = CloudMaskConfig(
        method="threshold",
        threshold_band="B08",
        threshold_value=4000  # Clouds above this
    )

    processor = ThresholdCloudMaskProcessor(config)
    result = processor.process(dummy_scene)

    assert result is not None
    assert result.data is not None
    assert result.data.shape == (100, 100)
    assert result.data.dtype in [np.uint8, np.int32, np.float32]
    print(f"✓ ThresholdCloudMaskProcessor: detected {np.sum(result.data > 0)} cloudy pixels")


def test_albedo_estimator(dummy_scene):
    """Smoke test: AlbedoEstimator can run with new interface."""
    from clouds_decoded.modules.albedo_estimator import AlbedoEstimator, AlbedoEstimatorConfig

    # Test 1: Constant fallback (no cloud mask, fallback="constant")
    config = AlbedoEstimatorConfig(fallback="constant")

    estimator = AlbedoEstimator(config)
    result = estimator.process(dummy_scene)

    assert result is not None
    assert result.data is not None
    assert result.data.ndim == 3  # (n_bands, h, w)
    assert result.data.shape[0] == len(dummy_scene.bands)  # One per band

    # Verify metadata
    assert hasattr(result.metadata, 'band_names')
    assert len(result.metadata.band_names) == result.data.shape[0]
    assert result.metadata.fallback_used is True
    assert result.metadata.method == "constant"

    # Test 2: GP fit with cloud mask
    from clouds_decoded.data import CloudMaskData, CloudMaskMetadata
    # Create a mask where top half is clear (0), bottom half is cloud (1)
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr[50:, :] = 1
    cloud_mask = CloudMaskData(
        data=mask_arr,
        transform=dummy_scene.transform,
        crs=dummy_scene.crs,
        metadata=CloudMaskMetadata(categorical=True, classes={0: 'Clear', 1: 'Cloud'}),
    )

    gp_config = AlbedoEstimatorConfig(method="gp")
    gp_estimator = AlbedoEstimator(gp_config)
    gp_result = gp_estimator.process(dummy_scene, cloud_mask=cloud_mask)

    assert gp_result.data.ndim == 3
    assert gp_result.data.shape[0] == len(dummy_scene.bands)
    assert gp_result.metadata.method == "gp"
    assert gp_result.metadata.fallback_used is False
    assert gp_result.metadata.clear_fraction > 0

    print(f"✓ AlbedoEstimator: constant shape {result.data.shape}, "
          f"GP shape {gp_result.data.shape}, "
          f"bands {len(result.metadata.band_names)}")


def test_albedo_estimator_idw(dummy_scene):
    """Smoke test: IDW albedo estimator runs and produces valid output."""
    from clouds_decoded.modules.albedo_estimator import AlbedoEstimator, AlbedoEstimatorConfig
    from clouds_decoded.data import CloudMaskData, CloudMaskMetadata

    # Create a mask where top half is clear (0), bottom half is cloud (1)
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr[50:, :] = 1
    cloud_mask = CloudMaskData(
        data=mask_arr,
        transform=dummy_scene.transform,
        crs=dummy_scene.crs,
        metadata=CloudMaskMetadata(categorical=True, classes={0: 'Clear', 1: 'Cloud'}),
    )

    idw_config = AlbedoEstimatorConfig(method="idw")
    idw_estimator = AlbedoEstimator(idw_config)
    idw_result = idw_estimator.process(dummy_scene, cloud_mask=cloud_mask)

    assert idw_result.data.ndim == 3
    assert idw_result.data.shape[0] == len(dummy_scene.bands)
    assert idw_result.metadata.method == "idw"
    assert idw_result.metadata.fallback_used is False
    assert idw_result.metadata.clear_fraction > 0

    print(f"✓ AlbedoEstimator IDW: shape {idw_result.data.shape}, "
          f"samples={idw_result.metadata.n_training_samples}")


def test_imports_work():
    """Smoke test: All processor imports work."""
    from clouds_decoded.modules.cloud_height import CloudHeightProcessor, CloudHeightConfig
    from clouds_decoded.modules.cloud_mask import CloudMaskProcessor, ThresholdCloudMaskProcessor
    from clouds_decoded.modules.albedo_estimator import AlbedoEstimator, AlbedoEstimatorConfig

    # Unified import
    from clouds_decoded.processors import (
        CloudHeightProcessor as CHP,
        CloudMaskProcessor as CMP,
        AlbedoEstimator as AE
    )

    print("✓ All imports successful")
    assert CHP is not None
    assert CMP is not None
    assert AE is not None


def test_get_band_reflectance(dummy_scene):
    """Smoke test: get_band() with reflectance conversion."""
    # Default quantification_value=10000, radio_add_offset={} (empty = offset 0)
    raw = dummy_scene.get_band('B02', reflectance=False)
    refl = dummy_scene.get_band('B02', reflectance=True)

    assert raw.shape == refl.shape
    assert refl.dtype == np.float32
    # With empty offset dict, quant=10000: reflectance = raw / 10000
    np.testing.assert_allclose(refl, raw / 10000.0, rtol=1e-5)

    # With per-band offset (typical for N0400+ products)
    dummy_scene.radio_add_offset = {'B02': -1000.0, 'B04': -1000.0}
    refl_offset = dummy_scene.get_band('B02', reflectance=True)
    np.testing.assert_allclose(refl_offset, (raw - 1000.0) / 10000.0, rtol=1e-5)

    # Band without an explicit offset still gets offset=0
    raw_b03 = dummy_scene.get_band('B03', reflectance=False)
    refl_b03 = dummy_scene.get_band('B03', reflectance=True)
    np.testing.assert_allclose(refl_b03, raw_b03 / 10000.0, rtol=1e-5)

    # Missing band raises KeyError
    with pytest.raises(KeyError):
        dummy_scene.get_band('B99')


def test_config_from_yaml(tmp_path):
    """Smoke test: Config loading from YAML works."""
    from clouds_decoded.modules.cloud_height import CloudHeightConfig

    # Create temp YAML
    yaml_content = """
stride: 300
max_height: 10000
n_workers: 4
"""
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(yaml_content)

    # Load it
    config = CloudHeightConfig.from_yaml(str(yaml_path))

    assert config.stride == 300
    assert config.max_height == 10000
    assert config.n_workers == 4
    print("✓ Config YAML loading works")


def test_cloud_mask_to_binary():
    """Smoke test: CloudMaskData.to_binary() works."""
    from clouds_decoded.data import CloudMaskData, CloudMaskMetadata
    from rasterio.transform import Affine

    # Create categorical mask with all classes
    mask_data = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 0],
        [3, 0, 0, 0]
    ], dtype=np.uint8)

    mask = CloudMaskData(
        data=mask_data,
        transform=Affine.identity(),
        crs=None,
        metadata=CloudMaskMetadata()
    )

    # Convert to binary (default: thick + thin cloud = classes 1, 2)
    binary = mask.to_binary()

    assert binary.data.shape == (4, 4)
    assert set(np.unique(binary.data)) == {0, 1}
    # Classes 1 and 2 should be 1, rest should be 0
    expected = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)
    np.testing.assert_array_equal(binary.data, expected)
    print("✓ CloudMaskData.to_binary() works")


if __name__ == "__main__":
    """Run smoke tests directly (no pytest needed)."""
    print("\n=== Running Smoke Tests ===\n")

    # Create dummy scene
    scene = create_dummy_scene()
    print(f"Created dummy scene with {len(scene.bands)} bands\n")

    # Run tests
    try:
        test_imports_work()
        test_cloud_height_basic(scene)
        test_cloud_mask_threshold(scene)
        test_albedo_estimator(scene)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            test_config_from_yaml(Path(tmpdir))

        test_cloud_mask_to_binary()

        print("\n=== ✓ All Smoke Tests Passed ===\n")
    except Exception as e:
        print(f"\n=== ✗ Smoke Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
