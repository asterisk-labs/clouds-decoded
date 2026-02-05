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

    config = AlbedoEstimatorConfig(
        percentile=1.0,
        default_albedo=0.05
    )

    estimator = AlbedoEstimator(config)
    result = estimator.process(dummy_scene)

    # Verify new interface
    assert result is not None
    assert result.data is not None
    assert result.data.ndim == 3  # (n_bands, h, w)
    assert result.data.shape[0] == len(dummy_scene.bands)  # One per band
    assert result.data.shape[1:] == (100, 100)

    # Verify metadata
    assert hasattr(result.metadata, 'band_names')
    assert len(result.metadata.band_names) == result.data.shape[0]
    assert hasattr(result.metadata, 'albedo_values')

    print(f"✓ AlbedoEstimator: shape {result.data.shape}, "
          f"bands {len(result.metadata.band_names)}")


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

        print("\n=== ✓ All Smoke Tests Passed ===\n")
    except Exception as e:
        print(f"\n=== ✗ Smoke Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
