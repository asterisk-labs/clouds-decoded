import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from pathlib import Path
import yaml
from rasterio.transform import Affine

from clouds_decoded.modules.cloud_height_emulator.processor import CloudHeightEmulatorProcessor
from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.normalization import CloudHeightNormalizationWrapper
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudMaskData, AlbedoData
from clouds_decoded.project import Project


@pytest.fixture
def mock_weights_file(tmp_path, monkeypatch):
    """Create a minimal valid checkpoint and point CLOUDS_DECODED_ASSETS_DIR at tmp_path."""
    monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))
    weights_path = tmp_path / "models" / "cloud_height_emulator" / "default.pth"
    weights_path.parent.mkdir(parents=True)
    # Minimal state dict matching CloudHeightNormalizationWrapper's 4 buffers.
    # The inner model is a MagicMock so it contributes no keys.
    state = {
        "in_min": torch.tensor([0.0]),
        "in_max": torch.tensor([1.0]),
        "out_min": torch.tensor([0.0]),
        "out_max": torch.tensor([1.0]),
    }
    torch.save(state, weights_path)
    return weights_path


@pytest.fixture
def mock_scene():
    """Create a mock Sentinel2Scene with 100x100 bands."""
    scene = MagicMock(spec=Sentinel2Scene)
    scene.product_uri = "S2A_MSIL1C_20230101T100000_N0509_R022_T31UFU_20230101T100000"

    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    scene.bands = {name: np.random.rand(100, 100).astype(np.float32) for name in band_names}

    def get_band(name, reflectance=True, resolution=None, cache=True):
        return scene.bands.get(name, np.random.rand(100, 100).astype(np.float32))

    def get_bands(names, reflectance=True, resolution=None, cache=True, n_workers=1):
        result = []
        for name in names:
            band_mock = MagicMock()
            band_mock.data = scene.bands.get(name, np.random.rand(100, 100).astype(np.float32))
            result.append(band_mock)
        return result

    scene.get_band.side_effect = get_band
    scene.get_bands.side_effect = get_bands
    scene.transform = Affine(10, 0, 0, 0, -10, 1000)
    scene.crs = "EPSG:32631"
    scene.shape = (100, 100)
    return scene


@pytest.fixture
def mock_unet():
    """Mock the Res34_Unet model to return a dict with regression/segmentation tensors."""
    with patch("clouds_decoded.modules.cloud_height_emulator.processor.Res34_Unet") as mock:
        model_instance = MagicMock()
        model_instance.to.return_value = model_instance
        model_instance.eval.return_value = model_instance

        def forward(x):
            B, _, H, W = x.shape
            return {
                # Return positive regression values and high segmentation confidence
                "regression": torch.ones((B, H, W), device=x.device) * 0.5,
                "segmentation": torch.ones((B, H, W), device=x.device) * 2.0,  # sigmoid(2) > 0.5
            }

        model_instance.side_effect = forward
        mock.return_value = model_instance
        yield mock


@pytest.fixture(autouse=True)
def mock_dataloader():
    """Ensure DataLoader uses num_workers=0 to avoid multiprocessing issues in tests."""
    with patch("clouds_decoded.sliding_window.DataLoader") as mock:
        def side_effect(dataset, **kwargs):
            kwargs['num_workers'] = 0
            kwargs['pin_memory'] = False
            from torch.utils.data import DataLoader as RealDataLoader
            return RealDataLoader(dataset, **kwargs)
        mock.side_effect = side_effect
        yield mock


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_config_initialization(tmp_path, monkeypatch):
    """Verify CloudHeightEmulatorConfig defaults and validation."""
    monkeypatch.setenv("CLOUDS_DECODED_ASSETS_DIR", str(tmp_path))
    config = CloudHeightEmulatorConfig()
    assert config.in_channels == 8
    assert config.window_size == (1024, 1024)
    # model_path is now auto-resolved to the managed assets directory
    assert config.model_path == str(tmp_path / "models" / "cloud_height_emulator" / "default.pth")

    with pytest.raises(ValueError, match="Overlap .* must be smaller than window size"):
        CloudHeightEmulatorConfig(window_size=(100, 100), overlap=100)


def test_config_strict_mode():
    """Verify extra fields are rejected."""
    with pytest.raises(Exception):
        CloudHeightEmulatorConfig(nonexistent_field="bad")


def test_processor_initialization(mock_unet, mock_weights_file):
    """Verify processor loads model and respects device."""
    config = CloudHeightEmulatorConfig(device="cpu")
    processor = CloudHeightEmulatorProcessor(config)

    assert processor.device == "cpu"
    processor._load_model()
    mock_unet.assert_called_once()
    assert processor.model is not None


def test_emulator_input_output_shapes(mock_unet, mock_scene, mock_weights_file):
    """Verify that output grid matches input scene dimensions."""
    config = CloudHeightEmulatorConfig(
        window_size=(64, 64), overlap=16, device="cpu", working_resolution=10,
    )
    processor = CloudHeightEmulatorProcessor(config)

    result = processor.process(mock_scene)

    assert isinstance(result, CloudHeightGridData)
    assert result.data.shape == (100, 100)
    # Mock returns regression=0.5 with cloud confidence high, so heights should be positive
    assert result.data.max() > 0


def test_emulator_no_cloud_mask_on_output(mock_unet, mock_scene, mock_weights_file):
    """Verify CloudHeightGridData output does not carry a cloud_mask attribute."""
    config = CloudHeightEmulatorConfig(
        window_size=(64, 64), overlap=16, device="cpu", working_resolution=10,
    )
    processor = CloudHeightEmulatorProcessor(config)

    result = processor.process(mock_scene)

    assert not hasattr(result, 'cloud_mask') or getattr(result, 'cloud_mask', None) is None


def test_cloud_height_normalization_wrapper():
    """Verify CloudHeightNormalizationWrapper normalize/denormalize and state_dict roundtrip."""
    from clouds_decoded.modules.cloud_height_emulator.resunet import Res34_Unet

    in_min, in_max = 0.0, 1.0
    out_min, out_max = 0.0, 20_000.0

    core = Res34_Unet(in_channels=6, out_channels=[1, 1],
                      heads=["regression", "segmentation"],
                      heads_hidden_channels=[48, 48], pretrained=False)

    wrapper = CloudHeightNormalizationWrapper(
        model=core,
        input_stats={"min": [in_min], "max": [in_max]},
        output_stats={"min": [out_min], "max": [out_max]},
    )

    # normalize_input: min-max to [-1, 1]
    x = torch.tensor([0.0, 0.5, 1.0])
    normed = wrapper.normalize_input(x)
    expected = 2.0 * (x - in_min) / (in_max - in_min) - 1.0
    assert torch.allclose(normed, expected)

    # denormalize_output: [-1, 1] back to physical units
    y = torch.tensor([-1.0, 0.0, 1.0])
    denormed = wrapper.denormalize_output(y)
    expected_d = (y + 1) / 2 * (out_max - out_min) + out_min
    assert torch.allclose(denormed, expected_d)

    # State dict roundtrip — buffers must persist
    sd = wrapper.state_dict()
    assert "in_max" in sd
    assert "out_max" in sd

    wrapper2 = CloudHeightNormalizationWrapper(
        model=Res34_Unet(in_channels=6, out_channels=[1, 1],
                         heads=["regression", "segmentation"],
                         heads_hidden_channels=[48, 48], pretrained=False),
        input_stats={"min": [0.0], "max": [0.0]},  # dummy values
        output_stats={"min": [0.0], "max": [0.0]},
    )
    wrapper2.load_state_dict(sd)
    assert torch.allclose(wrapper2.in_max, torch.tensor([in_max]))
    assert torch.allclose(wrapper2.out_max, torch.tensor([out_max]))


def test_cloud_height_norm_wrapper_forward_dict_output(mock_unet):
    """Verify forward() adds regression_norm key to dict output."""
    core = mock_unet.return_value

    wrapper = CloudHeightNormalizationWrapper(
        model=core,
        input_stats={"min": [0.0], "max": [1.0]},
        output_stats={"min": [0.0], "max": [20_000.0]},
    )

    # Simulate model returning a dict
    raw_regression = torch.tensor([0.5])
    raw_seg = torch.tensor([2.0])
    core.side_effect = lambda x: {"regression": raw_regression, "segmentation": raw_seg}

    x = torch.zeros(1, 6, 4, 4)
    output = wrapper(x)

    assert "regression_norm" in output
    assert "regression" in output
    # regression_norm should be the raw (pre-denorm) value
    assert torch.allclose(output["regression_norm"], raw_regression)
    # regression should be denormalized
    expected = (raw_regression + 1) / 2 * 20_000.0
    assert torch.allclose(output["regression"], expected)


# ---------------------------------------------------------------------------
# Integration Tests (Project / Workflow)
# ---------------------------------------------------------------------------

def test_emulator_within_project(tmp_path, mock_unet, mock_scene, mock_weights_file):
    """Verify emulator integration within the Project abstraction."""
    project_dir = tmp_path / "test_proj"

    project = Project.init(str(project_dir), name="Test Emulator Project")

    config_path = project_dir / "configs" / "cloud_height.yaml"
    assert config_path.exists()

    with open(config_path) as f:
        cfg_data = yaml.safe_load(f)
    assert "model_path" in cfg_data

    with patch("clouds_decoded.data.Sentinel2Scene.read"), \
         patch("clouds_decoded.data.CloudMaskData.from_file"), \
         patch("clouds_decoded.modules.cloud_mask.processor.CloudMaskProcessor.postprocess"), \
         patch("clouds_decoded.data.CloudHeightGridData.write"):

        mock_mask = MagicMock(spec=CloudMaskData)
        mock_mask.data = np.zeros((100, 100), dtype=np.uint8)

        with patch.object(Project, "_load_step_config") as mock_load_cfg:
            mock_load_cfg.return_value = CloudHeightEmulatorConfig(
                window_size=(64, 64), overlap=16, device="cpu", working_resolution=10,
            )

            output_path = project._run_step(
                "cloud_height",
                mock_scene,
                "mock_scene_path",
                project_dir / "scenes" / "mock_scene",
                mask_result=mock_mask
            )

            assert "cloud_height.tif" in output_path


def test_emulator_full_workflow(tmp_path, mock_unet, mock_scene):
    """Verify the full-workflow pipeline correctly utilizes the emulator."""
    project_dir = tmp_path / "workflow_proj"
    project = Project.init(str(project_dir))

    with patch("clouds_decoded.data.Sentinel2Scene.read", return_value=None), \
         patch("clouds_decoded.data.CloudMaskData.from_file"), \
         patch("clouds_decoded.data.CloudHeightGridData.from_file"), \
         patch("clouds_decoded.data.AlbedoData.from_file"), \
         patch("clouds_decoded.data.CloudHeightGridData.write"), \
         patch("clouds_decoded.data.CloudMaskData.write"), \
         patch("clouds_decoded.data.AlbedoData.write"), \
         patch("clouds_decoded.data.refl2prop.CloudPropertiesData.write"), \
         patch("clouds_decoded.cli.entry.run_cloud_mask"), \
         patch("clouds_decoded.cli.entry.run_cloud_height") as mock_run_height, \
         patch("clouds_decoded.cli.entry.run_albedo"), \
         patch("clouds_decoded.cli.entry.run_refocus"), \
         patch("clouds_decoded.cli.entry.run_cloud_properties"):

        mock_run_height.return_value = MagicMock(spec=CloudHeightGridData)
        mock_run_height.return_value.metadata = MagicMock()
        mock_run_height.return_value.metadata.provenance = {}

        project.add_scene("mock_scene.SAFE")

        with patch("clouds_decoded.data.Sentinel2Scene", return_value=mock_scene):
            project.run(force=True)

        mock_run_height.assert_called()
        args, kwargs = mock_run_height.call_args
        assert isinstance(args[1], CloudHeightEmulatorConfig)
        assert args[1].use_emulator is True


def test_emulator_cloud_height_only_workflow(tmp_path, mock_unet, mock_scene):
    """Verify standalone cloud-height step via Project."""
    project_dir = tmp_path / "height_only_proj"
    project = Project.init(str(project_dir))

    with patch("clouds_decoded.cli.entry.run_cloud_height") as mock_run_height, \
         patch("clouds_decoded.data.CloudHeightGridData.write"):

        mock_run_height.return_value = MagicMock(spec=CloudHeightGridData)
        mock_run_height.return_value.metadata = MagicMock()

        project._run_step(
            "cloud_height",
            mock_scene,
            "path",
            project_dir / "scenes" / "test"
        )

        args, kwargs = mock_run_height.call_args
        assert isinstance(args[1], CloudHeightEmulatorConfig)
        assert args[1].use_emulator is True
