import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from pathlib import Path
import yaml
from rasterio.transform import Affine

from clouds_decoded.modules.cloud_height_emulator.processor import CloudHeightEmulatorProcessor
from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudMaskData, AlbedoData
from clouds_decoded.project import Project

@pytest.fixture
def mock_scene():
    """Create a mock Sentinel2Scene with 100x100 bands."""
    scene = MagicMock(spec=Sentinel2Scene)
    scene.product_uri = "S2A_MSIL1C_20230101T100000_N0509_R022_T31UFU_20230101T100000"
    
    # Mock bands (13 channels as often expected)
    band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    scene.bands = {name: np.random.rand(100, 100).astype(np.float32) for name in band_names}
    
    def get_band(name, reflectance=True):
        return scene.bands.get(name, np.random.rand(100, 100).astype(np.float32))
    
    scene.get_band.side_effect = get_band
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
            # Input x is (B, C, H, W). Output should be (B, H, W) for each head
            B, _, H, W = x.shape
            return {
                "regression": torch.zeros((B, H, W), device=x.device),
                "segmentation": torch.zeros((B, H, W), device=x.device)
            }
        
        model_instance.side_effect = forward
        mock.return_value = model_instance
        yield mock

@pytest.fixture(autouse=True)
def mock_dataloader():
    """Ensure DataLoader uses num_workers=0 to avoid multiprocessing issues in tests."""
    with patch("clouds_decoded.modules.cloud_height_emulator.processor.DataLoader") as mock:
        def side_effect(dataset, **kwargs):
            # Force num_workers to 0
            kwargs['num_workers'] = 0
            # Disable pin_memory for tests to avoid noise from internal PyTorch warnings
            kwargs['pin_memory'] = False
            # Import here to avoid circular dependencies if any
            from torch.utils.data import DataLoader as RealDataLoader
            return RealDataLoader(dataset, **kwargs)
        mock.side_effect = side_effect
        yield mock

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_config_initialization():
    """Verify CloudHeightEmulatorConfig defaults and validation."""
    config = CloudHeightEmulatorConfig()
    assert config.in_channels == 6  # Default based on src
    assert config.window_size == (1024, 1024)
    
    # Validation test
    with pytest.raises(ValueError, match="Overlap .* must be smaller than window dimensions"):
        CloudHeightEmulatorConfig(window_size=(100, 100), overlap=100)

def test_processor_initialization(mock_unet):
    """Verify processor loads model and respects device."""
    config = CloudHeightEmulatorConfig(device="cpu")
    processor = CloudHeightEmulatorProcessor(config)
    
    assert processor.device == "cpu"
    processor._load_model()
    mock_unet.assert_called_once()
    assert processor.model is not None

def test_emulator_input_output_shapes(mock_unet, mock_scene):
    """Verify that output grid matches input scene dimensions."""
    config = CloudHeightEmulatorConfig(window_size=(64, 64), overlap=16)
    processor = CloudHeightEmulatorProcessor(config)
    
    result = processor.process(mock_scene)
    
    assert isinstance(result, CloudHeightGridData)
    # Input was 100x100, output must be 100x100
    assert result.data.shape == (100, 100)
    # Note: the mock returns zeros, and processor averages windows.
    # The processor multiplies result by 20000 in _sliding_window_inference.
    # So if mock returns 0, result should be 0.
    assert np.all(result.data == 0)

# ---------------------------------------------------------------------------
# Integration Tests (Project / Workflow)
# ---------------------------------------------------------------------------

def test_emulator_within_project(tmp_path, mock_unet, mock_scene):
    """Verify emulator integration within the Project abstraction."""
    project_dir = tmp_path / "test_proj"
    
    # 1. Initialize project with emulator enabled
    project = Project.init(str(project_dir), name="Test Emulator Project", use_emulator=True)
    
    # 2. Check if the correct config was written
    config_path = project_dir / "configs" / "cloud_height.yaml"
    assert config_path.exists()
    
    with open(config_path) as f:
        cfg_data = yaml.safe_load(f)
    # Check for an emulator-specific field
    assert "pth_path" in cfg_data
    
    # 3. Mock data I/O to avoid real files
    with patch("clouds_decoded.data.Sentinel2Scene.read"), \
         patch("clouds_decoded.data.CloudMaskData.from_file"), \
         patch("clouds_decoded.modules.cloud_mask.processor.CloudMaskProcessor.postprocess"), \
         patch("clouds_decoded.data.CloudHeightGridData.write"):
        
        # Setup intermediate mocks
        mock_mask = MagicMock(spec=CloudMaskData)
        mock_mask.data = np.zeros((100, 100), dtype=np.uint8)
        
        # Run the cloud_height step
        # Note: we need to make sure the Project uses our mock_scene
        # In a real run, it would read from disk. For this test, let's mock _run_step inputs.
        # Run the cloud_height step
        with patch.object(Project, "_load_step_config") as mock_load_cfg:
            mock_load_cfg.return_value = CloudHeightEmulatorConfig(window_size=(64, 64), overlap=16)
            
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
    project = Project.init(str(project_dir), use_emulator=True)
    
    # Mock all processors and data objects
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
        
        # Setup returned data objects for pipeline flow
        mock_run_height.return_value = MagicMock(spec=CloudHeightGridData)
        mock_run_height.return_value.metadata = MagicMock()
        mock_run_height.return_value.metadata.provenance = {}
        
        # Add a scene
        project.add_scene("mock_scene.SAFE")
        
        # Run project
        with patch("clouds_decoded.data.Sentinel2Scene", return_value=mock_scene):
             project.run(force=True)
        
        # Verify run_cloud_height was called with use_emulator=True
        mock_run_height.assert_called()
        args, kwargs = mock_run_height.call_args
        assert kwargs["use_emulator"] is True
        # Project._run_step passes config as second positional argument
        assert isinstance(args[1], CloudHeightEmulatorConfig)

def test_emulator_cloud_height_only_workflow(tmp_path, mock_unet, mock_scene):
    """Verify standalone cloud-height step via Project."""
    # This is similar to test_emulator_within_project but ensures the project logic
    # correctly dispatches to run_cloud_height with use_emulator=True.
    project_dir = tmp_path / "height_only_proj"
    project = Project.init(str(project_dir), use_emulator=True)
    
    with patch("clouds_decoded.cli.entry.run_cloud_height") as mock_run_height, \
         patch("clouds_decoded.data.CloudHeightGridData.write"):
        
        mock_run_height.return_value = MagicMock(spec=CloudHeightGridData)
        mock_run_height.return_value.metadata = MagicMock()
        
        # Call the step
        project._run_step(
            "cloud_height", 
            mock_scene, 
            "path", 
            project_dir / "scenes" / "test"
        )
        
        # Check if emulator flag was passed
        assert mock_run_height.call_args[1]["use_emulator"] is True