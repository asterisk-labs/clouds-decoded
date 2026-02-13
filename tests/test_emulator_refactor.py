import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append("/home/paul/clouds-decoded/src")

from modules.cloud_height_emulator.processor import CloudHeightEmulatorProcessor
from modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from rasterio.transform import Affine

@pytest.fixture
def mock_scene():
    scene = MagicMock(spec=Sentinel2Scene)
    # Mock bands
    scene.bands = {f"B{i:02d}": np.random.rand(100, 100) for i in range(1, 13)}
    scene.bands["B8A"] = np.random.rand(100, 100)
    
    # Mock get_band
    def get_band(name, reflectance=True):
        return np.random.rand(100, 100).astype(np.float32)
    scene.get_band.side_effect = get_band
    
    # Mock transform and crs
    scene.transform = Affine(10, 0, 0, 0, -10, 1000)
    scene.crs = "EPSG:32631"
    
    return scene

def test_config_initialization():
    config = CloudHeightEmulatorConfig(in_channels=13)
    assert config.in_channels == 13
    assert config.window_size == (1024, 1024)

@patch("modules.cloud_height_emulator.processor.Res34_Unet")
def test_processor_initialization(mock_unet):
    config = CloudHeightEmulatorConfig()
    processor = CloudHeightEmulatorProcessor(config)
    
    # Check device default
    assert processor.device == ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check model loading
    processor._load_model()
    mock_unet.assert_called_once()
    assert processor.model is not None

@patch("modules.cloud_height_emulator.processor.Res34_Unet")
def test_processing_flow(mock_unet, mock_scene):
    # Setup mock model output
    mock_model = MagicMock()
    # Return a tensor of shape (B, 1, H, W)
    # Window size default is 1024, but our scene is 100x100.
    # Sliding window logic should handle padding.
    # Input to model will be padded to at least window size?
    # Actually, let's use a small window size for test
    
    config = CloudHeightEmulatorConfig(window_size=(64, 64), overlap=16)
    processor = CloudHeightEmulatorProcessor(config)
    
    # Mock model return
    def forward(x):
        # x shape (B, C, H, W) -> (1, 1, 64, 64)
        return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])
    
    mock_model.side_effect = forward
    mock_unet.return_value = mock_model
    
    result = processor.process(mock_scene)
    
    assert isinstance(result, CloudHeightGridData)
    assert result.data.shape == (100, 100)
    assert result.metadata.processing_config["in_channels"] == 13

if __name__ == "__main__":
    # Manually run tests if not using pytest runner
    try:
        # We need to construct mocks manually if running as script
        print("Running manual verification...")
        
        # 1. Config
        cfg = CloudHeightEmulatorConfig(window_size=(32, 32), overlap=8)
        print("Config created.")
        
        # 2. Processor
        proc = CloudHeightEmulatorProcessor(cfg)
        print("Processor created.")
        
        # 3. Process with mocks
        # We'll use unittest.mock.patch as context manager
        with patch("modules.cloud_height_emulator.processor.Res34_Unet") as MockUnet:
            mock_net_instance = MagicMock()
            mock_net_instance.to.return_value = None
            mock_net_instance.eval.return_value = None
            
            # Forward pass mock
            def mock_forward(x):
                # Return random tensor matching spatial dims
                return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            
            mock_net_instance.side_effect = mock_forward
            
            MockUnet.return_value = mock_net_instance
            
            # Scene mock
            scene = MagicMock(spec=Sentinel2Scene)
            scene.get_band.return_value = np.zeros((100, 100), dtype=np.float32)
            scene.bands = {"B02": np.zeros((100, 100))}
            scene.transform = Affine(10, 0, 0, 0, -10, 1000)
            scene.crs = "EPSG:32631"
            # make sure model is on same device

            result = proc.process(scene)
            print("Process finished.")
            # import pdb; pdb.set_trace()
            assert isinstance(result, CloudHeightGridData)
            assert result.data.shape == (100, 100)
            print("Verification Successful!")
            
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
