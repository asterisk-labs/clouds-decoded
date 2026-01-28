import typer
from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np
import logging
import rasterio

# Standardized Imports
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from clouds_decoded.processors import (
    CloudHeightProcessor,
    CloudHeightConfig,
    CloudPropertyInverter,
    CloudMaskProcessor
)

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLI")

app = typer.Typer(help="Clouds Decoded Command Line Interface")

@app.command()
def cloud_height(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    config_path: str = typer.Option(None, help="Path to config.yaml"),
    output_path: str = typer.Option("height_output.tif", help="Output path (e.g. .tif or .nc)"),
):
    """
    Calculate Cloud Height from Sentinel-2 data.
    """
    logger.info(f"Starting Cloud Height processing for: {scene_path}")
    
    # 1. Load Data
    scene = Sentinel2Scene()
    scene.read(scene_path)
    
    # 2. Configure
    # CloudHeightConfig uses scene_dir in init, likely for defaults.
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    config = CloudHeightConfig.from_yaml(config_path)
    # Note: scene_dir is no longer stored in config, but used for processing logging/paths if needed
    
    # 3. Process
    processor = CloudHeightProcessor(config)
    result = processor.process(scene)
    
    # 4. Save
    result.write(output_path)
    logger.info(f"Cloud Height saved to {output_path}")

@app.command()
def cloud_mask(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_path: str = typer.Option("mask_output.tif", help="Output path (e.g. .tif)"),
    threshold_band: str = typer.Option("B08", help="Band used for thresholding"),
    threshold_value: float = typer.Option(1600, help="Reflectance threshold (Raw DN)"),
):
    """
    Calculate Cloud Mask from Sentinel-2 data using simple thresholding.
    """
    logger.info(f"Starting Cloud Mask processing for: {scene_path}")
    
    # 1. Load Data
    scene = Sentinel2Scene()
    scene.read(scene_path)

    # 2. Process
    processor = CloudMaskProcessor()
    result = processor.process(
        scene, 
        threshold_band=threshold_band, 
        threshold_value=threshold_value
    )
    
    # 3. Save
    result.write(output_path)
    logger.info(f"Cloud Mask saved to {output_path}")

@app.command()
def cloud_properties(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    height_path: str = typer.Option(..., help="Path to Cloud Height raster (.tif/.nc) matching the scene"),
    model_path: str = typer.Option(..., help="Path to trained .pth model"),
    output_path: str = typer.Option("properties_output.nc", help="Output NetCDF path"),
):
    """
    Run Cloud Property Inversion (Refl2Prop). Requires pre-calculated cloud heights.
    """
    logger.info(f"Starting Cloud Properties Inversion for: {scene_path}")
    
    # 1. Load Scene
    scene = Sentinel2Scene()
    scene.read(scene_path)
    
    # 2. Load Heights
    # We assume the height raster matches the scene resolution/bounds
    height_data = CloudHeightGridData.from_file(height_path)
    if height_data.data is None:
        raise ValueError("Could not read height data")
        
    # 3. Configure
    # Create configuration object from arguments, or update defaults
    # For now, we manually construct it to maintain CLI compatibility without a config file
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig
    config = Refl2PropConfig(model_path=model_path)
        
    # 4. Process
    processor = CloudPropertyInverter(config)
    result = processor.process(scene, height_data)
    
    # 5. Save
    result.write(output_path)
    logger.info(f"Cloud Properties saved to {output_path}")

@app.command()
def workflow(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    model_path: str = typer.Option(..., help="Refl2Prop Model Path"),
    output_dir: str = typer.Option("output", help="Directory for outputs"),
):
    """
    End-to-end workflow: Cloud Mask -> Cloud Height -> Cloud Properties
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    
    mask_out = str(out / "cloud_mask.tif")
    height_out = str(out / "cloud_height.tif")
    props_out = str(out / "properties.tif")
    
    # 0. Mask
    logger.info("Running Step 0: Cloud Mask")
    cloud_mask(
        scene_path=scene_path, 
        output_path=mask_out,
        threshold_band="B08",
        threshold_value=1600.0
    )

    # 1. Height
    logger.info("Running Step 1: Cloud Height")
    # TODO: Pass mask to cloud height once supported
    cloud_height(scene_path=scene_path, config_path=None, output_path=height_out)

    # 2. Properties
    logger.info("Running Step 2: Cloud Properties")
    cloud_properties(
        scene_path=scene_path, 
        height_path=height_out, 
        model_path=model_path, 
        output_path=props_out
    )

if __name__ == "__main__":
    app()