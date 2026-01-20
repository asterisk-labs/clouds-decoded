import typer
from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np
import logging
import rasterio

# Standardized Imports
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from clouds_decoded.processors import (
    # CloudHeightProcessor,
    # CloudHeightConfig,
    CloudPropertyInverter
)

from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
from clouds_decoded.modules.cloud_height.config import CloudHeightConfig


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
    config = CloudHeightConfig(config_file=config_path, scene_dir=scene_path)
    
    # 3. Process
    processor = CloudHeightProcessor(config)
    result = processor.process(scene)
    
    # 4. Save
    result.write(output_path)
    logger.info(f"Cloud Height saved to {output_path}")

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
    
    cloud_top_height = height_data.data
    
    # 3. Process
    inverter = CloudPropertyInverter(checkpoint_path=model_path, device='cpu')
    result = inverter.process(scene, height_data)
    
    # 5. Save Results
    logger.info("Properties calculated. Saving results.")
    result.write(output_path)

@app.command()
def workflow(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    model_path: str = typer.Option(..., help="Refl2Prop Model Path"),
    output_dir: str = typer.Option("output", help="Directory for outputs"),
):
    """
    End-to-end workflow: Height -> Properties
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    
    height_out = str(out / "cloud_height.tif")
    props_out = str(out / "properties.nc")
    
    # 1. Height
    logger.info("Running Step 1: Cloud Height")
    # We call the function logic, but we need to match signature or invoke command
    # Calling the decorated command function directly in typer usually works if arguments match
    # Or better, just extract the logic. For now calling the function (it acts as the function if used pythonically)
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