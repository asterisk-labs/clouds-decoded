import typer
from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np
import logging
import rasterio

# Standardized Imports
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData

# Direct imports to avoid stale package issues
from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter
# No Refl2PropConfig import needed here as we use it inside function or assume it's available
from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor, ThresholdCloudMaskProcessor
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig, PostProcessParams


# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLI")

app = typer.Typer(help="Clouds Decoded Command Line Interface")

# --- Core Processing Logic (Decoupled from CLI) ---

def run_cloud_height(
    scene: Sentinel2Scene,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> CloudHeightGridData:
    logger.info(f"Processing Cloud Height...")
    
    # Configure
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    
    if hasattr(CloudHeightConfig, 'from_yaml'):
        config = CloudHeightConfig.from_yaml(config_path)
    elif hasattr(CloudHeightConfig, 'load_yaml'):
         if config_path:
             config = CloudHeightConfig.load_yaml(config_path)
         else:
             config = CloudHeightConfig()
    else:
         config = CloudHeightConfig()

    processor = CloudHeightProcessor(config)
    result = processor.process(scene)
    
    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Height saved to {output_path}")
        
    return result

def run_cloud_mask(
    scene: Sentinel2Scene,
    output_path: Optional[str] = None,
    method: str = "senseiv2",
    threshold_band: str = "B08",
    threshold_value: float = 1600,
    resolution: int = 10,
    postprocess: bool = True,
    buffer_size: int = 0,
    output_resolution: Optional[int] = None,
    classes: str = "1,2,3"
):
    logger.info(f"Processing Cloud Mask (Method: {method})...")
    
    if method == "threshold":
        config = CloudMaskConfig(
            method="threshold", 
            threshold_band=threshold_band, 
            threshold_value=threshold_value
        )
        processor = ThresholdCloudMaskProcessor(config)
        result = processor.process(scene)
    else:
        # SEnSeIv2
        config = CloudMaskConfig(
            method="senseiv2", 
            resolution=resolution
        )
        processor = CloudMaskProcessor(config)
        
        # Raw inference
        result = processor.process(scene)
        
        # Post-processing
        if postprocess:
            classes_list = [int(x) for x in classes.split(",")]
            pp_params = PostProcessParams(
                output_resolution=output_resolution,
                buffer_size=buffer_size,
                classes_to_mask=classes_list,
                binary_mask=True
            )
            result = processor.postprocess(result, pp_params)
    
    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Mask saved to {output_path}")
        
    return result

def run_cloud_properties(
    scene: Sentinel2Scene,
    height_input: Union[str, CloudHeightGridData],
    model_path: str,
    output_path: Optional[str] = None,
    return_uncertainty: bool = False
):
    logger.info(f"Processing Cloud Properties...")
    
    # Load Heights if path provided
    if isinstance(height_input, (str, Path)):
        height_data = CloudHeightGridData.from_file(str(height_input))
        if height_data.data is None:
            raise ValueError("Could not read height data")
    else:
        height_data = height_input
        
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig
    config = Refl2PropConfig(model_path=model_path, return_uncertainty=return_uncertainty)
        
    processor = CloudPropertyInverter(config)
    result = processor.process(scene, height_data)
    
    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Properties saved to {output_path}")
        
    return result


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
    
    # 2. Run
    run_cloud_height(scene, config_path, output_path)

@app.command()
def cloud_mask(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_path: str = typer.Option("mask_output.tif", help="Output path (e.g. .tif)"),
    method: str = typer.Option("senseiv2", help="Method: 'senseiv2' or 'threshold'"),
    threshold_band: str = typer.Option("B08", help="Band used for thresholding"),
    threshold_value: float = typer.Option(1600, help="Reflectance threshold (Raw DN)"),
    resolution: int = typer.Option(10, help="Model input resolution in meters (for SEnSeIv2)"),
    # Post-processing options
    postprocess: bool = typer.Option(True, help="Apply post-processing (resizing, buffering)"),
    buffer_size: int = typer.Option(0, help="Buffer size in meters"),
    output_resolution: Optional[int] = typer.Option(None, help="Output resolution in meters. If None, uses model resolution."),
    classes: str = typer.Option("1,2,3", help="Comma-separated class indices to consider as cloud (e.g. '1,2,3')"),
):
    """
    Calculate Cloud Mask from Sentinel-2 data.
    Supports SEnSeIv2 (Deep Learning) and simple thresholding.
    """
    logger.info(f"Starting Cloud Mask processing for: {scene_path}")
    
    # 1. Load Data
    scene = Sentinel2Scene()
    scene.read(scene_path)

    # 2. Run
    run_cloud_mask(
        scene, 
        output_path, 
        method, 
        threshold_band, 
        threshold_value, 
        resolution, 
        postprocess, 
        buffer_size, 
        output_resolution, 
        classes
    )

@app.command()
def cloud_properties(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    height_path: str = typer.Option(..., help="Path to Cloud Height raster (.tif/.nc) matching the scene"),
    model_path: str = typer.Option(..., help="Path to trained .pth model"),
    output_path: str = typer.Option("properties_output.nc", help="Output NetCDF path"),
    return_uncertainty: bool = typer.Option(False, help="Calculate and include OOD uncertainty"),
):
    """
    Run Cloud Property Inversion (Refl2Prop). Requires pre-calculated cloud heights.
    """
    logger.info(f"Starting Cloud Properties Inversion for: {scene_path}")
    
    # 1. Load Scene
    scene = Sentinel2Scene()
    scene.read(scene_path)
    
    # 2. Run
    run_cloud_properties(
        scene,
        height_path,
        model_path,
        output_path,
        return_uncertainty
    )

@app.command()
def workflow(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    model_path: str = typer.Option(..., help="Refl2Prop Model Path"),
    output_dir: str = typer.Option("output", help="Directory for outputs"),
    return_uncertainty: bool = typer.Option(False, help="Calculate and output uncertainty maps"),
):
    """
    End-to-end workflow: Cloud Mask -> Cloud Height -> Cloud Properties
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    
    mask_out = str(out / "cloud_mask.tif")
    height_out = str(out / "cloud_height.tif")
    props_out = str(out / "properties.tif")
    
    # 0. Load Scene (ONCE)
    logger.info(f"Loading Scene: {scene_path}")
    scene = Sentinel2Scene()
    scene.read(scene_path)
    
    # 1. Mask
    logger.info("Running Step 1: Cloud Mask")
    run_cloud_mask(
        scene, 
        output_path=mask_out,
        method="senseiv2",
        resolution=10,
    )

    # 2. Height
    logger.info("Running Step 2: Cloud Height")
    # Store result in memory to pass to properties
    height_result = run_cloud_height(scene, config_path=None, output_path=height_out)

    # 3. Properties
    logger.info("Running Step 3: Cloud Properties")
    run_cloud_properties(
        scene, 
        height_input=height_result,  # Pass in-memory result
        model_path=model_path, 
        output_path=props_out,
        return_uncertainty=return_uncertainty
    )

if __name__ == "__main__":
    app()