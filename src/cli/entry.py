from __future__ import annotations

import typer
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Union
import logging

# Lightweight config imports only — processors and data classes are
# imported lazily inside the functions that need them so that
# --help / autocomplete stay fast.
from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
from clouds_decoded.modules.cloud_height_emulator.processor import CloudHeightEmulatorProcessor
from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter, ShadingPropertyInverter
from clouds_decoded.modules.refl2prop.config import Refl2PropConfig, ShadingRefl2PropConfig
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig, PostProcessParams
from clouds_decoded.modules.refocus.config import RefocusConfig
from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CLI")

app = typer.Typer(help="Clouds Decoded Command Line Interface")


# --- Private Helpers ---

def _load_scene(scene_path: str, crop_window: Optional[str] = None) -> Sentinel2Scene:
    """Load a Sentinel2Scene, optionally applying a spatial crop."""
    from clouds_decoded.data import Sentinel2Scene

    logger.info(f"Loading Scene: {scene_path}")
    scene = Sentinel2Scene()
    if crop_window:
        logger.info(f"Applying crop window: {crop_window}")
        try:
            col_off, row_off, width, height = map(int, crop_window.split(","))
            scene.read(scene_path, crop_window=(col_off, row_off, width, height))
        except Exception as e:
            logger.error(f"Failed to parse crop window: {e}. Format: 'col_off,row_off,width,height'. Processing full scene.")
            scene.read(scene_path)
    else:
        scene.read(scene_path)
    return scene


def _resolve_height_input(height_input: Union[str, Path, CloudHeightGridData]) -> CloudHeightGridData:
    """Load height data from file path, or pass through if already in memory."""
    from clouds_decoded.data import CloudHeightGridData

    if isinstance(height_input, (str, Path)):
        height_data = CloudHeightGridData.from_file(str(height_input))
        if height_data.data is None:
            raise ValueError(f"Could not read height data from {height_input}")
        return height_data
    return height_input


def _load_pipeline_config(config_path: str) -> Dict:
    """Load pipeline YAML config with optional sections: cloud_mask, cloud_height, cloud_properties."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data


# --- Core Processing Logic (Config-Driven) ---

def run_cloud_mask(
    scene: Sentinel2Scene,
    config: CloudMaskConfig,
    output_path: Optional[str] = None,
    pp_params: Optional[PostProcessParams] = None,
) -> CloudMaskData:
    """Run cloud masking with explicit config."""
    from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor, ThresholdCloudMaskProcessor

    logger.info(f"Processing Cloud Mask (Method: {config.method})...")

    if config.method == "threshold":
        processor = ThresholdCloudMaskProcessor(config)
        result = processor.process(scene)
    else:
        processor = CloudMaskProcessor(config)
        result = processor.process(scene)

    if pp_params:
        # postprocess() lives on CloudMaskProcessor; reuse if already that type,
        # otherwise create one with the user's config.
        postprocessor = processor if isinstance(processor, CloudMaskProcessor) else CloudMaskProcessor(config)
        result = postprocessor.postprocess(result, pp_params)

    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Mask saved to {output_path}")

    return result


def run_cloud_height(
    scene: Sentinel2Scene,
    config: Union[CloudHeightConfig, CloudHeightEmulatorConfig],
    output_path: Optional[str] = None,
    cloud_mask: Optional[Union[CloudMaskData, str, Path]] = None,
    use_emulator: bool = False,
) -> CloudHeightGridData:
    """Run cloud height retrieval with explicit config."""
    logger.info(f"Processing Cloud Height (Emulator: {use_emulator})...")

    if use_emulator:
        # Emulator does not use cloud mask argument in process()
        if not isinstance(config, CloudHeightEmulatorConfig):
             logger.warning("Config is not CloudHeightEmulatorConfig but use_emulator=True. Instantiating default emulator config.")
             config = CloudHeightEmulatorConfig()
        
        processor = CloudHeightEmulatorProcessor(config)
        result = processor.process(scene)
    else:
        processor = CloudHeightProcessor(config)
        result = processor.process(scene, cloud_mask=cloud_mask)

    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Height saved to {output_path}")

    return result


def run_cloud_properties(
    scene: Sentinel2Scene,
    height_input: Union[str, Path, CloudHeightGridData],
    config: Refl2PropConfig,
    output_path: Optional[str] = None,
    albedo_data: Optional[AlbedoData] = None,
) -> 'CloudPropertiesData':
    """Run standard cloud property inversion with explicit config."""
    from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter

    logger.info("Processing Cloud Properties...")
    height_data = _resolve_height_input(height_input)

    processor = CloudPropertyInverter(config)
    result = processor.process(scene, height_data, albedo_data=albedo_data)

    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Properties saved to {output_path}")

    return result


def run_shading_cloud_properties(
    scene: Sentinel2Scene,
    height_input: Union[str, Path, CloudHeightGridData],
    config: ShadingRefl2PropConfig,
    output_path: Optional[str] = None,
    albedo_data: Optional[AlbedoData] = None,
) -> 'CloudPropertiesData':
    """Run shading-aware cloud property inversion with explicit config."""
    from clouds_decoded.modules.refl2prop.processor import ShadingPropertyInverter

    logger.info("Processing Shading-Aware Cloud Properties...")
    height_data = _resolve_height_input(height_input)

    processor = ShadingPropertyInverter(config)
    result = processor.process(scene, height_data, albedo_data=albedo_data)

    if output_path:
        result.write(output_path)
        logger.info(f"Shading Cloud Properties saved to {output_path}")

    return result


def run_albedo(
    scene: Sentinel2Scene,
    config: AlbedoEstimatorConfig,
    cloud_mask: Optional[CloudMaskData] = None,
    output_path: Optional[str] = None,
) -> 'AlbedoData':
    """Run albedo estimation with explicit config."""
    from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator

    logger.info(f"Processing Albedo (method={config.method}, order={config.polynomial_order})...")

    processor = AlbedoEstimator(config)
    result = processor.process(scene, cloud_mask=cloud_mask)

    if output_path:
        result.write(output_path)
        logger.info(f"Albedo saved to {output_path}")

    return result


def run_refocus(
    scene: Sentinel2Scene,
    height_input: Union[str, Path, CloudHeightGridData],
    config: RefocusConfig,
    output_dir: Optional[str] = None,
) -> Sentinel2Scene:
    """Run parallax correction (refocusing) with explicit config.

    Args:
        output_dir: If provided, save refocused bands as individual GeoTIFFs
            in this directory.
    """
    from clouds_decoded.modules.refocus.processor import RefocusProcessor

    logger.info("Processing Refocus (parallax correction)...")
    height_data = _resolve_height_input(height_input)

    processor = RefocusProcessor(config)
    result = processor.process(scene, height_data)

    if output_dir:
        _save_refocused_bands(result, config, output_dir)

    logger.info("Refocus complete")
    return result


def _save_refocused_bands(
    refocused_scene: Sentinel2Scene,
    config: RefocusConfig,
    output_dir: str,
):
    """Save refocused bands as individual GeoTIFFs."""
    import numpy as np
    import rasterio as rio
    from clouds_decoded.constants import BAND_RESOLUTIONS

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    for band_name, band_data in refocused_scene.bands.items():
        band_res = config.output_resolution or BAND_RESOLUTIONS.get(band_name, 10)
        band_transform = rio.transform.Affine(
            band_res, 0, refocused_scene.transform.c,
            0, -band_res, refocused_scene.transform.f,
        ) if refocused_scene.transform else None

        band_path = out / f"{band_name}_refocused.tif"
        is_float = np.issubdtype(band_data.dtype, np.floating)
        profile = {
            'driver': 'GTiff',
            'height': band_data.shape[0],
            'width': band_data.shape[1],
            'count': 1,
            'dtype': band_data.dtype,
            'crs': refocused_scene.crs,
            'transform': band_transform,
            'compress': 'deflate',
            'predictor': 3 if is_float else 2,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
        }
        with rio.open(band_path, 'w', **profile) as dst:
            dst.write(band_data[np.newaxis, ...] if band_data.ndim == 2 else band_data)

    logger.info(f"Refocused bands saved to {output_dir}/")


# --- CLI Commands ---

@app.command()
def cloud_height(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_path: str = typer.Option("height_output.tif", help="Output path"),
    config_path: Optional[str] = typer.Option(None, help="Config YAML (overrides flags)"),
    mask_path: Optional[str] = typer.Option(None, help="Path to cloud mask file"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
    use_emulator: bool = typer.Option(False, help="Use Deep Learning Emulator for height retrieval"),
):
    """Calculate Cloud Height from Sentinel-2 data."""
    scene = _load_scene(scene_path, crop_window)

    if config_path:
        if use_emulator:
            config = CloudHeightEmulatorConfig.from_yaml(config_path)
        else:
            config = CloudHeightConfig.from_yaml(config_path)
    else:
        if use_emulator:
            config = CloudHeightEmulatorConfig()
        else:
            config = CloudHeightConfig()

    run_cloud_height(scene, config, output_path, cloud_mask=mask_path, use_emulator=use_emulator)


@app.command()
def cloud_mask(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_path: str = typer.Option("mask_output.tif", help="Output path"),
    config_path: Optional[str] = typer.Option(None, help="Config YAML (overrides flags)"),
    method: str = typer.Option("senseiv2", help="Method: 'senseiv2' or 'threshold'"),
    threshold_band: str = typer.Option("B08", help="Band for thresholding"),
    threshold_value: float = typer.Option(1600, help="Reflectance threshold (DN)"),
    resolution: int = typer.Option(10, help="Model resolution in meters (SEnSeIv2)"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
):
    """
    Calculate Cloud Mask from Sentinel-2 data.
    Supports SEnSeIv2 (Deep Learning) and simple thresholding.
    """
    scene = _load_scene(scene_path, crop_window)

    if config_path:
        mask_config = CloudMaskConfig.from_yaml(config_path)
    else:
        mask_config = CloudMaskConfig(
            method=method,
            threshold_band=threshold_band,
            threshold_value=threshold_value,
            resolution=resolution,
        )

    run_cloud_mask(scene, mask_config, output_path)


@app.command()
def cloud_properties(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    height_path: str = typer.Option(..., help="Path to Cloud Height raster (.tif)"),
    output_path: str = typer.Option("properties_output.tif", help="Output path"),
    config_path: Optional[str] = typer.Option(None, help="Refl2Prop config YAML (sets model_path, bands, etc.)"),
    properties_method: str = typer.Option("standard", help="Properties method: 'standard' or 'shading'"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
):
    """Run Cloud Property Inversion (Refl2Prop). Requires pre-calculated cloud heights."""
    scene = _load_scene(scene_path, crop_window)

    if properties_method == "shading":
        config = ShadingRefl2PropConfig.from_yaml(config_path)
        run_shading_cloud_properties(scene, height_path, config, output_path)
    else:
        config = Refl2PropConfig.from_yaml(config_path)
        run_cloud_properties(scene, height_path, config, output_path)


@app.command()
def refocus(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    height_path: str = typer.Option(..., help="Path to Cloud Height raster (.tif)"),
    output_dir: str = typer.Option("refocused", help="Output directory for refocused bands"),
    output_resolution: Optional[int] = typer.Option(None, help="Common output resolution (meters). None=native."),
    interpolation_order: int = typer.Option(1, help="Interpolation order (0=nearest, 1=bilinear, 3=cubic)"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
):
    """
    Refocus (parallax-correct) Sentinel-2 bands using cloud height data.

    Removes height-dependent misalignment between bands caused by push-broom
    acquisition. The reference band (B02) is unchanged; all other bands are
    warped to align with it.
    """
    scene = _load_scene(scene_path, crop_window)

    config = RefocusConfig(
        output_resolution=output_resolution,
        interpolation_order=interpolation_order,
    )

    run_refocus(scene, height_path, config, output_dir=output_dir)


@app.command()
def albedo(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_path: str = typer.Option("albedo_output.tif", help="Output path"),
    mask_path: Optional[str] = typer.Option(None, help="Path to cloud mask file (.tif). Enables polynomial fitting."),
    config_path: Optional[str] = typer.Option(None, help="Config YAML (overrides flags)"),
    method: str = typer.Option("polynomial", help="Method: 'polynomial' (needs mask) or 'percentile'"),
    polynomial_order: int = typer.Option(2, help="Polynomial order (1=linear, 2=quadratic, 3=cubic)"),
    output_resolution: int = typer.Option(300, help="Output resolution in meters/pixel"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
):
    """
    Estimate surface albedo from Sentinel-2 data.

    Fits a 2D polynomial to clear-sky pixels (when a cloud mask is provided),
    or falls back to a simple percentile method.
    """
    scene = _load_scene(scene_path, crop_window)

    if config_path:
        albedo_config = AlbedoEstimatorConfig.from_yaml(config_path)
    else:
        albedo_config = AlbedoEstimatorConfig(
            method=method,
            polynomial_order=polynomial_order,
            output_resolution=output_resolution,
        )

    cloud_mask = None
    if mask_path:
        from clouds_decoded.data import CloudMaskData
        cloud_mask = CloudMaskData.from_file(mask_path)
        logger.info(f"Loaded cloud mask from {mask_path}")

    run_albedo(scene, albedo_config, cloud_mask=cloud_mask, output_path=output_path)




@app.command()
def full_workflow(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_dir: str = typer.Option("output", help="Directory for outputs"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
    mask_method: str = typer.Option("senseiv2", help="Mask method: 'senseiv2' or 'threshold'"),
    use_emulator: bool = typer.Option(False, help="Use Deep Learning Emulator for height retrieval"),
    config: Optional[str] = typer.Option(None, help="Pipeline config YAML (overrides defaults)"),
):
    """
    Full pipeline with refocusing and albedo estimation.

    Cloud Mask -> Cloud Height + Albedo -> Refocus -> Cloud Properties.

    Uses the cloud mask to fit spatially-varying albedo, refocuses bands
    to correct parallax, then runs cloud property inversion on the
    refocused scene with the pre-computed albedo.

    All module settings (model_path, method, bands, etc.) come from the
    pipeline config YAML or their defaults. Use --config to customize.

    Usage:
        clouds-decoded full-workflow scene.SAFE
        clouds-decoded full-workflow scene.SAFE --config pipeline.yaml
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    # Load pipeline config sections (if provided)
    mask_cfg_dict: Dict = {}
    height_cfg_dict: Dict = {}
    props_cfg_dict: Dict = {}
    albedo_cfg_dict: Dict = {}
    refocus_cfg_dict: Dict = {}

    if config:
        pipeline = _load_pipeline_config(config)
        mask_cfg_dict = pipeline.get("cloud_mask", {})
        height_cfg_dict = pipeline.get("cloud_height", {})
        props_cfg_dict = pipeline.get("cloud_properties", {})
        albedo_cfg_dict = pipeline.get("albedo", {})
        refocus_cfg_dict = pipeline.get("refocus", {})
        mask_method = mask_cfg_dict.pop("method", mask_method)

    # Build configs
    cloud_mask_config = CloudMaskConfig(method=mask_method, **mask_cfg_dict)
    if use_emulator:
        cloud_height_config = CloudHeightEmulatorConfig(**height_cfg_dict)
    else:
        cloud_height_config = CloudHeightConfig(**height_cfg_dict)
    albedo_config = AlbedoEstimatorConfig(method="polynomial", **albedo_cfg_dict)
    refocus_config = RefocusConfig(**refocus_cfg_dict)

    # Load scene once
    scene = _load_scene(scene_path, crop_window)

    # Step 1: Cloud Mask
    logger.info("Step 1/5: Cloud Mask")
    raw_mask = run_cloud_mask(
        scene, cloud_mask_config,
        output_path=str(out / "cloud_mask.tif"),
    )
    # Postprocess to binary for downstream consumers (height, albedo, etc.)
    from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor
    postprocessor = CloudMaskProcessor()
    mask_result = postprocessor.postprocess(raw_mask, PostProcessParams())

    # Step 2: Cloud Height
    logger.info("Step 2/5: Cloud Height")
    height_result = run_cloud_height(
        scene, cloud_height_config,
        output_path=str(out / "cloud_height.tif"),
        cloud_mask=mask_result,
        use_emulator=use_emulator,
    )

    # Step 3: Albedo (uses cloud mask for clear-sky polynomial fit)
    logger.info("Step 3/5: Albedo Estimation")
    albedo_result = run_albedo(
        scene, albedo_config,
        cloud_mask=mask_result,
        output_path=str(out / "albedo.tif"),
    )

    # Step 4: Refocus (parallax correction using cloud height)
    logger.info("Step 4/5: Refocus")
    refocus_out = str(out / "refocused") if refocus_config.save_refocused else None
    refocused_scene = run_refocus(scene, height_result, refocus_config, output_dir=refocus_out)

    # Step 5: Cloud Properties (on refocused scene, with pre-computed albedo)
    logger.info("Step 5/5: Cloud Properties")
    props_method = props_cfg_dict.get("method", "standard")
    if props_method == "shading":
        props_config = ShadingRefl2PropConfig(**props_cfg_dict)
        run_shading_cloud_properties(
            refocused_scene, height_result, props_config,
            output_path=str(out / "properties_shading.tif"),
            albedo_data=albedo_result,
        )
    else:
        props_config = Refl2PropConfig(**props_cfg_dict)
        run_cloud_properties(
            refocused_scene, height_result, props_config,
            output_path=str(out / "properties.tif"),
            albedo_data=albedo_result,
        )

    logger.info(f"Full pipeline complete. Outputs in {output_dir}/")


# --- Project Commands ---

project_app = typer.Typer(help="Project management: configs, outputs, resumability")
app.add_typer(project_app, name="project")


@project_app.command("init")
def project_init(
    project_dir: str = typer.Argument(..., help="Directory for the new project"),
    name: Optional[str] = typer.Option(None, help="Project name (defaults to directory name)"),
    pipeline: str = typer.Option("full-workflow", help="Pipeline type"),
    use_emulator: bool = typer.Option(False, help="Use emulator for cloud height retrieval"),
    clone: Optional[str] = typer.Option(None, help="Clone configs from an existing project directory"),
):
    """
    Initialize a new project directory with default configs.

    Creates the directory, writes editable config YAMLs for each module,
    and sets up the project metadata. Edit configs in <dir>/configs/ before running.

    Example:
        clouds-decoded project init ./my_analysis
        clouds-decoded project init ./new_analysis --clone ./existing_analysis
    """
    from clouds_decoded.project import Project

    try:
        project = Project.init(
            project_dir,
            name=name,
            pipeline=pipeline,
            clone_from=clone,
            use_emulator=use_emulator,
        )
        logger.info(f"\nEdit configs in {project.configs_dir}/ then run:")
        logger.info(f"  clouds-decoded project run {project_dir}")
    except FileExistsError as e:
        logger.error(str(e))
        raise typer.Exit(1)


@project_app.command("run")
def project_run(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
    scenes: Optional[List[str]] = typer.Argument(None, help="Scene .SAFE path(s) to process (auto-registered)"),
    force: bool = typer.Option(False, help="Reprocess all steps (ignore cache)"),
    unsafe: bool = typer.Option(False, help="Skip file provenance validation on cached outputs"),
    crop_window: Optional[str] = typer.Option(None, help="Spatial crop: 'col_off,row_off,width,height'"),
):
    """
    Run the project pipeline on one or more scenes.

    Pass scene paths directly — they are auto-registered in the project.
    If no scenes are given, all previously registered scenes are processed.
    Skips steps that are already complete (output exists, config unchanged).

    Example:
        clouds-decoded project run ./my_analysis /data/S2A_scene.SAFE
        clouds-decoded project run ./my_analysis /data/scene1.SAFE /data/scene2.SAFE
        clouds-decoded project run ./my_analysis --force
    """
    from clouds_decoded.project import Project

    try:
        project = Project.load(project_dir)
        project.run(scenes=scenes, force=force, unsafe=unsafe, crop_window=crop_window)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)


@project_app.command("status")
def project_status(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
):
    """Show processing status for all scenes in the project."""
    from clouds_decoded.project import Project

    try:
        project = Project.load(project_dir)
        print(project.status())
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)


@project_app.command("add-scene")
def project_add_scene(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
    scene: List[str] = typer.Option(..., help="Path(s) to .SAFE scene(s) to add"),
):
    """Add one or more scenes to an existing project."""
    from clouds_decoded.project import Project

    try:
        project = Project.load(project_dir)
        for s in scene:
            project.add_scene(s)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
