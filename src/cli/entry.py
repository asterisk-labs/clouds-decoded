from __future__ import annotations

import typer
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, List, Union
import logging

# All config/processor/data imports are lazy (inside functions) so that
# --help and autocomplete stay fast. TYPE_CHECKING-only imports allow
# type checkers and IDEs to resolve annotations without runtime cost.
if TYPE_CHECKING:
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig, ShadingRefl2PropConfig
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
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
        except (ValueError, TypeError) as e:
            raise typer.BadParameter(
                f"Invalid crop window '{crop_window}': {e}. "
                f"Expected format: 'col_off,row_off,width,height' (4 integers)."
            )
        scene.read(scene_path, crop_window=(col_off, row_off, width, height))
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

    if output_path:
        result.write(output_path)
        logger.info(f"Cloud Mask saved to {output_path}")

    return result


def run_cloud_height(
    scene: Sentinel2Scene,
    config: Union[CloudHeightConfig, CloudHeightEmulatorConfig],
    output_path: Optional[str] = None,
    cloud_mask: Optional[Union[CloudMaskData, str, Path]] = None,
) -> CloudHeightGridData:
    """Run cloud height retrieval with explicit config."""
    from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig as _EmulatorConfig

    use_emulator = getattr(config, "use_emulator", False)
    logger.info(f"Processing Cloud Height (Emulator: {use_emulator})...")

    if use_emulator:
        from clouds_decoded.modules.cloud_height_emulator.processor import CloudHeightEmulatorProcessor
        processor = CloudHeightEmulatorProcessor(config)
        result = processor.process(scene, cloud_mask=cloud_mask)
    else:
        from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
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

    logger.info(f"Processing Albedo (method={config.method})...")

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
    use_emulator: bool = typer.Option(True, help="Use Deep Learning Emulator for height retrieval"),
):
    """Calculate Cloud Height from Sentinel-2 data."""
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
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

    run_cloud_height(scene, config, output_path, cloud_mask=mask_path)


@app.command()
def cloud_mask(
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),
    output_path: str = typer.Option("mask_output.tif", help="Output path"),
    config_path: Optional[str] = typer.Option(None, help="Config YAML (overrides flags)"),
    method: str = typer.Option("senseiv2", help="Method: 'senseiv2' or 'threshold'"),
    threshold_band: str = typer.Option("B08", help="Band for thresholding"),
    threshold_value: float = typer.Option(0.06, help="Reflectance threshold (0-1)"),
    resolution: int = typer.Option(20, help="Model resolution in meters (default 20 for deep learning)"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
):
    """
    Calculate Cloud Mask from Sentinel-2 data.
    Supports deep learning (SegFormer-B2) and simple thresholding.
    """
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
    scene = _load_scene(scene_path, crop_window)

    if config_path:
        mask_config = CloudMaskConfig.from_yaml(config_path)
    else:
        mask_config = CloudMaskConfig(
            method=method,
            threshold_band=threshold_band,
            threshold_value=threshold_value,
            working_resolution=resolution,
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
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig, ShadingRefl2PropConfig
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
    from clouds_decoded.modules.refocus.config import RefocusConfig
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
    mask_path: Optional[str] = typer.Option(None, help="Path to cloud mask file (.tif). Enables IDW fitting."),
    config_path: Optional[str] = typer.Option(None, help="Config YAML (overrides flags)"),
    method: str = typer.Option("idw", help="Method: 'idw' (inverse-distance weighting, default) or 'datadriven' (trained MLP). idw requires --mask-path."),
    fallback: str = typer.Option("datadriven", help="Fallback when insufficient clear pixels: 'datadriven' or 'constant'"),
    model_path: Optional[str] = typer.Option(None, help="Path to trained data-driven albedo model checkpoint"),
    output_resolution: int = typer.Option(300, help="Output resolution in meters/pixel"),
    crop_window: Optional[str] = typer.Option(None, help="Crop: 'col_off,row_off,width,height'"),
):
    """
    Estimate surface albedo from Sentinel-2 data.

    Uses inverse-distance weighted interpolation of clear-sky pixels (when a
    cloud mask is provided), or a trained MLP for data-driven estimation.
    """
    from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
    scene = _load_scene(scene_path, crop_window)

    if config_path:
        albedo_config = AlbedoEstimatorConfig.from_yaml(config_path)
    else:
        kwargs = dict(
            method=method,
            fallback=fallback,
            output_resolution=output_resolution,
        )
        if model_path is not None:
            kwargs["model_path"] = model_path
        albedo_config = AlbedoEstimatorConfig(**kwargs)

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
    use_emulator: bool = typer.Option(True, help="Use Deep Learning Emulator for height retrieval"),
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
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig, ShadingRefl2PropConfig
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
    from clouds_decoded.modules.refocus.config import RefocusConfig
    from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
    cloud_mask_config = CloudMaskConfig(method=mask_method, **mask_cfg_dict)
    if use_emulator:
        cloud_height_config = CloudHeightEmulatorConfig(**height_cfg_dict)
    else:
        cloud_height_config = CloudHeightConfig(**height_cfg_dict)
    albedo_method = albedo_cfg_dict.pop("method", "idw")
    albedo_config = AlbedoEstimatorConfig(method=albedo_method, **albedo_cfg_dict)
    refocus_config = RefocusConfig(**refocus_cfg_dict)

    # Load scene once
    scene = _load_scene(scene_path, crop_window)

    # Step 1: Cloud Mask
    logger.info("Step 1: Cloud Mask")
    raw_mask = run_cloud_mask(
        scene, cloud_mask_config,
        output_path=str(out / "cloud_mask.tif"),
    )
    # Postprocess to binary for downstream consumers (height, albedo, etc.)
    mask_result = raw_mask.to_binary(positive_classes=[1, 2, 3], threshold=0.5)

    # Step 2: Cloud Height
    logger.info("Step 2: Cloud Height")
    height_result = run_cloud_height(
        scene, cloud_height_config,
        output_path=str(out / "cloud_height.tif"),
        cloud_mask=mask_result,
    )

    # Step 3: Albedo (uses cloud mask for clear-sky IDW interpolation)
    logger.info("Step 3: Albedo Estimation")
    albedo_result = run_albedo(
        scene, albedo_config,
        cloud_mask=mask_result,
        output_path=str(out / "albedo.tif"),
    )

    # Step 4: Refocus (parallax correction using cloud height)
    logger.info("Step 4: Refocus")
    refocus_out = str(out / "refocused") if refocus_config.save_refocused else None
    refocused_scene = run_refocus(scene, height_result, refocus_config, output_dir=refocus_out)

    # Step 5: Cloud Properties (on refocused scene, with pre-computed albedo)
    logger.info("Step 5: Cloud Properties")
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
    pipeline: str = typer.Option(
        "full-workflow",
        help="Recipe name. Available: full-workflow, cloud-height-comparison. "
             "The chosen recipe is embedded in project.yaml and can be edited directly.",
    ),
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
    parallel: bool = typer.Option(
        False, "--parallel",
        help=(
            "Enable parallel pipeline: scenes are read and processed concurrently across "
            "stages. Use --workers / --queue-depth to tune. Default is serial (one scene "
            "at a time)."
        ),
    ),
    workers: Optional[List[str]] = typer.Option(
        None, "--workers", "-j",
        help=(
            "Per-stage worker count as STAGE=N (repeatable). Requires --parallel. "
            "Stage names: reader, cloud_mask, cloud_height_emulator, cloud_height, albedo, refocus, cloud_properties. "
            "A bare integer sets all stages to that count. "
            "Example: -j cloud_height=2 -j albedo=2"
        ),
    ),
    queue_depth: int = typer.Option(
        2, "--queue-depth",
        help="Max scenes buffered between adjacent pipeline stages (requires --parallel).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help=(
            "Print INFO-level logs to the terminal. By default only warnings and errors "
            "are shown on screen; full logs always go to <project>/logs/<scene>/pipeline.log."
        ),
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress",
        help="Disable the live pipeline progress display (only relevant with --parallel).",
    ),
    no_stats: bool = typer.Option(
        False, "--no-stats",
        help="Skip automatic stats computation after the pipeline finishes.",
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite",
        help=(
            "Reset 'done' scenes whose configs have changed since they were processed "
            "and re-run them with the current config."
        ),
    ),
    ignore_integrity: bool = typer.Option(
        False, "--ignore-integrity",
        help=(
            "Skip config integrity check. Stale 'done' scenes will not be re-run "
            "(not recommended)."
        ),
    ),
):
    """
    Run the project pipeline on one or more scenes.

    Pass scene paths directly — they are staged and then processed.
    If no scenes are given, all scenes with status ``staged`` or ``failed``
    are processed (use --force to reprocess ``done`` scenes too).
    Skips steps that are already complete (output exists, config unchanged).

    By default, scenes are processed serially (one at a time). Use --parallel
    to enable concurrent scene/stage processing:

    \b
        clouds-decoded project run ./analysis scene.SAFE
        clouds-decoded project run ./analysis --force --verbose
        clouds-decoded project run ./analysis --parallel -j cloud_height=2
        clouds-decoded project run ./analysis --parallel -j 4
    """
    from clouds_decoded.project import Project, _ALL_STAGE_NAMES

    # Parse --workers values: accept bare int (all stages) or STAGE=N pairs
    parallelism: Optional[Dict[str, int]] = None
    if workers:
        parallelism = {}
        for w in workers:
            if w.isdigit():
                # Bare integer — apply to all stages
                n = int(w)
                parallelism = {stage: n for stage in _ALL_STAGE_NAMES}
            elif "=" in w:
                stage, _, val = w.partition("=")
                stage = stage.strip()
                if stage not in _ALL_STAGE_NAMES:
                    logger.error(
                        f"Unknown stage '{stage}'. Valid stages: {', '.join(_ALL_STAGE_NAMES)}"
                    )
                    raise typer.Exit(1)
                try:
                    parallelism[stage] = int(val)
                except ValueError:
                    logger.error(f"Invalid worker count '{val}' for stage '{stage}'.")
                    raise typer.Exit(1)
            else:
                logger.error(
                    f"Invalid --workers value '{w}'. Use a bare integer or STAGE=N."
                )
                raise typer.Exit(1)

    try:
        project = Project.load(project_dir)
        project.run(
            scenes=scenes,
            force=force,
            unsafe=unsafe,
            crop_window=crop_window,
            parallel=parallel,
            parallelism=parallelism,
            queue_depth=queue_depth,
            verbose=verbose,
            progress=not no_progress,
            run_stats=not no_stats,
            force_overwrite=force_overwrite,
            ignore_integrity=ignore_integrity,
        )
    except (FileNotFoundError, RuntimeError) as e:
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


@project_app.command("stage")
def project_stage(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
    scenes: List[str] = typer.Argument(..., help="Path(s) to .SAFE scene(s) or directories to scan for .SAFE dirs"),
):
    """Register scenes in the project without processing them.

    Each argument can be a path to a .SAFE directory or a parent directory
    containing multiple .SAFE directories (scanned with *.SAFE glob).

    \b
        clouds-decoded project stage ./analysis ./scene.SAFE
        clouds-decoded project stage ./analysis /data/sentinel2/
    """
    from clouds_decoded.project import Project

    try:
        project = Project.load(project_dir)
        resolved: List[str] = []
        for s in scenes:
            p = Path(s)
            if p.is_dir() and not s.endswith(".SAFE"):
                found = sorted(p.glob("*.SAFE"))
                if not found:
                    logger.warning(f"No .SAFE directories found in {p}")
                resolved.extend(str(f.resolve()) for f in found)
            else:
                resolved.append(str(p.resolve()))
        before = len(project.db.get_all())
        project.stage(*resolved)
        after = len(project.db.get_all())
        new_count = after - before
        logger.info(f"Staged {new_count} new scene(s) ({after - new_count} already registered).")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)


@project_app.command("list")
def project_list(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
    status: Optional[str] = typer.Option(None, help="Filter by status: staged, started, done, failed"),
    crop_window: Optional[str] = typer.Option(None, "--crop-window", help="Filter by crop window"),
):
    """List all registered runs and their status.

    \b
        clouds-decoded project list ./analysis
        clouds-decoded project list ./analysis --status staged
        clouds-decoded project list ./analysis --crop-window 0,0,512,512
    """
    from clouds_decoded.project import Project

    try:
        project = Project.load(project_dir)
        rows = project.db.get_all()
        if status:
            rows = [r for r in rows if r["status"] == status]
        if crop_window is not None:
            rows = [r for r in rows if r.get("crop_window") == crop_window]
        if not rows:
            print("No runs found.")
            return
        header = (f"{'run_id':<18} {'scene_id':<50} {'crop_window':<16} "
                  f"{'status':<10} {'staged_at':<22} {'completed_at'}")
        print(header)
        print("-" * len(header))
        for r in rows:
            completed = r.get("completed_at") or "-"
            cw = r.get("crop_window") or "(full)"
            print(f"{r['run_id']:<18} {r['scene_id']:<50} {cw:<16} "
                  f"{r['status']:<10} {r['staged_at']:<22} {completed}")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)


@project_app.command("stats")
def project_stats(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
    force: bool = typer.Option(False, help="Re-compute stats even if already stored"),
    method: Optional[List[str]] = typer.Option(None, "--method",
                                                help="Stats method(s) to run, e.g. cloud_height::percentiles"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Compute stats for a single run_id only"),
):
    """Compute and store statistics for all completed runs.

    Stats are written to per-step tables in project.db. Already-computed stats
    are skipped unless --force is passed.

    \b
        clouds-decoded project stats ./analysis
        clouds-decoded project stats ./analysis --force
        clouds-decoded project stats ./analysis --method cloud_mask::class_fractions
        clouds-decoded project stats ./analysis --run-id abc123def456abcd
    """
    from clouds_decoded.project import Project

    try:
        project = Project.load(project_dir)
        project.run_stats(
            force=force,
            methods=list(method) if method else None,
            run_id_filter=run_id,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)


@project_app.command("delete")
def project_delete(
    project_dir: str = typer.Argument(..., help="Path to project directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete a project and all its outputs, logs, and database.

    Shows a summary of directories that will be removed and asks for
    confirmation before deleting anything.

    \b
        clouds-decoded project delete ./my_analysis
        clouds-decoded project delete ./my_analysis --yes
    """
    import shutil
    from pathlib import Path
    from clouds_decoded.project import Project

    project_path = Path(project_dir).resolve()
    config_path = project_path / "project.yaml"
    if not config_path.exists():
        logger.error(f"No project found at {project_path} (missing project.yaml)")
        raise typer.Exit(1)

    try:
        project = Project.load(str(project_path))
    except Exception as e:
        logger.error(f"Failed to load project: {e}")
        raise typer.Exit(1)

    # Collect directories to delete.  The output_dir may live outside the
    # project directory (absolute path in config), so track it separately.
    dirs_to_delete: list[tuple[str, Path]] = []
    output_dir = project.output_dir
    logs_dir = project.logs_dir

    # Count scenes and outputs for the summary
    n_scenes = 0
    n_outputs = 0
    try:
        runs = project.db.get_all()
        n_scenes = len(runs)
    except Exception:
        pass

    if output_dir.exists():
        n_outputs = sum(1 for _ in output_dir.rglob("*.tif"))

    # If output_dir is outside the project tree, list it separately
    output_is_external = False
    try:
        output_dir.relative_to(project_path)
    except ValueError:
        output_is_external = True

    if output_is_external and output_dir.exists():
        dirs_to_delete.append(("Outputs (external)", output_dir))

    dirs_to_delete.append(("Project", project_path))

    # Print summary
    typer.echo(f"\nProject: {project.config.name}")
    typer.echo(f"  Scenes registered:  {n_scenes}")
    typer.echo(f"  Output files (.tif): {n_outputs}")
    typer.echo(f"\nDirectories to delete:")
    for label, path in dirs_to_delete:
        size = _dir_size_human(path)
        typer.echo(f"  {label}: {path}  ({size})")

    typer.echo("")

    if not yes:
        confirm = typer.confirm("Are you sure you want to delete this project?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    # Close the database connection before deleting
    try:
        project.db._db.close()
    except Exception:
        pass

    for label, path in dirs_to_delete:
        shutil.rmtree(path)
        typer.echo(f"  Deleted {path}")

    typer.echo("Done.")


def _dir_size_human(path: "Path") -> str:
    """Return the total size of a directory in a human-readable string."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} PB"


@app.command()
def view(
    project_dir: str = typer.Argument(..., help="Path to a clouds-decoded project directory"),
    host: str = typer.Option("0.0.0.0", help="Hostname for the viser server"),
    port: int = typer.Option(8080, help="Port for the viser server"),
    max_grid_dim: int = typer.Option(800, help="Max grid dimension for display resolution"),
):
    """Launch a 3D point-cloud viewer for a project's cloud height outputs (viser).

    Opens a web-based 3D viewer showing cloud height surfaces for all
    scenes in the project. Supports interactive z-scale, point size,
    and texture controls (cloud mask, true colour, albedo, properties).

    Port-forward for remote servers:
        ssh -L 8080:localhost:8080 user@server
    """
    import warnings
    warnings.warn(
        "The 3D viser viewer is deprecated. Use 'clouds-decoded serve' for the 2D viewer.",
        DeprecationWarning,
        stacklevel=1,
    )
    from clouds_decoded.visualisation._deprecated.viser_viewer import ViserViewer

    try:
        viewer = ViserViewer(
            project_dir=project_dir,
            host=host,
            port=port,
            max_grid_dim=max_grid_dim,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        raise typer.Exit(1)

    viewer.serve()


@app.command()
def serve(
    project_dir: str = typer.Argument(..., help="Path to a clouds-decoded project directory"),
    port: int = typer.Option(5006, help="Port to serve on"),
    show: bool = typer.Option(False, help="Open browser automatically (disable for remote)"),
):
    """Launch an interactive 2D viewer for a project (Panel + Bokeh).

    Provides scene navigation, layer selection, overlay compositing, and
    RGB contrast controls in the browser.

    Ideal for remote servers — port-forward with:
        ssh -L 5006:localhost:5006 user@server

    Then open http://localhost:5006 in your browser.
    """
    from clouds_decoded.visualisation.project_visualiser import ProjectVisualiser

    try:
        pv = ProjectVisualiser(project_dir)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        raise typer.Exit(1)

    if not pv.scene_ids:
        logger.error("No scenes with outputs found. Run 'project run' first.")
        raise typer.Exit(1)

    logger.info(f"Serving {len(pv)} scene(s) on port {port}")
    pv.serve(port=port, show=show)


@app.command()
def overview(
    project_dir: str = typer.Argument(..., help="Path to a clouds-decoded project directory"),
    scene_id: Optional[str] = typer.Option(None, "--scene-id", help="Scene ID (default: first scene)"),
    output: str = typer.Option("overview.png", "--output", "-o", help="Output file path"),
    dpi: int = typer.Option(150, help="Figure resolution"),
):
    """Save a static overview figure for a scene's outputs."""
    from clouds_decoded.visualisation.project_visualiser import ProjectVisualiser

    try:
        pv = ProjectVisualiser(project_dir)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        raise typer.Exit(1)

    if not pv.scene_ids:
        logger.error("No scenes with outputs found. Run 'project run' first.")
        raise typer.Exit(1)

    sid = scene_id or pv.scene_ids[0]
    if sid not in pv.scene_ids:
        logger.error(f"Scene {sid!r} not found. Available: {pv.scene_ids}")
        raise typer.Exit(1)

    fig = pv.overview(sid)
    from clouds_decoded.visualisation.static import save_figure
    save_figure(fig, output, dpi=dpi)
    typer.echo(f"Saved overview to {output}")


# --- Asset Management Commands ---

@app.command()
def setup():
    """Configure the directory used to store large binary assets.

    clouds-decoded needs a directory to store large asset files:
      - Height emulator model weights    ~500 MB
      - Refl2prop model weights          ~50 MB
      - GEBCO bathymetry (optional)      ~2.7 GB

    Writes the chosen path to the user config file so it persists across
    sessions.  Can be overridden at any time with the
    CLOUDS_DECODED_ASSETS_DIR environment variable.
    """
    import platformdirs
    from clouds_decoded.assets import (
        _write_config_assets_dir,
        _read_config_assets_dir,
        KNOWN_ASSETS,
    )

    default = platformdirs.user_data_dir("clouds-decoded", appauthor=False)
    current = _read_config_assets_dir()

    typer.echo("\nclouds-decoded needs a directory to store large asset files:")
    for asset in KNOWN_ASSETS.values():
        typer.echo(f"  - {asset.description:<40} {asset.size_hint}")
    typer.echo("")

    if current:
        typer.echo(f"Current location: {current}")
    typer.echo(f"Default:          {default}")
    typer.echo("")

    use_default = typer.confirm("Use default location?", default=True)

    if use_default:
        chosen = default
    else:
        chosen = typer.prompt("Enter path", default=current or default)

    _write_config_assets_dir(chosen)
    typer.echo(f"\nAssets directory set to: {chosen}")
    typer.echo("\nNext steps:")
    typer.echo("  clouds-decoded download emulator   # height emulator weights")
    typer.echo("  clouds-decoded download refl2prop  # cloud property weights")
    typer.echo("  clouds-decoded download gebco      # GEBCO bathymetry (optional)")


@app.command()
def download(
    asset: str = typer.Argument(
        ...,
        help="Asset key: cloud_mask | emulator | refl2prop | albedo_datadriven | gebco | sample_scene | all",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if file exists"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """Download managed binary assets (model weights, data).

    \b
    clouds-decoded download emulator      # height emulator weights (~500 MB)
    clouds-decoded download refl2prop     # refl2prop weights (~50 MB)
    clouds-decoded download gebco         # GEBCO bathymetry (~2.7 GB)
    clouds-decoded download sample_scene  # sample Sentinel-2 scene (~705 MB)
    clouds-decoded download all           # everything
    """
    from clouds_decoded.assets import KNOWN_ASSETS, download_asset, get_asset

    keys = list(KNOWN_ASSETS.keys()) if asset == "all" else [asset]

    for k in keys:
        if k not in KNOWN_ASSETS:
            typer.echo(
                f"Unknown asset '{k}'. Choose from: {', '.join(KNOWN_ASSETS)} or 'all'",
                err=True,
            )
            raise typer.Exit(1)

        a = KNOWN_ASSETS[k]
        dest = get_asset(a.relative_path)

        if dest.exists() and not force:
            typer.echo(f"[{k}] Already present at {dest}. Use --force to re-download.")
            continue

        if not a.url:
            typer.echo(
                f"[{k}] No download URL configured. Please supply the file manually at:\n"
                f"  {dest}",
                err=True,
            )
            continue

        typer.echo(f"\n[{k}] {a.description}  {a.size_hint}")
        if not yes:
            confirmed = typer.confirm("Download now?", default=True)
            if not confirmed:
                typer.echo(f"Skipped {k}.")
                continue

        try:
            download_asset(k, force=force)
        except Exception as exc:
            typer.echo(f"Error downloading {k}: {exc}", err=True)
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
