"""Load layers from a project scene output directory."""
import json
import logging
from pathlib import Path
from typing import List, Optional

from .layers import (
    Layer,
    RGBConfig,
    layer_from_cloud_mask,
    layer_from_cloud_height,
    layer_from_rgb,
    layers_from_albedo,
    layers_from_cloud_properties,
)

logger = logging.getLogger(__name__)


def load_scene_layers(
    scene_dir: str,
    scene_path: Optional[str] = None,
    rgb_config: Optional[RGBConfig] = None,
) -> List[Layer]:
    """Load all available layers from a project scene output directory.

    Reads the GeoTIFFs produced by the pipeline (cloud_mask.tif,
    cloud_height.tif, albedo.tif, properties.tif) and converts them
    into Layer objects ready for plotting or interactive viewing.

    Optionally loads the original .SAFE scene for RGB composites.

    Args:
        scene_dir: Path to the project scene output directory
            (e.g. ``mynewproject/scenes/S2A_MSIL1C_.../``).
        scene_path: Path to the .SAFE directory for RGB composites.
            If None, tries to read from manifest.json. Skipped if the
            path doesn't exist on disk.
        rgb_config: Gamma/gain/offset for RGB composites.

    Returns:
        List of Layer objects.
    """
    scene_dir = Path(scene_dir)
    layers: List[Layer] = []

    # Try to read manifest for scene_path fallback
    manifest_path = scene_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    # Resolve scene_path from manifest if not provided
    if scene_path is None and manifest:
        scene_path = manifest.get("scene_path")

    # --- RGB composites (loaded first so they appear at the top) ---
    if scene_path and Path(scene_path).exists():
        try:
            from clouds_decoded.data import Sentinel2Scene

            scene = Sentinel2Scene()
            scene.read(scene_path)
            for composite in ("true_color", "ice"):
                try:
                    layers.append(layer_from_rgb(scene, composite, rgb_config=rgb_config))
                except Exception as e:
                    logger.warning(f"Could not build {composite} RGB: {e}")
        except Exception as e:
            logger.warning(f"Could not load scene for RGB composites: {e}")
    elif scene_path:
        logger.info(f"Scene path not found on disk, skipping RGB: {scene_path}")

    # --- Pipeline outputs ---
    _load_tif(scene_dir / "cloud_mask.tif", "cloud_mask", layers)
    _load_tif(scene_dir / "cloud_height.tif", "cloud_height", layers)
    _load_tif(scene_dir / "albedo.tif", "albedo", layers)
    _load_tif(scene_dir / "properties.tif", "cloud_properties", layers)

    if not layers:
        logger.warning(f"No layers loaded from {scene_dir}")

    return layers


def _load_tif(path: Path, step: str, layers: List[Layer]):
    """Load a single GeoTIFF and append layers. Skips if file missing."""
    if not path.exists():
        return

    try:
        if step == "cloud_mask":
            from clouds_decoded.data import CloudMaskData
            data = CloudMaskData.from_file(str(path))
            layers.append(layer_from_cloud_mask(data))

        elif step == "cloud_height":
            from clouds_decoded.data import CloudHeightGridData
            data = CloudHeightGridData.from_file(str(path))
            layers.append(layer_from_cloud_height(data))

        elif step == "albedo":
            from clouds_decoded.data import AlbedoData
            data = AlbedoData.from_file(str(path))
            layers.extend(layers_from_albedo(data))

        elif step == "cloud_properties":
            from clouds_decoded.data import CloudPropertiesData
            data = CloudPropertiesData.from_file(str(path))
            layers.extend(layers_from_cloud_properties(data))

    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
