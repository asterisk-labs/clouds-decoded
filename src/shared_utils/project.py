"""Project management: config directories, per-scene outputs, and resumability.

A Project is an optional directory structure that holds editable module configs,
organizes outputs per Sentinel-2 scene, and supports resuming interrupted runs.

Usage:
    project = Project.init("./my_analysis",
                           scenes=["/data/S2A_....SAFE"], pipeline="full-workflow")
    project.run()

    # Later, resume or add scenes:
    project = Project.load("./my_analysis")
    project.add_scene("/data/S2B_....SAFE")
    project.run()
"""
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field
from clouds_decoded.constants import METADATA_TAG

logger = logging.getLogger(__name__)

# Step order for each pipeline type
PIPELINE_STEPS = {
    "full-workflow": ["cloud_mask", "cloud_height", "albedo", "refocus", "cloud_properties"],
}

# Maps step name -> (config YAML filename, config class import path)
STEP_CONFIG_FILE = {
    "cloud_mask": "cloud_mask.yaml",
    "cloud_height": "cloud_height.yaml",
    "albedo": "albedo.yaml",
    "refocus": "refocus.yaml",
    "cloud_properties": "refl2prop.yaml",
}

# Maps step name -> default output filename(s)
STEP_OUTPUT_FILE = {
    "cloud_mask": "cloud_mask.tif",
    "cloud_height": "cloud_height.tif",
    "albedo": "albedo.tif",
    "refocus": None,  # Refocus produces a modified scene, not a single file
    "cloud_properties": "properties.tif",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Root project configuration, stored as project.yaml."""
    name: str = Field(..., description="User-friendly project name")
    pipeline: Literal["full-workflow"] = Field(
        default="full-workflow", description="Pipeline type to run"
    )
    scenes: List[str] = Field(default_factory=list, description="Absolute paths to .SAFE directories")
    created_at: str = Field(default="", description="ISO timestamp of project creation")
    use_emulator: bool = Field(default=False, description="Use emulator for cloud height retrieval")

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)


class StepResult(BaseModel):
    """Record of a single processing step for a scene."""
    status: Literal["pending", "completed", "failed"] = "pending"
    output_file: Optional[str] = Field(
        default=None, description="Output filename relative to scene directory"
    )
    config_hash: Optional[str] = Field(
        default=None, description="SHA-256 prefix of config YAML content"
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class Provenance(BaseModel):
    """Processing provenance, embedded in output metadata and manifest."""
    repo_url: str = "https://github.com/asterisk-labs/clouds-decoded"
    project_name: str
    codebase_version: str = "unknown"
    python_version: str = ""
    git_hash: Optional[str] = None
    timestamp: str = ""
    scene_path: str = ""
    product_id: Optional[str] = Field(
        default=None,
        description="Sentinel-2 PRODUCT_URI from ESA metadata"
    )
    pipeline: str = ""
    step_name: str = ""
    step_config: Dict[str, Any] = Field(default_factory=dict)


class SceneManifest(BaseModel):
    """Per-scene processing manifest, stored as manifest.json."""
    scene_id: str
    scene_path: str
    provenance: Optional[Provenance] = None
    steps: Dict[str, StepResult] = Field(default_factory=dict)
    last_updated: Optional[str] = None

    def is_step_complete(self, step: str, current_config_hash: str) -> bool:
        """Check whether a step can be skipped.

        Returns True only if the step is marked completed, the output file
        still exists on disk, and the config hash matches (i.e. the user
        hasn't edited the config since the last run).
        """
        if step not in self.steps:
            return False
        result = self.steps[step]
        if result.status != "completed":
            return False
        # Refocus doesn't produce a single output file
        if result.output_file is not None and not Path(result.output_file).exists():
            return False
        if result.config_hash != current_config_hash:
            return False
        return True

    @classmethod
    def from_json(cls, path: Path) -> "SceneManifest":
        with open(path) as f:
            return cls.model_validate_json(f.read())

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Project Class
# ---------------------------------------------------------------------------

class Project:
    """Manages a clouds-decoded project directory.

    Responsibilities:
    - Initialize project directory with default config YAMLs
    - Load/save project config and per-scene manifests
    - Determine which processing steps to skip (resumability)
    - Run the pipeline for each scene
    - Embed provenance metadata in outputs
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).resolve()
        self.config_path = self.project_dir / "project.yaml"
        self.configs_dir = self.project_dir / "configs"
        self.scenes_dir = self.project_dir / "scenes"
        self._config: Optional[ProjectConfig] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        project_dir: str,
        name: Optional[str] = None,
        pipeline: str = "full-workflow",
        clone_from: Optional[str] = None,
        use_emulator: bool = False,
    ) -> "Project":
        """Create a new project directory with default config YAMLs.

        Args:
            project_dir: Directory path for the new project.
            name: Project name. Defaults to the directory basename.
            pipeline: Pipeline type ('workflow' or 'full-workflow').
            clone_from: Path to an existing project directory whose configs
                will be copied into the new project.
        """
        project_dir = Path(project_dir).resolve()
        if (project_dir / "project.yaml").exists():
            raise FileExistsError(f"Project already exists at {project_dir}")

        if name is None:
            name = project_dir.name

        # If cloning, load the source project to copy its pipeline and configs
        source_project: Optional["Project"] = None
        if clone_from:
            source_dir = Path(clone_from).resolve()
            if not (source_dir / "project.yaml").exists():
                raise FileNotFoundError(
                    f"Cannot clone: no project.yaml found in {source_dir}"
                )
            source_project = cls(source_dir)
            pipeline = source_project.config.pipeline
            use_emulator = source_project.config.use_emulator

        project_dir.mkdir(parents=True, exist_ok=True)

        config = ProjectConfig(
            name=name,
            pipeline=pipeline,
            created_at=datetime.now().isoformat(),
            use_emulator=use_emulator,
        )
        config.to_yaml(project_dir / "project.yaml")

        project = cls(project_dir)
        project._config = config

        if source_project:
            project._clone_configs(source_project)
        else:
            project._write_default_configs()

        (project_dir / "scenes").mkdir(exist_ok=True)

        if source_project:
            logger.info(f"Project '{name}' initialized at {project_dir} (cloned from {clone_from})")
        else:
            logger.info(f"Project '{name}' initialized at {project_dir}")
        logger.info(f"  Pipeline: {pipeline}")
        logger.info(f"  Edit configs in {project.configs_dir}/")

        return project

    @classmethod
    def load(cls, project_dir: str) -> "Project":
        """Load an existing project from its directory."""
        project_dir = Path(project_dir).resolve()
        if not (project_dir / "project.yaml").exists():
            raise FileNotFoundError(
                f"No project.yaml found in {project_dir}. "
                f"Use 'project init' to create a new project."
            )
        project = cls(project_dir)
        # Eagerly load config to validate
        _ = project.config
        logger.info(f"Loaded project '{project.config.name}' from {project_dir}")
        return project

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            self._config = ProjectConfig.from_yaml(self.config_path)
        return self._config

    @property
    def steps(self) -> List[str]:
        return PIPELINE_STEPS[self.config.pipeline]

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_scene(self, scene_path: str):
        """Add a scene to the project."""
        resolved = str(Path(scene_path).resolve())
        if resolved in self.config.scenes:
            logger.warning(f"Scene already in project: {resolved}")
            return
        self.config.scenes.append(resolved)
        self.config.to_yaml(self.config_path)
        logger.info(f"Added scene: {self._scene_id(resolved)}")

    def _scene_id(self, scene_path: str) -> str:
        """Extract scene ID from .SAFE path (basename without .SAFE)."""
        return Path(scene_path).stem

    def _scene_output_dir(self, scene_id: str) -> Path:
        return self.scenes_dir / scene_id

    def _load_manifest(self, scene_id: str, scene_path: str) -> SceneManifest:
        manifest_path = self._scene_output_dir(scene_id) / "manifest.json"
        if manifest_path.exists():
            return SceneManifest.from_json(manifest_path)
        return SceneManifest(scene_id=scene_id, scene_path=scene_path)

    def _save_manifest(self, scene_id: str, manifest: SceneManifest):
        manifest.last_updated = datetime.now().isoformat()
        manifest_path = self._scene_output_dir(scene_id) / "manifest.json"
        manifest.to_json(manifest_path)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _config_yaml_path(self, step: str) -> Path:
        """Get the config YAML path for a given step."""
        return self.configs_dir / STEP_CONFIG_FILE[step]

    def _config_hash(self, step: str) -> str:
        """SHA-256 hash (first 16 chars) of the config YAML file content."""
        path = self._config_yaml_path(step)
        if not path.exists():
            return "no_config"
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def _load_step_config(self, step: str):
        """Load the appropriate config object for a step."""
        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
        from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
        from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
        from clouds_decoded.modules.refocus.config import RefocusConfig
        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig, ShadingRefl2PropConfig

        config_path = str(self._config_yaml_path(step))

        if step == "cloud_mask":
            return CloudMaskConfig.from_yaml(config_path)
        elif step == "cloud_height":
            if self.config.use_emulator:
                return CloudHeightEmulatorConfig.from_yaml(config_path)
            return CloudHeightConfig.from_yaml(config_path)
        elif step == "albedo":
            return AlbedoEstimatorConfig.from_yaml(config_path)
        elif step == "refocus":
            return RefocusConfig.from_yaml(config_path)
        elif step == "cloud_properties":
            # Peek at the YAML to determine which config class to use
            base = Refl2PropConfig.from_yaml(config_path)
            if base.method == "shading":
                return ShadingRefl2PropConfig.from_yaml(config_path)
            return base
        else:
            raise ValueError(f"Unknown step: {step}")

    def _write_default_configs(self):
        """Write default YAML configs for each module."""
        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
        from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
        from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
        from clouds_decoded.modules.refocus.config import RefocusConfig
        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig

        self.configs_dir.mkdir(parents=True, exist_ok=True)

        CloudMaskConfig().to_yaml(str(self.configs_dir / "cloud_mask.yaml"))
        if self.config.use_emulator:
            CloudHeightEmulatorConfig().to_yaml(str(self.configs_dir / "cloud_height.yaml"))
        else:
            CloudHeightConfig().to_yaml(str(self.configs_dir / "cloud_height.yaml"))
        AlbedoEstimatorConfig().to_yaml(str(self.configs_dir / "albedo.yaml"))
        RefocusConfig().to_yaml(str(self.configs_dir / "refocus.yaml"))
        Refl2PropConfig().to_yaml(
            str(self.configs_dir / "refl2prop.yaml")
        )

        logger.info(f"Default configs written to {self.configs_dir}/")

    def _clone_configs(self, source: "Project"):
        """Copy config YAMLs from another project."""
        import shutil

        self.configs_dir.mkdir(parents=True, exist_ok=True)
        for src_file in source.configs_dir.glob("*.yaml"):
            shutil.copy2(src_file, self.configs_dir / src_file.name)
        logger.info(f"Configs cloned from {source.project_dir}/")

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _get_codebase_version(self) -> str:
        try:
            return importlib.metadata.version("clouds-decoded")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def _get_git_hash(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(Path(__file__).parent),
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _build_provenance(
        self,
        scene_path: str,
        product_id: Optional[str],
        step_name: str,
        step_config_dict: Dict[str, Any],
    ) -> Provenance:
        import sys
        return Provenance(
            project_name=self.config.name,
            codebase_version=self._get_codebase_version(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            git_hash=self._get_git_hash(),
            timestamp=datetime.now().isoformat(),
            scene_path=scene_path,
            product_id=product_id,
            pipeline=self.config.pipeline,
            step_name=step_name,
            step_config=step_config_dict,
        )

    # ------------------------------------------------------------------
    # File provenance validation
    # ------------------------------------------------------------------

    def _read_file_provenance(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Read provenance metadata from a GeoTIFF's tags (header-only, no pixel I/O).

        Returns the provenance dict if present, None otherwise.
        """
        import rasterio as rio

        try:
            with rio.open(filepath) as src:
                tags = src.tags()
                extra = tags.get(METADATA_TAG)
                if not extra:
                    return None
                meta = json.loads(extra)
                return meta.get('provenance') if isinstance(meta, dict) else None
        except Exception as e:
            logger.debug(f"Could not read provenance from {filepath}: {e}")
            return None

    def _validate_step_file(
        self,
        step: str,
        scene_out: Path,
        scene_path: str,
        current_config_dict: Dict[str, Any],
    ) -> Optional[str]:
        """Validate that an output file's embedded provenance matches the current config.

        Returns None if valid, or an error message string if there's a mismatch.
        Skips validation for steps with no output file (e.g. refocus).
        """
        output_file = STEP_OUTPUT_FILE.get(step)
        if output_file is None:
            return None  # refocus has no file to validate

        filepath = scene_out / output_file
        if not filepath.exists():
            return f"Output file missing: {filepath}"

        provenance = self._read_file_provenance(filepath)

        if provenance is None:
            return (
                f"[{step}] {filepath.name} has no provenance metadata. "
                f"File may not have been produced by this project."
            )

        # Check project name
        file_project = provenance.get("project_name")
        if file_project and file_project != self.config.name:
            return (
                f"[{step}] {filepath.name} was produced by project '{file_project}', "
                f"but current project is '{self.config.name}'."
            )

        # Check scene identity (product_id is authoritative; fall back to path)
        file_product = provenance.get("product_id")
        if file_product:
            # Compare product IDs if the current scene has one
            current_product = Path(scene_path).name.removesuffix(".SAFE")
            file_product_stem = file_product.removesuffix(".SAFE")
            if current_product != file_product_stem:
                return (
                    f"[{step}] {filepath.name} was produced from product '{file_product}', "
                    f"but current scene is '{current_product}'."
                )
        else:
            # Fall back to scene_path check
            file_scene = provenance.get("scene_path", "")
            if file_scene and str(Path(file_scene).resolve()) != str(Path(scene_path).resolve()):
                return (
                    f"[{step}] {filepath.name} was produced from a different scene path."
                )

        # Check step_config matches current config
        file_config = provenance.get("step_config")
        if file_config is not None and file_config != current_config_dict:
            # Find which keys differ for a useful message
            changed_keys = []
            all_keys = set(file_config.keys()) | set(current_config_dict.keys())
            for k in sorted(all_keys):
                if file_config.get(k) != current_config_dict.get(k):
                    changed_keys.append(k)
            return (
                f"[{step}] {filepath.name} config mismatch. "
                f"Differing keys: {', '.join(changed_keys)}. "
                f"File was produced with different settings than the current config."
            )

        return None

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        scenes: Optional[List[str]] = None,
        force: bool = False,
        unsafe: bool = False,
        crop_window: Optional[str] = None,
    ):
        """Run the pipeline for the given scenes (or all registered scenes).

        Args:
            scenes: Scene .SAFE paths to process. Auto-registered if not
                already in the project. If None, processes all registered scenes.
            force: If True, reprocess all steps regardless of manifest state.
            unsafe: If True, skip file provenance validation (use existing
                intermediate files even if their metadata doesn't match).
            crop_window: Optional spatial crop 'col_off,row_off,width,height'
                applied when reading each scene.
        """
        # Lazy imports to avoid circular dependencies and heavy torch import
        from clouds_decoded.modules.cloud_mask.config import PostProcessParams
        from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor

        if scenes:
            # Auto-register any new scenes
            for s in scenes:
                self.add_scene(s)
            scene_list = [str(Path(s).resolve()) for s in scenes]
        else:
            scene_list = self.config.scenes

        if not scene_list:
            logger.warning("No scenes to process. Pass scene paths to 'project run'.")
            return

        logger.info(f"Running project '{self.config.name}' ({self.config.pipeline})")
        logger.info(f"Processing {len(scene_list)} scene(s)")

        for scene_path in scene_list:
            scene_id = self._scene_id(scene_path)
            logger.info(f"\n{'='*60}")
            logger.info(f"Scene: {scene_id}")
            logger.info(f"{'='*60}")

            scene_out = self._scene_output_dir(scene_id)
            scene_out.mkdir(parents=True, exist_ok=True)

            manifest = self._load_manifest(scene_id, scene_path)

            # Determine which steps need running
            steps_to_run = self.steps
            invalidate_from: Optional[int] = None

            if not force:
                for i, step in enumerate(steps_to_run):
                    config_hash = self._config_hash(step)
                    if manifest.is_step_complete(step, config_hash):
                        # Validate file provenance unless --unsafe
                        if not unsafe:
                            step_config = self._load_step_config(step)
                            computed = type(step_config).model_computed_fields
                            exclude = set(computed.keys()) if computed else set()
                            config_dict = step_config.model_dump(exclude=exclude)
                            error = self._validate_step_file(step, scene_out, scene_path, config_dict)
                            if error is not None:
                                logger.error(
                                    f"  {error}\n"
                                    f"  Use --unsafe to skip validation and use this file anyway, "
                                    f"or --force to reprocess all steps."
                                )
                                raise RuntimeError(
                                    f"File provenance mismatch for step '{step}' "
                                    f"in scene '{scene_id}'. {error}"
                                )
                        logger.info(f"  [{step}] Skipping (already complete, config unchanged)")
                    else:
                        # Check if it was complete but config changed
                        if (step in manifest.steps
                                and manifest.steps[step].status == "completed"
                                and manifest.steps[step].config_hash != config_hash):
                            logger.warning(
                                f"  [{step}] Config changed since last run — "
                                f"re-running this and all downstream steps"
                            )
                        invalidate_from = i
                        break
            else:
                invalidate_from = 0

            if invalidate_from is None:
                logger.info(f"All steps complete for {scene_id}")
                continue

            # Load scene
            from clouds_decoded.data import Sentinel2Scene
            scene = Sentinel2Scene()
            if crop_window:
                try:
                    parts = crop_window.split(",")
                    if len(parts) != 4:
                        raise ValueError(
                            f"crop_window must have 4 comma-separated integers, got {len(parts)}"
                        )
                    col_off, row_off, width, height = map(int, parts)
                    scene.read(scene_path, crop_window=(col_off, row_off, width, height))
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid crop_window '{crop_window}': {e}. "
                        f"Expected format: 'col_off,row_off,width,height'"
                    ) from e
            else:
                scene.read(scene_path)

            # Track intermediate results for pipeline data flow
            mask_result = None
            height_result = None
            albedo_result = None
            refocused_scene = None

            # Load already-completed intermediate results if needed
            if invalidate_from > 0:
                mask_result, height_result, albedo_result, refocused_scene = (
                    self._load_intermediates(scene_out, steps_to_run[:invalidate_from])
                )
                # Refocus produces an in-memory scene that can't be persisted.
                # If it was completed but we still need it, re-run from refocus.
                if (refocused_scene is None
                        and "refocus" in steps_to_run[:invalidate_from]):
                    invalidate_from = steps_to_run.index("refocus")
                    logger.info("  [refocus] Re-running (in-memory result not persisted to disk)")

            # Run steps from invalidate_from onwards
            for step in steps_to_run[invalidate_from:]:
                config_hash = self._config_hash(step)

                # Record start
                step_result = StepResult(
                    status="pending",
                    config_hash=config_hash,
                    started_at=datetime.now().isoformat(),
                )
                manifest.steps[step] = step_result
                self._save_manifest(scene_id, manifest)

                t_start = time.time()
                try:
                    output_file = self._run_step(
                        step, scene, scene_path, scene_out,
                        mask_result=mask_result,
                        height_result=height_result,
                        albedo_result=albedo_result,
                        refocused_scene=refocused_scene,
                    )

                    # Capture results for downstream steps
                    step_output = STEP_OUTPUT_FILE.get(step)
                    if step == "cloud_mask":
                        from clouds_decoded.data import CloudMaskData
                        mask_result = CloudMaskData.from_file(str(scene_out / step_output))
                        # Postprocess to binary for downstream
                        postprocessor = CloudMaskProcessor()
                        mask_result = postprocessor.postprocess(mask_result, PostProcessParams())
                    elif step == "cloud_height":
                        from clouds_decoded.data import CloudHeightGridData
                        height_result = CloudHeightGridData.from_file(
                            str(scene_out / step_output)
                        )
                    elif step == "albedo":
                        from clouds_decoded.data import AlbedoData
                        albedo_result = AlbedoData.from_file(str(scene_out / step_output))
                    elif step == "refocus":
                        # Refocus returns a modified scene in memory
                        refocused_scene = output_file  # _run_step returns scene for refocus

                    duration = time.time() - t_start
                    step_result.status = "completed"
                    step_result.completed_at = datetime.now().isoformat()
                    step_result.duration_seconds = round(duration, 1)
                    if isinstance(output_file, str):
                        step_result.output_file = output_file
                    logger.info(f"  [{step}] Completed in {duration:.1f}s")

                except Exception as e:
                    duration = time.time() - t_start
                    step_result.status = "failed"
                    step_result.completed_at = datetime.now().isoformat()
                    step_result.duration_seconds = round(duration, 1)
                    step_result.error = str(e)
                    logger.error(f"  [{step}] FAILED: {e}")
                    self._save_manifest(scene_id, manifest)
                    break  # Stop pipeline on failure

                self._save_manifest(scene_id, manifest)

        logger.info(f"\nProject run complete.")

    def _run_step(
        self,
        step: str,
        scene,
        scene_path: str,
        scene_out: Path,
        mask_result=None,
        height_result=None,
        albedo_result=None,
        refocused_scene=None,
    ):
        """Run a single processing step. Returns output path or scene object."""
        # Lazy imports
        from clouds_decoded.cli.entry import (
            run_cloud_mask, run_cloud_height, run_cloud_properties,
            run_shading_cloud_properties, run_albedo, run_refocus,
        )
        from clouds_decoded.modules.cloud_mask.config import PostProcessParams
        from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor

        step_config = self._load_step_config(step)

        # Build provenance
        computed = type(step_config).model_computed_fields
        exclude = set(computed.keys()) if computed else set()
        config_dict = step_config.model_dump(exclude=exclude)
        product_id = getattr(scene, 'product_uri', None)
        provenance = self._build_provenance(scene_path, product_id, step, config_dict)

        if step == "cloud_mask":
            output_path = str(scene_out / STEP_OUTPUT_FILE[step])
            result = run_cloud_mask(scene, step_config, output_path=output_path)
            result.metadata.provenance = provenance.model_dump()
            result.write(output_path)
            return output_path

        elif step == "cloud_height":
            output_path = str(scene_out / STEP_OUTPUT_FILE[step])
            result = run_cloud_height(
                scene, step_config,
                output_path=output_path,
                cloud_mask=mask_result,
                use_emulator=self.config.use_emulator,
            )
            result.metadata.provenance = provenance.model_dump()
            result.write(output_path)
            return output_path

        elif step == "albedo":
            output_path = str(scene_out / STEP_OUTPUT_FILE[step])
            result = run_albedo(
                scene, step_config,
                cloud_mask=mask_result,
                output_path=output_path,
            )
            result.metadata.provenance = provenance.model_dump()
            result.write(output_path)
            return output_path

        elif step == "refocus":
            if height_result is None:
                raise ValueError("Refocus requires cloud height data")
            refocus_out = str(scene_out / "refocused") if step_config.save_refocused else None
            result_scene = run_refocus(scene, height_result, step_config, output_dir=refocus_out)
            return result_scene  # Return scene object, not path

        elif step == "cloud_properties":
            # Use refocused scene if available, otherwise original
            input_scene = refocused_scene if refocused_scene is not None else scene
            output_file = self._get_properties_output_file(step_config)
            output_path = str(scene_out / output_file)

            if step_config.method == "shading":
                result = run_shading_cloud_properties(
                    input_scene, height_result, step_config,
                    output_path=output_path,
                    albedo_data=albedo_result,
                )
            else:
                result = run_cloud_properties(
                    input_scene, height_result, step_config,
                    output_path=output_path,
                    albedo_data=albedo_result,
                )
            result.metadata.provenance = provenance.model_dump()
            result.write(output_path)
            return output_path

        else:
            raise ValueError(f"Unknown step: {step}")

    def _get_properties_output_file(self, step_config=None) -> str:
        """Get the output filename for cloud properties based on method."""
        method = getattr(step_config, 'method', 'standard') if step_config else 'standard'
        if method == "shading":
            return "properties_shading.tif"
        return STEP_OUTPUT_FILE["cloud_properties"]

    def _load_intermediates(self, scene_out: Path, completed_steps: List[str]):
        """Load intermediate results from completed steps for data flow."""
        mask_result = None
        height_result = None
        albedo_result = None
        refocused_scene = None

        from clouds_decoded.modules.cloud_mask.config import PostProcessParams
        from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor

        for step in completed_steps:
            output_file = STEP_OUTPUT_FILE.get(step)
            if output_file is None:
                continue  # refocus has no output file

            output_path = scene_out / output_file
            if not output_path.exists():
                continue

            if step == "cloud_mask":
                from clouds_decoded.data import CloudMaskData
                raw_mask = CloudMaskData.from_file(str(output_path))
                postprocessor = CloudMaskProcessor()
                mask_result = postprocessor.postprocess(raw_mask, PostProcessParams())
            elif step == "cloud_height":
                from clouds_decoded.data import CloudHeightGridData
                height_result = CloudHeightGridData.from_file(str(output_path))
            elif step == "albedo":
                from clouds_decoded.data import AlbedoData
                albedo_result = AlbedoData.from_file(str(output_path))
            # Note: refocused scene can't be loaded from disk (it's an in-memory scene)
            # If refocus was completed but cloud_properties wasn't, we'd need to re-run refocus

        return mask_result, height_result, albedo_result, refocused_scene

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Return a formatted status table for the project."""
        lines = []
        lines.append(f"Project: {self.config.name} ({self.config.pipeline})")
        lines.append(f"Scenes: {len(self.config.scenes)}")
        lines.append("")

        steps = self.steps
        # Header
        header = f"{'Scene':<50} " + " ".join(f"{s:<12}" for s in steps)
        lines.append(header)
        lines.append("-" * len(header))

        for scene_path in self.config.scenes:
            scene_id = self._scene_id(scene_path)
            manifest = self._load_manifest(scene_id, scene_path)

            cols = []
            for step in steps:
                if step in manifest.steps:
                    result = manifest.steps[step]
                    if result.status == "completed":
                        cols.append("done")
                    elif result.status == "failed":
                        cols.append("FAILED")
                    else:
                        cols.append("--")
                else:
                    cols.append("--")

            # Truncate scene ID for display
            display_id = scene_id[:48] + ".." if len(scene_id) > 50 else scene_id
            line = f"{display_id:<50} " + " ".join(f"{c:<12}" for c in cols)
            lines.append(line)

        return "\n".join(lines)
