"""Tests for the Project abstraction: config management, manifests, resumability."""
import json
import pytest
import numpy as np
from pathlib import Path

import yaml
from rasterio.transform import Affine
from rasterio.crs import CRS


# ---------------------------------------------------------------------------
# to_yaml() tests
# ---------------------------------------------------------------------------

class TestToYaml:
    """Tests for BaseProcessorConfig.to_yaml()."""

    def test_roundtrip_cloud_height(self, tmp_path):
        """CloudHeightConfig survives YAML roundtrip."""
        from clouds_decoded.modules.cloud_height.config import CloudHeightConfig

        original = CloudHeightConfig(stride=250, max_height=8000, n_workers=2)
        yaml_path = tmp_path / "cloud_height.yaml"
        original.to_yaml(str(yaml_path))

        loaded = CloudHeightConfig.from_yaml(str(yaml_path))
        assert loaded.stride == 250
        assert loaded.max_height == 8000
        assert loaded.n_workers == 2

    def test_roundtrip_cloud_mask(self, tmp_path):
        """CloudMaskConfig survives YAML roundtrip."""
        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig

        original = CloudMaskConfig(method="threshold", threshold_band="B04", threshold_value=5000)
        yaml_path = tmp_path / "cloud_mask.yaml"
        original.to_yaml(str(yaml_path))

        loaded = CloudMaskConfig.from_yaml(str(yaml_path))
        assert loaded.method == "threshold"
        assert loaded.threshold_band == "B04"
        assert loaded.threshold_value == 5000

    def test_roundtrip_albedo(self, tmp_path):
        """AlbedoEstimatorConfig survives YAML roundtrip."""
        from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig

        original = AlbedoEstimatorConfig(method="polynomial", polynomial_order=3)
        yaml_path = tmp_path / "albedo.yaml"
        original.to_yaml(str(yaml_path))

        loaded = AlbedoEstimatorConfig.from_yaml(str(yaml_path))
        assert loaded.method == "polynomial"
        assert loaded.polynomial_order == 3

    def test_roundtrip_refl2prop(self, tmp_path):
        """Refl2PropConfig survives YAML roundtrip (with computed fields excluded)."""
        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig

        original = Refl2PropConfig(model_path="/tmp/model.pth", batch_size=16384)
        yaml_path = tmp_path / "refl2prop.yaml"
        original.to_yaml(str(yaml_path))

        loaded = Refl2PropConfig.from_yaml(str(yaml_path))
        assert loaded.model_path == "/tmp/model.pth"
        assert loaded.batch_size == 16384
        # Computed fields should be re-derived, not loaded from YAML
        assert loaded.input_size == original.input_size
        assert loaded.num_bands == original.num_bands

    def test_excludes_computed_fields(self, tmp_path):
        """to_yaml() excludes @computed_field properties from output."""
        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig

        config = Refl2PropConfig(model_path="/tmp/model.pth")
        yaml_path = tmp_path / "refl2prop.yaml"
        config.to_yaml(str(yaml_path))

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        # These computed fields should NOT appear in the YAML
        assert "input_size" not in raw
        assert "num_bands" not in raw
        assert "noise_output_size" not in raw
        assert "noise_indices" not in raw
        assert "input_feature_names" not in raw

        # Regular fields should still be there
        assert "model_path" in raw
        assert "bands" in raw
        assert "batch_size" in raw

    def test_excludes_bag_size_from_shading(self, tmp_path):
        """ShadingRefl2PropConfig also excludes bag_size computed field."""
        from clouds_decoded.modules.refl2prop.config import ShadingRefl2PropConfig

        config = ShadingRefl2PropConfig(model_path="/tmp/model.pth")
        yaml_path = tmp_path / "shading.yaml"
        config.to_yaml(str(yaml_path))

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        assert "bag_size" not in raw
        assert "window_size" in raw  # Regular field should be present

    def test_creates_parent_directories(self, tmp_path):
        """to_yaml() creates parent directories if needed."""
        from clouds_decoded.modules.cloud_height.config import CloudHeightConfig

        config = CloudHeightConfig()
        nested_path = tmp_path / "a" / "b" / "c" / "config.yaml"
        config.to_yaml(str(nested_path))
        assert nested_path.exists()


# ---------------------------------------------------------------------------
# Provenance on Metadata
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for the provenance field on Metadata base class."""

    def test_metadata_default_provenance_is_none(self):
        """Metadata.provenance defaults to None."""
        from clouds_decoded.data.base import Metadata

        meta = Metadata()
        assert meta.provenance is None

    def test_metadata_accepts_provenance_dict(self):
        """Metadata accepts a provenance dict."""
        from clouds_decoded.data.base import Metadata

        prov = {"project_name": "test", "git_hash": "abc123"}
        meta = Metadata(provenance=prov)
        assert meta.provenance == prov

    def test_provenance_survives_geotiff_roundtrip(self, tmp_path):
        """Provenance dict is preserved through GeoTIFF write/read."""
        from clouds_decoded.data.base import GeoRasterData, Metadata

        prov = {
            "project_name": "roundtrip-test",
            "codebase_version": "0.1.0",
            "git_hash": "abc123def456",
            "pipeline": "workflow",
        }

        data = np.random.rand(1, 50, 50).astype(np.float32)
        raster = GeoRasterData(
            data=data,
            transform=Affine.translation(0, 0) * Affine.scale(10, -10),
            crs=CRS.from_epsg(32633),
            metadata=Metadata(provenance=prov),
        )

        filepath = tmp_path / "test_provenance.tif"
        raster.write(str(filepath))

        loaded = GeoRasterData.from_file(str(filepath))
        assert loaded.metadata.provenance is not None
        assert loaded.metadata.provenance["project_name"] == "roundtrip-test"
        assert loaded.metadata.provenance["git_hash"] == "abc123def456"

    def test_provenance_backward_compatible(self, tmp_path):
        """Files written without provenance load fine (provenance=None)."""
        from clouds_decoded.data.base import GeoRasterData, Metadata

        data = np.random.rand(1, 20, 20).astype(np.float32)
        raster = GeoRasterData(
            data=data,
            transform=Affine.translation(0, 0) * Affine.scale(10, -10),
            crs=CRS.from_epsg(32633),
            metadata=Metadata(),  # No provenance
        )

        filepath = tmp_path / "no_provenance.tif"
        raster.write(str(filepath))

        loaded = GeoRasterData.from_file(str(filepath))
        # Should not crash, provenance stays None
        assert loaded.metadata.provenance is None


# ---------------------------------------------------------------------------
# SceneManifest tests
# ---------------------------------------------------------------------------

class TestSceneManifest:
    """Tests for SceneManifest serialization and step completion logic."""

    def test_manifest_json_roundtrip(self, tmp_path):
        """SceneManifest survives JSON serialization/deserialization."""
        from clouds_decoded.project import SceneManifest, StepResult

        manifest = SceneManifest(
            scene_id="S2A_TEST",
            scene_path="/data/S2A_TEST.SAFE",
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file="/out/cloud_mask.tif",
                    config_hash="abc123",
                    duration_seconds=12.5,
                ),
                "cloud_height": StepResult(
                    status="failed",
                    error="Out of memory",
                    config_hash="def456",
                ),
            },
        )

        json_path = tmp_path / "manifest.json"
        manifest.to_json(json_path)
        loaded = SceneManifest.from_json(json_path)

        assert loaded.scene_id == "S2A_TEST"
        assert loaded.steps["cloud_mask"].status == "completed"
        assert loaded.steps["cloud_mask"].output_file == "/out/cloud_mask.tif"
        assert loaded.steps["cloud_height"].status == "failed"
        assert loaded.steps["cloud_height"].error == "Out of memory"

    def test_step_complete_true(self, tmp_path):
        """is_step_complete returns True when status, file, and hash all match."""
        from clouds_decoded.project import SceneManifest, StepResult

        # Create a dummy output file
        output_file = tmp_path / "cloud_mask.tif"
        output_file.touch()

        manifest = SceneManifest(
            scene_id="test",
            scene_path="/data/test.SAFE",
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file=str(output_file),
                    config_hash="abc123",
                ),
            },
        )

        assert manifest.is_step_complete("cloud_mask", "abc123") is True

    def test_step_complete_false_missing_step(self):
        """is_step_complete returns False for a step not in manifest."""
        from clouds_decoded.project import SceneManifest

        manifest = SceneManifest(scene_id="test", scene_path="/data/test.SAFE")
        assert manifest.is_step_complete("cloud_mask", "abc123") is False

    def test_step_complete_false_not_completed(self):
        """is_step_complete returns False when status is not 'completed'."""
        from clouds_decoded.project import SceneManifest, StepResult

        manifest = SceneManifest(
            scene_id="test",
            scene_path="/data/test.SAFE",
            steps={"cloud_mask": StepResult(status="failed", config_hash="abc123")},
        )
        assert manifest.is_step_complete("cloud_mask", "abc123") is False

    def test_step_complete_false_file_missing(self, tmp_path):
        """is_step_complete returns False when output file doesn't exist."""
        from clouds_decoded.project import SceneManifest, StepResult

        manifest = SceneManifest(
            scene_id="test",
            scene_path="/data/test.SAFE",
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file=str(tmp_path / "nonexistent.tif"),
                    config_hash="abc123",
                ),
            },
        )
        assert manifest.is_step_complete("cloud_mask", "abc123") is False

    def test_step_complete_false_config_changed(self, tmp_path):
        """is_step_complete returns False when config hash differs."""
        from clouds_decoded.project import SceneManifest, StepResult

        output_file = tmp_path / "cloud_mask.tif"
        output_file.touch()

        manifest = SceneManifest(
            scene_id="test",
            scene_path="/data/test.SAFE",
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file=str(output_file),
                    config_hash="old_hash",
                ),
            },
        )
        assert manifest.is_step_complete("cloud_mask", "new_hash") is False

    def test_refocus_step_complete_no_output_file(self):
        """Refocus step (output_file=None) is complete if status+hash match."""
        from clouds_decoded.project import SceneManifest, StepResult

        manifest = SceneManifest(
            scene_id="test",
            scene_path="/data/test.SAFE",
            steps={
                "refocus": StepResult(
                    status="completed",
                    output_file=None,
                    config_hash="abc123",
                ),
            },
        )
        assert manifest.is_step_complete("refocus", "abc123") is True


# ---------------------------------------------------------------------------
# ProjectConfig tests
# ---------------------------------------------------------------------------

class TestProjectConfig:
    """Tests for ProjectConfig serialization."""

    def test_project_config_yaml_roundtrip(self, tmp_path):
        """ProjectConfig survives YAML roundtrip."""
        from clouds_decoded.project import ProjectConfig

        config = ProjectConfig(
            name="Test Project",
            pipeline="full-workflow",
            scenes=["/data/scene1.SAFE", "/data/scene2.SAFE"],
            created_at="2025-01-01T00:00:00",
        )

        yaml_path = tmp_path / "project.yaml"
        config.to_yaml(yaml_path)

        loaded = ProjectConfig.from_yaml(yaml_path)
        assert loaded.name == "Test Project"
        assert loaded.pipeline == "full-workflow"
        assert len(loaded.scenes) == 2


# ---------------------------------------------------------------------------
# Project.init() tests
# ---------------------------------------------------------------------------

class TestProjectInit:
    """Tests for Project.init() directory structure."""

    def test_init_creates_directory_structure(self, tmp_path):
        """Project.init() creates configs/, scenes/, and project.yaml."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "my_project"
        project = Project.init(
            str(project_dir),
            name="Test",
        )

        assert (project_dir / "project.yaml").exists()
        assert (project_dir / "configs").is_dir()
        assert (project_dir / "scenes").is_dir()

    def test_init_creates_config_yamls(self, tmp_path):
        """Project.init() writes one YAML per module with correct defaults."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "my_project"
        Project.init(str(project_dir), pipeline="full-workflow")

        configs_dir = project_dir / "configs"
        expected_files = [
            "cloud_mask.yaml",
            "cloud_height.yaml",
            "albedo.yaml",
            "refocus.yaml",
            "refl2prop.yaml",
        ]
        for fname in expected_files:
            assert (configs_dir / fname).exists(), f"Missing config: {fname}"

    def test_init_config_yamls_are_loadable(self, tmp_path):
        """Config YAMLs produced by init() can be loaded by their config classes."""
        from clouds_decoded.project import Project
        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
        from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
        from clouds_decoded.modules.refocus.config import RefocusConfig
        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig

        project_dir = tmp_path / "my_project"
        Project.init(str(project_dir), pipeline="full-workflow")

        configs_dir = project_dir / "configs"
        CloudMaskConfig.from_yaml(str(configs_dir / "cloud_mask.yaml"))
        CloudHeightConfig.from_yaml(str(configs_dir / "cloud_height.yaml"))
        AlbedoEstimatorConfig.from_yaml(str(configs_dir / "albedo.yaml"))
        RefocusConfig.from_yaml(str(configs_dir / "refocus.yaml"))
        r2p = Refl2PropConfig.from_yaml(str(configs_dir / "refl2prop.yaml"))
        # model_path should default to the bundled model
        assert "model.pth" in r2p.model_path

    def test_init_refuses_existing_project(self, tmp_path):
        """Project.init() raises if project.yaml already exists."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "my_project"
        Project.init(str(project_dir), name="First")

        with pytest.raises(FileExistsError):
            Project.init(str(project_dir), name="Second")

    def test_init_name_defaults_to_basename(self, tmp_path):
        """Project.init() uses directory basename when no name given."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "uk_summer_2021"
        project = Project.init(str(project_dir))
        assert project.config.name == "uk_summer_2021"

    def test_load_existing_project(self, tmp_path):
        """Project.load() loads an existing project."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "my_project"
        Project.init(str(project_dir), name="Load Test")

        loaded = Project.load(str(project_dir))
        assert loaded.config.name == "Load Test"

    def test_load_nonexistent_raises(self, tmp_path):
        """Project.load() raises FileNotFoundError for missing project."""
        from clouds_decoded.project import Project

        with pytest.raises(FileNotFoundError):
            Project.load(str(tmp_path / "nonexistent"))

    def test_clone_copies_configs(self, tmp_path):
        """Project.init(clone_from=...) copies configs from source project."""
        from clouds_decoded.project import Project

        # Create source project and modify a config
        source_dir = tmp_path / "source"
        Project.init(str(source_dir), name="Source")
        config_path = source_dir / "configs" / "cloud_mask.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        data["threshold_value"] = 7777
        with open(config_path, 'w') as f:
            yaml.dump(data, f)

        # Clone into a new project
        clone_dir = tmp_path / "clone"
        clone = Project.init(str(clone_dir), clone_from=str(source_dir))

        # Name defaults to clone dir basename, not source name
        assert clone.config.name == "clone"

        # Config should have the modified value
        cloned_config = clone_dir / "configs" / "cloud_mask.yaml"
        with open(cloned_config) as f:
            cloned_data = yaml.safe_load(f)
        assert cloned_data["threshold_value"] == 7777

        # Scenes should be empty (not copied from source)
        assert len(clone.config.scenes) == 0

    def test_clone_inherits_pipeline(self, tmp_path):
        """Cloned project inherits pipeline type from source."""
        from clouds_decoded.project import Project

        source_dir = tmp_path / "source"
        Project.init(str(source_dir))

        clone_dir = tmp_path / "clone"
        clone = Project.init(str(clone_dir), clone_from=str(source_dir))
        assert clone.config.pipeline == "full-workflow"

    def test_clone_nonexistent_raises(self, tmp_path):
        """Cloning from nonexistent project raises FileNotFoundError."""
        from clouds_decoded.project import Project

        with pytest.raises(FileNotFoundError, match="Cannot clone"):
            Project.init(str(tmp_path / "new"), clone_from=str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Project.add_scene() and config hash
# ---------------------------------------------------------------------------

class TestProjectSceneManagement:
    """Tests for scene management and config hashing."""

    def test_add_scene(self, tmp_path):
        """add_scene() appends to scenes list."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.add_scene("/data/S2A_TEST.SAFE")

        assert len(project.config.scenes) == 1
        # Reload to verify persistence
        reloaded = Project.load(str(project_dir))
        assert len(reloaded.config.scenes) == 1

    def test_add_duplicate_scene_ignored(self, tmp_path):
        """add_scene() doesn't add duplicates."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.add_scene("/data/S2A_TEST.SAFE")
        project.add_scene("/data/S2A_TEST.SAFE")

        assert len(project.config.scenes) == 1

    def test_config_hash_changes_with_content(self, tmp_path):
        """_config_hash() changes when config content changes."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        hash1 = project._config_hash("cloud_mask")

        # Modify the config file
        config_path = project.configs_dir / "cloud_mask.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        data["threshold_value"] = 9999
        with open(config_path, 'w') as f:
            yaml.dump(data, f)

        hash2 = project._config_hash("cloud_mask")
        assert hash1 != hash2

    def test_config_hash_stable_for_same_content(self, tmp_path):
        """_config_hash() returns same value for unchanged config."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        hash1 = project._config_hash("cloud_height")
        hash2 = project._config_hash("cloud_height")
        assert hash1 == hash2

    def test_status_output(self, tmp_path):
        """status() returns a formatted string without crashing."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="Status Test")
        project.add_scene("/data/scene1.SAFE")
        output = project.status()
        assert "Status Test" in output
        assert "scene1" in output


# ---------------------------------------------------------------------------
# Provenance model
# ---------------------------------------------------------------------------

class TestProvenanceModel:
    """Tests for the Provenance pydantic model."""

    def test_provenance_creation(self):
        """Provenance model can be created with all fields."""
        from clouds_decoded.project import Provenance

        prov = Provenance(
            project_name="test",
            codebase_version="0.1.0",
            python_version="3.12.0",
            git_hash="abc123",
            timestamp="2025-01-01T00:00:00",
            scene_path="/data/scene.SAFE",
            product_id="S2B_MSIL1C_20250104T185019_N0511_R127_T09KVQ_20250104T220125.SAFE",
            pipeline="workflow",
            step_name="cloud_mask",
            step_config={"stride": 300},
        )
        assert prov.project_name == "test"
        assert prov.step_config["stride"] == 300
        assert prov.step_name == "cloud_mask"
        assert prov.python_version == "3.12.0"
        assert prov.product_id.startswith("S2B_MSIL1C")
        assert "github.com" in prov.repo_url

    def test_provenance_model_dump(self):
        """Provenance.model_dump() produces a plain dict suitable for metadata."""
        from clouds_decoded.project import Provenance

        prov = Provenance(
            project_name="test",
            codebase_version="0.1.0",
            step_name="cloud_height",
        )
        d = prov.model_dump()
        assert isinstance(d, dict)
        assert d["project_name"] == "test"
        assert d["repo_url"] == "https://github.com/asterisk-labs/clouds-decoded"
        assert d["step_name"] == "cloud_height"

    def test_pipeline_steps_constants(self):
        """PIPELINE_STEPS has correct step orders."""
        from clouds_decoded.project import PIPELINE_STEPS

        assert PIPELINE_STEPS["full-workflow"] == [
            "cloud_mask", "cloud_height", "albedo", "refocus", "cloud_properties"
        ]


# ---------------------------------------------------------------------------
# File provenance validation tests
# ---------------------------------------------------------------------------

class TestFileProvenanceValidation:
    """Tests for _validate_step_file() and _read_file_provenance()."""

    def _write_tif_with_provenance(self, filepath, provenance_dict):
        """Helper: write a minimal GeoTIFF with provenance in its metadata."""
        from clouds_decoded.data.base import GeoRasterData, Metadata

        data = np.random.rand(1, 10, 10).astype(np.float32)
        raster = GeoRasterData(
            data=data,
            transform=Affine.translation(0, 0) * Affine.scale(10, -10),
            crs=CRS.from_epsg(32633),
            metadata=Metadata(provenance=provenance_dict),
        )
        raster.write(str(filepath))

    def _write_tif_without_provenance(self, filepath):
        """Helper: write a minimal GeoTIFF with no provenance metadata."""
        from clouds_decoded.data.base import GeoRasterData, Metadata

        data = np.random.rand(1, 10, 10).astype(np.float32)
        raster = GeoRasterData(
            data=data,
            transform=Affine.translation(0, 0) * Affine.scale(10, -10),
            crs=CRS.from_epsg(32633),
            metadata=Metadata(),
        )
        raster.write(str(filepath))

    def test_read_file_provenance_present(self, tmp_path):
        """_read_file_provenance reads provenance from a TIF."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        tif_path = tmp_path / "test.tif"
        prov = {"project_name": "T", "step_config": {"stride": 300}}
        self._write_tif_with_provenance(tif_path, prov)

        result = project._read_file_provenance(tif_path)
        assert result is not None
        assert result["project_name"] == "T"
        assert result["step_config"]["stride"] == 300

    def test_read_file_provenance_absent(self, tmp_path):
        """_read_file_provenance returns None for files without provenance."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        tif_path = tmp_path / "no_prov.tif"
        self._write_tif_without_provenance(tif_path)

        result = project._read_file_provenance(tif_path)
        assert result is None

    def test_validate_matching_provenance(self, tmp_path):
        """_validate_step_file returns None when provenance matches."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        config_dict = {"method": "threshold", "threshold_value": 4000}
        prov = {"project_name": "T", "step_config": config_dict}
        self._write_tif_with_provenance(scene_out / "cloud_mask.tif", prov)

        error = project._validate_step_file("cloud_mask", scene_out, "/data/scene.SAFE", config_dict)
        assert error is None

    def test_validate_config_mismatch(self, tmp_path):
        """_validate_step_file detects config mismatch."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        file_config = {"method": "threshold", "threshold_value": 4000}
        prov = {"project_name": "T", "step_config": file_config}
        self._write_tif_with_provenance(scene_out / "cloud_mask.tif", prov)

        current_config = {"method": "threshold", "threshold_value": 5000}
        error = project._validate_step_file("cloud_mask", scene_out, "/data/scene.SAFE", current_config)
        assert error is not None
        assert "threshold_value" in error
        assert "mismatch" in error.lower()

    def test_validate_project_name_mismatch(self, tmp_path):
        """_validate_step_file detects project name mismatch."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="ProjectA")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        prov = {"project_name": "ProjectB", "step_config": {"x": 1}}
        self._write_tif_with_provenance(scene_out / "cloud_mask.tif", prov)

        error = project._validate_step_file("cloud_mask", scene_out, "/data/scene.SAFE", {"x": 1})
        assert error is not None
        assert "ProjectB" in error
        assert "ProjectA" in error

    def test_validate_no_provenance_in_file(self, tmp_path):
        """_validate_step_file flags files without provenance."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()
        self._write_tif_without_provenance(scene_out / "cloud_mask.tif")

        error = project._validate_step_file("cloud_mask", scene_out, "/data/scene.SAFE", {"x": 1})
        assert error is not None
        assert "no provenance" in error.lower()

    def test_validate_missing_file(self, tmp_path):
        """_validate_step_file flags missing output files."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        error = project._validate_step_file("cloud_mask", scene_out, "/data/scene.SAFE", {"x": 1})
        assert error is not None
        assert "missing" in error.lower()

    def test_validate_refocus_skipped(self, tmp_path):
        """_validate_step_file skips validation for refocus (no output file)."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        error = project._validate_step_file("refocus", scene_out, "/data/scene.SAFE", {"x": 1})
        assert error is None

    def test_validate_product_id_mismatch(self, tmp_path):
        """_validate_step_file detects product_id mismatch."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        prov = {
            "project_name": "T",
            "product_id": "S2A_MSIL1C_20210801T000000_DIFFERENT.SAFE",
            "step_config": {"x": 1},
        }
        self._write_tif_with_provenance(scene_out / "cloud_mask.tif", prov)

        error = project._validate_step_file(
            "cloud_mask", scene_out,
            "/data/S2B_MSIL1C_20250104T185019_ACTUAL.SAFE",
            {"x": 1},
        )
        assert error is not None
        assert "DIFFERENT" in error
