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

        original = AlbedoEstimatorConfig(method="idw", output_resolution=500)
        yaml_path = tmp_path / "albedo.yaml"
        original.to_yaml(str(yaml_path))

        loaded = AlbedoEstimatorConfig.from_yaml(str(yaml_path))
        assert loaded.method == "idw"
        assert loaded.output_resolution == 500

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
            created_at="2025-01-01T00:00:00",
        )

        yaml_path = tmp_path / "project.yaml"
        config.to_yaml(yaml_path)

        loaded = ProjectConfig.from_yaml(yaml_path)
        assert loaded.name == "Test Project"
        assert loaded.pipeline == "full-workflow"

    def test_project_config_ignores_legacy_scenes_key(self, tmp_path):
        """ProjectConfig silently ignores a legacy 'scenes' key in YAML."""
        from clouds_decoded.project import ProjectConfig

        # Write old-style YAML with a scenes list
        yaml_path = tmp_path / "project.yaml"
        import yaml as _yaml
        _yaml.dump(
            {"name": "Old", "pipeline": "full-workflow", "scenes": ["/data/s1.SAFE"]},
            open(yaml_path, "w"),
        )
        # Must not raise, scenes key is silently dropped
        loaded = ProjectConfig.from_yaml(yaml_path)
        assert loaded.name == "Old"
        assert not hasattr(loaded, "scenes")


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
        assert not (project_dir / "outputs").exists()  # output dir created on first use, not at init

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
        from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
        from clouds_decoded.modules.refocus.config import RefocusConfig
        from clouds_decoded.modules.refl2prop.config import Refl2PropConfig

        project_dir = tmp_path / "my_project"
        Project.init(str(project_dir), pipeline="full-workflow")

        configs_dir = project_dir / "configs"
        CloudMaskConfig.from_yaml(str(configs_dir / "cloud_mask.yaml"))
        from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
        CloudHeightEmulatorConfig.from_yaml(str(configs_dir / "cloud_height.yaml"))
        AlbedoEstimatorConfig.from_yaml(str(configs_dir / "albedo.yaml"))
        RefocusConfig.from_yaml(str(configs_dir / "refocus.yaml"))
        r2p = Refl2PropConfig.from_yaml(str(configs_dir / "refl2prop.yaml"))
        # model_path should default to the managed asset location
        assert r2p.model_path.endswith("refl2prop/default.pth")

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
        assert len(clone.db.get_all()) == 0

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
        """add_scene() registers scene in project.db."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.stage("/data/S2A_TEST.SAFE")

        assert len(project.db.get_all()) == 1
        # Reload to verify persistence
        reloaded = Project.load(str(project_dir))
        assert len(reloaded.db.get_all()) == 1

    def test_add_duplicate_scene_ignored(self, tmp_path):
        """add_scene() doesn't add duplicates."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.stage("/data/S2A_TEST.SAFE")
        project.stage("/data/S2A_TEST.SAFE")

        assert len(project.db.get_all()) == 1

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
        project.stage("/data/scene1.SAFE")
        output = project.status()
        assert "Status Test" in output
        assert "scene1" in output
        assert "Status" in output  # DB status column header


# ---------------------------------------------------------------------------
# SceneDB tests
# ---------------------------------------------------------------------------

class TestSceneDB:
    """Tests for the SceneDB SQLite-backed scene registry."""

    def test_stage_adds_to_db(self, tmp_path):
        """stage() inserts a new scene and returns True."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        inserted = db.stage("/data/S2A.SAFE", "S2A")
        assert inserted is True
        rows = db.get_all()
        assert len(rows) == 1
        assert rows[0]["scene_id"] == "S2A"
        assert rows[0]["status"] == "staged"

    def test_stage_deduplicates(self, tmp_path):
        """stage() returns False and does not duplicate on second call."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        inserted = db.stage("/data/S2A.SAFE", "S2A")
        assert inserted is False
        assert len(db.get_all()) == 1

    def test_stage_with_crop_window(self, tmp_path):
        """stage() with crop_window produces a different run_id than without."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        db.stage("/data/S2A.SAFE", "S2A", crop_window="0,0,512,512")
        rows = db.get_all()
        assert len(rows) == 2
        run_ids = {r["run_id"] for r in rows}
        assert _make_run_id("S2A", None) in run_ids
        assert _make_run_id("S2A", "0,0,512,512") in run_ids

    def test_stage_deduplicates_same_crop(self, tmp_path):
        """stage() with same (scene, crop) deduplicates on second call."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A", crop_window="0,0,512,512")
        inserted = db.stage("/data/S2A.SAFE", "S2A", crop_window="0,0,512,512")
        assert inserted is False
        assert len(db.get_all()) == 1

    def test_set_status_updates_row(self, tmp_path):
        """set_status() updates status using run_id; sets completed_at for done."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        db.set_status(run_id, "done")
        rows = db.get_all()
        assert rows[0]["status"] == "done"
        assert rows[0]["completed_at"] is not None

    def test_set_status_uses_run_id(self, tmp_path):
        """set_status() sets started_at for 'started', completed_at for others."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        db.set_status(run_id, "started")
        rows = db.get_all()
        assert rows[0]["status"] == "started"
        assert rows[0]["started_at"] is not None
        assert rows[0]["completed_at"] is None

    def test_set_status_stores_error(self, tmp_path):
        """set_status() stores error message when provided."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        db.set_status(run_id, "failed", error="Something went wrong")
        rows = db.get_all()
        assert rows[0]["status"] == "failed"
        assert rows[0]["error"] == "Something went wrong"

    def test_get_pending_returns_staged_and_failed(self, tmp_path):
        """get_pending() returns (path, crop_window) tuples for staged+failed runs."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/A.SAFE", "A")
        db.stage("/data/B.SAFE", "B")
        db.stage("/data/C.SAFE", "C")
        db.stage("/data/D.SAFE", "D")
        db.set_status(_make_run_id("B", None), "done")
        db.set_status(_make_run_id("C", None), "started")
        db.set_status(_make_run_id("D", None), "failed")

        pending = db.get_pending(force=False)
        paths = {p for p, _ in pending}
        assert paths == {"/data/A.SAFE", "/data/D.SAFE"}

    def test_get_pending_filters_by_crop_window(self, tmp_path):
        """get_pending(crop_window=None) returns only full-scene rows."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/A.SAFE", "A")
        db.stage("/data/A.SAFE", "A", crop_window="0,0,512,512")

        # Default: only full-scene (crop_window IS NULL)
        pending_full = db.get_pending(force=False)
        assert len(pending_full) == 1
        assert pending_full[0] == ("/data/A.SAFE", None)

        # With crop filter: only the crop row
        pending_crop = db.get_pending(force=False, crop_window="0,0,512,512")
        assert len(pending_crop) == 1
        assert pending_crop[0] == ("/data/A.SAFE", "0,0,512,512")

    def test_get_pending_force_returns_all(self, tmp_path):
        """get_pending(force=True) returns all full-scene rows regardless of status."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/A.SAFE", "A")
        db.stage("/data/B.SAFE", "B")
        db.set_status(_make_run_id("B", None), "done")

        pending = db.get_pending(force=True)
        assert len(pending) == 2

    def test_count_by_status(self, tmp_path):
        """count_by_status() returns correct counts per status."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/A.SAFE", "A")
        db.stage("/data/B.SAFE", "B")
        db.stage("/data/C.SAFE", "C")
        db.set_status(_make_run_id("B", None), "done")
        db.set_status(_make_run_id("C", None), "failed")

        counts = db.count_by_status()
        assert counts.get("staged") == 1
        assert counts.get("done") == 1
        assert counts.get("failed") == 1

    def test_scene_metadata_upsert(self, tmp_path):
        """upsert_scene_metadata() stores and retrieves scene-level metadata."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        db.upsert_scene_metadata("S2A_001", {
            "sensing_time": "2024-12-17T21:59:11",
            "satellite": "S2A",
            "tile_id": "T01KGB",
            "orbit_rel": 86,
            "lat_center": -0.5,
            "lon_center": -179.0,
        })
        # Verify via the SceneDB's own connection
        with db._conn() as conn:
            result = conn.execute(
                "SELECT satellite, tile_id, orbit_rel FROM scene_metadata WHERE scene_id='S2A_001'"
            )
            cols = [d[0] for d in result.description]
            row = dict(zip(cols, result.fetchone()))
        assert row["satellite"] == "S2A"
        assert row["tile_id"] == "T01KGB"
        assert row["orbit_rel"] == 86

    def test_write_stats_creates_table_and_columns(self, tmp_path):
        """write_stats() creates the table and columns dynamically."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        db.write_stats("abc123", "stats_cloud_mask", {
            "clear_frac": 0.6, "n_pixels": 1000
        })
        with db._conn() as conn:
            result = conn.execute(
                "SELECT clear_frac, n_pixels FROM stats_cloud_mask WHERE run_id='abc123'"
            )
            cols = [d[0] for d in result.description]
            row = dict(zip(cols, result.fetchone()))
        assert row is not None
        assert abs(row["clear_frac"] - 0.6) < 1e-6
        assert row["n_pixels"] == 1000

    def test_write_stats_adds_new_column(self, tmp_path):
        """write_stats() adds missing columns via ALTER TABLE."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        db.write_stats("run1", "stats_albedo", {"b02_mean": 0.1})
        db.write_stats("run2", "stats_albedo", {"b02_mean": 0.2, "b03_mean": 0.3})
        with db._conn() as conn:
            result = conn.execute(
                "SELECT b03_mean FROM stats_albedo WHERE run_id='run2'"
            )
            row2 = result.fetchone()
        assert abs(row2[0] - 0.3) < 1e-6

    def test_has_stats(self, tmp_path):
        """has_stats() returns True only after write_stats() has been called."""
        from clouds_decoded.project import SceneDB

        db = SceneDB(tmp_path / "project.db")
        assert db.has_stats("run1", "stats_cloud_mask") is False
        db.write_stats("run1", "stats_cloud_mask", {"clear_frac": 0.5})
        assert db.has_stats("run1", "stats_cloud_mask") is True
        assert db.has_stats("run2", "stats_cloud_mask") is False

    def test_project_init_creates_project_db(self, tmp_path):
        """Project.init() creates project.db eagerly."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        Project.init(str(project_dir), name="T")
        assert (project_dir / "project.db").exists()

    def test_project_stage_method(self, tmp_path):
        """Project.stage() registers scenes in the DB."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.stage("/data/S2A_001.SAFE", "/data/S2A_002.SAFE")
        rows = project.db.get_all()
        assert len(rows) == 2
        scene_ids = {r["scene_id"] for r in rows}
        assert "S2A_001" in scene_ids
        assert "S2A_002" in scene_ids

    def test_run_reads_staged_from_db(self, tmp_path, monkeypatch):
        """run() with no scenes argument processes staged scenes from DB."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.stage("/data/S2A_001.SAFE")

        processed = []

        def mock_run_serial(scene_list, **kwargs):
            processed.extend(scene_list)

        monkeypatch.setattr(project, "_run_serial", mock_run_serial)
        project.run()
        assert len(processed) == 1
        # scene_list is now (path, crop_window) tuples
        scene_path, crop_window = processed[0]
        assert scene_path.endswith("S2A_001.SAFE")
        assert crop_window is None

    def test_run_force_includes_done_scenes(self, tmp_path, monkeypatch):
        """run(force=True) with no scenes processes all scenes from DB."""
        from clouds_decoded.project import Project, _make_run_id

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")
        project.stage("/data/S2A_001.SAFE")
        run_id = _make_run_id("S2A_001", None)
        # Mark done with the current config hash so the integrity check passes
        project.db.set_status(run_id, "done",
                              pipeline_config_hash=project._pipeline_config_hash())

        processed = []

        def mock_run_serial(scene_list, **kwargs):
            processed.extend(scene_list)

        monkeypatch.setattr(project, "_run_serial", mock_run_serial)
        # Without force, done scene is skipped (integrity check passes)
        project.run()
        assert len(processed) == 0
        # With force, all scenes are included
        project.run(force=True)
        assert len(processed) == 1


class TestConfigIntegrity:
    """Tests for pipeline_config_hash storage and integrity checking in run()."""

    # ------------------------------------------------------------------
    # _pipeline_config_hash
    # ------------------------------------------------------------------

    def test_pipeline_config_hash_stable(self, tmp_path):
        """_pipeline_config_hash() returns the same value for unchanged configs."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        assert project._pipeline_config_hash() == project._pipeline_config_hash()

    def test_pipeline_config_hash_changes_with_config(self, tmp_path):
        """_pipeline_config_hash() changes when any config file is modified."""
        import yaml
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        hash1 = project._pipeline_config_hash()

        config_path = project.configs_dir / "cloud_mask.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        data["__test_sentinel__"] = 12345
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        hash2 = project._pipeline_config_hash()
        assert hash1 != hash2

    # ------------------------------------------------------------------
    # SceneDB.get_stale_done_runs / reset_to_staged
    # ------------------------------------------------------------------

    def test_get_stale_done_runs_finds_mismatch(self, tmp_path):
        """get_stale_done_runs() returns done rows with a different hash."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        db.set_status(run_id, "done", pipeline_config_hash="oldhash12345678")

        stale = db.get_stale_done_runs("newhash12345678")
        assert len(stale) == 1
        assert stale[0]["scene_id"] == "S2A"

    def test_get_stale_done_runs_null_hash_is_stale(self, tmp_path):
        """get_stale_done_runs() treats a NULL pipeline_config_hash as stale."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        # Mark done without supplying a hash (simulates pre-feature run)
        db.set_status(run_id, "done")

        stale = db.get_stale_done_runs("anyhash00000000")
        assert len(stale) == 1

    def test_get_stale_done_runs_excludes_fresh(self, tmp_path):
        """get_stale_done_runs() does not return a run whose hash matches."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        db.set_status(run_id, "done", pipeline_config_hash="currenthash1234")

        stale = db.get_stale_done_runs("currenthash1234")
        assert stale == []

    def test_get_stale_done_runs_ignores_non_done(self, tmp_path):
        """get_stale_done_runs() ignores staged/failed/started rows."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        for scene_id in ["A", "B", "C"]:
            db.stage(f"/data/{scene_id}.SAFE", scene_id)
        db.set_status(_make_run_id("B", None), "failed")
        db.set_status(_make_run_id("C", None), "started")

        stale = db.get_stale_done_runs("anyhash")
        assert stale == []

    def test_reset_to_staged(self, tmp_path):
        """reset_to_staged() resets specified runs to 'staged' and clears hash."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        db.stage("/data/S2B.SAFE", "S2B")
        rid_a = _make_run_id("S2A", None)
        rid_b = _make_run_id("S2B", None)
        db.set_status(rid_a, "done", pipeline_config_hash="oldhash12345678")
        db.set_status(rid_b, "done", pipeline_config_hash="oldhash12345678")

        db.reset_to_staged([rid_a])

        rows = {r["scene_id"]: r for r in db.get_all()}
        assert rows["S2A"]["status"] == "staged"
        assert rows["S2A"]["pipeline_config_hash"] is None
        assert rows["S2A"]["completed_at"] is None
        assert rows["S2B"]["status"] == "done"  # untouched

    def test_set_status_stores_pipeline_config_hash(self, tmp_path):
        """set_status('done', pipeline_config_hash=...) writes the hash column."""
        from clouds_decoded.project import SceneDB, _make_run_id

        db = SceneDB(tmp_path / "project.db")
        db.stage("/data/S2A.SAFE", "S2A")
        run_id = _make_run_id("S2A", None)
        db.set_status(run_id, "done", pipeline_config_hash="abc12345678abcde")

        with db._conn() as conn:
            row = conn.execute(
                "SELECT pipeline_config_hash FROM runs WHERE run_id=?", [run_id]
            ).fetchone()
        assert row[0] == "abc12345678abcde"

    # ------------------------------------------------------------------
    # Project.run() integrity check
    # ------------------------------------------------------------------

    def test_run_raises_on_stale_done(self, tmp_path, monkeypatch):
        """run() raises RuntimeError when done scenes have a stale config hash."""
        import pytest
        from clouds_decoded.project import Project, _make_run_id

        project = Project.init(str(tmp_path / "proj"), name="T")
        project.stage("/data/S2A_001.SAFE")
        run_id = _make_run_id("S2A_001", None)
        project.db.set_status(run_id, "done", pipeline_config_hash="stale000deadbeef")

        monkeypatch.setattr(project, "_run_serial", lambda *a, **kw: None)

        with pytest.raises(RuntimeError, match="configs have changed"):
            project.run()

    def test_run_force_overwrite_resets_stale(self, tmp_path, monkeypatch):
        """run(force_overwrite=True) resets stale done scenes to 'staged'."""
        from clouds_decoded.project import Project, _make_run_id

        project = Project.init(str(tmp_path / "proj"), name="T")
        project.stage("/data/S2A_001.SAFE")
        run_id = _make_run_id("S2A_001", None)
        project.db.set_status(run_id, "done", pipeline_config_hash="stale000deadbeef")

        processed = []
        monkeypatch.setattr(project, "_run_serial",
                            lambda scene_list, **kw: processed.extend(scene_list))

        project.run(force_overwrite=True)

        # Scene should have been reset to staged and re-queued
        assert len(processed) == 1
        row = next(r for r in project.db.get_all() if r["run_id"] == run_id)
        # After re-queuing it will have been passed to _run_serial; DB update
        # is done by _run_serial which we've mocked, so status stays 'staged'.
        assert row["status"] == "staged"

    def test_run_ignore_integrity_proceeds(self, tmp_path, monkeypatch):
        """run(ignore_integrity=True) skips the check and proceeds without error."""
        from clouds_decoded.project import Project, _make_run_id

        project = Project.init(str(tmp_path / "proj"), name="T")
        project.stage("/data/S2A_001.SAFE")
        run_id = _make_run_id("S2A_001", None)
        project.db.set_status(run_id, "done", pipeline_config_hash="stale000deadbeef")

        processed = []
        monkeypatch.setattr(project, "_run_serial",
                            lambda scene_list, **kw: processed.extend(scene_list))

        # Should not raise; stale done scene stays done, no new scenes staged
        project.run(ignore_integrity=True)
        assert len(processed) == 0  # done scene not re-queued

    def test_run_stores_hash_on_completion(self, tmp_path, monkeypatch):
        """run() passes pipeline_config_hash to _run_serial for storage on done."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        project.stage("/data/S2A_001.SAFE")

        captured: dict = {}

        def mock_run_serial(scene_list, **kwargs):
            captured["pipeline_config_hash"] = kwargs.get("pipeline_config_hash")

        monkeypatch.setattr(project, "_run_serial", mock_run_serial)
        project.run()

        expected_hash = project._pipeline_config_hash()
        assert captured["pipeline_config_hash"] == expected_hash

    def test_run_force_bypasses_integrity_check(self, tmp_path, monkeypatch):
        """run(force=True) bypasses the integrity check (re-runs everything anyway)."""
        from clouds_decoded.project import Project, _make_run_id

        project = Project.init(str(tmp_path / "proj"), name="T")
        project.stage("/data/S2A_001.SAFE")
        run_id = _make_run_id("S2A_001", None)
        project.db.set_status(run_id, "done", pipeline_config_hash="stale000deadbeef")

        processed = []
        monkeypatch.setattr(project, "_run_serial",
                            lambda scene_list, **kw: processed.extend(scene_list))

        # force=True skips integrity check entirely and re-runs everything
        project.run(force=True)
        assert len(processed) == 1


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

    def test_validate_crop_window_mismatch(self, tmp_path):
        """_validate_step_file detects crop_window mismatch."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        # File was produced with a crop
        prov = {
            "project_name": "T",
            "step_config": {"x": 1},
            "crop_window": "100,200,512,512",
        }
        self._write_tif_with_provenance(scene_out / "cloud_mask.tif", prov)

        # Current run uses no crop
        error = project._validate_step_file(
            "cloud_mask", scene_out, "/data/scene.SAFE", {"x": 1}, crop_window=None
        )
        assert error is not None
        assert "crop_window" in error

    def test_validate_crop_window_match(self, tmp_path):
        """_validate_step_file passes when crop_window matches."""
        from clouds_decoded.project import Project

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="T")

        scene_out = tmp_path / "scene_out"
        scene_out.mkdir()

        prov = {
            "project_name": "T",
            "step_config": {"x": 1},
            "crop_window": "100,200,512,512",
        }
        self._write_tif_with_provenance(scene_out / "cloud_mask.tif", prov)

        error = project._validate_step_file(
            "cloud_mask", scene_out, "/data/scene.SAFE", {"x": 1},
            crop_window="100,200,512,512",
        )
        assert error is None


# ---------------------------------------------------------------------------
# Crop window directory structure tests
# ---------------------------------------------------------------------------

class TestCropWindowDirectoryStructure:
    """Tests for crop-window-separated output directories."""

    def test_scene_output_dir_no_crop(self, tmp_path):
        """_scene_output_dir without crop returns scenes/<scene_id>/."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"))
        out = project._scene_output_dir("S2A_TEST")
        assert out == project.output_dir / "S2A_TEST"

    def test_scene_output_dir_with_crop(self, tmp_path):
        """_scene_output_dir with crop returns scenes/<scene_id>/crops/<crop_id>/."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"))
        out = project._scene_output_dir("S2A_TEST", crop_window="100,200,512,512")
        assert out == project.output_dir / "S2A_TEST" / "crops" / "100_200_512_512"

    def test_different_crops_different_dirs(self, tmp_path):
        """Two different crop windows produce different output directories."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"))
        out_a = project._scene_output_dir("S2A_TEST", crop_window="0,0,512,512")
        out_b = project._scene_output_dir("S2A_TEST", crop_window="512,512,512,512")
        assert out_a != out_b

    def test_full_scene_and_crop_manifests_coexist(self, tmp_path):
        """Full-scene and crop manifests coexist independently for the same scene."""
        from clouds_decoded.project import Project, SceneManifest

        project = Project.init(str(tmp_path / "proj"))

        full = SceneManifest(scene_id="S2A_TEST", scene_path="/data/test.SAFE")
        project._save_manifest("S2A_TEST", full)

        crop = SceneManifest(
            scene_id="S2A_TEST", scene_path="/data/test.SAFE", crop_window="0,0,256,256"
        )
        project._save_manifest("S2A_TEST", crop, crop_window="0,0,256,256")

        full_path = project.output_dir / "S2A_TEST" / "manifest.json"
        crop_path = project.output_dir / "S2A_TEST" / "crops" / "0_0_256_256" / "manifest.json"
        assert full_path.exists()
        assert crop_path.exists()

        loaded_full = project._load_manifest("S2A_TEST", "/data/test.SAFE")
        loaded_crop = project._load_manifest("S2A_TEST", "/data/test.SAFE", crop_window="0,0,256,256")
        assert loaded_full.crop_window is None
        assert loaded_crop.crop_window == "0,0,256,256"

    def test_status_shows_crop_rows(self, tmp_path):
        """status() lists crop subdirectory runs beneath the full-scene row."""
        from clouds_decoded.project import Project, SceneManifest, StepResult

        project = Project.init(str(tmp_path / "proj"), name="CropStatus")
        project.stage("/data/scene1.SAFE")

        crop = SceneManifest(
            scene_id="scene1",
            scene_path="/data/scene1.SAFE",
            crop_window="100,200,512,512",
            steps={"cloud_mask": StepResult(status="completed", config_hash="abc")},
        )
        project._save_manifest("scene1", crop, crop_window="100,200,512,512")

        output = project.status()
        assert "crop:100,200,512,512" in output

    def test_manifest_stores_crop_window(self, tmp_path):
        """SceneManifest stores and round-trips crop_window."""
        from clouds_decoded.project import SceneManifest

        manifest = SceneManifest(
            scene_id="test",
            scene_path="/data/test.SAFE",
            crop_window="100,200,512,512",
        )
        path = tmp_path / "manifest.json"
        manifest.to_json(path)
        loaded = SceneManifest.from_json(path)
        assert loaded.crop_window == "100,200,512,512"

    def test_manifest_crop_window_defaults_none(self):
        """SceneManifest.crop_window defaults to None for full-scene runs."""
        from clouds_decoded.project import SceneManifest

        manifest = SceneManifest(scene_id="test", scene_path="/data/test.SAFE")
        assert manifest.crop_window is None

    def test_provenance_stores_crop_window(self):
        """Provenance model stores crop_window."""
        from clouds_decoded.project import Provenance

        prov = Provenance(project_name="test", step_name="cloud_mask", crop_window="0,0,256,256")
        assert prov.crop_window == "0,0,256,256"
        assert prov.model_dump()["crop_window"] == "0,0,256,256"


# ---------------------------------------------------------------------------
# Parallel processing and in-memory data flow tests
# ---------------------------------------------------------------------------

def _make_mock_run_step(scene_out_root, steps_with_files=None):
    """Return a fake _run_step that touches dummy output files without processing."""
    import numpy as np
    from unittest.mock import MagicMock
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from clouds_decoded.data.base import GeoRasterData, Metadata

    if steps_with_files is None:
        steps_with_files = {"cloud_mask", "cloud_height", "albedo", "cloud_properties"}

    def _write_dummy_tif(path):
        """Write a real GeoTIFF so is_step_complete can find the file."""
        data = np.zeros((1, 4, 4), dtype=np.float32)
        raster = GeoRasterData(
            data=data,
            transform=Affine.scale(10, -10),
            crs=CRS.from_epsg(32633),
            metadata=Metadata(),
        )
        raster.write(str(path))

    def fake_run_step(self_proj, step, ctx, processors=None):
        mock_data = MagicMock()
        if step.name in steps_with_files:
            fname = step.output_file
            if fname:
                out = ctx.scene_out / fname
                _write_dummy_tif(out)
                return str(out), mock_data
        return None, mock_data  # refocus

    return fake_run_step


class TestNoReloadOnFreshRun:
    """Verify that intermediate results are passed in-memory, not reloaded from disk."""

    def test_from_file_not_called_during_fresh_run(self, tmp_path, monkeypatch):
        """CloudMaskData.from_file() must not be called during a fresh pipeline run."""
        from unittest.mock import MagicMock, patch
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")

        # Patch _run_step and scene loading so no real processing happens
        fake_step = _make_mock_run_step(tmp_path)
        monkeypatch.setattr(Project, "_run_step", fake_step)
        monkeypatch.setattr(
            "clouds_decoded.data.sentinel.Sentinel2Scene.read",
            lambda self, *a, **kw: None,
        )

        with patch("clouds_decoded.data.cloud_mask.CloudMaskData.from_file") as mock_ff:
            project.run(scenes=["/fake/S2A_scene.SAFE"], run_stats=False)

        mock_ff.assert_not_called()

    def test_run_step_returns_tuple(self, tmp_path, monkeypatch):
        """_process_one_scene unpacks (output_path, data) from _run_step."""
        from unittest.mock import MagicMock
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")

        captured = {}

        def fake_step(self_proj, step, ctx, processors=None):
            fname = step.output_file
            mock_data = MagicMock(name=f"{step.name}_data")
            captured[step.name] = mock_data
            if fname:
                out = ctx.scene_out / fname
                out.touch()
                return str(out), mock_data
            return None, mock_data

        monkeypatch.setattr(Project, "_run_step", fake_step)
        monkeypatch.setattr(
            "clouds_decoded.data.sentinel.Sentinel2Scene.read",
            lambda self, *a, **kw: None,
        )

        # Should complete without KeyError or AttributeError
        project.run(scenes=["/fake/S2A_scene.SAFE"])
        assert "cloud_mask" in captured


class TestParallelWorkerCount:
    """Verify thread-pipeline parallelism behaviour."""

    def test_multi_reader_completes_gracefully_with_invalid_scenes(self, tmp_path):
        """Multiple reader workers: run() must complete without raising when all scenes fail."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")

        scenes = [f"/nonexistent/S2A_MSIL1C_2023010{i}T100000_N0509_R022_T31UFU_2023010{i}T100000.SAFE"
                  for i in range(1, 5)]
        for s in scenes:
            project.stage(s)

        # max_workers=2 shorthand: sets all stages to 2 workers.
        # All scenes are invalid — readers will fail and mark ctx.failed.
        project.run(parallel=True, max_workers=2, force=True)  # must not raise

    def test_processors_created_once_per_step_at_startup(self, tmp_path, monkeypatch):
        """_create_processor_for_step is called once per step at run() startup, not per scene."""
        from unittest.mock import MagicMock
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        call_count = []

        def mock_create_for_step(self_proj, step, device=None):
            call_count.append(step)
            m = MagicMock()
            m.process.return_value = MagicMock()
            return {step: m, "_cloud_mask_postprocessor": m}

        monkeypatch.setattr(Project, "_create_processor_for_step", mock_create_for_step)

        def fake_prepare(self_proj, ctx):
            # Mark all steps done so no step workers run — we only care about creation count
            ctx.first_step_idx = len(self_proj.steps)
            ctx.steps_to_run = list(self_proj.steps)
            ctx.scene_out = self_proj.output_dir / self_proj._scene_id(ctx.scene_path)
            ctx.scene_out.mkdir(parents=True, exist_ok=True)
            from clouds_decoded.project import SceneManifest
            ctx.manifest = SceneManifest(
                scene_id=self_proj._scene_id(ctx.scene_path),
                scene_path=ctx.scene_path,
            )

        monkeypatch.setattr(Project, "_prepare_scene_context", fake_prepare)

        n_scenes = 3
        for i in range(n_scenes):
            project.stage(f"/fake/S2A_scene{i}.SAFE")

        project.run(parallel=True)

        n_steps = len(project.steps)
        # With parallel=True and 1 worker per step (default), _create_processor_for_step is
        # called exactly once per step at startup regardless of how many scenes there are.
        assert len(call_count) == n_steps, (
            f"Expected {n_steps} calls (one per step), got {len(call_count)}: {call_count}"
        )
        assert set(call_count) == set(project.steps)


class TestErrorIsolation:
    """One failing scene must not prevent other scenes from completing."""

    def test_failed_scene_does_not_block_others(self, tmp_path, monkeypatch):
        """A failing reader must not block subsequent scenes from completing."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        completed = []

        def fake_prepare(self_proj, ctx):
            if "scene0" in ctx.scene_path:
                raise RuntimeError("Simulated failure")
            # Mark scene as fully complete so no steps run
            ctx.first_step_idx = len(self_proj.steps)
            ctx.steps_to_run = list(self_proj.steps)
            ctx.scene_out = self_proj.output_dir / self_proj._scene_id(ctx.scene_path)
            ctx.scene_out.mkdir(parents=True, exist_ok=True)
            from clouds_decoded.project import SceneManifest
            ctx.manifest = SceneManifest(
                scene_id=self_proj._scene_id(ctx.scene_path),
                scene_path=ctx.scene_path,
            )
            completed.append(ctx.scene_path)

        monkeypatch.setattr(Project, "_prepare_scene_context", fake_prepare)

        scenes = ["/fake/S2A_scene0.SAFE", "/fake/S2A_scene1.SAFE"]
        for s in scenes:
            project.stage(s)

        project.run()

        # scene1 must have completed despite scene0 failing
        assert any("scene1" in p for p in completed)

    def test_failed_scene_with_multiple_readers(self, tmp_path):
        """Multiple reader workers: one failing scene must not prevent others from reporting."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")

        scenes = [
            "/nonexistent/S2A_MSIL1C_20230101T100000_N0509_R022_T31UFU_20230101T100000.SAFE",
            "/nonexistent/S2A_MSIL1C_20230102T100000_N0509_R022_T31UFU_20230102T100000.SAFE",
        ]
        for s in scenes:
            project.stage(s)

        # Must complete without raising even though both scenes are invalid.
        project.run(parallel=True, parallelism={"reader": 2}, force=True)


class TestGitHashCaching:
    """git hash should be computed once per run(), not once per step."""

    def test_get_git_hash_called_once(self, tmp_path, monkeypatch):
        """_get_git_hash is called exactly once per run(), regardless of scene count."""
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        call_count = []

        def fake_git_hash(self_proj):
            call_count.append(1)
            return "abc123"

        def fake_prepare(self_proj, ctx):
            ctx.first_step_idx = len(self_proj.steps)
            ctx.steps_to_run = list(self_proj.steps)
            ctx.scene_out = self_proj.output_dir / self_proj._scene_id(ctx.scene_path)
            ctx.scene_out.mkdir(parents=True, exist_ok=True)
            from clouds_decoded.project import SceneManifest
            ctx.manifest = SceneManifest(
                scene_id=self_proj._scene_id(ctx.scene_path),
                scene_path=ctx.scene_path,
            )

        monkeypatch.setattr(Project, "_get_git_hash", fake_git_hash)
        monkeypatch.setattr(Project, "_prepare_scene_context", fake_prepare)
        # Prevent actual processor creation
        from unittest.mock import MagicMock
        monkeypatch.setattr(
            Project, "_create_processor_for_step",
            lambda s, step, device=None: {step: MagicMock(), "_cloud_mask_postprocessor": MagicMock()},
        )

        for i in range(3):
            project.stage(f"/fake/S2A_scene{i}.SAFE")

        project.run()
        assert len(call_count) == 1


class TestCreateProcessors:
    """Unit tests for _create_processors()."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        """_create_processors() returns a key for each step plus postprocessor helper."""
        from clouds_decoded.project import Project
        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig

        project = Project.init(str(tmp_path / "proj"), name="T")
        # Override cloud_mask config to use threshold (no model weights needed)
        CloudMaskConfig(method="threshold").to_yaml(
            str(project.configs_dir / "cloud_mask.yaml")
        )

        procs = project._create_processors()

        # All pipeline steps should be present
        for step in project.steps:
            assert step in procs, f"Missing processor for step '{step}'"
        # Postprocessor helper
        assert "_cloud_mask_postprocessor" in procs

    def test_threshold_mask_uses_correct_processor_type(self, tmp_path):
        """Threshold cloud-mask config produces ThresholdCloudMaskProcessor."""
        from clouds_decoded.project import Project
        from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
        from clouds_decoded.modules.cloud_mask.processor import (
            ThresholdCloudMaskProcessor, CloudMaskProcessor,
        )

        project = Project.init(str(tmp_path / "proj"), name="T")
        CloudMaskConfig(method="threshold").to_yaml(
            str(project.configs_dir / "cloud_mask.yaml")
        )

        procs = project._create_processors()

        assert isinstance(procs["cloud_mask"], ThresholdCloudMaskProcessor)
        # Postprocessor is always a CloudMaskProcessor
        assert isinstance(procs["_cloud_mask_postprocessor"], CloudMaskProcessor)
