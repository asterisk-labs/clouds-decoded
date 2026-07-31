"""Tests for the multitemporal albedo extension and its orchestrator hooks."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from clouds_decoded.extensions.multitemporal_albedo.config import (
    MultitemporalAlbedoParams,
)
from clouds_decoded.extensions.multitemporal_albedo.model import (
    fit_cluster_model,
    predict_reflectance,
)
from clouds_decoded.extensions.multitemporal_albedo.validation import (
    TimeSeriesValidationError,
    validate_time_series,
)


# ---------------------------------------------------------------------------
# Synthetic stack helpers
# ---------------------------------------------------------------------------

def make_synthetic_stack(T=100, B=3, H=16, W=16, seed=0):
    """Piecewise-constant surface classes with seasonal cycles + noise."""
    rng = np.random.default_rng(seed)
    ords = np.sort(rng.uniform(0, 730, T))
    doy = ords % 365
    year = (2022 + ords // 365).astype(np.int16)
    cls = rng.integers(0, 3, (H, W))
    base = np.array([[.1, .15, .2], [.3, .28, .26], [.05, .4, .1]])[:, :B]
    amp = np.array([.05, .02, .08])
    off = rng.normal(0, .01, (H, W, B))
    refl = np.empty((T, B, H, W), np.float16)
    truth = np.empty((T, B, H, W), np.float32)
    for t in range(T):
        seas = 1 + amp[cls][None] * np.sin(2 * np.pi * doy[t] / 365)
        truth[t] = (base[cls].transpose(2, 0, 1)
                    + off.transpose(2, 0, 1)) * seas
        refl[t] = truth[t] + rng.normal(0, .005, (B, H, W))
    clear = rng.random((T, H, W)) < 0.4
    return {
        "refl": refl, "clear": clear, "truth": truth,
        "times": (ords - ords.min()).astype(np.float64),
        "ord0": float(ords.min()), "doy": doy.astype(np.float64),
        "year": year, "cls": cls,
    }


def default_params(**kw):
    kw.setdefault("n_clusters", 5)
    kw.setdefault("min_clear_obs", 10)
    kw.setdefault("n_kmeans_iter", 15)
    return MultitemporalAlbedoParams(**kw)


def fake_scene_ids(n, tile="T37VCC", start=datetime(2022, 3, 1),
                   step_days=8):
    ids = []
    for i in range(n):
        dt = start + timedelta(days=i * step_days)
        ids.append(
            f"S2A_MSIL1C_{dt:%Y%m%dT%H%M%S}_N0510_R064_{tile}_"
            f"{dt:%Y%m%dT%H%M%S}")
    return ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestClusterModel:
    def test_fit_and_predict_recovers_truth(self):
        stack = make_synthetic_stack()
        params = default_params()
        model = fit_cluster_model(stack["refl"], stack["clear"],
                                  stack["times"], stack["doy"],
                                  stack["year"], params)
        t = 50
        pred = predict_reflectance(model, float(stack["times"][t]),
                                   float(stack["doy"][t]),
                                   int(stack["year"][t]))
        m = np.isfinite(pred)
        assert m.any()
        mae = np.abs(pred - stack["truth"][t])[m].mean()
        assert mae < 0.01

    def test_fit_deterministic(self):
        stack = make_synthetic_stack()
        params = default_params()
        m1 = fit_cluster_model(stack["refl"], stack["clear"], stack["times"],
                               stack["doy"], stack["year"], params)
        m2 = fit_cluster_model(stack["refl"], stack["clear"], stack["times"],
                               stack["doy"], stack["year"], params)
        assert (m1["assign"] == m2["assign"]).all()
        assert np.allclose(m1["offset_v"], m2["offset_v"])

    def test_sparse_pixels_excluded(self):
        stack = make_synthetic_stack()
        # A pixel with almost no clear obs must be NaN in predictions.
        stack["clear"][:, 0, 0] = False
        stack["clear"][:2, 0, 0] = True
        params = default_params()
        model = fit_cluster_model(stack["refl"], stack["clear"],
                                  stack["times"], stack["doy"],
                                  stack["year"], params)
        pred = predict_reflectance(model, 10.0, 100.0, 2022)
        assert np.isnan(pred[:, 0, 0]).all()

    def test_hierarchical_clustering_runs(self):
        stack = make_synthetic_stack()
        params = default_params(clustering="hierarchical", n_clusters=3,
                                max_clusters=12,
                                split_mse_threshold=1e-6,
                                min_postsplit_obs=1)
        model = fit_cluster_model(stack["refl"], stack["clear"],
                                  stack["times"], stack["doy"],
                                  stack["year"], params)
        assert int(model["K"]) >= 3
        pred = predict_reflectance(model, 10.0, 100.0, 2022)
        assert np.isfinite(pred).any()

    def test_too_few_valid_pixels_raises(self):
        stack = make_synthetic_stack(H=4, W=4)
        stack["clear"][:] = False
        params = default_params()
        with pytest.raises(ValueError, match="clear"):
            fit_cluster_model(stack["refl"], stack["clear"], stack["times"],
                              stack["doy"], stack["year"], params)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestAlbedoConfigWiring:
    def test_multitemporal_method_fills_default_params(self):
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        cfg = AlbedoEstimatorConfig(method="multitemporal")
        assert cfg.multitemporal is not None
        assert cfg.multitemporal.n_clusters == 100

    def test_params_without_method_rejected(self):
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        with pytest.raises(ValueError, match="multitemporal"):
            AlbedoEstimatorConfig(
                method="idw",
                multitemporal=MultitemporalAlbedoParams())

    def test_params_change_config_hash(self, tmp_path):
        """Fit hyperparameters must be part of the albedo step's semantic
        dump so pre-populated outputs are invalidated on change."""
        from clouds_decoded.config import BaseProcessorConfig
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        a = BaseProcessorConfig._semantic_dump(
            AlbedoEstimatorConfig(method="multitemporal"))
        b = BaseProcessorConfig._semantic_dump(
            AlbedoEstimatorConfig(
                method="multitemporal",
                multitemporal=MultitemporalAlbedoParams(n_clusters=50)))
        assert a != b

    def test_yaml_roundtrip(self, tmp_path):
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        cfg = AlbedoEstimatorConfig(
            method="multitemporal",
            multitemporal=MultitemporalAlbedoParams(n_clusters=42))
        path = tmp_path / "albedo.yaml"
        cfg.to_yaml(path)
        loaded = AlbedoEstimatorConfig.from_yaml(path)
        assert loaded.method == "multitemporal"
        assert loaded.multitemporal.n_clusters == 42

    def test_processor_refuses_multitemporal(self):
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        from clouds_decoded.modules.albedo_estimator.processor import (
            AlbedoEstimator,
        )
        proc = AlbedoEstimator(AlbedoEstimatorConfig(method="multitemporal"))
        with pytest.raises(RuntimeError, match="multitemporal"):
            proc.process(object())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestTimeSeriesValidation:
    def _rows(self, sids):
        return [(f"/data/{s}.SAFE", None) for s in sids]

    def test_valid_series_passes(self):
        rows = self._rows(fake_scene_ids(80))
        summary = validate_time_series(rows, default_params(min_scenes=50))
        assert summary["tile"] == "T37VCC"
        assert summary["n_scenes"] == 80

    def test_empty_rejected(self):
        with pytest.raises(TimeSeriesValidationError, match="No scenes"):
            validate_time_series([], default_params())

    def test_crop_window_rejected(self):
        rows = [(p, "0,0,100,100") for p, _ in
                self._rows(fake_scene_ids(80))]
        with pytest.raises(TimeSeriesValidationError, match="crop"):
            validate_time_series(rows, default_params())

    def test_mixed_tiles_rejected(self):
        rows = self._rows(fake_scene_ids(40)
                          + fake_scene_ids(40, tile="T37UCB"))
        with pytest.raises(TimeSeriesValidationError, match="single-tile"):
            validate_time_series(rows, default_params(min_scenes=10))

    def test_too_few_scenes_rejected(self):
        rows = self._rows(fake_scene_ids(10))
        with pytest.raises(TimeSeriesValidationError, match="min_scenes"):
            validate_time_series(rows, default_params(min_scenes=50))

    def test_short_span_rejected(self):
        rows = self._rows(fake_scene_ids(60, step_days=2))
        with pytest.raises(TimeSeriesValidationError, match="year"):
            validate_time_series(rows, default_params(min_scenes=10))


# ---------------------------------------------------------------------------
# Orchestrator: only_steps + reordered recipe
# ---------------------------------------------------------------------------

class TestOnlySteps:
    def _project(self, tmp_path, pipeline="full-workflow"):
        from clouds_decoded.project import Project
        return Project.init(str(tmp_path / "proj"), name="T",
                            pipeline=pipeline)

    def test_non_prefix_rejected(self, tmp_path):
        project = self._project(tmp_path)
        project.stage("/data/S2A_001.SAFE")
        with pytest.raises(ValueError, match="prefix"):
            project.run(only_steps=["albedo"])
        with pytest.raises(ValueError, match="prefix"):
            project.run(only_steps=[])

    def test_prefix_truncates_workflow(self, tmp_path, monkeypatch):
        project = self._project(tmp_path)
        project.stage("/data/S2A_001.SAFE")
        seen = {}

        def mock_run_serial(scene_list, **kwargs):
            seen["steps"] = list(project.steps)

        monkeypatch.setattr(project, "_run_serial", mock_run_serial)
        project.run(only_steps=["cloud_mask", "cloud_height"])
        assert seen["steps"] == ["cloud_mask", "cloud_height"]
        # Restriction cleared after the run.
        assert len(project.steps) == 5

    def test_full_prefix_is_noop(self, tmp_path, monkeypatch):
        project = self._project(tmp_path)
        project.stage("/data/S2A_001.SAFE")
        monkeypatch.setattr(project, "_run_serial",
                            lambda scene_list, **kw: None)
        project.run(only_steps=list(project.steps))
        assert project._only_steps is None

    def test_multitemporal_recipe_order(self):
        from clouds_decoded.project import _get_recipe
        steps = [s.name for s in
                 _get_recipe("full-workflow-multitemporal").steps]
        assert steps == ["cloud_mask", "albedo", "cloud_height", "refocus",
                         "cloud_properties"]

    def test_restricted_run_keeps_scene_staged(self, tmp_path):
        """A restricted run over a scene whose restricted steps are already
        complete finishes via the manifest fast path and leaves the scene
        'staged' (not 'done') so a later full run picks it up."""
        from clouds_decoded.project import (
            Project, StepResult, _make_run_id,
        )

        project = self._project(tmp_path)
        sid = "S2A_MSIL1C_20230715T100031_N0510_R064_T37VCC_20230715T100031"
        scene_path = str(tmp_path / f"{sid}.SAFE")
        project.stage(scene_path)

        # Seed a completed cloud_mask manifest entry (no output_file so the
        # existence check is skipped; hash must match current config).
        scene_out = project._scene_output_dir(sid)
        scene_out.mkdir(parents=True)
        manifest = project._load_manifest(sid, scene_path)
        manifest.steps["cloud_mask"] = StepResult(
            status="completed",
            config_hash=project._config_hash("cloud_mask"),
        )
        project._save_manifest(sid, manifest)

        project.run(only_steps=["cloud_mask"], unsafe=True)
        rows = {r["scene_id"]: r for r in project.db.get_all()}
        assert rows[sid]["status"] == "staged"


# ---------------------------------------------------------------------------
# Pre-population end-to-end (gates: manifest + file + provenance)
# ---------------------------------------------------------------------------

class TestPrepopulate:
    @pytest.fixture
    def project(self, tmp_path):
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T",
                               pipeline="full-workflow-multitemporal")
        cfg = AlbedoEstimatorConfig(
            method="multitemporal",
            multitemporal=MultitemporalAlbedoParams(
                n_clusters=5, min_clear_obs=10, min_scenes=5),
            output_resolution=360,
        )
        cfg.to_yaml(project.configs_dir / "albedo.yaml")
        return project

    @pytest.fixture
    def model(self):
        stack = make_synthetic_stack()
        params = default_params()
        model = fit_cluster_model(stack["refl"], stack["clear"],
                                  stack["times"], stack["doy"],
                                  stack["year"], params)
        model["ord0"] = stack["ord0"]
        model["transform"] = np.array([180.0, 0, 300000.0,
                                       0, -180.0, 6200000.0])
        model["crs"] = "EPSG:32637"
        model["bands"] = ["B02", "B03", "B04"]
        return model

    def test_prepopulate_passes_all_resume_gates(self, project, model,
                                                 tmp_path):
        from clouds_decoded.config import BaseProcessorConfig
        from clouds_decoded.extensions.multitemporal_albedo.prepopulate import (
            prepopulate_scene,
        )

        sid = "S2A_MSIL1C_20230715T100031_N0510_R064_T37VCC_20230715T100031"
        scene_path = str(tmp_path / f"{sid}.SAFE")
        albedo_cfg = project._load_step_config("albedo")

        written = prepopulate_scene(project, model, scene_path, albedo_cfg)
        assert written

        out = project._scene_output_dir(sid) / "albedo.tif"
        assert out.exists()

        # Gate 1: manifest complete under the current config hash.
        manifest = project._load_manifest(sid, scene_path)
        assert manifest.is_step_complete(
            "albedo", project._config_hash("albedo"))

        # Gate 3: file provenance validates against the current config.
        err = project._validate_step_file(
            "albedo", project._scene_output_dir(sid), scene_path,
            BaseProcessorConfig._semantic_dump(albedo_cfg))
        assert err is None

        # Written raster is resampled to output_resolution and loadable.
        from clouds_decoded.data.base import AlbedoData
        data = AlbedoData.from_file(str(out))
        assert abs(data.transform.a - 360.0) < 0.5
        assert data.metadata.method == "multitemporal"
        assert data.metadata.band_names == ["B02", "B03", "B04"]

        # Idempotent: second call is a no-op.
        assert not prepopulate_scene(project, model, scene_path, albedo_cfg)

    def test_config_change_invalidates(self, project, model, tmp_path):
        from clouds_decoded.extensions.multitemporal_albedo.prepopulate import (
            prepopulate_scene,
        )
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )

        sid = "S2A_MSIL1C_20230715T100031_N0510_R064_T37VCC_20230715T100031"
        scene_path = str(tmp_path / f"{sid}.SAFE")
        albedo_cfg = project._load_step_config("albedo")
        prepopulate_scene(project, model, scene_path, albedo_cfg)

        # Change a fit hyperparameter -> hash changes -> step incomplete.
        cfg2 = AlbedoEstimatorConfig(
            method="multitemporal",
            multitemporal=MultitemporalAlbedoParams(
                n_clusters=7, min_clear_obs=10, min_scenes=5),
            output_resolution=360,
        )
        cfg2.to_yaml(project.configs_dir / "albedo.yaml")
        manifest = project._load_manifest(sid, scene_path)
        assert not manifest.is_step_complete(
            "albedo", project._config_hash("albedo"))

    def test_footprint_masks_nodata(self, project, model, tmp_path):
        from clouds_decoded.extensions.multitemporal_albedo.prepopulate import (
            predict_scene_albedo,
        )

        sid = "S2A_MSIL1C_20230715T100031_N0510_R064_T37VCC_20230715T100031"
        H, W = int(model["H"]), int(model["W"])
        footprint = np.ones((H, W), bool)
        footprint[:, : W // 2] = False
        data, _ = predict_scene_albedo(
            model, sid, {"B02": 0.05, "B03": 0.05, "B04": 0.05},
            footprint=footprint)
        assert np.isnan(data[:, :, : W // 2]).all()
        assert np.isfinite(data[:, :, W // 2:]).all()


# ---------------------------------------------------------------------------
# Cloud-mask checkpoint format tolerance
# ---------------------------------------------------------------------------

class TestCheckpointNormalisation:
    def test_senseiv2_prefixed(self):
        from clouds_decoded.modules.cloud_mask.processor import (
            _normalise_state_dict,
        )
        sd = {"segmenter.segformer.encoder.x": 1, "segmenter.decode_head.y": 2}
        out = _normalise_state_dict(sd)
        assert set(out) == {"segformer.encoder.x", "decode_head.y"}

    def test_training_wrapper_bare_keys(self):
        from clouds_decoded.modules.cloud_mask.processor import (
            _normalise_state_dict,
        )
        sd = {"model": {"segformer.encoder.x": 1, "decode_head.y": 2},
              "epoch": 3, "cloud_iou": 0.99}
        out = _normalise_state_dict(sd)
        assert set(out) == {"segformer.encoder.x", "decode_head.y"}

    def test_plain_bare_keys_passthrough(self):
        from clouds_decoded.modules.cloud_mask.processor import (
            _normalise_state_dict,
        )
        sd = {"segformer.encoder.x": 1}
        assert _normalise_state_dict(sd) == sd


# ---------------------------------------------------------------------------
# Stage selection
# ---------------------------------------------------------------------------

class TestStageSelection:
    def test_not_selected_for_default_config(self, tmp_path):
        from clouds_decoded.extensions.multitemporal_albedo.stage import (
            MultitemporalAlbedoStage,
        )
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T")
        assert not MultitemporalAlbedoStage.is_selected(project)

    def test_selected_for_multitemporal_config(self, tmp_path):
        from clouds_decoded.extensions.multitemporal_albedo.stage import (
            MultitemporalAlbedoStage,
        )
        from clouds_decoded.modules.albedo_estimator.config import (
            AlbedoEstimatorConfig,
        )
        from clouds_decoded.project import Project

        project = Project.init(str(tmp_path / "proj"), name="T",
                               pipeline="full-workflow-multitemporal")
        AlbedoEstimatorConfig(method="multitemporal").to_yaml(
            project.configs_dir / "albedo.yaml")
        assert MultitemporalAlbedoStage.is_selected(project)
