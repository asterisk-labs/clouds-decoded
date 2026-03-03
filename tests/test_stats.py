"""Tests for the stats plugin module (clouds_decoded.stats)."""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tif(path: Path, data: np.ndarray, band_names=None):
    """Write a minimal GeoTIFF with optional band_names in METADATA_TAG."""
    from clouds_decoded.constants import METADATA_TAG

    count = data.shape[0] if data.ndim == 3 else 1
    arr = data if data.ndim == 3 else data[np.newaxis]
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": count,
        "dtype": arr.dtype,
        "crs": CRS.from_epsg(32633),
        "transform": Affine.translation(0, 0) * Affine.scale(10, -10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if band_names is not None:
            meta = json.dumps({"band_names": band_names})
            dst.update_tags(**{METADATA_TAG: meta})


def _make_manifest(tmp_path: Path, step_name: str, output_file: str):
    """Create a SceneManifest with a single completed step."""
    from clouds_decoded.project import SceneManifest, StepResult

    return SceneManifest(
        scene_id="TEST",
        scene_path=str(tmp_path / "scene.SAFE"),
        steps={
            step_name: StepResult(
                status="completed",
                output_file=output_file,
                config_hash="abc123",
            )
        },
    )


def _make_caller(tmp_path: Path, step_name: str, output_file: str):
    """Return a StatsCaller for a single-step manifest."""
    from clouds_decoded.stats import StatsCaller
    return StatsCaller(_make_manifest(tmp_path, step_name, output_file))


# ---------------------------------------------------------------------------
# _generic.mean
# ---------------------------------------------------------------------------

class TestGenericMean:
    def test_single_band_no_names(self, tmp_path):
        """mean() on a single-band tif with no band_names returns flat keys."""
        from clouds_decoded.stats._generic import mean

        data = np.full((1, 10, 10), 5.0, dtype=np.float32)
        tif = tmp_path / "out.tif"
        _write_tif(tif, data)
        caller = _make_caller(tmp_path, "cloud_height_emulator", str(tif))

        result = mean(caller, "cloud_height_emulator")
        assert "mean" in result
        assert abs(result["mean"] - 5.0) < 1e-4
        assert result["n_pixels"] == 100

    def test_multiband_with_names(self, tmp_path):
        """mean() on a multi-band tif with band_names uses 'name__mean' keys."""
        from clouds_decoded.stats._generic import mean

        data = np.stack([
            np.full((10, 10), 0.2, dtype=np.float32),
            np.full((10, 10), 0.4, dtype=np.float32),
        ])
        tif = tmp_path / "albedo.tif"
        _write_tif(tif, data, band_names=["B02", "B03"])
        caller = _make_caller(tmp_path, "albedo", str(tif))

        result = mean(caller, "albedo")
        assert "B02__mean" in result
        assert "B03__mean" in result
        assert abs(result["B02__mean"] - 0.2) < 1e-4
        assert abs(result["B03__mean"] - 0.4) < 1e-4

    def test_returns_empty_on_missing_file(self, tmp_path):
        """mean() returns {} when the output file does not exist."""
        from clouds_decoded.stats._generic import mean
        from clouds_decoded.project import SceneManifest, StepResult
        from clouds_decoded.stats import StatsCaller

        manifest = SceneManifest(
            scene_id="TEST",
            scene_path=str(tmp_path / "scene.SAFE"),
            steps={
                "cloud_height_emulator": StepResult(
                    status="completed",
                    output_file=str(tmp_path / "nonexistent.tif"),
                    config_hash="abc",
                )
            },
        )
        result = mean(StatsCaller(manifest), "cloud_height_emulator")
        assert result == {}

    def test_returns_empty_on_incomplete_step(self, tmp_path):
        """mean() returns {} when step status is not 'completed'."""
        from clouds_decoded.stats._generic import mean
        from clouds_decoded.project import SceneManifest, StepResult
        from clouds_decoded.stats import StatsCaller

        manifest = SceneManifest(
            scene_id="TEST",
            scene_path=str(tmp_path / "scene.SAFE"),
            steps={
                "cloud_height_emulator": StepResult(
                    status="failed",
                    output_file=str(tmp_path / "out.tif"),
                    config_hash="abc",
                )
            },
        )
        result = mean(StatsCaller(manifest), "cloud_height_emulator")
        assert result == {}


# ---------------------------------------------------------------------------
# _generic.percentiles
# ---------------------------------------------------------------------------

class TestGenericPercentiles:
    def test_percentile_keys_single_band(self, tmp_path):
        """percentiles() produces p-keys, mean, and n_pixels for a single-band tif."""
        from clouds_decoded.stats._generic import percentiles, _DEFAULT_PERCENTILES

        vals = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
        tif = tmp_path / "height.tif"
        _write_tif(tif, vals + 1)  # all non-zero
        caller = _make_caller(tmp_path, "cloud_height_emulator", str(tif))

        result = percentiles(caller, "cloud_height_emulator")
        assert "p050" in result
        assert "p000" in result
        assert "p100" in result
        assert "mean" in result
        assert "n_pixels" in result
        # one key per default percentile, plus mean and n_pixels
        assert len(result) == len(_DEFAULT_PERCENTILES) + 2

    def test_p050_approx_median(self, tmp_path):
        """p050 is approximately the median of valid pixels."""
        from clouds_decoded.stats._generic import percentiles

        vals = np.arange(1, 101, dtype=np.float32).reshape(1, 10, 10)
        tif = tmp_path / "height.tif"
        _write_tif(tif, vals)
        caller = _make_caller(tmp_path, "cloud_height_emulator", str(tif))

        result = percentiles(caller, "cloud_height_emulator")
        assert abs(result["p050"] - 50.5) < 1.0  # median of 1..100


# ---------------------------------------------------------------------------
# cloud_mask.class_fractions
# ---------------------------------------------------------------------------

class TestClassFractions:
    def test_fractions_sum_to_one(self, tmp_path):
        """class_fractions() returns values that sum to approximately 1."""
        from clouds_decoded.stats.cloud_mask import class_fractions

        data = np.zeros((10, 10), dtype=np.uint8)
        data[:5, :] = 0   # clear
        data[5:, :5] = 2  # thick
        data[5:, 5:] = 3  # shadow
        tif = tmp_path / "mask.tif"
        _write_tif(tif, data[np.newaxis])
        caller = _make_caller(tmp_path, "cloud_mask", str(tif))

        result = class_fractions(caller, "cloud_mask")
        assert "clear_frac" in result
        assert "thin_cloud_frac" in result
        assert "thick_cloud_frac" in result
        assert "cloud_shadow_frac" in result
        total = sum(v for k, v in result.items() if k.endswith("_frac"))
        assert abs(total - 1.0) < 1e-6

    def test_n_pixels_excludes_255(self, tmp_path):
        """class_fractions() excludes nodata=255 pixels from n_pixels."""
        from clouds_decoded.stats.cloud_mask import class_fractions

        data = np.zeros((10, 10), dtype=np.uint8)
        data[0, 0] = 255  # nodata
        tif = tmp_path / "mask.tif"
        _write_tif(tif, data[np.newaxis])
        caller = _make_caller(tmp_path, "cloud_mask", str(tif))

        result = class_fractions(caller, "cloud_mask")
        assert result["n_pixels"] == 99  # 100 - 1 nodata

    def test_returns_empty_on_missing_file(self, tmp_path):
        """class_fractions() returns {} when output file missing."""
        from clouds_decoded.stats.cloud_mask import class_fractions
        from clouds_decoded.project import SceneManifest, StepResult
        from clouds_decoded.stats import StatsCaller

        manifest = SceneManifest(
            scene_id="TEST",
            scene_path=str(tmp_path / "scene.SAFE"),
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file=str(tmp_path / "nonexistent.tif"),
                    config_hash="abc",
                )
            },
        )
        result = class_fractions(StatsCaller(manifest), "cloud_mask")
        assert result == {}


# ---------------------------------------------------------------------------
# StatsCaller.load caching
# ---------------------------------------------------------------------------

class TestStatsCaller:
    def test_caches_data_object(self, tmp_path):
        """StatsCaller loads each step output only once."""
        from clouds_decoded.stats import StatsCaller

        data = np.full((1, 10, 10), 1.0, dtype=np.float32)
        tif = tmp_path / "out.tif"
        _write_tif(tif, data)
        caller = _make_caller(tmp_path, "cloud_height_emulator", str(tif))

        obj1 = caller.load("cloud_height_emulator")
        obj2 = caller.load("cloud_height_emulator")
        assert obj1 is obj2  # same object, not reloaded

    def test_multi_step_load(self, tmp_path):
        """StatsCaller can load outputs from two different steps."""
        from clouds_decoded.stats import StatsCaller
        from clouds_decoded.project import SceneManifest, StepResult

        tif_a = tmp_path / "height.tif"
        tif_b = tmp_path / "mask.tif"
        _write_tif(tif_a, np.ones((1, 4, 4), dtype=np.float32))
        _write_tif(tif_b, np.zeros((1, 4, 4), dtype=np.uint8))

        manifest = SceneManifest(
            scene_id="TEST",
            scene_path=str(tmp_path / "scene.SAFE"),
            steps={
                "cloud_height_emulator": StepResult(
                    status="completed", output_file=str(tif_a), config_hash="a"
                ),
                "cloud_mask": StepResult(
                    status="completed", output_file=str(tif_b), config_hash="b"
                ),
            },
        )
        caller = StatsCaller(manifest)
        height = caller.load("cloud_height_emulator")
        mask   = caller.load("cloud_mask")
        assert height is not None
        assert mask is not None
        assert height is not mask


# ---------------------------------------------------------------------------
# resolve_stats_fn
# ---------------------------------------------------------------------------

class TestResolveStatsFn:
    def test_step_specific_wins_over_generic(self):
        """resolve_stats_fn prefers step-specific module over _generic."""
        from clouds_decoded.stats import resolve_stats_fn

        fn, step_name = resolve_stats_fn("cloud_mask::class_fractions")
        assert step_name == "cloud_mask"
        import clouds_decoded.stats.cloud_mask as cm_mod
        assert fn is cm_mod.class_fractions

    def test_falls_back_to_generic(self):
        """resolve_stats_fn falls back to _generic for unknown step modules."""
        from clouds_decoded.stats import resolve_stats_fn

        fn, step_name = resolve_stats_fn("cloud_height_emulator::percentiles")
        assert step_name == "cloud_height_emulator"
        import clouds_decoded.stats._generic as gen
        assert fn is gen.percentiles

    def test_raises_on_unknown_function(self):
        """resolve_stats_fn raises AttributeError for a nonexistent function."""
        from clouds_decoded.stats import resolve_stats_fn

        with pytest.raises(AttributeError, match="not found"):
            resolve_stats_fn("cloud_mask::nonexistent_function_xyz")


# ---------------------------------------------------------------------------
# Project.run_stats() integration
# ---------------------------------------------------------------------------

class TestRunStats:
    def test_run_stats_skips_non_done_runs(self, tmp_path):
        """run_stats() only computes stats for runs with status='done'."""
        from clouds_decoded.project import Project, _make_run_id

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="Stats Test")
        project.stage("/data/S2A_001.SAFE")
        # Leave as 'staged', not 'done'

        project.run_stats(methods=["cloud_mask::class_fractions"])
        run_id = _make_run_id("S2A_001", None)
        assert project.db.has_stats(run_id, "stats_cloud_mask") is False

    def test_run_stats_idempotent(self, tmp_path):
        """run_stats() without force writes stats only once."""
        from clouds_decoded.project import Project, _make_run_id, SceneManifest, StepResult

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="Stats Test")
        project.stage("/data/S2A_001.SAFE")

        run_id = _make_run_id("S2A_001", None)
        project.db.set_status(run_id, "done")

        scene_dir = project.output_dir / "S2A_001"
        scene_dir.mkdir(parents=True, exist_ok=True)
        mask_tif = scene_dir / "cloud_mask.tif"
        data = np.zeros((10, 10), dtype=np.uint8)
        _write_tif(mask_tif, data[np.newaxis])

        manifest = SceneManifest(
            scene_id="S2A_001",
            scene_path="/data/S2A_001.SAFE",
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file=str(mask_tif),
                    config_hash="abc",
                )
            },
        )
        manifest.to_json(scene_dir / "manifest.json")

        project.run_stats(methods=["cloud_mask::class_fractions"])
        assert project.db.has_stats(run_id, "stats_cloud_mask") is True

        project.run_stats(methods=["cloud_mask::class_fractions"])
        with project.db._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM stats_cloud_mask").fetchone()[0]
        assert count == 1

    def test_run_stats_force_overwrites(self, tmp_path):
        """run_stats(force=True) replaces existing stats rows."""
        from clouds_decoded.project import Project, _make_run_id, SceneManifest, StepResult

        project_dir = tmp_path / "proj"
        project = Project.init(str(project_dir), name="Stats Force Test")
        project.stage("/data/S2A_001.SAFE")

        run_id = _make_run_id("S2A_001", None)
        project.db.set_status(run_id, "done")
        project.db.write_stats(run_id, "stats_cloud_mask", {"clear_frac": 0.99})

        scene_dir = project.output_dir / "S2A_001"
        scene_dir.mkdir(parents=True, exist_ok=True)
        mask_tif = scene_dir / "cloud_mask.tif"
        data = np.zeros((10, 10), dtype=np.uint8)  # all clear
        _write_tif(mask_tif, data[np.newaxis])

        manifest = SceneManifest(
            scene_id="S2A_001",
            scene_path="/data/S2A_001.SAFE",
            steps={
                "cloud_mask": StepResult(
                    status="completed",
                    output_file=str(mask_tif),
                    config_hash="abc",
                )
            },
        )
        manifest.to_json(scene_dir / "manifest.json")

        project.run_stats(force=True, methods=["cloud_mask::class_fractions"])

        with project.db._conn() as conn:
            row = conn.execute(
                "SELECT clear_frac FROM stats_cloud_mask WHERE run_id=?", [run_id]
            ).fetchone()
        assert abs(row[0] - 1.0) < 1e-6
