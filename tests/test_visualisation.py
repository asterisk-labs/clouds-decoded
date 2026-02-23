"""Tests for the viser-based 3D visualisation module.

Tests SceneData loading, texture computation, and ViserViewer instantiation
using synthetic GeoTIFFs written to tmp_path. Does NOT start a viser server.
"""
import json

import numpy as np
import pytest
import rasterio as rio
import yaml
from rasterio.crs import CRS
from rasterio.transform import Affine

from clouds_decoded.visualisation.viser_viewer import (
    SceneData,
    ViserViewer,
    _build_grid_faces,
    _squeeze_2d,
    _stride_downsample,
    _viridis_colors,
)


# ---------------------------------------------------------------------------
# Helpers — write synthetic GeoTIFFs
# ---------------------------------------------------------------------------

def _write_geotiff(
    path, data: np.ndarray, transform: Affine = None, crs=None, metadata: dict = None
):
    """Write a numpy array as a GeoTIFF with optional metadata."""
    if transform is None:
        transform = Affine.translation(500_000.0, 6_000_000.0) * Affine.scale(10.0, -10.0)
    if crs is None:
        crs = CRS.from_epsg(32633)

    if data.ndim == 2:
        count = 1
        height, width = data.shape
        write_data = data[np.newaxis, ...]
    else:
        count, height, width = data.shape
        write_data = data

    with rio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=write_data.dtype,
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(write_data)
        if metadata:
            dst.update_tags(METADATA=json.dumps(metadata))


def _make_scene_output_dir(tmp_path, scene_id="TEST_SCENE", include_mask=True, include_albedo=False):
    """Create a minimal project scene output directory with cloud_height.tif.

    Returns:
        (output_dir, scene_id)
    """
    output_dir = tmp_path / "scenes" / scene_id
    output_dir.mkdir(parents=True)

    # Cloud height: 50x40 grid of heights in [0, 5000]
    np.random.seed(42)
    height_arr = np.random.uniform(0, 5000, (50, 40)).astype(np.float32)
    _write_geotiff(output_dir / "cloud_height.tif", height_arr)

    if include_mask:
        # Cloud mask: categorical 0-3
        mask_arr = np.random.randint(0, 4, (50, 40)).astype(np.uint8)
        meta = {"categorical": True, "classes": {"0": "Clear", "1": "Thick Cloud", "2": "Thin Cloud", "3": "Cloud Shadow"}}
        _write_geotiff(output_dir / "cloud_mask.tif", mask_arr, metadata=meta)

    if include_albedo:
        # Albedo: 3 bands
        albedo_arr = np.random.uniform(0.0, 0.5, (3, 50, 40)).astype(np.float32)
        meta = {"band_names": ["B02", "B03", "B04"]}
        _write_geotiff(output_dir / "albedo.tif", albedo_arr, metadata=meta)

    return output_dir, scene_id


def _make_project(tmp_path, scene_ids=None, include_mask=True, include_albedo=False):
    """Create a minimal project directory structure.

    Returns:
        (project_dir, {scene_id: output_dir})
    """
    if scene_ids is None:
        scene_ids = ["SCENE_A"]

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # project.yaml
    config = {
        "name": "test_project",
        "pipeline": "full-workflow",
        "scenes": [f"/data/{sid}.SAFE" for sid in scene_ids],
        "created_at": "2024-01-01T00:00:00",
    }
    with open(project_dir / "project.yaml", "w") as f:
        yaml.dump(config, f)

    # configs directory (required by Project.load)
    (project_dir / "configs").mkdir()

    # scenes with outputs
    scene_dirs = {}
    for sid in scene_ids:
        output_dir = project_dir / "scenes" / sid
        output_dir.mkdir(parents=True)

        np.random.seed(hash(sid) % 2**31)
        height_arr = np.random.uniform(0, 5000, (50, 40)).astype(np.float32)
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        if include_mask:
            mask_arr = np.random.randint(0, 4, (50, 40)).astype(np.uint8)
            _write_geotiff(output_dir / "cloud_mask.tif", mask_arr)

        if include_albedo:
            albedo_arr = np.random.uniform(0.0, 0.5, (3, 50, 40)).astype(np.float32)
            meta = {"band_names": ["B02", "B03", "B04"]}
            _write_geotiff(output_dir / "albedo.tif", albedo_arr, metadata=meta)

        # manifest.json
        manifest = {
            "scene_id": sid,
            "scene_path": f"/data/{sid}.SAFE",
            "steps": {
                "cloud_height": {"status": "completed", "output_file": str(output_dir / "cloud_height.tif")},
            },
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        scene_dirs[sid] = output_dir

    return project_dir, scene_dirs


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_squeeze_2d_3d_input(self):
        arr = np.zeros((1, 10, 20))
        result = _squeeze_2d(arr)
        assert result.shape == (10, 20)

    def test_squeeze_2d_already_2d(self):
        arr = np.zeros((10, 20))
        result = _squeeze_2d(arr)
        assert result.shape == (10, 20)

    def test_squeeze_2d_multiband(self):
        arr = np.zeros((3, 10, 20))
        result = _squeeze_2d(arr)
        assert result.shape == (3, 10, 20)  # unchanged

    def test_stride_downsample_no_change(self):
        arr = np.ones((50, 50))
        result, stride = _stride_downsample(arr, max_dim=100)
        assert stride == 1
        assert result.shape == (50, 50)

    def test_stride_downsample_halves(self):
        arr = np.ones((200, 100))
        result, stride = _stride_downsample(arr, max_dim=100)
        assert stride == 2
        assert result.shape == (100, 50)

    def test_viridis_colors_shape(self):
        vals = np.linspace(0, 1, 100)
        colors = _viridis_colors(vals)
        assert colors.shape == (100, 3)
        assert colors.dtype == np.uint8

    def test_viridis_colors_range(self):
        vals = np.array([0.0, 0.5, 1.0])
        colors = _viridis_colors(vals)
        # All values should be valid uint8
        assert colors.min() >= 0
        assert colors.max() <= 255

    def test_build_grid_faces_shape(self):
        H, W = 5, 8
        faces = _build_grid_faces(H, W)
        # 2 triangles per quad, (H-1)*(W-1) quads
        assert faces.shape == (2 * (H - 1) * (W - 1), 3)
        assert faces.dtype == np.uint32

    def test_build_grid_faces_index_range(self):
        H, W = 4, 6
        faces = _build_grid_faces(H, W)
        assert faces.min() >= 0
        assert faces.max() < H * W

    def test_build_grid_faces_minimal(self):
        # 2x2 grid → 2 triangles
        faces = _build_grid_faces(2, 2)
        assert faces.shape == (2, 3)


# ---------------------------------------------------------------------------
# SceneData tests
# ---------------------------------------------------------------------------


class TestSceneData:
    """Tests for SceneData loading and texture computation."""

    def test_load_basic(self, tmp_path):
        """SceneData.load() reads cloud_height.tif and creates points."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        assert sd.points is not None
        assert sd.points.ndim == 2
        assert sd.points.shape[1] == 3
        assert sd.base_z is not None
        assert sd.grid_shape is not None
        assert len(sd.base_z) == sd.points.shape[0]

    def test_load_creates_height_texture(self, tmp_path):
        """Height texture is always created."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()

        assert "Height" in sd.texture_names
        colors = sd.get_colors("Height")
        assert colors.shape == (sd.points.shape[0], 3)
        assert colors.dtype == np.uint8

    def test_load_creates_cloud_mask_texture(self, tmp_path):
        """Cloud mask texture is loaded when cloud_mask.tif exists."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=True)

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()

        assert "Cloud Mask" in sd.texture_names
        colors = sd.get_colors("Cloud Mask")
        assert colors.shape == (sd.points.shape[0], 3)
        assert colors.dtype == np.uint8

    def test_load_creates_albedo_texture(self, tmp_path):
        """Albedo texture is loaded when albedo.tif exists."""
        output_dir, scene_id = _make_scene_output_dir(
            tmp_path, include_mask=False, include_albedo=True
        )

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()

        assert "Albedo" in sd.texture_names
        colors = sd.get_colors("Albedo")
        assert colors.shape == (sd.points.shape[0], 3)

    def test_missing_height_raises(self, tmp_path):
        """load() raises FileNotFoundError when cloud_height.tif is missing."""
        output_dir = tmp_path / "empty_scene"
        output_dir.mkdir()

        sd = SceneData(scene_id="MISSING", output_dir=str(output_dir))
        with pytest.raises(FileNotFoundError, match="cloud_height.tif"):
            sd.load()

    def test_get_points_z_scale(self, tmp_path):
        """get_points(z_scale) multiplies z by scale factor."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()

        pts_1 = sd.get_points(z_scale=1.0)
        pts_2 = sd.get_points(z_scale=2.0)

        # x, y unchanged
        np.testing.assert_array_equal(pts_1[:, :2], pts_2[:, :2])
        # z doubled
        np.testing.assert_allclose(pts_2[:, 2], pts_1[:, 2] * 2.0, rtol=1e-6)

    def test_get_colors_invalid_texture_raises(self, tmp_path):
        """get_colors() raises KeyError for unknown texture name."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()

        with pytest.raises(KeyError, match="nonexistent"):
            sd.get_colors("nonexistent")

    def test_points_centered_near_origin(self, tmp_path):
        """xy coordinates should be centered around (0, 0)."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)

        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()

        mean_x = np.mean(sd.points[:, 0])
        mean_y = np.mean(sd.points[:, 1])
        # Should be very close to zero after centering
        assert abs(mean_x) < 1.0
        assert abs(mean_y) < 1.0

    def test_downsampling_respects_max_dim(self, tmp_path):
        """Grid is downsampled when larger than max_grid_dim."""
        output_dir = tmp_path / "scenes" / "LARGE"
        output_dir.mkdir(parents=True)

        # Write a 500x600 height grid
        height_arr = np.random.uniform(0, 3000, (500, 600)).astype(np.float32)
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="LARGE", output_dir=str(output_dir))
        sd.load(max_grid_dim=100)

        # Grid should be <= 100 in both dimensions
        assert sd.grid_shape[0] <= 100
        assert sd.grid_shape[1] <= 100
        assert sd.stride > 1

    def test_cloud_mask_colors_are_correct(self, tmp_path):
        """Cloud mask texture uses the expected categorical colours."""
        output_dir = tmp_path / "scenes" / "MASK_CHECK"
        output_dir.mkdir(parents=True)

        # Uniform height
        height_arr = np.ones((10, 10), dtype=np.float32) * 1000.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        # All-clear mask (class 0)
        mask_arr = np.zeros((10, 10), dtype=np.uint8)
        _write_geotiff(output_dir / "cloud_mask.tif", mask_arr)

        sd = SceneData(scene_id="MASK_CHECK", output_dir=str(output_dir))
        sd.load(max_grid_dim=100)

        colors = sd.get_colors("Cloud Mask")
        # All pixels should be green (44, 160, 44) for class 0
        assert np.all(colors[:, 0] == 44)
        assert np.all(colors[:, 1] == 160)
        assert np.all(colors[:, 2] == 44)

    def test_xy_extent_positive(self, tmp_path):
        """xy_extent is positive after load()."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)
        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()
        assert sd.xy_extent > 0

    def test_mesh_faces_shape(self, tmp_path):
        """mesh_faces has the right shape for the grid dimensions."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)
        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        H, W = sd.grid_shape
        expected_faces = 2 * (H - 1) * (W - 1)
        assert sd.mesh_faces is not None
        assert sd.mesh_faces.shape == (expected_faces, 3)
        assert sd.mesh_faces.dtype == np.uint32

    def test_mesh_faces_index_range(self, tmp_path):
        """mesh_faces indices are all within the point array bounds."""
        output_dir, scene_id = _make_scene_output_dir(tmp_path, include_mask=False)
        sd = SceneData(scene_id=scene_id, output_dir=str(output_dir))
        sd.load()
        n_points = sd.points.shape[0]
        assert sd.mesh_faces.max() < n_points

    def test_get_points_nan_replaced_with_zero(self, tmp_path):
        """get_points() replaces NaN heights with 0 for safe rendering."""
        output_dir = tmp_path / "scenes" / "NAN_SCENE"
        output_dir.mkdir(parents=True)

        height_arr = np.ones((10, 10), dtype=np.float32) * 500.0
        height_arr[3, 3] = np.nan
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="NAN_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=100)

        pts = sd.get_points(z_scale=1.0)
        assert np.all(np.isfinite(pts))  # no NaNs in output

    def test_get_mesh_data_shapes(self, tmp_path):
        """get_mesh_data() returns correctly shaped vertices and RGBA; faces ≤ max."""
        output_dir = tmp_path / "scenes" / "MESH_SCENE"
        output_dir.mkdir(parents=True)
        # Uniform height so no faces are filtered by z-diff
        height_arr = np.ones((20, 30), dtype=np.float32) * 1000.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="MESH_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        vertices, faces, rgba = sd.get_mesh_data(z_scale=1.0, texture_name="Height", step=1)
        H, W = sd.grid_shape
        N = H * W
        assert vertices.shape == (N, 3)
        assert faces.shape[1] == 3
        assert faces.shape[0] <= 2 * (H - 1) * (W - 1)
        assert rgba.shape == (N, 4)
        assert rgba.dtype == np.uint8
        assert np.all(rgba[:, 3] == 255)  # alpha channel is opaque

    def test_get_mesh_data_step_reduces_vertices(self, tmp_path):
        """A step > 1 produces fewer vertices than step=1."""
        output_dir = tmp_path / "scenes" / "STEP_SCENE"
        output_dir.mkdir(parents=True)
        height_arr = np.ones((20, 20), dtype=np.float32) * 500.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="STEP_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        v1, f1, _ = sd.get_mesh_data(z_scale=1.0, texture_name="Height", step=1)
        v2, f2, _ = sd.get_mesh_data(z_scale=1.0, texture_name="Height", step=2)

        assert v2.shape[0] < v1.shape[0]
        assert f2.shape[0] < f1.shape[0]

    def test_get_mesh_data_z_scale_applied(self, tmp_path):
        """get_mesh_data() applies z_scale to vertex z-coordinates."""
        output_dir = tmp_path / "scenes" / "ZSCALE_SCENE"
        output_dir.mkdir(parents=True)
        height_arr = np.ones((10, 10), dtype=np.float32) * 1000.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="ZSCALE_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        v1, _, _ = sd.get_mesh_data(z_scale=1.0, texture_name="Height", step=1)
        v2, _, _ = sd.get_mesh_data(z_scale=3.0, texture_name="Height", step=1)

        # z-coords should be 3× larger
        np.testing.assert_allclose(v2[:, 2], v1[:, 2] * 3.0, rtol=1e-5)

    def test_get_mesh_data_drops_curtain_faces(self, tmp_path):
        """Faces with physical z-range > max_z_diff are removed."""
        output_dir = tmp_path / "scenes" / "CURTAIN_SCENE"
        output_dir.mkdir(parents=True)
        # Left half at 5000 m, right half at 10000 m (both non-zero so only
        # the curtain filter triggers, not the zero-edge filter)
        height_arr = np.full((10, 10), 5000.0, dtype=np.float32)
        height_arr[:, 5:] = 10000.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="CURTAIN_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        H, W = sd.grid_shape
        max_faces = 2 * (H - 1) * (W - 1)
        _, faces, _ = sd.get_mesh_data(
            z_scale=1.0, texture_name="Height", step=1,
            median_kernel=1, max_z_diff=1000.0,
        )
        assert faces.shape[0] < max_faces  # curtain faces removed

    def test_get_mesh_data_drops_zero_edge_faces(self, tmp_path):
        """Faces bridging z=0 and z>0 are removed; all-zero faces are kept."""
        output_dir = tmp_path / "scenes" / "ZERO_EDGE_SCENE"
        output_dir.mkdir(parents=True)
        # Top half at 500 m, bottom half at 0 m — 500 m jump is within
        # max_z_diff=1000 so only the zero-edge filter should remove boundaries
        height_arr = np.zeros((10, 10), dtype=np.float32)
        height_arr[:5, :] = 500.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="ZERO_EDGE_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        H, W = sd.grid_shape
        max_faces = 2 * (H - 1) * (W - 1)
        _, faces, _ = sd.get_mesh_data(
            z_scale=1.0, texture_name="Height", step=1,
            median_kernel=1, max_z_diff=1000.0,
        )
        assert faces.shape[0] < max_faces  # boundary faces removed

    def test_get_mesh_data_no_faces_dropped_when_uniform(self, tmp_path):
        """No faces are dropped when height is uniform and non-zero."""
        output_dir = tmp_path / "scenes" / "UNIFORM_SCENE"
        output_dir.mkdir(parents=True)
        height_arr = np.ones((10, 10), dtype=np.float32) * 500.0
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="UNIFORM_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        H, W = sd.grid_shape
        expected_faces = 2 * (H - 1) * (W - 1)
        _, faces, _ = sd.get_mesh_data(
            z_scale=1.0, texture_name="Height", step=1,
            median_kernel=1, max_z_diff=20000.0,
        )
        assert faces.shape[0] == expected_faces

    def test_get_mesh_data_all_zero_faces_kept(self, tmp_path):
        """Faces entirely at z=0 are preserved (the ground plane is valid)."""
        output_dir = tmp_path / "scenes" / "ALL_ZERO_SCENE"
        output_dir.mkdir(parents=True)
        height_arr = np.zeros((10, 10), dtype=np.float32)
        _write_geotiff(output_dir / "cloud_height.tif", height_arr)

        sd = SceneData(scene_id="ALL_ZERO_SCENE", output_dir=str(output_dir))
        sd.load(max_grid_dim=800)

        H, W = sd.grid_shape
        expected_faces = 2 * (H - 1) * (W - 1)
        _, faces, _ = sd.get_mesh_data(
            z_scale=1.0, texture_name="Height", step=1,
            median_kernel=1, max_z_diff=20000.0,
        )
        assert faces.shape[0] == expected_faces


# ---------------------------------------------------------------------------
# ViserViewer instantiation test
# ---------------------------------------------------------------------------


class TestViserViewer:
    """Tests for ViserViewer construction (no server started)."""

    def test_instantiation_with_mock_project(self, tmp_path):
        """ViserViewer loads scenes from a valid project directory."""
        project_dir, _ = _make_project(
            tmp_path, scene_ids=["SCENE_A", "SCENE_B"]
        )

        viewer = ViserViewer(
            project_dir=str(project_dir),
            max_grid_dim=100,
        )

        assert len(viewer._scene_data) == 2
        assert "SCENE_A" in viewer._scene_data
        assert "SCENE_B" in viewer._scene_data

    def test_instantiation_skips_scenes_without_height(self, tmp_path):
        """Scenes without cloud_height.tif are skipped."""
        project_dir, scene_dirs = _make_project(
            tmp_path, scene_ids=["HAS_HEIGHT", "NO_HEIGHT"]
        )

        # Remove cloud_height.tif from NO_HEIGHT
        (scene_dirs["NO_HEIGHT"] / "cloud_height.tif").unlink()

        viewer = ViserViewer(
            project_dir=str(project_dir),
            max_grid_dim=100,
        )

        assert len(viewer._scene_data) == 1
        assert "HAS_HEIGHT" in viewer._scene_data
        assert "NO_HEIGHT" not in viewer._scene_data

    def test_instantiation_fails_no_scenes(self, tmp_path):
        """ViserViewer raises RuntimeError when no scenes have cloud height."""
        project_dir, scene_dirs = _make_project(
            tmp_path, scene_ids=["EMPTY"]
        )
        (scene_dirs["EMPTY"] / "cloud_height.tif").unlink()

        with pytest.raises(RuntimeError, match="No scenes"):
            ViserViewer(project_dir=str(project_dir), max_grid_dim=100)

    def test_scene_order_matches_project(self, tmp_path):
        """Scenes are ordered as listed in the project config."""
        project_dir, _ = _make_project(
            tmp_path, scene_ids=["ALPHA", "BETA", "GAMMA"]
        )

        viewer = ViserViewer(
            project_dir=str(project_dir),
            max_grid_dim=100,
        )

        assert viewer._scene_order == ["ALPHA", "BETA", "GAMMA"]

    def test_scene_data_has_textures(self, tmp_path):
        """Loaded scenes have at least the Height texture."""
        project_dir, _ = _make_project(
            tmp_path,
            scene_ids=["TEXTURED"],
            include_mask=True,
            include_albedo=True,
        )

        viewer = ViserViewer(
            project_dir=str(project_dir),
            max_grid_dim=100,
        )

        sd = viewer._scene_data["TEXTURED"]
        assert "Height" in sd.texture_names
        assert "Cloud Mask" in sd.texture_names
        assert "Albedo" in sd.texture_names
