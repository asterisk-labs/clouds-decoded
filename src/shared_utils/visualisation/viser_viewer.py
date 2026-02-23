"""Viser-powered 3D viewer for project cloud height point clouds.

Renders cloud height GeoTIFFs as interactive 3D point clouds in the browser,
with per-vertex texturing from pipeline outputs (cloud mask, albedo, RGB, etc.).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cloud mask class colours — matches layers.py / static.py
_CLOUD_MASK_COLORS = {
    0: (44, 160, 44),     # Clear — green
    1: (214, 39, 40),     # Thick Cloud — red
    2: (255, 127, 14),    # Thin Cloud — orange
    3: (127, 127, 127),   # Cloud Shadow — gray
}


def _squeeze_2d(data: np.ndarray) -> np.ndarray:
    """Ensure (1, H, W) → (H, W)."""
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    return data


def _colormap_colors(values: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Map normalised [0,1] values to RGB uint8 (N, 3) using a matplotlib colormap."""
    import matplotlib

    rgba = matplotlib.colormaps[cmap](values)
    return (rgba[:, :3] * 255).astype(np.uint8)


def _viridis_colors(values: np.ndarray) -> np.ndarray:
    """Map normalised [0,1] values to viridis RGB uint8 (N, 3)."""
    return _colormap_colors(values, "viridis")


def _stride_downsample(arr: np.ndarray, max_dim: int) -> Tuple[np.ndarray, int]:
    """Stride-downsample a 2D array so the longest side ≤ max_dim.

    Returns:
        (downsampled_array, stride)
    """
    longest = max(arr.shape[0], arr.shape[1])
    stride = max(1, (longest + max_dim - 1) // max_dim)
    return arr[::stride, ::stride], stride


def _build_grid_faces(H: int, W: int) -> np.ndarray:
    """Build triangle face indices for an H×W regular grid.

    Each quad is split into two CCW triangles (normals point in the +Z / up
    direction so the surface is visible when viewed from above):
        ul-ll-ur  and  ur-ll-lr

    Note: In GeoTIFF rasters the y-pixel size is negative (north-up), so row
    index ii maps to decreasing northing.  The correct CCW winding from above
    (+Z looking down, x=easting, y=northing) is therefore ul→ll→ur and
    ur→ll→lr, not the naïve ul→ur→ll which would be clockwise.

    Returns:
        (2*(H-1)*(W-1), 3) uint32 face array.
    """
    ii, jj = np.meshgrid(np.arange(H - 1), np.arange(W - 1), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()
    ul = (ii * W + jj).astype(np.uint32)
    ur = (ii * W + jj + 1).astype(np.uint32)
    ll = ((ii + 1) * W + jj).astype(np.uint32)
    lr = ((ii + 1) * W + jj + 1).astype(np.uint32)
    return np.vstack([
        np.stack([ul, ll, ur], axis=1),   # CCW from above
        np.stack([ur, ll, lr], axis=1),   # CCW from above
    ]).astype(np.uint32)


# ---------------------------------------------------------------------------
# SceneData — loads and holds data for one scene
# ---------------------------------------------------------------------------


class SceneData:
    """Loads and caches point-cloud and texture data for a single scene.

    Args:
        scene_id: Scene identifier (without .SAFE).
        scene_path: Absolute path to the .SAFE directory (for RGB loading).
        output_dir: Path to the project scene output directory
            (e.g. ``project/scenes/S2A_…/``).
    """

    def __init__(
        self,
        scene_id: str,
        output_dir: str,
        scene_path: Optional[str] = None,
    ) -> None:
        self.scene_id = scene_id
        self.scene_path = scene_path
        self.output_dir = Path(output_dir)

        # Populated by load()
        self.points: Optional[np.ndarray] = None    # (N, 3) float32, xy centred
        self.base_z: Optional[np.ndarray] = None    # (N,) unscaled heights
        self.grid_shape: Optional[Tuple[int, int]] = None
        self.stride: int = 1
        self.xy_extent: float = 1.0                 # max(x_range, y_range) in metres
        self.mesh_faces: Optional[np.ndarray] = None  # (M, 3) uint32 triangle indices
        self._textures: Dict[str, np.ndarray] = {}  # name → (N, 3) uint8

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, max_grid_dim: int = 800) -> None:
        """Load cloud height grid and compute point positions and textures.

        Args:
            max_grid_dim: Maximum grid dimension (longest side) after
                stride downsampling. Controls display resolution.

        Raises:
            FileNotFoundError: If cloud_height.tif is missing.
        """
        from clouds_decoded.data import CloudHeightGridData

        height_path = self.output_dir / "cloud_height.tif"
        if not height_path.exists():
            raise FileNotFoundError(
                f"cloud_height.tif not found in {self.output_dir}"
            )

        height_data = CloudHeightGridData.from_file(str(height_path))
        height_arr = _squeeze_2d(height_data.data).astype(np.float32)
        transform = height_data.transform

        # Stride-downsample
        height_ds, stride = _stride_downsample(height_arr, max_grid_dim)
        self.grid_shape = height_ds.shape
        self.stride = stride

        H, W = height_ds.shape

        # Compute map coordinates via affine transform
        cols = np.arange(W) * stride
        rows = np.arange(H) * stride
        cc, rr = np.meshgrid(cols, rows)

        if transform is not None:
            eastings = transform.c + cc * transform.a
            northings = transform.f + rr * transform.e
        else:
            eastings = cc.astype(np.float32)
            northings = rr.astype(np.float32)

        # Center around (0, 0) to avoid float32 precision issues with large UTM coords
        east_mean = eastings.mean()
        north_mean = northings.mean()
        eastings = (eastings - east_mean).astype(np.float32)
        northings = (northings - north_mean).astype(np.float32)

        # Flatten to (N, 3) point cloud
        x = eastings.ravel()
        y = northings.ravel()
        z = height_ds.ravel().astype(np.float32)

        self.points = np.column_stack([x, y, z]).astype(np.float32)
        self.base_z = z.copy()

        # Scene extent for camera positioning
        self.xy_extent = max(
            float(x.max() - x.min()),
            float(y.max() - y.min()),
        )

        # Precompute triangle face indices for mesh rendering
        self.mesh_faces = _build_grid_faces(H, W)

        # Compute textures
        self._compute_height_texture()
        self._try_load_cloud_mask_texture()
        self._try_load_true_colour_texture()
        self._try_load_ice_composite_texture()
        self._try_load_albedo_texture()
        self._try_load_properties_textures()

        logger.info(
            f"  [{self.scene_id}] Loaded {self.points.shape[0]:,} points "
            f"({H}×{W} grid, stride={stride}), "
            f"xy_extent={self.xy_extent/1000:.1f} km, "
            f"textures: {list(self._textures.keys())}"
        )

    # ------------------------------------------------------------------
    # Texture computation helpers
    # ------------------------------------------------------------------

    def _resize_to_grid(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        """Resize a 2D array to exactly grid_shape using skimage.

        Args:
            arr: 2D input array.
            order: Interpolation order (0=nearest-neighbour, 1=bilinear).

        Returns:
            (H, W) float32 array.
        """
        from skimage.transform import resize as sk_resize

        H, W = self.grid_shape
        if arr.shape == (H, W):
            return arr.astype(np.float32)
        return sk_resize(
            arr, (H, W), order=order, preserve_range=True,
            anti_aliasing=(order > 0),
        ).astype(np.float32)

    def _load_rgb_composite(
        self, band_r: str, band_g: str, band_b: str, texture_name: str
    ) -> None:
        """Load an RGB composite from .SAFE scene bands and store as a texture.

        Applies joint normalization (all channels share one scale, preserving
        inter-band colour ratios / hue) followed by gamma correction.

        Args:
            band_r: Band name for red channel.
            band_g: Band name for green channel.
            band_b: Band name for blue channel.
            texture_name: Key to store in self._textures.
        """
        if self.scene_path is None or not Path(self.scene_path).exists():
            return
        try:
            from clouds_decoded.data import Sentinel2Scene

            scene = Sentinel2Scene()
            scene.read(self.scene_path)

            r = scene.get_band(band_r, reflectance=True).astype(np.float32)
            g = scene.get_band(band_g, reflectance=True).astype(np.float32)
            b = scene.get_band(band_b, reflectance=True).astype(np.float32)

            r_ds = self._resize_to_grid(r, order=1)
            g_ds = self._resize_to_grid(g, order=1)
            b_ds = self._resize_to_grid(b, order=1)

            rgb = np.stack([r_ds, g_ds, b_ds], axis=-1)

            # Clip negatives (nodata edges and anti-aliasing artefacts)
            rgb = np.clip(rgb, 0.0, None)

            # Joint normalisation: a single scale for all channels preserves
            # inter-band colour ratios so the result has correct hue.
            positive = rgb[rgb > 0]
            if positive.size > 0:
                p98 = float(np.percentile(positive, 98))
                rgb = np.clip(rgb / max(p98, 1e-9), 0.0, 1.0)

            # Gamma correction for perceptual brightness
            rgb = np.power(rgb, 0.7)
            self._textures[texture_name] = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
        except Exception as e:
            logger.warning(
                f"  [{self.scene_id}] Could not load {texture_name} texture: {e}"
            )

    # ------------------------------------------------------------------
    # Texture computation
    # ------------------------------------------------------------------

    def _compute_height_texture(self) -> None:
        """Viridis colourmap from cloud height values."""
        z = self.base_z
        z_min, z_max = np.nanmin(z), np.nanmax(z)
        denom = z_max - z_min
        if denom < 1e-12:
            denom = 1.0
        normalised = np.clip((z - z_min) / denom, 0, 1)
        normalised = np.nan_to_num(normalised, nan=0.0)
        self._textures["Height"] = _viridis_colors(normalised)

    def _try_load_cloud_mask_texture(self) -> None:
        """Categorical colours from cloud_mask.tif."""
        mask_path = self.output_dir / "cloud_mask.tif"
        if not mask_path.exists():
            return
        try:
            from clouds_decoded.data import CloudMaskData

            mask_data = CloudMaskData.from_file(str(mask_path))
            mask_arr = _squeeze_2d(mask_data.data)

            # Nearest-neighbour to preserve integer class labels
            H, W = self.grid_shape
            mask_ds = self._resize_to_grid(mask_arr, order=0)

            colors = np.zeros((H * W, 3), dtype=np.uint8)
            flat = mask_ds.ravel().astype(int)
            for cls_id, rgb in _CLOUD_MASK_COLORS.items():
                colors[flat == cls_id] = rgb

            self._textures["Cloud Mask"] = colors
        except Exception as e:
            logger.warning(f"  [{self.scene_id}] Could not load cloud mask texture: {e}")

    def _try_load_true_colour_texture(self) -> None:
        """RGB composite from B04/B03/B02 (true colour)."""
        self._load_rgb_composite("B04", "B03", "B02", "True Colour")

    def _try_load_ice_composite_texture(self) -> None:
        """SWIR false-colour composite from B12/B11/B04 (ice/snow detection)."""
        self._load_rgb_composite("B12", "B11", "B04", "Ice Composite")

    def _try_load_albedo_texture(self) -> None:
        """Mean albedo across bands, mapped to grayscale."""
        albedo_path = self.output_dir / "albedo.tif"
        if not albedo_path.exists():
            return
        try:
            from clouds_decoded.data import AlbedoData

            albedo_data = AlbedoData.from_file(str(albedo_path))
            arr = albedo_data.data  # (N_bands, H, W)
            mean_albedo = np.nanmean(arr, axis=0)  # (H, W)

            mean_ds = self._resize_to_grid(mean_albedo, order=1)

            p2, p98 = np.nanpercentile(mean_ds, [2, 98])
            denom = p98 - p2
            if denom < 1e-12:
                denom = 1.0
            normalised = np.clip((mean_ds - p2) / denom, 0, 1)
            normalised = np.nan_to_num(normalised, nan=0.0)

            gray = (normalised.ravel() * 255).astype(np.uint8)
            self._textures["Albedo"] = np.column_stack([gray, gray, gray])
        except Exception as e:
            logger.warning(f"  [{self.scene_id}] Could not load albedo texture: {e}")

    def _try_load_properties_textures(self) -> None:
        """Per-band properties textures with band-specific normalisations.

        - ``tau``:           log10, then percentile stretch → viridis
        - ``ice_liq_ratio``: fixed [0, 1] physical scale → RdBu
        - all others:        percentile stretch → viridis
        """
        props_path = self.output_dir / "properties.tif"
        if not props_path.exists():
            return
        try:
            from clouds_decoded.data import CloudPropertiesData

            props_data = CloudPropertiesData.from_file(str(props_path))
            arr = props_data.data  # (N_bands, H, W)
            band_names = getattr(
                props_data.metadata, "band_names",
                [f"band_{i}" for i in range(arr.shape[0])],
            )

            for i, name in enumerate(band_names):
                if i >= arr.shape[0]:
                    break
                band_ds = self._resize_to_grid(arr[i], order=1)

                if name == "tau":
                    # Log10 normalization — percentile-stretch the log values
                    positive = band_ds[np.isfinite(band_ds) & (band_ds > 0)]
                    if positive.size == 0:
                        continue
                    log_band = np.log10(np.maximum(band_ds, 1e-10))
                    log_valid = log_band[np.isfinite(band_ds) & (band_ds > 0)]
                    p2, p98 = np.percentile(log_valid, [2, 98])
                    denom = p98 - p2
                    if denom < 1e-12:
                        denom = 1.0
                    normalised = np.clip((log_band - p2) / denom, 0, 1)
                    normalised = np.nan_to_num(normalised, nan=0.0)
                    self._textures[f"Properties: {name}"] = _viridis_colors(
                        normalised.ravel()
                    )

                elif name == "ice_liq_ratio":
                    # Fixed physical [0, 1] scale; RdBu: blue=liquid, red=ice
                    normalised = np.clip(band_ds, 0.0, 1.0)
                    normalised = np.nan_to_num(normalised, nan=0.0)
                    self._textures[f"Properties: {name}"] = _colormap_colors(
                        normalised.ravel(), cmap="RdBu"
                    )

                else:
                    # Default: percentile stretch → viridis
                    valid = band_ds[np.isfinite(band_ds)]
                    if valid.size == 0:
                        continue
                    p2, p98 = np.percentile(valid, [2, 98])
                    denom = p98 - p2
                    if denom < 1e-12:
                        denom = 1.0
                    normalised = np.clip((band_ds - p2) / denom, 0, 1)
                    normalised = np.nan_to_num(normalised, nan=0.0)
                    self._textures[f"Properties: {name}"] = _viridis_colors(
                        normalised.ravel()
                    )

        except Exception as e:
            logger.warning(f"  [{self.scene_id}] Could not load properties textures: {e}")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def texture_names(self) -> List[str]:
        """Available texture names for this scene."""
        return list(self._textures.keys())

    def get_mesh_data(
        self,
        z_scale: float,
        texture_name: str,
        step: int = 2,
        median_kernel: int = 7,
        max_z_diff: float = 20000.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return coarsened mesh vertices, face indices, and RGBA vertex colors.

        Height is median-filtered before subsampling for smooth geometry.
        Two categories of faces are then dropped:

        1. **Curtain faces** — physical z-range across the triangle exceeds
           ``max_z_diff`` metres.
        2. **Zero-edge faces** — at least one vertex sits at z=0 (clear-sky /
           invalid pixel) while at least one other vertex is above z=0.
           Faces entirely at z=0 are kept so the ground plane is preserved.

        XY positions and vertex colors are sampled from the full-resolution
        grid at the coarse positions — no filtering applied to either.

        Args:
            z_scale: Vertical exaggeration factor (display only; all height
                thresholds are in physical metres).
            texture_name: Texture mode to use for vertex colors.
            step: Grid subsampling stride (2 = every other row/col).
            median_kernel: Size of the median-filter kernel applied to the
                height grid before subsampling (must be odd; if even, 1 is
                added). Larger values give smoother geometry.
            max_z_diff: Maximum physical height difference (metres) across any
                triangle; faces exceeding this are removed.

        Returns:
            vertices: (N, 3) float32 point positions.
            faces:    (M, 3) uint32 triangle indices (filtered).
            rgba:     (N, 4) uint8 RGBA colors (trimesh format).
        """
        from scipy.ndimage import median_filter

        H, W = self.grid_shape
        rows = np.arange(0, H, step)
        cols = np.arange(0, W, step)
        H2, W2 = len(rows), len(cols)

        # Median-filter the height grid — kernel must be odd
        kernel = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        height_2d = np.nan_to_num(self.base_z.reshape(H, W), nan=0.0)
        height_smooth = median_filter(height_2d, size=kernel).astype(np.float32)

        # Subsampled unscaled heights — used for all face filters
        z_unscaled = height_smooth[rows[:, None], cols[None, :]].ravel()

        # XY from the original (already centred) grid — no filtering
        pts_2d = self.points.reshape(H, W, 3)
        xy_sub = pts_2d[rows[:, None], cols[None, :], :2].reshape(-1, 2)
        vertices = np.column_stack(
            [xy_sub, (z_unscaled * z_scale)]
        ).astype(np.float32)

        # Build all candidate faces, then filter
        faces = _build_grid_faces(H2, W2)
        face_z = z_unscaled[faces]                    # (M, 3) physical heights
        face_min = face_z.min(axis=1)
        face_max = face_z.max(axis=1)

        # 1. Drop curtain faces (large height jump)
        keep = (face_max - face_min) <= max_z_diff

        # 2. Drop zero-edge faces: some vertices at z=0, some above
        #    (keeps the flat z=0 ground plane but removes walls rising from it)
        keep &= ~((face_min == 0.0) & (face_max > 0.0))

        faces = faces[keep]

        # Colors from the full-res texture sampled at coarse positions — no filter
        colors = self.get_colors(texture_name)
        colors_2d = colors.reshape(H, W, 3)
        rgb_sub = colors_2d[rows[:, None], cols[None, :], :].reshape(-1, 3)
        alpha = np.full((len(rgb_sub), 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb_sub, alpha], axis=1)

        return vertices, faces, rgba

    def get_points(self, z_scale: float = 1.0) -> np.ndarray:
        """Return points with z multiplied by z_scale.

        NaN heights (masked/invalid pixels) are replaced with 0 so they
        don't produce degenerate geometry.

        Args:
            z_scale: Vertical exaggeration factor.

        Returns:
            (N, 3) float32 array of [x, y, z*z_scale].
        """
        pts = self.points.copy()
        pts[:, 2] = np.nan_to_num(self.base_z * z_scale, nan=0.0)
        return pts

    def get_colors(self, texture_name: str) -> np.ndarray:
        """Return cached colour array for a texture mode.

        Args:
            texture_name: One of :attr:`texture_names`.

        Returns:
            (N, 3) uint8 colour array.

        Raises:
            KeyError: If texture_name is not available.
        """
        if texture_name not in self._textures:
            raise KeyError(
                f"Texture '{texture_name}' not available. "
                f"Choose from: {self.texture_names}"
            )
        return self._textures[texture_name]


# ---------------------------------------------------------------------------
# ViserViewer — manages the viser server
# ---------------------------------------------------------------------------


class ViserViewer:
    """3D viewer for a clouds-decoded project (viser-powered).

    Launches a viser web server showing cloud height surfaces for each
    scene in the project. Supports Wireframe / Surface / Points render
    modes and per-vertex texturing from any pipeline output.

    Args:
        project_dir: Path to the project directory (must contain project.yaml).
        host: Hostname for the viser server.
        port: Port for the viser server.
        max_grid_dim: Maximum grid dimension after downsampling.
    """

    def __init__(
        self,
        project_dir: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        max_grid_dim: int = 800,
    ) -> None:
        from clouds_decoded.project import Project

        self.host = host
        self.port = port
        self.max_grid_dim = max_grid_dim

        project = Project.load(project_dir)
        self._scene_data: Dict[str, SceneData] = {}
        self._scene_order: List[str] = []

        for scene_path in project.config.scenes:
            scene_id = project._scene_id(scene_path)
            output_dir = project._scene_output_dir(scene_id)

            if not (output_dir / "cloud_height.tif").exists():
                logger.info(f"  Skipping {scene_id} (no cloud_height.tif)")
                continue

            manifest = project._load_manifest(scene_id, scene_path)
            safe_path = manifest.scene_path or scene_path

            sd = SceneData(
                scene_id=scene_id,
                output_dir=str(output_dir),
                scene_path=safe_path,
            )
            try:
                sd.load(max_grid_dim=max_grid_dim)
                self._scene_data[scene_id] = sd
                self._scene_order.append(scene_id)
            except Exception as e:
                logger.warning(f"  Failed to load scene {scene_id}: {e}")

        if not self._scene_data:
            raise RuntimeError(
                "No scenes with cloud_height.tif found in project. "
                "Run the pipeline first."
            )

        logger.info(
            f"ViserViewer ready: {len(self._scene_data)} scene(s) loaded"
        )

    def serve(self) -> None:
        """Start the viser server and block until interrupted."""
        import viser

        server = viser.ViserServer(host=self.host, port=self.port)
        server.scene.set_background_image(np.zeros((1, 1, 3), dtype=np.uint8))
        logger.info(f"Viser server running at http://{self.host}:{self.port}")

        # Precompute union of all texture names in first-appearance order.
        # We fix the dropdown options once so viser doesn't reconnect on update.
        all_texture_names: List[str] = []
        seen: set = set()
        for scene_id in self._scene_order:
            for name in self._scene_data[scene_id].texture_names:
                if name not in seen:
                    all_texture_names.append(name)
                    seen.add(name)

        # Mutable state shared across callbacks
        state: Dict = {"handle": None, "scene_id": None}

        # -- GUI ---------------------------------------------------------------

        with server.gui.add_folder("Scene"):
            scene_select = server.gui.add_dropdown(
                "Scene",
                options=self._scene_order,
                initial_value=self._scene_order[0],
            )

        with server.gui.add_folder("Display"):
            render_mode_select = server.gui.add_dropdown(
                "Render Mode",
                options=["Points", "Surface"],
                initial_value="Points",
            )
            z_scale_slider = server.gui.add_slider(
                "Z-Scale",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=1.0,
            )
            point_size_slider = server.gui.add_slider(
                "Point Size",
                min=0.5,
                max=30.0,
                step=0.5,
                initial_value=5.0,
            )
            texture_select = server.gui.add_dropdown(
                "Texture",
                options=all_texture_names,
                initial_value=all_texture_names[0],
            )

        # -- Helpers -----------------------------------------------------------

        def _camera_params(sd: SceneData) -> Tuple[Tuple, float]:
            """Return (position, far) for the current scene and z_scale."""
            z_max = max(float(np.nanmax(sd.base_z)) * z_scale_slider.value, 1.0)
            # Place camera high enough to see the full horizontal extent.
            # For a ~60° FOV camera, visible radius ≈ height * tan(30°) ≈ 0.58*h.
            # So h ≈ xy_extent / 0.58 ≈ 1.7 * xy_extent.  Use 2× for margin.
            camera_height = z_max + sd.xy_extent * 2.0
            far = max(camera_height * 6.0, 50_000.0)
            return (0.0, 0.0, camera_height), far

        def _reset_cameras(sd: SceneData) -> None:
            pos, far = _camera_params(sd)
            for client in server.get_clients().values():
                client.camera.position = pos
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.up = (0.0, 1.0, 0.0)
                client.camera.far = far

        def _best_texture(sd: SceneData) -> str:
            """Return current dropdown value if available for this scene, else Height."""
            tex = texture_select.value
            return tex if tex in sd.texture_names else sd.texture_names[0]

        def _rebuild(sd: SceneData) -> None:
            """Remove existing handle and create a fresh point cloud or mesh."""
            if state["handle"] is not None:
                state["handle"].remove()
                state["handle"] = None

            tex = _best_texture(sd)
            if render_mode_select.value == "Surface":
                import trimesh as tm

                vertices, faces, rgba = sd.get_mesh_data(
                    z_scale_slider.value, tex
                )
                mesh = tm.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=rgba,
                    process=False,
                )
                state["handle"] = server.scene.add_mesh_trimesh(
                    name="/scene_object",
                    mesh=mesh,
                )
            else:
                pts = sd.get_points(z_scale_slider.value)
                colors = sd.get_colors(tex)
                state["handle"] = server.scene.add_point_cloud(
                    name="/scene_object",
                    points=pts,
                    colors=colors,
                    point_size=point_size_slider.value,
                    point_shape="square",
                )

        def _load_scene(scene_id: str) -> None:
            sd = self._scene_data[scene_id]
            state["scene_id"] = scene_id
            _rebuild(sd)
            _reset_cameras(sd)

        # -- Callbacks ---------------------------------------------------------

        @scene_select.on_update
        def _on_scene_change(ev) -> None:
            _load_scene(scene_select.value)

        @render_mode_select.on_update
        def _on_render_mode(ev) -> None:
            if state["scene_id"] is None:
                return
            _rebuild(self._scene_data[state["scene_id"]])

        @z_scale_slider.on_update
        def _on_z_scale(ev) -> None:
            if state["handle"] is None or state["scene_id"] is None:
                return
            sd = self._scene_data[state["scene_id"]]
            if render_mode_select.value == "Surface":
                _rebuild(sd)
            else:
                state["handle"].points = sd.get_points(z_scale_slider.value)

        @point_size_slider.on_update
        def _on_point_size(ev) -> None:
            if state["handle"] is None:
                return
            if render_mode_select.value == "Points":
                state["handle"].point_size = point_size_slider.value

        @texture_select.on_update
        def _on_texture(ev) -> None:
            if state["handle"] is None or state["scene_id"] is None:
                return
            sd = self._scene_data[state["scene_id"]]
            if render_mode_select.value == "Surface":
                _rebuild(sd)
            else:
                state["handle"].colors = sd.get_colors(_best_texture(sd))

        @server.on_client_connect
        def _on_client(client: viser.ClientHandle) -> None:
            if state["scene_id"] is not None:
                sd = self._scene_data[state["scene_id"]]
                pos, far = _camera_params(sd)
                client.camera.position = pos
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.up = (0.0, 1.0, 0.0)
                client.camera.far = far

        # -- Initial load ------------------------------------------------------

        _load_scene(self._scene_order[0])

        logger.info(
            "Viser viewer is live. "
            "Open the URL above in your browser. Press Ctrl+C to stop."
        )

        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Shutting down viser server.")
