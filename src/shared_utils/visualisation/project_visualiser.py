"""Project-level visualisation with scene navigation and layer compositing.

The :class:`ProjectVisualiser` wraps multiple :class:`Visualiser` instances
(one per scene) and provides both static overview methods and an interactive
browser-based viewer built on Panel + HoloViews.

Scene data is **lazy-loaded** — only the project database is read at init
time.  The first time a scene is accessed (via ``__getitem__``, ``overview``,
or a viewer interaction), its GeoTIFFs are read and the :class:`Visualiser`
is constructed.  In the Panel viewer, loading is done in a background thread
with parallel file reads so the UI stays responsive.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib.figure import Figure

from .layers import RGBConfig
from .visualiser import Visualiser

logger = logging.getLogger(__name__)

# Expected output filenames and the layer names they produce.
# Used to populate dropdowns *before* any data is read.
_EXPECTED_OUTPUTS = {
    "cloud_mask.tif": ["Cloud Mask"],
    "cloud_height.tif": ["Cloud Height"],
    "properties.tif": [
        "Properties: tau",
        "Properties: ice_liq_ratio",
        "Properties: r_eff_liq",
        "Properties: r_eff_ice",
        "Properties: uncertainty",
    ],
}

# Load priority: tifs first (small/fast), then RGB composites (read .SAFE bands).
_TIF_LOAD_ORDER = [
    "cloud_mask.tif",
    "cloud_height.tif",
    "properties.tif",
]


def _load_tif(path: Path):
    """Load a single GeoTIFF and return (filename, data_object) or None."""
    from clouds_decoded.data import (
        AlbedoData,
        CloudHeightGridData,
        CloudMaskData,
        CloudPropertiesData,
    )

    loaders = {
        "cloud_mask": CloudMaskData,
        "cloud_height": CloudHeightGridData,
        "albedo": AlbedoData,
        "properties": CloudPropertiesData,
    }
    cls = loaders.get(path.stem.lower())
    if cls is None:
        return None
    try:
        return cls.from_file(str(path))
    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None


class ProjectVisualiser:
    """Navigate and visualise all scenes in a clouds-decoded project.

    Args:
        project_dir: Path to the project root directory.
        rgb_config: Shared :class:`RGBConfig` for true-colour composites.
        ice_rgb_config: Shared :class:`RGBConfig` for ice composites.
        display_resolution_m: Target pixel size in metres for display.
            Defaults to 60 m.  Set to ``None`` for native resolution.
    """

    def __init__(
        self,
        project_dir: str,
        rgb_config: Optional[RGBConfig] = None,
        ice_rgb_config: Optional[RGBConfig] = None,
        display_resolution_m: Optional[float] = 60.0,
    ):
        self._project_dir = Path(project_dir)
        self._rgb_config = rgb_config
        self._ice_rgb_config = ice_rgb_config
        self._display_resolution_m = display_resolution_m

        # Lazy cache: populated on first access per scene.
        self._visualisers: Dict[str, Visualiser] = {}
        self._load_lock = threading.Lock()

        # Populated at init from DB + filesystem stat (no reads).
        self._scene_ids: List[str] = []          # complete scenes first
        self._scene_ids_complete: List[str] = []
        self._scene_ids_incomplete: List[str] = []
        self._scene_status: Dict[str, str] = {}  # key → "done"/"failed"/...
        self._scene_paths: Dict[str, Optional[str]] = {}
        self._output_dirs: Dict[str, str] = {}
        self._available_tifs: Dict[str, List[str]] = {}  # key → filenames on disk
        self._predicted_layers: Dict[str, List[str]] = {}

        self._discover_scenes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scene_ids(self) -> List[str]:
        """Available scene IDs (those with at least one output file on disk)."""
        return list(self._scene_ids)

    def __getitem__(self, scene_id: str) -> Visualiser:
        """Get the :class:`Visualiser` for a specific scene (lazy-loaded)."""
        if scene_id not in self._output_dirs:
            raise KeyError(f"Unknown scene {scene_id!r}. Available: {self._scene_ids}")
        return self._get_or_load(scene_id)

    def __len__(self) -> int:
        return len(self._scene_ids)

    def overview(self, scene_id: str, **kwargs) -> Figure:
        """Plot the overview for a single scene.

        Args:
            scene_id: Scene identifier.
            **kwargs: Forwarded to :meth:`Visualiser.overview`.

        Returns:
            The matplotlib Figure.
        """
        return self[scene_id].overview(**kwargs)

    def serve(
        self,
        port: int = 5006,
        show: bool = False,
        max_display_px: int = 1000,
    ) -> None:
        """Launch a browser-based viewer with scene navigation.

        Uses Panel + HoloViews. The viewer has a scene selector, base/overlay
        layer selectors, opacity slider, and RGB contrast controls.

        Args:
            port: Port to serve on.
            show: Open browser window automatically.
            max_display_px: Maximum pixel dimension for display performance.
        """
        layout = self._build_panel(max_display_px=max_display_px)
        import panel as pn

        pn.serve(layout, port=port, show=show, title="clouds-decoded Project Viewer")

    def panel(self, max_display_px: int = 1000):
        """Return the Panel layout for embedding in notebooks.

        Args:
            max_display_px: Maximum pixel dimension for display performance.

        Returns:
            A Panel layout object.
        """
        return self._build_panel(max_display_px=max_display_px)

    # ------------------------------------------------------------------
    # Internal — discovery (fast, no data reads)
    # ------------------------------------------------------------------

    def _discover_scenes(self) -> None:
        """Query the project DB and stat the filesystem for output files.

        No GeoTIFFs are opened — only ``Path.exists()`` checks.
        Scenes are partitioned into complete (status ``'done'``) and
        incomplete.  Complete scenes are listed first.
        """
        from clouds_decoded.project import Project

        project = Project.load(str(self._project_dir))
        rows = project.db.get_all()

        complete: List[str] = []
        incomplete: List[str] = []

        for row in rows:
            scene_id = row["scene_id"]
            crop_window = row.get("crop_window")
            output_dir = Path(str(project._scene_output_dir(scene_id, crop_window)))
            scene_path = row.get("path")
            status = row.get("status", "staged")
            key = f"{scene_id}_{crop_window}" if crop_window else scene_id

            # Check which tif files actually exist on disk.
            tifs_on_disk = [
                fname for fname in _EXPECTED_OUTPUTS
                if (output_dir / fname).exists()
            ]

            # Predict which layers will be available.
            predicted: List[str] = []
            if scene_path and Path(scene_path).exists():
                predicted.extend(["True Colour", "Ice Composite"])
            for fname in tifs_on_disk:
                layer_names = _EXPECTED_OUTPUTS.get(fname)
                if layer_names is not None:
                    predicted.extend(layer_names)

            if not predicted:
                continue

            self._scene_paths[key] = scene_path
            self._output_dirs[key] = str(output_dir)
            self._scene_status[key] = status
            self._available_tifs[key] = tifs_on_disk
            self._predicted_layers[key] = predicted

            if status == "done":
                complete.append(key)
            else:
                incomplete.append(key)

        self._scene_ids_complete = complete
        self._scene_ids_incomplete = incomplete
        self._scene_ids = complete + incomplete

        if not self._scene_ids:
            logger.warning("No scenes with output files found in project.")

    # ------------------------------------------------------------------
    # Internal — lazy loading with parallel reads
    # ------------------------------------------------------------------

    def _get_or_load(self, scene_id: str) -> Visualiser:
        """Return a cached Visualiser, or load it from disk on first access.

        GeoTIFFs are read in parallel via a thread pool.  The .SAFE scene
        (for RGB composites) is loaded concurrently alongside the tifs.
        """
        if scene_id in self._visualisers:
            return self._visualisers[scene_id]

        with self._load_lock:
            # Double-check after acquiring lock.
            if scene_id in self._visualisers:
                return self._visualisers[scene_id]

            output_dir = Path(self._output_dirs[scene_id])
            scene_path = self._scene_paths.get(scene_id)
            available_tifs = self._available_tifs.get(scene_id, [])

            logger.info(f"Loading scene {scene_id} …")
            vis = self._load_scene_parallel(output_dir, scene_path, available_tifs)
            self._visualisers[scene_id] = vis
            self._predicted_layers[scene_id] = vis.layer_names

        return vis

    def _load_scene_parallel(
        self,
        output_dir: Path,
        scene_path: Optional[str],
        available_tifs: Optional[List[str]] = None,
    ) -> Visualiser:
        """Load a scene's outputs using a thread pool for parallelism.

        Only tifs listed in *available_tifs* are attempted (determined
        during discovery).  This avoids errors from missing outputs in
        incomplete runs.
        """
        vis = Visualiser(
            rgb_config=self._rgb_config,
            ice_rgb_config=self._ice_rgb_config,
            display_resolution_m=self._display_resolution_m,
        )

        # Only load tifs that were confirmed to exist during discovery.
        tifs_to_load = set(available_tifs) if available_tifs is not None else set()
        tif_paths = [
            output_dir / fname
            for fname in _TIF_LOAD_ORDER
            if fname in tifs_to_load
        ]

        n_workers = len(tif_paths) + (1 if scene_path else 0)
        if n_workers == 0:
            return vis

        with ThreadPoolExecutor(max_workers=min(n_workers, 6)) as pool:
            # Submit tif reads.
            tif_futures = {
                pool.submit(_load_tif, p): p.name for p in tif_paths
            }

            # Submit .SAFE scene read (RGB composites) concurrently.
            scene_future = None
            if scene_path and Path(scene_path).exists():
                scene_future = pool.submit(self._load_safe_scene, scene_path)

            # Collect tif results as they complete.
            for future in as_completed(tif_futures):
                data = future.result()
                if data is not None:
                    vis._detect_and_add(data)

            # Collect RGB composites.
            if scene_future is not None:
                scene = scene_future.result()
                if scene is not None:
                    vis._add_scene_layers(scene)

        return vis

    @staticmethod
    def _load_safe_scene(scene_path: str):
        """Load a Sentinel2Scene from a .SAFE directory (for RGB composites)."""
        try:
            from clouds_decoded.data import Sentinel2Scene

            scene = Sentinel2Scene()
            scene.read(scene_path)
            return scene
        except Exception as e:
            logger.warning(f"Could not load scene for RGB composites: {e}")
            return None

    def _layer_names_for(self, scene_id: str) -> List[str]:
        """Return layer names: actual if loaded, predicted otherwise."""
        if scene_id in self._visualisers:
            return self._visualisers[scene_id].layer_names
        return self._predicted_layers.get(scene_id, [])

    # ------------------------------------------------------------------
    # Internal — Panel viewer
    # ------------------------------------------------------------------

    def _build_panel(self, max_display_px: int = 1000):
        """Build the Panel+HoloViews interactive layout."""
        import panel as pn

        # --- Widgets ---
        # Build scene options: complete first, then incomplete with a marker.
        scene_options = {}
        for key in self._scene_ids_complete:
            scene_options[key] = key
        for key in self._scene_ids_incomplete:
            status = self._scene_status.get(key, "?")
            scene_options[f"\u26a0 {key} ({status})"] = key

        first_value = self._scene_ids[0] if self._scene_ids else None

        scene_select = pn.widgets.Select(
            name="Scene",
            options=scene_options,
            value=first_value,
        )

        # Eagerly load the first scene so the initial render works.
        if first_value is not None:
            try:
                self._get_or_load(first_value)
            except Exception as e:
                logger.warning(f"Failed to load initial scene: {e}")

        def _overlay_options_for(scene_id):
            """Non-RGB, non-albedo layer names for the overlay dropdown.

            Only includes layers whose backing tif exists on disk.
            """
            names = self._layer_names_for(scene_id) if scene_id else []
            available = [
                n for n in names
                if n not in ("True Colour", "Ice Composite")
                and not n.startswith("Albedo")
            ]
            return available or ["(no layers)"]

        initial_overlays = _overlay_options_for(scene_select.value)

        # --- Base layer: toggle + radio ---
        base_toggle = pn.widgets.Toggle(
            name="On", value=True, button_type="success", width=60,
        )
        base_radio = pn.widgets.RadioButtonGroup(
            name="Base Layer",
            options=["True Colour", "Ice Composite"],
            value="True Colour",
        )

        # --- Overlay layer: toggle + select ---
        overlay_toggle = pn.widgets.Toggle(
            name="Off", value=False, button_type="default", width=60,
        )
        overlay_select = pn.widgets.Select(
            name="",
            options=initial_overlays,
            value=initial_overlays[0] if initial_overlays else None,
            width=230,
        )

        def _sync_toggle_label(toggle):
            def _cb(event):
                if event.new:
                    toggle.name = "On"
                    toggle.button_type = "success"
                else:
                    toggle.name = "Off"
                    toggle.button_type = "default"
            return _cb

        base_toggle.param.watch(_sync_toggle_label(base_toggle), "value")
        overlay_toggle.param.watch(_sync_toggle_label(overlay_toggle), "value")
        opacity_slider = pn.widgets.FloatSlider(
            name="Overlay Opacity", start=0.0, end=1.0, step=0.05, value=0.5,
        )
        gamma_slider = pn.widgets.FloatSlider(
            name="Gamma", start=0.1, end=3.0, step=0.05, value=0.65,
        )
        gain_slider = pn.widgets.FloatSlider(
            name="Gain", start=0.1, end=5.0, step=0.1, value=1.0,
        )

        loading_indicator = pn.indicators.LoadingSpinner(
            value=False, size=25, name="",
        )

        # Hidden trigger: bumped after a scene finishes loading to force
        # the bound render functions to re-execute.
        _render_trigger = pn.widgets.IntInput(value=0, name="")

        # --- Callbacks ---
        def _on_scene_change(event):
            scene_id = event.new
            if scene_id is None:
                return

            # Update overlay options immediately (predicted layers).
            opts = _overlay_options_for(scene_id)
            overlay_select.options = opts
            overlay_select.value = opts[0] if opts else None

            # If already cached, just bump the trigger to re-render.
            if scene_id in self._visualisers:
                _render_trigger.value += 1
                return

            # Load in the main thread (Panel serialises callbacks anyway).
            loading_indicator.value = True
            try:
                self._get_or_load(scene_id)
            except Exception as e:
                logger.warning(f"Failed to load scene {scene_id}: {e}")
            finally:
                loading_indicator.value = False

            # If the user switched away while we were loading, don't
            # bother re-rendering — they'll get the new scene instead.
            if scene_select.value != scene_id:
                return

            # Update overlay options now that real layers are known.
            opts = _overlay_options_for(scene_id)
            overlay_select.options = opts
            overlay_select.value = opts[0] if opts else None

            # Bump trigger to re-render with loaded data.
            _render_trigger.value += 1

        scene_select.param.watch(_on_scene_change, "value")

        # --- Render functions ---
        # Image and colorbar are separate figures so they live in
        # independent Panel panes.  The image never shifts when the
        # colorbar appears or disappears.

        def _render_image(
            scene_id, base_on, base_name, overlay_on, overlay_name,
            opacity, gamma, gain, _trigger,
        ):
            import matplotlib
            matplotlib.use("agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 10))

            if scene_id is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                plt.close(fig)
                return fig

            vis = self._visualisers.get(scene_id)
            if vis is None:
                ax.text(0.5, 0.5, "Loading\u2026", ha="center",
                        va="center", transform=ax.transAxes, fontsize=14)
                plt.close(fig)
                return fig

            from .layers import Layer, RGBConfig as _RGBConfig, _apply_rgb_config
            from .static import render_to_axes

            ax.set_xticks([])
            ax.set_yticks([])

            # --- Base layer (RGB) ---
            if base_on and base_name:
                base_layer = vis._layers.get(base_name)
                if base_layer is not None and base_layer.is_rgb and base_layer.rgb_config is not None:
                    from .viewer import InteractiveViewer

                    raw = InteractiveViewer._recover_raw_rgb(base_layer)
                    base_cfg = base_layer.rgb_config
                    cfg = _RGBConfig(
                        gamma=gamma,
                        gain=base_cfg.gain * gain,
                        offset=base_cfg.offset * gain,
                    )
                    new_data = _apply_rgb_config(raw, cfg)
                    base_layer = Layer(
                        name=base_layer.name, data=new_data,
                        render=base_layer.render,
                        extent=base_layer.extent,
                        resolution_m=base_layer.resolution_m,
                        is_rgb=True, rgb_config=cfg,
                    )
                if base_layer is not None:
                    show_base_title = not (overlay_on and overlay_name)
                    render_to_axes(base_layer, ax, show_title=show_base_title)

            # --- Overlay layer ---
            if overlay_on and overlay_name:
                overlay_layer = vis._layers.get(overlay_name)
                if overlay_layer is not None:
                    alpha = opacity if base_on else None
                    im = render_to_axes(overlay_layer, ax, alpha=alpha)
                    if im is not None and overlay_layer.render.categorical:
                        labels = overlay_layer.render.category_labels or {}
                        if labels:
                            from .static import _build_cmap
                            cmap = _build_cmap(overlay_layer.render)
                            handles = [
                                plt.Line2D(
                                    [0], [0], marker="s", color="w",
                                    markerfacecolor=cmap(i), markersize=10,
                                    label=lbl,
                                )
                                for i, lbl in sorted(labels.items())
                            ]
                            ax.legend(handles=handles, loc="upper right",
                                      framealpha=0.8)

            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plt.close(fig)
            return fig

        def _render_colorbar(
            scene_id, overlay_on, overlay_name, _trigger,
        ):
            """Render a standalone vertical colorbar figure, or empty."""
            import matplotlib
            matplotlib.use("agg")
            import matplotlib.pyplot as plt
            from .static import _build_cmap, _build_norm

            fig, ax = plt.subplots(figsize=(0.8, 8))
            ax.set_visible(False)

            vis = self._visualisers.get(scene_id) if scene_id else None
            if not (overlay_on and overlay_name and vis):
                plt.close(fig)
                return fig

            overlay_layer = vis._layers.get(overlay_name)
            if overlay_layer is None or overlay_layer.render.categorical:
                plt.close(fig)
                return fig

            import matplotlib.colors as mcolors
            render = overlay_layer.render
            norm = _build_norm(render) or mcolors.Normalize(
                vmin=render.vmin, vmax=render.vmax)
            cmap = _build_cmap(render)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar_ax = fig.add_axes([0.05, 0.05, 0.35, 0.90])
            plt.colorbar(sm, cax=cbar_ax, orientation="vertical")
            cbar_ax.tick_params(labelsize=8, length=2, pad=1)

            plt.close(fig)
            return fig

        image_pane = pn.pane.Matplotlib(
            pn.bind(
                _render_image, scene_select,
                base_toggle, base_radio,
                overlay_toggle, overlay_select,
                opacity_slider, gamma_slider, gain_slider,
                _render_trigger,
            ),
            tight=True,
            dpi=150,
            sizing_mode="scale_height",
        )

        colorbar_pane = pn.pane.Matplotlib(
            pn.bind(
                _render_colorbar, scene_select,
                overlay_toggle, overlay_select,
                _render_trigger,
            ),
            tight=True,
            dpi=150,
            sizing_mode="scale_height",
        )

        sidebar = pn.Column(
            "## Project Viewer",
            pn.Row(scene_select, loading_indicator),
            "---",
            "### Base",
            pn.Row(base_toggle, base_radio, align="center"),
            "---",
            "### Overlay",
            pn.Row(overlay_toggle, overlay_select, align="center"),
            opacity_slider,
            "---",
            "### RGB Controls",
            gamma_slider,
            gain_slider,
            width=320,
            sizing_mode="fixed",
        )

        return pn.Row(
            sidebar,
            image_pane,
            colorbar_pane,
            sizing_mode="stretch_both",
        )
