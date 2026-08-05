"""The multitemporal albedo pre-run stage.

Runs before the ordinary per-scene project run when the albedo step's config
selects ``method: multitemporal``:

1. Validate the staged scenes form a usable time series (cheap, fail fast).
2. Run the ``cloud_mask`` step for all pending scenes through the ordinary
   orchestrator (``Project.run(only_steps=["cloud_mask"])``) — scenes stay
   ``staged`` in the DB with a completed cloud_mask manifest entry.
3. Build / refresh the 180 m stack from the masks (full tile, or the shared
   crop window when the stage is run with one).
4. Fit (or load the cached) cluster + temporal model.
5. Pre-populate ``albedo.tif`` + manifest for every scene with a mask.

After the stage, the ordinary run resumes each scene at the first incomplete
step — with the ``full-workflow-multitemporal`` recipe that is
``cloud_height``, and the pre-populated albedo is honoured, not recomputed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _albedo_model_key(path: Path) -> Optional[str]:
    """Read the model_key stamped into an albedo.tif's metadata tag
    (None if absent or unreadable)."""
    import json

    import rasterio

    from clouds_decoded.constants import METADATA_TAG

    try:
        with rasterio.open(path) as src:
            meta = json.loads(src.tags().get(METADATA_TAG, "{}"))
        return meta.get("model_key")
    except Exception:
        return None


class MultitemporalAlbedoStage:
    """Orchestrates the tile-level fit + per-scene pre-population."""

    def __init__(self, project):
        self.project = project
        self.dir = project.project_dir / "multitemporal"

    def _albedo_config(self):
        return self.project._load_step_config("albedo")

    @staticmethod
    def is_selected(project) -> bool:
        """True when the project's albedo step selects the multitemporal
        method (and the workflow has an albedo step at all)."""
        try:
            if "albedo" not in project.steps:
                return False
            cfg = project._load_step_config("albedo")
        except Exception:
            return False
        return getattr(cfg, "method", None) == "multitemporal"

    def run(self, parallel: bool = False, verbose: bool = False,
            progress: bool = True, force: bool = False,
            parallelism: Optional[dict] = None,
            crop_window: Optional[str] = None) -> None:
        """Execute the full pre-run stage (phases 1-5).

        Args:
            crop_window: Restrict the whole stage to scenes staged with this
                crop window (``'col,row,w,h'`` in B02 pixels). None = full
                scenes. A cropped stage fits on the crop's analysis grid —
                much faster, useful for validation runs.
        """
        from .fit import fit_and_save
        from .prepopulate import prepopulate_scene
        from .stack import build_stack, stack_signature
        from .validation import validate_time_series

        project = self.project
        albedo_cfg = self._albedo_config()
        params = albedo_cfg.multitemporal
        workflow_steps = project.steps
        if ("cloud_height" in workflow_steps
                and workflow_steps.index("albedo")
                > workflow_steps.index("cloud_height")):
            logger.warning(
                "Workflow orders albedo after cloud_height — pre-populated "
                "albedo would be recomputed on resume. Use the "
                "'full-workflow-multitemporal' recipe.")

        rows = [r for r in project.db.get_all()
                if r["crop_window"] == crop_window]
        scene_rows = [(r["path"], r["crop_window"]) for r in rows]
        summary = validate_time_series(scene_rows, params)
        logger.warning(
            "Multitemporal albedo stage: tile %s, %d scenes, %d-%d%s",
            summary["tile"], summary["n_scenes"],
            summary["years"][0], summary["years"][-1],
            f", crop {crop_window}" if crop_window else "")

        logger.warning("Stage 1/4: cloud masks (restricted project run)...")
        project.run(parallel=parallel, verbose=verbose, progress=progress,
                    parallelism=parallelism, force=force,
                    crop_window=crop_window,
                    only_steps=["cloud_mask"], run_stats=False)

        logger.warning("Stage 2/4: building %d m stack...", params.grid_res)
        mask_rows = []
        for r in rows:
            mask_path = (project._scene_output_dir(r["scene_id"], crop_window)
                         / "cloud_mask.tif")
            if mask_path.exists():
                mask_rows.append((r["path"], mask_path))
            else:
                logger.warning(
                    "no cloud mask for %s — excluded from the fit and not "
                    "pre-populated", r["scene_id"])
        # Scene extractions depend on the masks, so key the cache (and the
        # model signature) on the cloud_mask config hash — a mask config
        # change invalidates both automatically.
        cm_hash = project._config_hash("cloud_mask")
        crop_tag = (f"_crop{crop_window.replace(',', '_')}"
                    if crop_window else "")
        stack = build_stack(
            mask_rows, self.dir / f"scene_cache_{cm_hash}{crop_tag}",
            params.grid_res, crop_window)
        sig = f"{stack_signature(stack)}_{cm_hash}"

        # Post-stack coverage check (needs the masks, so runs here).
        cnt = stack["clear"].sum(0)
        covered = float((cnt >= params.min_clear_obs).mean())
        if covered < 0.5:
            logger.warning(
                "Only %.0f%% of tile pixels have >= %d clear observations — "
                "fit quality will be poor; consider staging more scenes.",
                100 * covered, params.min_clear_obs)

        logger.warning("Stage 3/4: fitting cluster model...")
        model = fit_and_save(stack, sig, params,
                             self.dir / f"model{crop_tag}.npz")

        logger.warning("Stage 4/4: pre-populating albedo.tif per scene...")
        git_hash = project._get_git_hash()
        sid_to_t = {str(s): i for i, s in enumerate(stack["sids"])}
        n_written = n_skipped = 0
        stale: list = []
        for scene_path, mask_path in mask_rows:
            sid = project._scene_id(scene_path)
            footprint = None
            t_idx = sid_to_t.get(sid)
            if t_idx is not None:
                footprint = np.asarray(stack["valid"][t_idx])
            try:
                written = prepopulate_scene(
                    project, model, scene_path, albedo_cfg,
                    git_hash=git_hash, footprint=footprint,
                    crop_window=crop_window, force=force)
            except Exception as exc:
                logger.error("pre-populate failed for %s: %s", sid, exc)
                continue
            n_written += int(written)
            if not written:
                n_skipped += 1
                # Skipped via config hash — but was it produced by the
                # *current* fit? The data signature is not part of the
                # config hash, so outputs can silently outlive their model.
                out = (project._scene_output_dir(sid, crop_window)
                       / "albedo.tif")
                if _albedo_model_key(out) != model.get("key"):
                    stale.append(sid)
        logger.warning(
            "Multitemporal albedo stage complete: %d albedo.tif written, "
            "%d already up to date.", n_written, n_skipped)
        if stale:
            logger.warning(
                "%d scene(s) have albedo.tif from a SUPERSEDED multitemporal "
                "fit (the model has since been refit on different data). "
                "Differences are typically far below the model's own error, "
                "but for outputs consistent with the current fit re-run with "
                "--force ('project run --force' or 'project prefit --force'). "
                "Affected: %s%s",
                len(stale), ", ".join(stale[:5]),
                f" (+{len(stale) - 5} more)" if len(stale) > 5 else "")
