"""Project management: config directories, per-scene outputs, and resumability.

A Project is an optional directory structure that holds editable module configs,
organizes outputs per Sentinel-2 scene, and supports resuming interrupted runs.

Usage:
    project = Project.init("./my_analysis",
                           scenes=["/data/S2A_....SAFE"], pipeline="full-workflow")
    project.run()

    # Later, resume or add scenes:
    project = Project.load("./my_analysis")
    project.stage("/data/S2B_....SAFE")
    project.run()
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import logging
import duckdb
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import re

import yaml
from pydantic import BaseModel, ConfigDict, Field
from clouds_decoded.constants import METADATA_TAG

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def _require_safe_sql_id(name: str, label: str = "identifier") -> None:
    """Raise ValueError if *name* is not a safe SQL identifier (lowercase, alphanumeric + underscore)."""
    if not _SAFE_ID_RE.match(name):
        raise ValueError(
            f"Unsafe SQL {label} {name!r}: must match [a-z][a-z0-9_]*"
        )


# ---------------------------------------------------------------------------
# run_id helper
# ---------------------------------------------------------------------------

def _make_run_id(scene_id: str, crop_window: Optional[str]) -> str:
    """Content-addressed identifier for a (scene_id, crop_window) pair.

    Args:
        scene_id: The scene ID string (e.g. ``"S2A_MSIL1C_..."``).
        crop_window: The crop window string (e.g. ``"0,0,512,512"``), or None.

    Returns:
        First 16 hex characters of the SHA-256 of ``"{scene_id}:{crop_window}"``.
    """
    key = f"{scene_id}:{crop_window or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-scene log routing
# ---------------------------------------------------------------------------

_scene_log_path_var: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "_scene_log_path", default=None
)

_file_handler_cache: Dict[Path, logging.FileHandler] = {}
_file_handler_lock = threading.Lock()


def _get_or_create_file_handler(path: Path) -> logging.FileHandler:
    with _file_handler_lock:
        if path not in _file_handler_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(path), mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            ))
            _file_handler_cache[path] = fh
        return _file_handler_cache[path]


def _close_all_file_handlers() -> None:
    with _file_handler_lock:
        for fh in _file_handler_cache.values():
            try:
                fh.close()
            except Exception:
                pass
        _file_handler_cache.clear()


class _SceneRoutingHandler(logging.Handler):
    """Routes log records to the per-scene log file for the current thread context."""

    def emit(self, record: logging.LogRecord) -> None:
        path = _scene_log_path_var.get(None)
        if path is None:
            return
        try:
            _get_or_create_file_handler(path).emit(record)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Pipeline data structures
# ---------------------------------------------------------------------------

_STOP = object()  # End-of-stream sentinel for inter-stage queues.


@dataclasses.dataclass
class _PipelineCtx:
    """Per-scene state packet that flows through the pipeline stages."""

    scene_path: str
    crop_window: Optional[str]
    log_path: Path
    force: bool
    unsafe: bool
    git_hash: Optional[str]

    # Set by reader stage:
    scene_out: Optional[Path] = None
    manifest: Optional["SceneManifest"] = None
    steps_to_run: List[str] = dataclasses.field(default_factory=list)
    first_step_idx: int = 0

    # Generic token store: holds "scene", "cloud_mask", "cloud_height", etc.
    # Replaces the old named fields (mask_result, height_result, albedo_result,
    # refocused_scene, scene). Token names come from the workflow YAML.
    intermediates: Dict[str, Any] = dataclasses.field(default_factory=dict)

    failed: bool = False
    error: Optional[Exception] = None


# ---------------------------------------------------------------------------
# Workflow graph models
# ---------------------------------------------------------------------------

class WorkflowStepDef(BaseModel):
    """Declarative definition of one step in the workflow graph.

    The ``inputs`` and ``keyword_inputs`` fields name tokens in the shared
    intermediates dict; ``output`` names the token this step writes back.
    ``output_file`` is the disk filename (None = ephemeral, no write/resume).
    """

    name: str                           # unique step ID within this workflow
    processor: str                      # key into PROCESSORS registry
    config: str                         # config YAML filename (relative to configs/)
    inputs: List[str] = []              # positional token names (first = scene arg)
    keyword_inputs: Dict[str, str] = {} # {param_name: token_name}
    output: Optional[str] = None        # output token name; None = terminal step
    output_file: Optional[str] = None   # disk filename; None = ephemeral
    postprocess: bool = False           # apply processor's postprocess_fn to result
    model_config = ConfigDict(extra="forbid")


class WorkflowDef(BaseModel):
    """An ordered sequence of WorkflowStepDef nodes defining the pipeline graph."""

    steps: List[WorkflowStepDef]
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Processor registry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProcessorDef:
    """Implementation descriptor for one processor type.

    Contains only what the executor cannot derive from the workflow YAML —
    factory functions, data-class constructors, and side-effect hooks.
    Graph structure (which tokens flow in/out) lives in WorkflowStepDef.
    """

    config_loader: Callable[[str], Any]             # (yaml_path) -> config
    processor_factory: Callable[[Any], Dict[str, Any]]  # (config) -> {key: proc}
    config_factory: Callable[[], Any]               # () -> default config instance
    output_loader: Optional[Callable[[str], Any]] = None  # (path) -> result; for resume
    postprocess_fn: Optional[Callable] = None       # (result) -> result
    prefetch_fn: Optional[Callable] = None          # (scene, config) -> None
    on_complete: Optional[Callable] = None          # (result, config, scene_out)
    get_output_file: Optional[Callable] = None      # (config) -> str; dynamic filename
    clears_scene_caches: Tuple[str, ...] = ()       # scene dict attrs to clear
    clears_scene_if_refocused: bool = False         # clear original caches post-refocus


# -- Config loaders ----------------------------------------------------------

def _load_cloud_mask_config(path: str):
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
    return CloudMaskConfig.from_yaml(path)


def _load_cloud_height_emulator_config(path: str):
    from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
    return CloudHeightEmulatorConfig.from_yaml(path)


def _load_cloud_height_physics_config(path: str):
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    return CloudHeightConfig.from_yaml(path)


def _load_albedo_config(path: str):
    from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
    return AlbedoEstimatorConfig.from_yaml(path)


def _load_refocus_config(path: str):
    from clouds_decoded.modules.refocus.config import RefocusConfig
    return RefocusConfig.from_yaml(path)


def _load_cloud_properties_config(path: str):
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig, ShadingRefl2PropConfig
    base = Refl2PropConfig.from_yaml(path)
    return ShadingRefl2PropConfig.from_yaml(path) if base.method == "shading" else base


# -- Band prefetch functions -------------------------------------------------

def _prefetch_cloud_mask(scene, config) -> None:
    """Pre-compute all 13 bands at cloud_mask working_resolution into scene cache."""
    from clouds_decoded.constants import BANDS
    ref = scene.bands.get("B02") or next(iter(scene.bands.values()))
    scale = abs(scene.transform[0]) / float(config.working_resolution)
    target_shape = (int(ref.shape[0] * scale), int(ref.shape[1] * scale))
    scene.prefetch_at_shape(BANDS, target_shape)


def _prefetch_cloud_height(scene, config) -> None:
    """Pre-compute emulator bands at cloud_height working_resolution into scene cache."""
    ref_name = "B02" if "B02" in config.bands else config.bands[0]
    ref = scene.get_band(ref_name, reflectance=True)
    scale = abs(scene.transform.a) / float(config.working_resolution)
    target_shape = (max(1, round(ref.shape[0] * scale)), max(1, round(ref.shape[1] * scale)))
    scene.prefetch_at_shape(config.bands, target_shape)


# -- Processor factories -----------------------------------------------------

def _make_cloud_mask_processors(config) -> Dict[str, Any]:
    from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor, ThresholdCloudMaskProcessor
    if config.method == "threshold":
        return {"cloud_mask": ThresholdCloudMaskProcessor(config),
                "_cloud_mask_postprocessor": CloudMaskProcessor(config)}
    cm = CloudMaskProcessor(config)
    return {"cloud_mask": cm, "_cloud_mask_postprocessor": cm}


def _make_cloud_height_emulator_processors(config) -> Dict[str, Any]:
    from clouds_decoded.modules.cloud_height_emulator.processor import CloudHeightEmulatorProcessor
    return {"cloud_height_emulator": CloudHeightEmulatorProcessor(config)}


def _make_cloud_height_physics_processors(config) -> Dict[str, Any]:
    from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
    return {"cloud_height": CloudHeightProcessor(config)}


def _make_albedo_processors(config) -> Dict[str, Any]:
    from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator
    return {"albedo": AlbedoEstimator(config)}


def _make_refocus_processors(config) -> Dict[str, Any]:
    from clouds_decoded.modules.refocus.processor import RefocusProcessor
    return {"refocus": RefocusProcessor(config)}


def _make_cloud_properties_processors(config) -> Dict[str, Any]:
    if config.method == "shading":
        from clouds_decoded.modules.refl2prop.processor import ShadingPropertyInverter
        return {"cloud_properties": ShadingPropertyInverter(config)}
    from clouds_decoded.modules.refl2prop.processor import CloudPropertyInverter
    return {"cloud_properties": CloudPropertyInverter(config)}


# -- Config factories (lazy; used by _write_default_configs) ----------------

def _config_factory_cloud_mask():
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
    return CloudMaskConfig()


def _config_factory_cloud_height_emulator():
    from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
    return CloudHeightEmulatorConfig()


def _config_factory_cloud_height_physics():
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
    return CloudHeightConfig()


def _config_factory_albedo():
    from clouds_decoded.modules.albedo_estimator.config import AlbedoEstimatorConfig
    return AlbedoEstimatorConfig()


def _config_factory_refocus():
    from clouds_decoded.modules.refocus.config import RefocusConfig
    return RefocusConfig()


def _config_factory_cloud_properties():
    from clouds_decoded.modules.refl2prop.config import Refl2PropConfig
    return Refl2PropConfig()


# -- Output loaders (lazy; used by _load_intermediates on resume) ------------

def _load_cloud_mask_result(path: str):
    from clouds_decoded.data import CloudMaskData
    return CloudMaskData.from_file(path)


def _load_cloud_height_result(path: str):
    from clouds_decoded.data import CloudHeightGridData
    return CloudHeightGridData.from_file(path)


def _load_albedo_result(path: str):
    from clouds_decoded.data import AlbedoData
    return AlbedoData.from_file(path)


# -- Postprocess function ----------------------------------------------------

def _postprocess_cloud_mask(result):
    """Convert raw 4-class mask to binary for downstream consumers."""
    from clouds_decoded.modules.cloud_mask.config import PostProcessParams
    from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor
    return CloudMaskProcessor().postprocess(result, PostProcessParams())


# -- Side-effect hooks -------------------------------------------------------

def _refocus_on_complete(result_scene, config, scene_out: Path) -> None:
    if config.save_refocused:
        from clouds_decoded.cli.entry import _save_refocused_bands
        _save_refocused_bands(result_scene, config, str(scene_out / "refocused"))


def _cloud_properties_output_file(config) -> str:
    return "properties_shading.tif" if config.method == "shading" else "properties.tif"


# -- Processor registry ------------------------------------------------------

PROCESSORS: Dict[str, ProcessorDef] = {
    "cloud_mask": ProcessorDef(
        config_loader=_load_cloud_mask_config,
        processor_factory=_make_cloud_mask_processors,
        config_factory=_config_factory_cloud_mask,
        output_loader=_load_cloud_mask_result,
        postprocess_fn=_postprocess_cloud_mask,
        prefetch_fn=_prefetch_cloud_mask,
    ),
    "cloud_height_emulator": ProcessorDef(
        config_loader=_load_cloud_height_emulator_config,
        processor_factory=_make_cloud_height_emulator_processors,
        config_factory=_config_factory_cloud_height_emulator,
        output_loader=_load_cloud_height_result,
        prefetch_fn=_prefetch_cloud_height,
        clears_scene_caches=("_resized_band_cache",),
    ),
    "cloud_height": ProcessorDef(
        config_loader=_load_cloud_height_physics_config,
        processor_factory=_make_cloud_height_physics_processors,
        config_factory=_config_factory_cloud_height_physics,
        output_loader=_load_cloud_height_result,
    ),
    "albedo": ProcessorDef(
        config_loader=_load_albedo_config,
        processor_factory=_make_albedo_processors,
        config_factory=_config_factory_albedo,
        output_loader=_load_albedo_result,
    ),
    "refocus": ProcessorDef(
        config_loader=_load_refocus_config,
        processor_factory=_make_refocus_processors,
        config_factory=_config_factory_refocus,
        output_loader=None,   # ephemeral — no disk write, cannot resume
        on_complete=_refocus_on_complete,
        clears_scene_if_refocused=True,
    ),
    "cloud_properties": ProcessorDef(
        config_loader=_load_cloud_properties_config,
        processor_factory=_make_cloud_properties_processors,
        config_factory=_config_factory_cloud_properties,
        output_loader=None,   # terminal step — no in-memory resume needed
        get_output_file=_cloud_properties_output_file,
    ),
}

_ALL_STAGE_NAMES: List[str] = list(PROCESSORS.keys())


# ---------------------------------------------------------------------------
# Recipe loading
# ---------------------------------------------------------------------------

_RECIPES_DIR = Path(__file__).parent / "workflows" / "recipes"


def _get_recipe(name: str) -> WorkflowDef:
    """Load a bundled workflow recipe by name.

    Args:
        name: Recipe filename stem (e.g. ``"full-workflow"``).

    Returns:
        Parsed :class:`WorkflowDef`.

    Raises:
        ValueError: If the recipe name is not found in the bundled recipes.
    """
    path = _RECIPES_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _RECIPES_DIR.glob("*.yaml"))
        raise ValueError(
            f"Unknown recipe '{name}'. Available: {available}"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return WorkflowDef(steps=data["steps"])


_default_wf = _get_recipe("full-workflow")

PIPELINE_STEPS: Dict[str, List[str]] = {
    "full-workflow": [s.name for s in _default_wf.steps],
}

STEP_OUTPUT_FILE: Dict[str, Optional[str]] = {
    s.name: s.output_file for s in _default_wf.steps
}

STEP_CONFIG_FILE: Dict[str, str] = {
    s.name: s.config for s in _default_wf.steps
}


# ---------------------------------------------------------------------------
# Scene database
# ---------------------------------------------------------------------------

class SceneDB:
    """DuckDB-backed scene registry for a project.

    Each row in the ``runs`` table represents one ``(scene, crop_window)``
    processing unit, identified by a content-addressed ``run_id``.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._db = duckdb.connect(str(db_path))
        try:
            self._db.execute("INSTALL spatial")
            self._db.execute("LOAD spatial")
            self._spatial_available = True
        except Exception as exc:
            logger.warning(
                "DuckDB spatial extension unavailable: %s. "
                "Spatial footprint queries will not work.",
                exc,
            )
            self._spatial_available = False
        self._init_schema()

    @contextlib.contextmanager
    def _conn(self):
        with self._lock:
            self._db.begin()
            try:
                yield self._db
                self._db.commit()
            except Exception:
                try:
                    self._db.rollback()
                except Exception:
                    pass
                raise

    def _init_schema(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id                TEXT PRIMARY KEY,
                    scene_id              TEXT NOT NULL,
                    path                  TEXT NOT NULL,
                    crop_window           TEXT,
                    staged_at             TEXT NOT NULL,
                    status                TEXT NOT NULL DEFAULT 'staged',
                    started_at            TEXT,
                    completed_at          TEXT,
                    error                 TEXT,
                    footprint             BLOB,
                    pipeline_config_hash  TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS scene_metadata (
                    scene_id      TEXT PRIMARY KEY,
                    sensing_time  TEXT,
                    satellite     TEXT,
                    tile_id       TEXT,
                    orbit_rel     INTEGER,
                    lat_center    REAL,
                    lon_center    REAL,
                    sun_zenith    REAL,
                    sun_azimuth   REAL,
                    footprint     BLOB,
                    crs           TEXT DEFAULT 'EPSG:4326'
                )
            """)

    def stage(self, path: str, scene_id: str,
              crop_window: Optional[str] = None, status: str = "staged") -> bool:
        """Insert a new run row. Returns True if a new row was inserted.

        Args:
            path: Absolute path to the .SAFE directory.
            scene_id: Scene ID string (stem of the .SAFE filename).
            crop_window: Optional crop window string ``'col,row,w,h'``.
            status: Initial status (default ``'staged'``).

        Returns:
            ``True`` if a new row was inserted, ``False`` if it already existed.
        """
        run_id = _make_run_id(scene_id, crop_window)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?", [run_id]
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO runs "
                "(run_id,scene_id,path,crop_window,staged_at,status) VALUES (?,?,?,?,?,?)",
                (run_id, scene_id, path, crop_window,
                 datetime.now().isoformat(), status),
            )
            return True

    def set_status(self, run_id: str, status: str,
                   error: Optional[str] = None,
                   pipeline_config_hash: Optional[str] = None) -> None:
        """Update the status of a run.

        Sets ``started_at`` when status is ``'started'``, ``completed_at`` otherwise.
        When status is ``'done'`` and ``pipeline_config_hash`` is provided, stores
        it for future integrity checks.

        Args:
            run_id: The run identifier (from :func:`_make_run_id`).
            status: New status string.
            error: Optional error message to store.
            pipeline_config_hash: Combined hash of all pipeline config files at
                completion time. Stored only when status is ``'done'``.
        """
        now = datetime.now().isoformat()
        time_col = "started_at" if status == "started" else "completed_at"
        extra = ""
        params: List[Any] = [status, now, error]
        if status == "done" and pipeline_config_hash is not None:
            extra = ", pipeline_config_hash=?"
            params.append(pipeline_config_hash)
        params.append(run_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE runs SET status=?, {time_col}=?, error=?{extra} WHERE run_id=?",
                params,
            )

    def set_footprint(self, run_id: str, footprint_wkb: bytes) -> None:
        """Store a WKB footprint geometry for a run.

        Args:
            run_id: The run identifier.
            footprint_wkb: WKB-encoded polygon bytes.
        """
        with self._conn() as conn:
            conn.execute("UPDATE runs SET footprint=? WHERE run_id=?",
                         (footprint_wkb, run_id))

    def upsert_scene_metadata(self, scene_id: str, data: Dict[str, Any]) -> None:
        """Insert or replace scene-level metadata.

        Args:
            scene_id: The scene identifier (PRIMARY KEY).
            data: Dict of column → value pairs to upsert.
        """
        cols = ["scene_id"] + list(data.keys())
        vals = [scene_id] + list(data.values())
        ph = ", ".join(["?" for _ in cols])
        update_set = ", ".join(f"{c}=EXCLUDED.{c}" for c in data.keys())
        with self._conn() as conn:
            conn.execute(
                f"INSERT INTO scene_metadata ({', '.join(cols)}) VALUES ({ph}) "
                f"ON CONFLICT (scene_id) DO UPDATE SET {update_set}",
                vals,
            )

    def reset_stale_running(self) -> int:
        """Reset any 'started' rows back to 'staged'.

        A 'started' row that persists across process restarts is stale — it
        means the previous run was killed before it could mark the scene as
        'done' or 'failed'.  Call this at the start of each run() so those
        scenes are picked up again.

        Returns:
            Number of rows reset.
        """
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status='started'"
            ).fetchone()[0]
            if count:
                conn.execute(
                    "UPDATE runs SET status='staged', started_at=NULL, error=NULL "
                    "WHERE status='started'"
                )
            return count

    def get_pending(self, force: bool = False,
                    crop_window: Optional[str] = None) -> List[Tuple[str, Optional[str]]]:
        """Return ``(path, crop_window)`` pairs for runs to process.

        When ``crop_window`` is None, returns only full-scene runs
        (``crop_window IS NULL``). Pass a crop string to filter by that crop.

        Args:
            force: If True, include all runs regardless of status.
            crop_window: Filter by this crop window (None = full scenes only).

        Returns:
            List of ``(path, crop_window)`` tuples.
        """
        cw_filter = "AND crop_window IS NULL" if crop_window is None else "AND crop_window=?"
        cw_params: List[Any] = [] if crop_window is None else [crop_window]
        with self._conn() as conn:
            if force:
                q = f"SELECT path, crop_window FROM runs WHERE 1=1 {cw_filter} ORDER BY staged_at"
            else:
                q = (f"SELECT path, crop_window FROM runs "
                     f"WHERE status IN ('staged','failed') {cw_filter} ORDER BY staged_at")
            result = conn.execute(q, cw_params)
            rows = result.fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all runs as a list of dicts."""
        with self._conn() as conn:
            result = conn.execute(
                "SELECT run_id,scene_id,path,crop_window,status,"
                "staged_at,started_at,completed_at,error,pipeline_config_hash "
                "FROM runs ORDER BY staged_at"
            )
            cols = [d[0] for d in result.description]
            return [dict(zip(cols, row)) for row in result.fetchall()]

    def count_by_status(self) -> Dict[str, int]:
        """Return a dict mapping status → count across all runs."""
        with self._conn() as conn:
            result = conn.execute(
                "SELECT status, COUNT(*) AS n FROM runs GROUP BY status"
            )
            rows = result.fetchall()
        return {r[0]: r[1] for r in rows}

    def write_stats(self, run_id: str, table_name: str,
                    stats: Dict[str, Any]) -> None:
        """Write computed stats for a run into a dynamic stats table.

        The table is created if it doesn't exist; missing columns are added
        via ``ALTER TABLE``.

        Args:
            run_id: The run identifier.
            table_name: The stats table name (e.g. ``'stats_cloud_mask'``).
            stats: Dict of column → float/int value pairs.
        """
        if not stats:
            return
        _require_safe_sql_id(table_name, "table name")
        for col in stats:
            _require_safe_sql_id(col, "column name")
        with self._conn() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} "
                f"(run_id TEXT PRIMARY KEY)"
            )
            existing = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name=? AND table_schema='main'",
                    [table_name],
                ).fetchall()
            }
            for col in stats:
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
            cols = ["run_id"] + list(stats.keys())
            ph = ", ".join(["?" for _ in cols])
            update_set = ", ".join(f"{c}=EXCLUDED.{c}" for c in stats.keys())
            conn.execute(
                f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({ph}) "
                f"ON CONFLICT (run_id) DO UPDATE SET {update_set}",
                [run_id] + list(stats.values()),
            )

    def get_stale_done_runs(self, current_hash: str) -> List[Dict[str, Any]]:
        """Return done runs whose stored pipeline_config_hash differs from current_hash.

        Runs with a NULL pipeline_config_hash are also included — a missing hash
        means the run predates integrity tracking and its config state is unknown,
        so it is treated as stale.

        Args:
            current_hash: The current pipeline config hash from
                :meth:`Project._pipeline_config_hash`.

        Returns:
            List of dicts with ``run_id``, ``scene_id``, ``path``, ``crop_window``
            keys for each stale run.
        """
        with self._conn() as conn:
            result = conn.execute(
                "SELECT run_id, scene_id, path, crop_window, pipeline_config_hash "
                "FROM runs WHERE status='done' "
                "AND (pipeline_config_hash IS NULL OR pipeline_config_hash != ?)",
                [current_hash],
            )
            cols = [d[0] for d in result.description]
            return [dict(zip(cols, row)) for row in result.fetchall()]

    def reset_to_staged(self, run_ids: List[str]) -> None:
        """Reset specific runs back to 'staged' status.

        Clears ``completed_at``, ``error``, and ``pipeline_config_hash`` so the
        runs are picked up fresh by the next :meth:`get_pending` call.

        Args:
            run_ids: List of run_id values to reset.
        """
        if not run_ids:
            return
        with self._conn() as conn:
            for run_id in run_ids:
                conn.execute(
                    "UPDATE runs SET status='staged', completed_at=NULL, error=NULL, "
                    "pipeline_config_hash=NULL WHERE run_id=?",
                    (run_id,),
                )

    def has_stats(self, run_id: str, table_name: str) -> bool:
        """Return True if stats already exist for this run in the given table.

        Args:
            run_id: The run identifier.
            table_name: The stats table name.

        Returns:
            ``True`` if a row exists, ``False`` otherwise.
        """
        with self._conn() as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            if table_name not in tables:
                return False
            result = conn.execute(
                f"SELECT 1 FROM {table_name} WHERE run_id=?", [run_id]
            )
            return result.fetchone() is not None



# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Root project configuration, stored as project.yaml."""
    model_config = ConfigDict(extra="ignore")
    name: str = Field(..., description="User-friendly project name")
    pipeline: str = Field(
        default="full-workflow",
        description="Recipe name (see clouds_decoded/workflows/recipes/)",
    )
    created_at: str = Field(default="", description="ISO timestamp of project creation")
    workflow: Optional[WorkflowDef] = Field(
        default=None,
        description=(
            "Workflow DAG embedded at project init. If absent, the named "
            "recipe is loaded from the bundled recipes directory."
        ),
    )
    output_dir: str = Field(
        default="outputs",
        description=(
            "Directory where per-scene output files and logs are written. "
            "Relative paths are resolved against the project directory; "
            "absolute paths are used as-is. The directory is created on "
            "first use, not during project init."
        ),
    )
    stats: List[str] = Field(
        default_factory=lambda: [
            "cloud_mask::class_fractions",
            "cloud_height_emulator::percentiles",
            "cloud_properties::percentiles",
            "albedo::mean",
        ],
        description=(
            "Stats methods to compute. Format: 'step_name::function_name'. "
            "Resolved as clouds_decoded.stats.{step_name}.{fn} with fallback "
            "to clouds_decoded.stats._generic.{fn}."
        ),
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "ProjectConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)


class StepResult(BaseModel):
    """Record of a single processing step for a scene."""
    status: str = "pending"
    output_file: Optional[str] = Field(default=None)
    config_hash: Optional[str] = Field(default=None)
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
    product_id: Optional[str] = Field(default=None)
    pipeline: str = ""
    step_name: str = ""
    step_config: Dict[str, Any] = Field(default_factory=dict)
    crop_window: Optional[str] = Field(default=None)


class SceneManifest(BaseModel):
    """Per-scene processing manifest, stored as manifest.json."""
    scene_id: str
    scene_path: str
    crop_window: Optional[str] = Field(default=None)
    provenance: Optional[Provenance] = None
    steps: Dict[str, StepResult] = Field(default_factory=dict)
    last_updated: Optional[str] = None

    def is_step_complete(self, step: str, current_config_hash: str) -> bool:
        if step not in self.steps:
            return False
        result = self.steps[step]
        if result.status != "completed":
            return False
        if result.output_file is not None and not Path(result.output_file).exists():
            return False
        return result.config_hash == current_config_hash

    @classmethod
    def from_json(cls, path: Path) -> "SceneManifest":
        with open(path) as f:
            return cls.model_validate_json(f.read())

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class Project:
    """Manages a clouds-decoded project directory."""

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).resolve()
        self.config_path = self.project_dir / "project.yaml"
        self.configs_dir = self.project_dir / "configs"
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
    ) -> "Project":
        """Create a new project directory with default config YAMLs.

        The chosen workflow recipe is embedded verbatim into project.yaml so
        users can customise the pipeline graph by editing that file directly.
        """
        project_dir = Path(project_dir).resolve()
        if (project_dir / "project.yaml").exists():
            raise FileExistsError(f"Project already exists at {project_dir}")

        if name is None:
            name = project_dir.name

        source_project: Optional["Project"] = None
        if clone_from:
            source_dir = Path(clone_from).resolve()
            if not (source_dir / "project.yaml").exists():
                raise FileNotFoundError(f"Cannot clone: no project.yaml found in {source_dir}")
            source_project = cls(source_dir)
            pipeline = source_project.config.pipeline

        project_dir.mkdir(parents=True, exist_ok=True)
        workflow = _get_recipe(pipeline)
        config = ProjectConfig(
            name=name, pipeline=pipeline, workflow=workflow,
            created_at=datetime.now().isoformat(),
        )
        config.to_yaml(project_dir / "project.yaml")

        project = cls(project_dir)
        project._config = config

        if source_project:
            project._clone_configs(source_project)
            logger.info(f"Project '{name}' initialized at {project_dir} (cloned from {clone_from})")
        else:
            project._write_default_configs()
            logger.info(f"Project '{name}' initialized at {project_dir}")

        logger.info(f"  Pipeline: {pipeline}")
        logger.info(f"  Edit configs in {project.configs_dir}/")
        logger.info(f"  Outputs will be written to {project.output_dir}/")
        _ = project.db  # eagerly create project.db
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
    def workflow(self) -> WorkflowDef:
        """The active workflow definition (embedded or loaded from recipe)."""
        return self.config.workflow or _get_recipe(self.config.pipeline)

    @property
    def steps(self) -> List[str]:
        return [s.name for s in self.workflow.steps]

    @property
    def output_dir(self) -> Path:
        """Resolved path to the output directory for per-scene files."""
        p = Path(self.config.output_dir)
        if not p.is_absolute():
            p = self.project_dir / p
        return p.resolve()

    @property
    def logs_dir(self) -> Path:
        return self.project_dir / "logs"

    # ------------------------------------------------------------------
    # Scene database
    # ------------------------------------------------------------------

    @functools.cached_property
    def db(self) -> SceneDB:
        return SceneDB(self.project_dir / "project.db")

    def stage(self, *paths: str, crop_window: Optional[str] = None) -> None:
        """Register scenes in project.db without processing.

        Args:
            *paths: Absolute or relative paths to .SAFE directories.
            crop_window: Optional crop window string ``'col,row,w,h'``.
        """
        for path in paths:
            resolved = str(Path(path).resolve())
            scene_id = self._scene_id(resolved)
            is_new = self.db.stage(resolved, scene_id, crop_window=crop_window)
            if is_new:
                logger.info(f"Staged: {scene_id}" +
                            (f" [{crop_window}]" if crop_window else ""))
            else:
                logger.debug(f"Already registered: {scene_id}")

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def _scene_id(self, scene_path: str) -> str:
        return Path(scene_path).stem

    def _scene_output_dir(self, scene_id: str, crop_window: Optional[str] = None) -> Path:
        base = self.output_dir / scene_id
        if crop_window is not None:
            return base / "crops" / crop_window.replace(",", "_")
        return base

    def _scene_log_path(self, scene_id: str, crop_window: Optional[str] = None) -> Path:
        base = self.logs_dir / scene_id
        if crop_window is not None:
            return base / "crops" / crop_window.replace(",", "_") / "pipeline.log"
        return base / "pipeline.log"

    def _load_manifest(self, scene_id: str, scene_path: str, crop_window: Optional[str] = None) -> SceneManifest:
        manifest_path = self._scene_output_dir(scene_id, crop_window) / "manifest.json"
        if manifest_path.exists():
            return SceneManifest.from_json(manifest_path)
        return SceneManifest(scene_id=scene_id, scene_path=scene_path, crop_window=crop_window)

    def _save_manifest(self, scene_id: str, manifest: SceneManifest, crop_window: Optional[str] = None):
        manifest.last_updated = datetime.now().isoformat()
        (self._scene_output_dir(scene_id, crop_window) / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        manifest.to_json(self._scene_output_dir(scene_id, crop_window) / "manifest.json")

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    def _get_workflow_step(self, step_name: str) -> WorkflowStepDef:
        """Return the WorkflowStepDef for ``step_name``, or raise KeyError."""
        for s in self.workflow.steps:
            if s.name == step_name:
                return s
        raise KeyError(f"Step '{step_name}' not in workflow")

    @functools.cached_property
    def _token_lifetimes(self) -> Dict[str, str]:
        """Map token_name -> name of the last step that consumes it.

        Used by _execute_step_in_ctx to free intermediates as soon as they
        are no longer needed (replaces the old ``releases_ctx`` field on Step).
        """
        last_use: Dict[str, str] = {}
        for s in self.workflow.steps:
            for t in list(s.inputs) + list(s.keyword_inputs.values()):
                last_use[t] = s.name
        return last_use

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _config_yaml_path(self, step_name: str) -> Path:
        return self.configs_dir / self._get_workflow_step(step_name).config

    def _config_hash(self, step_name: str) -> str:
        path = self._config_yaml_path(step_name)
        if not path.exists():
            return "no_config"
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _pipeline_config_hash(self) -> str:
        """Compute a combined hash of all config files for the current workflow.

        Combines per-step config hashes in workflow order so that a change to
        any config file produces a different pipeline hash. Reuses the existing
        :meth:`_config_hash` per-step hash for consistency.

        Returns:
            First 16 hex characters of the combined SHA-256.
        """
        h = hashlib.sha256()
        for step_name in self.steps:
            h.update(self._config_hash(step_name).encode())
        return h.hexdigest()[:16]

    def _load_step_config(self, step_name: str) -> Any:
        step_def = self._get_workflow_step(step_name)
        return PROCESSORS[step_def.processor].config_loader(str(self._config_yaml_path(step_name)))

    def _write_default_configs(self):
        """Write one default config YAML per unique config file in the workflow."""
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        written: set = set()
        for step in self.workflow.steps:
            if step.config not in written:
                PROCESSORS[step.processor].config_factory().to_yaml(
                    str(self.configs_dir / step.config)
                )
                written.add(step.config)
        logger.info(f"Default configs written to {self.configs_dir}/")

    def _clone_configs(self, source: "Project"):
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
        config_dict: Dict[str, Any],
        crop_window: Optional[str] = None,
        git_hash: Optional[str] = None,
    ) -> Provenance:
        import sys
        return Provenance(
            project_name=self.config.name,
            codebase_version=self._get_codebase_version(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            git_hash=git_hash if git_hash is not None else self._get_git_hash(),
            timestamp=datetime.now().isoformat(),
            scene_path=scene_path,
            product_id=product_id,
            pipeline=self.config.pipeline,
            step_name=step_name,
            step_config=config_dict,
            crop_window=crop_window,
        )

    # ------------------------------------------------------------------
    # File provenance validation
    # ------------------------------------------------------------------

    def _read_file_provenance(self, filepath: Path) -> Optional[Dict[str, Any]]:
        import rasterio as rio
        try:
            with rio.open(filepath) as src:
                extra = src.tags().get(METADATA_TAG)
                if not extra:
                    return None
                meta = json.loads(extra)
                return meta.get("provenance") if isinstance(meta, dict) else None
        except Exception as e:
            logger.debug(f"Could not read provenance from {filepath}: {e}")
            return None

    def _validate_step_file(
        self,
        step_name: str,
        scene_out: Path,
        scene_path: str,
        current_config_dict: Dict[str, Any],
        crop_window: Optional[str] = None,
    ) -> Optional[str]:
        step_def = self._get_workflow_step(step_name)
        proc_def = PROCESSORS[step_def.processor]

        if step_def.output_file is None:
            return None

        # Resolve dynamic filename (e.g. cloud_properties shading variant)
        if proc_def.get_output_file:
            config = proc_def.config_loader(str(self.configs_dir / step_def.config))
            actual_file = proc_def.get_output_file(config)
        else:
            actual_file = step_def.output_file

        filepath = scene_out / actual_file
        if not filepath.exists():
            return f"Output file missing: {filepath}"

        prov = self._read_file_provenance(filepath)
        if prov is None:
            return f"[{step_name}] {filepath.name} has no provenance metadata."

        file_project = prov.get("project_name")
        if file_project and file_project != self.config.name:
            return (f"[{step_name}] {filepath.name} was produced by project '{file_project}', "
                    f"but current project is '{self.config.name}'.")

        file_product = prov.get("product_id")
        if file_product:
            current_product = Path(scene_path).name.removesuffix(".SAFE")
            if current_product != file_product.removesuffix(".SAFE"):
                return (f"[{step_name}] {filepath.name} was produced from product '{file_product}', "
                        f"but current scene is '{current_product}'.")
        else:
            file_scene = prov.get("scene_path", "")
            if file_scene and str(Path(file_scene).resolve()) != str(Path(scene_path).resolve()):
                return f"[{step_name}] {filepath.name} was produced from a different scene path."

        file_config = prov.get("step_config")
        if file_config is not None and file_config != current_config_dict:
            changed = sorted(k for k in set(file_config) | set(current_config_dict)
                             if file_config.get(k) != current_config_dict.get(k))
            return (f"[{step_name}] {filepath.name} config mismatch. "
                    f"Differing keys: {', '.join(changed)}.")

        if prov.get("crop_window") != crop_window:
            return (f"[{step_name}] {filepath.name} was produced with "
                    f"crop_window={prov.get('crop_window')!r}, "
                    f"but current run uses crop_window={crop_window!r}.")

        return None

    # ------------------------------------------------------------------
    # Processor creation
    # ------------------------------------------------------------------

    def _create_processor_for_step(
        self, step_name: str, device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Instantiate processor(s) for one step via its registered factory.

        The returned dict is keyed by ``step_name`` (not the processor registry
        key) so that ``_run_step`` can look it up with ``processors.get(step.name)``.
        Auxiliary keys (e.g. ``_cloud_mask_postprocessor``) are preserved as-is.

        Args:
            step_name: Step name in the active workflow.
            device: If provided and the config exposes a ``device`` field,
                override it (used for round-robin GPU assignment).
        """
        step_def = self._get_workflow_step(step_name)
        proc_def = PROCESSORS[step_def.processor]
        config = proc_def.config_loader(str(self._config_yaml_path(step_name)))
        if device is not None and hasattr(config, "device"):
            config = config.model_copy(update={"device": device})
        raw = proc_def.processor_factory(config)
        # Remap main processor key from proc-registry key to step name
        result: Dict[str, Any] = {}
        for k, v in raw.items():
            if k == step_def.processor:
                result[step_name] = v
            else:
                result[k] = v  # preserve auxiliary keys
        return result

    def _create_processors(self) -> Dict[str, Any]:
        """Create all processors for the current workflow, merged into a single dict."""
        result: Dict[str, Any] = {}
        for step_name in self.steps:
            result.update(self._create_processor_for_step(step_name))
        return result

    # ------------------------------------------------------------------
    # Core step execution
    # ------------------------------------------------------------------

    def _run_step(
        self,
        step: WorkflowStepDef,
        ctx: _PipelineCtx,
        processors: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Any]:
        """Run one pipeline step generically via its ProcessorDef.

        Inputs are resolved from ``ctx.intermediates`` using the token names
        declared in ``step.inputs`` and ``step.keyword_inputs``.  The result
        is returned (and written to disk when ``step.output_file`` is set) but
        NOT stored back into ``ctx`` — that is done by ``_execute_step_in_ctx``.

        Returns:
            ``(output_path, result_data)`` where ``output_path`` is the file
            written to disk (or ``None`` for ephemeral steps).
        """
        proc_def = PROCESSORS[step.processor]
        config = self._load_step_config(step.name)

        computed = type(config).model_computed_fields
        config_dict = config.model_dump(mode="json", exclude=set(computed.keys()) if computed else set())

        scene_obj = ctx.intermediates.get("scene")
        provenance = self._build_provenance(
            ctx.scene_path, getattr(scene_obj, "product_uri", None),
            step.name, config_dict,
            crop_window=ctx.crop_window, git_hash=ctx.git_hash,
        )

        # Resolve positional and keyword arguments from the token store
        pos_args = [ctx.intermediates[t] for t in step.inputs]
        kw_args = {k: ctx.intermediates[v] for k, v in step.keyword_inputs.items()}

        # Get or create processor (pre-created dict uses step name as key)
        proc = (processors or {}).get(step.name)
        if proc is None:
            raw = proc_def.processor_factory(config)
            proc = raw.get(step.processor) or next(iter(raw.values()))

        result = proc.process(*pos_args, **kw_args)

        # Write output to disk when this step produces a file
        out_file = proc_def.get_output_file(config) if proc_def.get_output_file else step.output_file
        if out_file:
            output_path = str(ctx.scene_out / out_file)
            result.metadata.provenance = provenance.model_dump()
            result.write(output_path)
        else:
            output_path = None

        # Postprocess pass (e.g. cloud_mask: binary mask for downstream consumers)
        if step.postprocess and proc_def.postprocess_fn:
            result = proc_def.postprocess_fn(result)

        # Optional side-effect (e.g. refocus: save individual band GeoTIFFs)
        if proc_def.on_complete:
            proc_def.on_complete(result, config, ctx.scene_out)

        return output_path, result

    # ------------------------------------------------------------------
    # Pipeline: reader stage
    # ------------------------------------------------------------------

    def _prepare_scene_context(self, ctx: _PipelineCtx) -> None:
        """Validate manifest, load scene, pre-load completed intermediates."""
        scene_id = self._scene_id(ctx.scene_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Scene: {scene_id}")
        logger.info(f"{'='*60}")

        scene_out = self._scene_output_dir(scene_id, ctx.crop_window)
        scene_out.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest(scene_id, ctx.scene_path, ctx.crop_window)

        steps_to_run = self.steps
        invalidate_from: Optional[int] = None

        if not ctx.force:
            for i, step_name in enumerate(steps_to_run):
                config_hash = self._config_hash(step_name)
                if manifest.is_step_complete(step_name, config_hash):
                    if not ctx.unsafe:
                        cfg = self._load_step_config(step_name)
                        computed = type(cfg).model_computed_fields
                        cfg_dict = cfg.model_dump(mode="json", exclude=set(computed.keys()) if computed else set())
                        error = self._validate_step_file(step_name, scene_out, ctx.scene_path,
                                                         cfg_dict, crop_window=ctx.crop_window)
                        if error is not None:
                            logger.error(f"  {error}\n  Use --unsafe or --force to proceed.")
                            raise RuntimeError(
                                f"File provenance mismatch for '{step_name}' in '{scene_id}'. {error}"
                            )
                    logger.info(f"  [{step_name}] Skipping (already complete, config unchanged)")
                else:
                    if (step_name in manifest.steps
                            and manifest.steps[step_name].status == "completed"
                            and manifest.steps[step_name].config_hash != config_hash):
                        logger.warning(f"  [{step_name}] Config changed — re-running from here")
                    invalidate_from = i
                    break
        else:
            invalidate_from = 0

        ctx.scene_out = scene_out
        ctx.manifest = manifest
        ctx.steps_to_run = list(steps_to_run)

        if invalidate_from is None:
            logger.info(f"All steps complete for {scene_id}")
            ctx.first_step_idx = len(steps_to_run)
            return

        manifest.crop_window = ctx.crop_window
        self._save_manifest(scene_id, manifest, ctx.crop_window)

        # Generic ephemeral-step detection: any completed step with no disk output
        # must be re-run (its in-memory result cannot be restored from disk).
        for i, step_name in enumerate(steps_to_run[:invalidate_from]):
            step_def = self._get_workflow_step(step_name)
            if step_def.output_file is None:
                invalidate_from = i
                logger.info(f"  [{step_name}] Re-running (in-memory result not persisted to disk)")
                break

        # Load scene
        from clouds_decoded.data import Sentinel2Scene
        scene = Sentinel2Scene()
        if ctx.crop_window:
            parts = ctx.crop_window.split(",")
            if len(parts) != 4:
                raise ValueError(f"crop_window must have 4 comma-separated integers, got {len(parts)}")
            col_off, row_off, width, height = map(int, parts)
            scene.read(ctx.scene_path, crop_window=(col_off, row_off, width, height))
        else:
            scene.read(ctx.scene_path)

        # Populate scene_metadata and run footprint (best-effort, never raises)
        self._write_scene_metadata(scene, scene_id)
        run_id = _make_run_id(scene_id, ctx.crop_window)
        self._write_run_footprint(scene, run_id, ctx.crop_window)

        # Kick off band prefetch for upcoming GPU steps in background threads.
        def _safe_prefetch(fn, sc, cfg):
            try:
                fn(sc, cfg)
            except Exception:
                pass  # scene may have been torn down; processor will compute on demand

        for step_name in steps_to_run[invalidate_from:]:
            step_def = self._get_workflow_step(step_name)
            proc_def = PROCESSORS[step_def.processor]
            if proc_def.prefetch_fn:
                cfg = self._load_step_config(step_name)
                threading.Thread(
                    target=_safe_prefetch, args=(proc_def.prefetch_fn, scene, cfg),
                    daemon=True, name=f"prefetch-{step_name}",
                ).start()

        # Load completed intermediates for resuming
        loaded = self._load_intermediates(scene_out, steps_to_run[:invalidate_from])
        ctx.intermediates.update(loaded)
        ctx.intermediates["scene"] = scene
        ctx.first_step_idx = invalidate_from

    def _load_intermediates(
        self, scene_out: Path, completed_step_names: List[str]
    ) -> Dict[str, Any]:
        """Load intermediate results from completed steps for pipeline resume.

        Returns a dict of token_name -> loaded data object, ready to be
        merged into ``ctx.intermediates``.
        """
        results: Dict[str, Any] = {}
        for step in self.workflow.steps:
            if step.name not in completed_step_names:
                break
            if not step.output_file or not step.output:
                continue  # ephemeral or terminal step
            path = scene_out / step.output_file
            if not path.exists():
                continue
            proc_def = PROCESSORS[step.processor]
            if proc_def.output_loader is None:
                continue
            data = proc_def.output_loader(str(path))
            if step.postprocess and proc_def.postprocess_fn:
                data = proc_def.postprocess_fn(data)
            results[step.output] = data
        return results

    # ------------------------------------------------------------------
    # Pipeline: per-step stage
    # ------------------------------------------------------------------

    def _execute_step_in_ctx(
        self,
        ctx: _PipelineCtx,
        step_name: str,
        step_idx: int,
        proc_dict: Dict[str, Any],
    ) -> None:
        """Execute one step for one scene context; mutates ctx.intermediates in-place."""
        if step_idx < ctx.first_step_idx:
            return

        step_def = self._get_workflow_step(step_name)
        proc_def = PROCESSORS[step_def.processor]
        scene_id = self._scene_id(ctx.scene_path)
        config_hash = self._config_hash(step_name)

        step_result = StepResult(
            status="pending",
            config_hash=config_hash,
            started_at=datetime.now().isoformat(),
        )
        ctx.manifest.steps[step_name] = step_result
        self._save_manifest(scene_id, ctx.manifest, ctx.crop_window)

        t_start = time.time()
        try:
            output_path, step_data = self._run_step(step_def, ctx, proc_dict)
        except Exception:
            duration = time.time() - t_start
            import traceback
            step_result.status = "failed"
            step_result.completed_at = datetime.now().isoformat()
            step_result.duration_seconds = round(duration, 1)
            step_result.error = traceback.format_exc()
            self._save_manifest(scene_id, ctx.manifest, ctx.crop_window)
            raise

        # Store result in token store
        if step_def.output:
            ctx.intermediates[step_def.output] = step_data

        duration = time.time() - t_start
        step_result.status = "completed"
        step_result.completed_at = datetime.now().isoformat()
        step_result.duration_seconds = round(duration, 1)
        step_result.output_file = output_path
        logger.info(f"  [{step_name}] Completed in {duration:.1f}s")
        self._save_manifest(scene_id, ctx.manifest, ctx.crop_window)

        # Automatic memory management: free tokens whose last consumer was this step.
        for token, last_step in self._token_lifetimes.items():
            if last_step == step_name and token != step_def.output:
                ctx.intermediates.pop(token, None)

        # Scene band-cache clearing (processor-specific)
        scene = ctx.intermediates.get("scene")
        if scene is not None:
            for attr in proc_def.clears_scene_caches:
                cache = getattr(scene, attr, None)
                if isinstance(cache, dict):
                    cache.clear()
        if proc_def.clears_scene_if_refocused and ctx.intermediates.get("refocused_scene") is not None:
            scene_orig = ctx.intermediates.get("scene")
            if scene_orig is not None:
                for attr in ("_band_cache", "_resized_band_cache"):
                    c = getattr(scene_orig, attr, None)
                    if isinstance(c, dict):
                        c.clear()

    # ------------------------------------------------------------------
    # Serial pipeline
    # ------------------------------------------------------------------

    def _run_serial(
        self,
        scene_list: List[Tuple[str, Optional[str]]],
        force: bool = False,
        unsafe: bool = False,
        verbose: bool = False,
        pipeline_config_hash: str = "",
    ) -> None:
        """Process scenes one at a time, steps in sequence — no threading.

        Args:
            scene_list: List of ``(scene_path, crop_window)`` tuples.
            force: Reprocess all steps even if cached.
            unsafe: Skip file provenance validation.
            verbose: Show INFO logs on the terminal.
            pipeline_config_hash: Combined hash of all pipeline config files,
                stored in the DB when each scene completes.
        """
        import clouds_decoded.sliding_window as _sw
        _sw.suppress_inference_progress = True

        scene_handler = _SceneRoutingHandler()
        scene_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(scene_handler)

        git_hash = self._get_git_hash()
        logger.warning(
            f"Running project '{self.config.name}' — {len(scene_list)} scene(s)"
        )

        # Suppress INFO logs on the terminal; keep WARNING+ visible so users can
        # follow progress without the noise of per-band / per-window messages.
        saved_term_levels: Dict[logging.Handler, int] = {}
        if not verbose:
            for h in root_logger.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, (_SceneRoutingHandler, logging.FileHandler)
                ):
                    saved_term_levels[h] = h.level
                    h.setLevel(logging.WARNING)

        failed_scenes: List[str] = []
        try:
            for scene_path, cw in scene_list:
                scene_id = self._scene_id(scene_path)
                run_id = _make_run_id(scene_id, cw)
                log_path = self._scene_log_path(scene_id, cw)
                ctx = _PipelineCtx(
                    scene_path=scene_path, crop_window=cw, log_path=log_path,
                    force=force, unsafe=unsafe, git_hash=git_hash,
                )
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as flog:
                        flog.write(f"\n--- Run started {datetime.now().isoformat()} ---\n")
                except Exception:
                    pass
                self.db.set_status(run_id, "started")
                logger.warning(f"[{scene_id}] Processing...")
                token = _scene_log_path_var.set(log_path)
                try:
                    try:
                        self._prepare_scene_context(ctx)
                    except Exception as exc:
                        ctx.failed = True
                        ctx.error = exc
                        logger.error(f"[{scene_id}] Reader failed: {exc}")
                    if not ctx.failed:
                        for step_idx, step_name in enumerate(self.steps):
                            if not ctx.failed:
                                try:
                                    proc_dict = self._create_processor_for_step(step_name)
                                    self._execute_step_in_ctx(ctx, step_name, step_idx, proc_dict)
                                except Exception as exc:
                                    ctx.failed = True
                                    ctx.error = exc
                                    logger.error(f"[{step_name}][{scene_id}] FAILED: {exc}")
                finally:
                    _scene_log_path_var.reset(token)
                if ctx.failed:
                    logger.error(f"[{scene_id}] FAILED: {ctx.error}")
                    self.db.set_status(run_id, "failed", error=str(ctx.error))
                    failed_scenes.append(scene_id)
                else:
                    logger.warning(f"[{scene_id}] Complete.")
                    self.db.set_status(run_id, "done",
                                       pipeline_config_hash=pipeline_config_hash)
        finally:
            for h, level in saved_term_levels.items():
                h.setLevel(level)
            root_logger.removeHandler(scene_handler)
            _close_all_file_handlers()

        if failed_scenes:
            logger.warning(f"{len(failed_scenes)} scene(s) failed: {', '.join(failed_scenes)}")
        logger.warning(
            f"Project run complete "
            f"({len(scene_list) - len(failed_scenes)}/{len(scene_list)} succeeded)."
        )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        scenes: Optional[List[str]] = None,
        force: bool = False,
        unsafe: bool = False,
        crop_window: Optional[str] = None,
        parallel: bool = False,
        parallelism: Optional[Dict[str, int]] = None,
        queue_depth: int = 2,
        verbose: bool = False,
        progress: bool = True,
        max_workers: Optional[int] = None,  # deprecated shorthand
        force_overwrite: bool = False,
        ignore_integrity: bool = False,
        run_stats: bool = True,
    ):
        """Run the project pipeline for all scenes.

        By default runs serially (one scene at a time, steps in sequence).
        Pass ``parallel=True`` to enable a thread-based pipeline that overlaps
        scene reading and processing across stages.

        Args:
            scenes: Scene .SAFE paths (auto-registered). None = all registered scenes.
            force: Reprocess all steps (ignore manifest cache).
            unsafe: Skip file provenance validation.
            crop_window: Spatial crop ``'col_off,row_off,width,height'``.
            parallel: Enable thread-based parallel pipeline. Default is serial.
            parallelism: (parallel only) Stage name → worker count. Unspecified
                stages default to 1. Stage names: ``"reader"``, plus each step name.
            queue_depth: (parallel only) Max scenes buffered between adjacent stages.
            verbose: Show INFO logs on terminal (default: warnings only; full logs
                always written to ``<project>/logs/<scene>/pipeline.log``).
            progress: (parallel only) Show live pipeline progress display.
            max_workers: Deprecated — sets all stages to this count.
            force_overwrite: Reset any ``done`` scenes with a stale config hash back
                to ``staged`` and re-run them.  Mutually exclusive with
                ``ignore_integrity``.
            ignore_integrity: Skip the config integrity check.  Stale ``done``
                scenes are silently left as-is (not recommended).
            run_stats: Automatically compute and store statistics for all
                completed runs after the pipeline finishes.  Pass ``False``
                to skip (equivalent to the ``--no-stats`` CLI flag).
        """
        if crop_window is not None:
            crop_window = ",".join(p.strip() for p in crop_window.split(","))

        pipeline_config_hash = self._pipeline_config_hash()

        if scenes:
            self.stage(*[str(Path(s).resolve()) for s in scenes],
                       crop_window=crop_window)
            scene_list: List[Tuple[str, Optional[str]]] = [
                (str(Path(s).resolve()), crop_window) for s in scenes
            ]
        else:
            n_reset = self.db.reset_stale_running()
            if n_reset:
                logger.warning(
                    f"{n_reset} scene(s) were left in 'started' state from a previous "
                    f"interrupted run — resetting to 'staged' so they are retried."
                )

            if not force:
                stale = self.db.get_stale_done_runs(pipeline_config_hash)
                if stale:
                    scene_ids = [r["scene_id"] for r in stale]
                    if force_overwrite:
                        self.db.reset_to_staged([r["run_id"] for r in stale])
                        logger.warning(
                            f"Config changed: reset {len(stale)} 'done' scene(s) to "
                            f"'staged' for reprocessing: " + ", ".join(scene_ids)
                        )
                    elif ignore_integrity:
                        logger.warning(
                            f"Skipping config integrity check (--ignore-integrity). "
                            f"{len(stale)} 'done' scene(s) may have been processed "
                            "with a different config."
                        )
                    else:
                        raise RuntimeError(
                            f"{len(stale)} scene(s) have 'done' status but the project "
                            f"configs have changed since they were last processed:\n"
                            + "\n".join(f"  - {sid}" for sid in scene_ids)
                            + "\n\nTo re-run affected scenes with the current config, "
                            "use --force-overwrite.\n"
                            "To ignore this and skip those scenes, "
                            "use --ignore-integrity (not recommended)."
                        )

            scene_list = self.db.get_pending(force=force, crop_window=crop_window)

        if not scene_list:
            logger.warning("No scenes to process. Pass scene paths to 'project run'.")
            return

        if not parallel:
            self._run_serial(scene_list, force=force, unsafe=unsafe, verbose=verbose,
                             pipeline_config_hash=pipeline_config_hash)
            if run_stats:
                self.run_stats()
            return

        par: Dict[str, int] = {"reader": 1, **{step: 1 for step in self.steps}}
        if max_workers is not None and parallelism is None:
            par = {stage: max_workers for stage in par}
        if parallelism:
            par.update(parallelism)

        # Suppress per-window tqdm bars inside the pipeline (rich display handles progress).
        import clouds_decoded.sliding_window as _sw
        _sw.suppress_inference_progress = True

        # --- Logging setup ---
        scene_handler = _SceneRoutingHandler()
        scene_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(scene_handler)

        git_hash = self._get_git_hash()
        par_summary = ", ".join(f"{k}×{v}" for k, v in par.items() if v > 1)
        logger.warning(
            f"Running project '{self.config.name}' — {len(scene_list)} scene(s)"
            + (f" [{par_summary}]" if par_summary else "")
        )

        # Suppress all non-error terminal output during the run so that processor
        # warnings (e.g. "Clear fraction < threshold") don't corrupt the rich display.
        saved_term_levels: Dict[logging.Handler, int] = {}
        if not verbose:
            for h in root_logger.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, (_SceneRoutingHandler, logging.FileHandler)
                ):
                    saved_term_levels[h] = h.level
                    h.setLevel(logging.ERROR)

        failed_scenes: List[str] = []
        results_lock = threading.Lock()

        try:
            # Detect available GPUs for round-robin device assignment.
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

            def _effective_workers(step_name: str, requested: int) -> int:
                """Cap GPU-bound steps to n_gpus workers; CUDA serialises beyond that."""
                if n_gpus < 1:
                    return requested
                cfg = self._load_step_config(step_name)
                if not hasattr(cfg, "device"):
                    return requested
                base = cfg.device  # type: ignore[attr-defined]
                if base not in (None, "cuda"):
                    return requested
                if requested > n_gpus:
                    logger.warning(
                        f"  {step_name}: capping workers {requested} → {n_gpus} "
                        f"(CUDA serialises kernels beyond 1 per GPU)"
                    )
                return min(requested, n_gpus)

            # Global GPU counter: increments across steps so successive GPU
            # steps land on successive devices even when each has n=1.
            _next_gpu: List[int] = [0]

            def _device_list(step_name: str, n: int) -> List[Optional[str]]:
                """Return per-instance device overrides for a step, or None to use config."""
                if n_gpus <= 1:
                    return [None] * n
                cfg = self._load_step_config(step_name)
                if not hasattr(cfg, "device"):
                    return [None] * n
                base = cfg.device  # type: ignore[attr-defined]
                if base not in (None, "cuda"):
                    return [None] * n
                devices = [f"cuda:{(_next_gpu[0] + i) % n_gpus}" for i in range(n)]
                _next_gpu[0] = (_next_gpu[0] + n) % n_gpus
                logger.warning(
                    f"  {step_name}: {n} worker(s) → {devices}"
                )
                return devices

            # Apply GPU cap, then create processors with round-robin device assignment.
            for step in self.steps:
                par[step] = _effective_workers(step, par.get(step, 1))

            step_processor_lists: Dict[str, List[Dict[str, Any]]] = {
                step: [
                    self._create_processor_for_step(step, device=dev)
                    for dev in _device_list(step, par[step])
                ]
                for step in self.steps
            }

            # Build queue chain
            import queue as Q
            inter_queues: List[Q.Queue] = [Q.Queue(maxsize=queue_depth) for _ in self.steps]
            path_queue: Q.Queue = Q.Queue()
            sink_queue: Q.Queue = Q.Queue()

            scenes_done = [0]
            scenes_started = [0]
            scenes_done_lock = threading.Lock()

            def _make_worker(work_fn, in_q, out_q, remaining, remaining_lock, n_downstream):
                def worker():
                    while True:
                        item = in_q.get()
                        if item is _STOP:
                            with remaining_lock:
                                remaining[0] -= 1
                                is_last = remaining[0] == 0
                            if is_last:
                                for _ in range(n_downstream):
                                    out_q.put(_STOP)
                            return
                        out_q.put(work_fn(item))
                return threading.Thread(target=worker, daemon=True)

            all_threads: List[threading.Thread] = []

            # Reader workers
            n_reader = par["reader"]
            n_after_reader = par.get(self.steps[0], 1) if self.steps else 1

            def _reader_work(scene_path_cw: Tuple[str, Optional[str]]) -> _PipelineCtx:
                scene_path, cw = scene_path_cw
                with scenes_done_lock:
                    scenes_started[0] += 1
                scene_id = self._scene_id(scene_path)
                run_id = _make_run_id(scene_id, cw)
                self.db.set_status(run_id, "started")
                log_path = self._scene_log_path(scene_id, cw)
                ctx = _PipelineCtx(
                    scene_path=scene_path, crop_window=cw, log_path=log_path,
                    force=force, unsafe=unsafe, git_hash=git_hash,
                )
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as flog:
                        flog.write(f"\n--- Run started {datetime.now().isoformat()} ---\n")
                except Exception:
                    pass
                token = _scene_log_path_var.set(log_path)
                try:
                    self._prepare_scene_context(ctx)
                except Exception as exc:
                    ctx.failed = True
                    ctx.error = exc
                    logger.error(f"[{scene_id}] Reader failed: {exc}")
                finally:
                    _scene_log_path_var.reset(token)
                return ctx

            reader_remaining = [n_reader]
            reader_lock = threading.Lock()
            first_q = inter_queues[0] if self.steps else sink_queue
            for _ in range(n_reader):
                all_threads.append(_make_worker(
                    _reader_work, path_queue, first_q,
                    reader_remaining, reader_lock, n_after_reader if self.steps else 1,
                ))

            # Step workers
            for step_idx, step_name in enumerate(self.steps):
                n_this = par.get(step_name, 1)
                in_q = inter_queues[step_idx]
                out_q = inter_queues[step_idx + 1] if step_idx + 1 < len(self.steps) else sink_queue
                n_next = par.get(self.steps[step_idx + 1], 1) if step_idx + 1 < len(self.steps) else 1
                remaining = [n_this]
                rem_lock = threading.Lock()

                for worker_i, proc_dict in enumerate(step_processor_lists[step_name]):
                    def _make_step_work(pd, sn, si):
                        def work(ctx: _PipelineCtx) -> _PipelineCtx:
                            if not ctx.failed:
                                token = _scene_log_path_var.set(ctx.log_path)
                                try:
                                    self._execute_step_in_ctx(ctx, sn, si, pd)
                                except Exception as exc:
                                    ctx.failed = True
                                    ctx.error = exc
                                    logger.error(f"[{sn}][{self._scene_id(ctx.scene_path)}] FAILED: {exc}")
                                finally:
                                    _scene_log_path_var.reset(token)
                            return ctx
                        return work

                    all_threads.append(_make_worker(
                        _make_step_work(proc_dict, step_name, step_idx),
                        in_q, out_q, remaining, rem_lock, n_next,
                    ))

            # Sink thread
            def _sink():
                while True:
                    item = sink_queue.get()
                    if item is _STOP:
                        return
                    ctx: _PipelineCtx = item
                    sid = self._scene_id(ctx.scene_path)
                    rid = _make_run_id(sid, ctx.crop_window)
                    with scenes_done_lock:
                        scenes_done[0] += 1
                    if ctx.failed:
                        logger.error(f"[{sid}] FAILED: {ctx.error}")
                        self.db.set_status(rid, "failed", error=str(ctx.error))
                        with results_lock:
                            failed_scenes.append(sid)
                    else:
                        self.db.set_status(rid, "done",
                                           pipeline_config_hash=pipeline_config_hash)
                        logger.info(f"[{sid}] Complete.")

            all_threads.append(threading.Thread(target=_sink, daemon=True))

            # Progress display
            stop_progress = threading.Event()
            progress_thread: Optional[threading.Thread] = None
            if progress and not verbose:
                progress_thread = threading.Thread(
                    target=self._run_progress_display,
                    args=(scene_list, self.steps, par, inter_queues,
                          scenes_done, scenes_started, scenes_done_lock, stop_progress),
                    daemon=True,
                )

            for t in all_threads:
                t.start()
            if progress_thread:
                progress_thread.start()

            for sp_cw in scene_list:
                path_queue.put(sp_cw)  # each item is (path, crop_window) tuple
            for _ in range(n_reader):
                path_queue.put(_STOP)

            for t in all_threads:
                t.join()

            stop_progress.set()
            if progress_thread:
                progress_thread.join(timeout=2)

        finally:
            for h, level in saved_term_levels.items():
                h.setLevel(level)
            root_logger.removeHandler(scene_handler)
            _close_all_file_handlers()

        if failed_scenes:
            logger.warning(f"{len(failed_scenes)} scene(s) failed: {', '.join(failed_scenes)}")
        logger.warning(
            f"Project run complete "
            f"({len(scene_list) - len(failed_scenes)}/{len(scene_list)} succeeded)."
        )
        if run_stats:
            self.run_stats()

    # ------------------------------------------------------------------
    # Progress display
    # ------------------------------------------------------------------

    @staticmethod
    def _run_progress_display(scene_list, steps, par, inter_queues,
                               scenes_done, scenes_started, scenes_done_lock, stop_event) -> None:
        try:
            from rich.live import Live
            from rich.table import Table
            from rich.text import Text
            from rich import box
        except ImportError:
            return

        try:
            import psutil as _psutil
            _psutil.cpu_percent(interval=None)  # warm up
            _has_psutil = True
        except ImportError:
            _has_psutil = False

        n_scenes = len(scene_list)

        def _bar(q, width=20):
            try:
                size, cap = q.qsize(), q.maxsize or 1
            except Exception:
                size, cap = 0, 1
            frac = min(size / cap, 1.0)
            filled = int(frac * width)
            colour = "green" if frac < 0.33 else ("yellow" if frac < 0.67 else "red")
            return f"[{colour}]{'█' * filled}{'░' * (width - filled)}[/{colour}] {size}/{cap}"

        def _build():
            with scenes_done_lock:
                done = scenes_done[0]
                started = scenes_started[0]
            in_progress = max(0, started - done)
            w = 44
            done_cells = int(w * done / n_scenes) if n_scenes else 0
            prog_cells = min(int(w * started / n_scenes) if n_scenes else 0, w) - done_cells
            empty_cells = w - done_cells - prog_cells
            bar = (
                f"[green]{'█' * done_cells}[/green]"
                f"[yellow]{'·' * prog_cells}[/yellow]"
                f"[dim]{'░' * empty_cells}[/dim]"
            )
            prog_str = f" ({in_progress})" if in_progress > 0 else ""
            t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            t.add_column("Stage", style="cyan", width=18)
            t.add_column("Workers", justify="center", width=8)
            t.add_column("Queue →", width=30)
            for idx, step in enumerate(steps):
                bar_col = _bar(inter_queues[idx]) if idx < len(inter_queues) else ""
                t.add_row(step.replace("_", " "), str(par.get(step, 1)), bar_col)
            if _has_psutil:
                cpu = _psutil.cpu_percent(interval=None)
                ram = _psutil.virtual_memory()
                cc = "green" if cpu < 60 else ("yellow" if cpu < 85 else "red")
                rc = "green" if ram.percent < 60 else ("yellow" if ram.percent < 85 else "red")
                t.add_row("[dim]system[/dim]", "",
                          f"[{cc}]CPU {cpu:4.1f}%[/{cc}]  "
                          f"[{rc}]RAM {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB[/{rc}]")
            outer = Table.grid()
            outer.add_row(Text.from_markup(
                f"[bold]Scenes[/bold]  {bar}  {done}{prog_str} / {n_scenes}"
            ))
            outer.add_row(t)
            return outer

        from rich.console import Console
        with Live(_build(), console=Console(), refresh_per_second=2, transient=True) as live:
            while not stop_event.is_set():
                live.update(_build())
                stop_event.wait(timeout=0.5)
            live.update(_build())

    # ------------------------------------------------------------------
    # Scene metadata and footprint helpers
    # ------------------------------------------------------------------

    def _compute_scene_footprint(self, scene) -> Optional[bytes]:
        """Compute a WKB bounding-box polygon for a scene in EPSG:4326.

        Args:
            scene: A loaded :class:`~clouds_decoded.data.Sentinel2Scene`.

        Returns:
            WKB bytes of the bounding polygon, or ``None`` on failure.
        """
        try:
            import shapely.geometry
            import shapely
            from rasterio.transform import array_bounds
            from rasterio.warp import transform_bounds

            ref = scene.bands.get("B02") or next(iter(scene.bands.values()))
            h, w = ref.shape
            bounds = array_bounds(h, w, scene.transform)
            west, south, east, north = transform_bounds(
                scene.crs, "EPSG:4326", *bounds)
            poly = shapely.geometry.box(west, south, east, north)
            return shapely.to_wkb(poly)
        except Exception:
            return None

    def _write_scene_metadata(self, scene, scene_id: str) -> None:
        """Populate the ``scene_metadata`` table for a newly loaded scene.

        Args:
            scene: A loaded :class:`~clouds_decoded.data.Sentinel2Scene`.
            scene_id: The scene identifier string.
        """
        try:
            satellite = tile_id = None
            orbit_rel = None
            if scene.product_uri:
                parts = Path(scene.product_uri).stem.split("_")
                if len(parts) >= 6:
                    satellite = parts[0]
                    orbit_str = parts[4]
                    orbit_rel = int(orbit_str[1:]) if orbit_str.startswith("R") else None
                    tile_id = parts[5]

            footprint_wkb = self._compute_scene_footprint(scene)
            sensing_str = scene.sensing_time.isoformat() if scene.sensing_time else None

            self.db.upsert_scene_metadata(scene_id, {
                "sensing_time": sensing_str,
                "satellite":    satellite,
                "tile_id":      tile_id,
                "orbit_rel":    orbit_rel,
                "lat_center":   scene.latitude,
                "lon_center":   scene.longitude,
                "sun_zenith":   scene.sun_zenith,
                "sun_azimuth":  scene.sun_azimuth,
                "footprint":    footprint_wkb,
                "crs":          str(scene.crs) if scene.crs else None,
            })
        except Exception as exc:
            logger.debug(f"_write_scene_metadata failed for {scene_id}: {exc}")

    def _write_run_footprint(self, scene, run_id: str,
                              crop_window: Optional[str]) -> None:
        """Store a WKB footprint polygon for this run in the ``runs`` table.

        For crop runs the footprint is the crop window projected to EPSG:4326;
        for full-scene runs it reuses :meth:`_compute_scene_footprint`.

        Args:
            scene: A loaded :class:`~clouds_decoded.data.Sentinel2Scene`.
            run_id: The run identifier.
            crop_window: Optional crop window string ``'col,row,w,h'``.
        """
        try:
            if crop_window:
                import shapely.geometry
                import shapely
                from rasterio.warp import transform as warp_transform

                col_off, row_off, width, height = map(int, crop_window.split(","))
                tf = scene.transform
                corners_xy = [
                    tf * (col_off, row_off),
                    tf * (col_off + width, row_off),
                    tf * (col_off + width, row_off + height),
                    tf * (col_off, row_off + height),
                ]
                xs = [c[0] for c in corners_xy]
                ys = [c[1] for c in corners_xy]
                lons, lats = warp_transform(scene.crs, "EPSG:4326", xs, ys)
                poly = shapely.geometry.Polygon(zip(lons, lats))
                footprint_wkb = shapely.to_wkb(poly)
            else:
                footprint_wkb = self._compute_scene_footprint(scene)

            if footprint_wkb:
                self.db.set_footprint(run_id, footprint_wkb)
        except Exception as exc:
            logger.debug(f"_write_run_footprint failed for {run_id}: {exc}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def run_stats(
        self,
        force: bool = False,
        methods: Optional[List[str]] = None,
        run_id_filter: Optional[str] = None,
    ) -> None:
        """Compute and store statistics for all completed runs.

        Stats are written to per-step tables (e.g. ``stats_cloud_mask``) in
        ``project.db``. Already-computed stats are skipped unless *force* is True.

        Args:
            force: Re-compute stats even if they already exist.
            methods: Override list of ``'step::fn'`` identifiers. Defaults to
                ``config.stats``.
            run_id_filter: If provided, only compute stats for this run_id.
        """
        from clouds_decoded.stats import resolve_stats_fn, StatsCaller

        methods = methods or self.config.stats
        for row in self.db.get_all():
            if row["status"] != "done":
                continue
            if run_id_filter and row["run_id"] != run_id_filter:
                continue
            scene_id = row["scene_id"]
            crop_win  = row["crop_window"]
            run_id    = row["run_id"]
            manifest  = self._load_manifest(scene_id, row["path"], crop_win)
            caller    = StatsCaller(manifest)

            for ident in methods:
                table = f"stats_{ident.split('::')[0]}"
                if not force and self.db.has_stats(run_id, table):
                    continue
                try:
                    fn, step_name = resolve_stats_fn(ident)
                    stats = caller.call(fn, step_name)
                    self.db.write_stats(run_id, table, stats)
                except Exception as exc:
                    logger.warning(f"[{scene_id}] stats {ident} failed: {exc}")

    def _process_one_scene(
        self,
        scene_path: str,
        force: bool = False,
        unsafe: bool = False,
        crop_window: Optional[str] = None,
        processors: Optional[Dict[str, Any]] = None,
        git_hash: Optional[str] = None,
    ):
        """Serial wrapper around the pipeline stages."""
        scene_id = self._scene_id(scene_path)
        ctx = _PipelineCtx(
            scene_path=scene_path, crop_window=crop_window,
            log_path=self._scene_log_path(scene_id, crop_window),
            force=force, unsafe=unsafe, git_hash=git_hash,
        )
        self._prepare_scene_context(ctx)
        if ctx.failed:
            raise ctx.error  # type: ignore[misc]
        for step_idx, step_name in enumerate(ctx.steps_to_run):
            step_proc = (
                {k: v for k, v in processors.items()
                 if k == step_name or k == "_cloud_mask_postprocessor"}
                if processors else self._create_processor_for_step(step_name)
            )
            self._execute_step_in_ctx(ctx, step_name, step_idx, step_proc)
            if ctx.failed:
                raise ctx.error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _status_row(self, label: str, db_status: str, manifest: SceneManifest,
                    steps: List[str], crop_display: str = "(full)") -> str:
        cols = []
        for step in steps:
            if step in manifest.steps:
                s = manifest.steps[step].status
                cols.append("done" if s == "completed" else ("FAILED" if s == "failed" else "--"))
            else:
                cols.append("--")
        display = label[:48] + ".." if len(label) > 50 else label
        return (f"{display:<50} {crop_display:<14} {db_status:<10} "
                + " ".join(f"{c:<12}" for c in cols))

    def status(self) -> str:
        all_rows = self.db.get_all()
        counts = self.db.count_by_status()
        count_str = "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
        lines = [
            f"Project: {self.config.name} ({self.config.pipeline})",
            f"Scenes:  {len(all_rows)}",
        ]
        if count_str:
            lines.append(f"  {count_str}")
        lines.append("")
        steps = self.steps
        header = (f"{'Scene':<50} {'Crop':<14} {'Status':<10} "
                  + " ".join(f"{s:<12}" for s in steps))
        lines += [header, "-" * len(header)]

        # Collect (scene_id, crop_window) pairs already shown from the DB
        db_crop_pairs: set = set()
        for row in all_rows:
            cw = row.get("crop_window")
            if cw is not None:
                db_crop_pairs.add((row["scene_id"], cw))

        for row in all_rows:
            scene_id = row["scene_id"]
            scene_path = row["path"]
            db_status = row["status"]
            cw = row.get("crop_window")
            manifest = self._load_manifest(scene_id, scene_path, cw)
            cw_display = "(full)" if cw is None else cw
            label = scene_id if cw is None else f"  {scene_id}"
            lines.append(self._status_row(label, db_status, manifest, steps, cw_display))

            if cw is None:
                # Scan filesystem for old-style crop directories not yet in DB
                crops_dir = self.output_dir / scene_id / "crops"
                if crops_dir.exists():
                    for crop_dir in sorted(crops_dir.iterdir()):
                        if not crop_dir.is_dir():
                            continue
                        mp = crop_dir / "manifest.json"
                        if not mp.exists():
                            continue
                        cm = SceneManifest.from_json(mp)
                        cw_str = cm.crop_window or crop_dir.name.replace("_", ",")
                        if (scene_id, cw_str) in db_crop_pairs:
                            continue  # already shown from DB row
                        lines.append(self._status_row(
                            f"  crop:{cw_str}", "--", cm, steps,
                            crop_display=cw_str,
                        ))
        return "\n".join(lines)
