# Project System

The project system organises batch processing of multiple Sentinel-2 scenes with config management, resumability, and statistics. Source: `src/shared_utils/project.py`.

## Workflow

```
init  -->  edit configs  -->  stage scenes  -->  run  -->  status / stats  -->  delete
```

1. **Init**: Create a project directory with default config YAMLs.
2. **Edit configs**: Tune processing parameters in `configs/*.yaml`.
3. **Stage scenes**: Register `.SAFE` scene paths.
4. **Run**: Process all staged scenes through the pipeline.
5. **Status/Stats**: Monitor progress and compute summary statistics.

---

## Directory Structure

After `clouds-decoded project init ./my_analysis`, the project directory contains:

```
my_analysis/
  project.yaml          # Project metadata and pipeline definition
  project.db            # DuckDB database (runs, stats)
  configs/              # Editable YAML configs for each module
    cloud_mask.yaml
    cloud_height_emulator.yaml
    albedo.yaml
    refocus.yaml
    cloud_properties.yaml
  outputs/              # Per-scene output directories (created on first run)
    <scene_id>/
      cloud_mask.tif
      cloud_height.tif
      albedo.tif
      properties.tif
      manifest.json     # Per-scene processing manifest
    <scene_id>/
      ...
  logs/                 # Per-scene log files
    <scene_id>/
      pipeline.log
```

---

## Quick Start

```bash
# Create project
clouds-decoded project init ./analysis

# Edit configs as needed
# e.g. vim ./analysis/configs/cloud_mask.yaml

# Process scenes (staging + running in one step)
clouds-decoded project run ./analysis /data/scene1.SAFE /data/scene2.SAFE

# Check progress
clouds-decoded project status ./analysis

# Or stage first, run later
clouds-decoded project stage ./analysis /data/scenes/
clouds-decoded project run ./analysis
```

---

## Config Hashing and Resumability

Each processing step's config is hashed (SHA-256 of the JSON-serialised config). When a run is resumed:

- Steps with matching config hashes and existing output files are **skipped**.
- Steps where the config has changed are **re-run**.
- Use `--force` to reprocess all steps regardless of cache state.
- Use `--force-overwrite` to reset "done" scenes whose configs have changed and re-run them.

The per-scene `manifest.json` tracks the status, config hash, output file, and timing for each step.

---

## Pipeline Definition

Pipelines are defined as ordered sequences of [`WorkflowStepDef`][clouds_decoded.project.WorkflowStepDef] nodes, bundled into a [`WorkflowDef`][clouds_decoded.project.WorkflowDef]. Each step specifies:

| Field | Description |
|-------|-------------|
| `name` | Unique step ID |
| `processor` | Key into the processor registry |
| `config` | Config YAML filename (relative to `configs/`) |
| `inputs` | Positional input token names |
| `keyword_inputs` | Named input tokens (`{param: token}`) |
| `output` | Output token name (None = terminal) |
| `output_file` | Disk filename (None = ephemeral, no write/resume) |

The built-in recipes (e.g. `full-workflow`, `cloud-height-comparison`) are stored in the package and embedded into `project.yaml` at init time. You can edit the workflow directly in `project.yaml` for custom pipelines.

---

## Statistics System

Statistics are computed per-scene and stored in DuckDB tables inside `project.db`. The [`StatsCaller`][clouds_decoded.stats.StatsCaller] class manages loading data and dispatching stats functions.

### Configuration

Stats methods are listed in `project.yaml`:

```yaml
stats:
  - cloud_mask::class_fractions
  - cloud_height_emulator::percentiles
  - cloud_properties::percentiles
  - albedo::mean
```

The format is `step_name::function_name`. The [`resolve_stats_fn()`][clouds_decoded.stats.resolve_stats_fn] function resolves identifiers in this order:

1. Step-specific module: `src/stats/<step_name>.py` (e.g. `cloud_mask.py`)
2. Generic fallback: `src/stats/_generic.py`

### Built-in stats functions

**Generic** (work on any [`GeoRasterData`][clouds_decoded.data.base.GeoRasterData]):

- `mean` -- Mean of valid pixels, plus `n_pixels` count.
- `median` -- Median of valid pixels.
- `percentiles` -- Percentiles at [0, 5, 25, 50, 75, 95, 100] plus `mean` and `n_pixels`.

**Step-specific**:

- `cloud_mask::class_fractions` -- Per-class pixel fractions (`clear_frac`, `thin_cloud_frac`, `thick_cloud_frac`, `cloud_shadow_frac`) plus `n_pixels`. Nodata (255) is excluded.

### Running stats

```bash
# Compute all configured stats for completed runs
clouds-decoded project stats ./analysis

# Re-compute even if already stored
clouds-decoded project stats ./analysis --force

# Run specific methods only
clouds-decoded project stats ./analysis --method cloud_mask::class_fractions

# Single run
clouds-decoded project stats ./analysis --run-id abc123def456abcd
```

Stats are also computed automatically at the end of `project run` unless `--no-stats` is passed.

---

## Parallel Processing

Use `--parallel` to enable concurrent processing across scenes and stages:

```bash
# All stages use 1 worker by default with --parallel
clouds-decoded project run ./analysis --parallel

# Set all stages to 4 workers
clouds-decoded project run ./analysis --parallel -j 4

# Fine-grained control per stage
clouds-decoded project run ./analysis --parallel -j cloud_height=2 -j albedo=2
```

Valid stage names for `-j`: `reader`, `cloud_mask`, `cloud_height_emulator`, `cloud_height`, `albedo`, `refocus`, `cloud_properties`.

Use `--queue-depth` to control how many scenes are buffered between adjacent pipeline stages (default: 2).

---

## project.yaml Reference

```yaml
name: my_analysis
pipeline: full-workflow
created_at: "2025-01-15T10:30:00"
output_dir: outputs
stats:
  - cloud_mask::class_fractions
  - cloud_height_emulator::percentiles
  - cloud_properties::percentiles
  - albedo::mean
workflow:
  steps:
    - name: cloud_mask
      processor: cloud_mask
      config: cloud_mask.yaml
      inputs: [scene]
      output: cloud_mask
      output_file: cloud_mask.tif
    # ... additional steps
```

[`ProjectConfig`][clouds_decoded.project.ProjectConfig] uses `extra='ignore'` for forward/backward compatibility -- older project files with missing fields load without error.
