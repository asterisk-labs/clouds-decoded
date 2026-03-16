# CLI Reference

All commands are accessed through the `clouds-decoded` executable. Every command accepts `--help` for full option details.

```bash
clouds-decoded --help
clouds-decoded <command> --help
```

---

## Processing Commands

### `cloud-mask`

4-class cloud segmentation (clear, thick cloud, thin cloud, cloud shadow) using deep learning (SegFormer-B2) or simple reflectance thresholding.

```bash
clouds-decoded cloud-mask scene.SAFE
clouds-decoded cloud-mask scene.SAFE --method threshold --threshold-band B08 --threshold-value 0.06
clouds-decoded cloud-mask scene.SAFE --output-path mask.tif --crop-window 0,0,1024,1024
```

| Option | Default | Description |
|--------|---------|-------------|
| `scene_path` (arg) | required | Path to `.SAFE` directory |
| `--output-path` | `mask_output.tif` | Output file path |
| `--config-path` | None | Config YAML (overrides flags) |
| `--method` | `senseiv2` | `senseiv2` or `threshold` |
| `--threshold-band` | `B08` | Band for threshold method |
| `--threshold-value` | `0.06` | Reflectance threshold (0--1) |
| `--resolution` | `20` | Model resolution in metres |
| `--crop-window` | None | Spatial crop: `col_off,row_off,width,height` |

### `cloud-height`

Cloud top height retrieval via stereo parallax detection using multi-band time delays, or a deep learning emulator (ResUNet).

```bash
clouds-decoded cloud-height scene.SAFE
clouds-decoded cloud-height scene.SAFE --no-use-emulator --mask-path mask.tif
```

| Option | Default | Description |
|--------|---------|-------------|
| `scene_path` (arg) | required | Path to `.SAFE` directory |
| `--output-path` | `height_output.tif` | Output file path |
| `--config-path` | None | Config YAML |
| `--mask-path` | None | Path to cloud mask file |
| `--crop-window` | None | Spatial crop |
| `--use-emulator / --no-use-emulator` | `True` | Use DL emulator vs physics-based retrieval |

### `albedo`

Surface albedo estimation from clear-sky pixels using inverse-distance weighted interpolation (IDW) or a trained MLP.

```bash
clouds-decoded albedo scene.SAFE --mask-path mask.tif
clouds-decoded albedo scene.SAFE --method datadriven
```

| Option | Default | Description |
|--------|---------|-------------|
| `scene_path` (arg) | required | Path to `.SAFE` directory |
| `--output-path` | `albedo_output.tif` | Output file path |
| `--mask-path` | None | Cloud mask for clear-sky sampling (required for IDW) |
| `--config-path` | None | Config YAML |
| `--method` | `idw` | `idw` or `datadriven` |
| `--fallback` | `datadriven` | Fallback when insufficient clear pixels: `datadriven` or `constant` |
| `--model-path` | None | Path to trained MLP checkpoint |
| `--output-resolution` | `300` | Output resolution in metres/pixel |
| `--crop-window` | None | Spatial crop |

### `refocus`

Parallax correction that warps Sentinel-2 bands to a common reference (B02) using cloud height data. Removes height-dependent misalignment between bands caused by push-broom acquisition.

```bash
clouds-decoded refocus scene.SAFE --height-path height.tif
clouds-decoded refocus scene.SAFE --height-path height.tif --output-dir refocused/ --output-resolution 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `scene_path` (arg) | required | Path to `.SAFE` directory |
| `--height-path` | required | Path to cloud height raster |
| `--output-dir` | `refocused` | Output directory for refocused band GeoTIFFs |
| `--output-resolution` | None | Common output resolution in metres (None = native) |
| `--interpolation-order` | `1` | Interpolation: 0=nearest, 1=bilinear, 3=cubic |
| `--crop-window` | None | Spatial crop |

### `cloud-properties`

Neural inversion of reflectance to cloud optical thickness, effective radii, and ice-liquid ratio. Requires pre-calculated cloud heights.

```bash
clouds-decoded cloud-properties scene.SAFE --height-path height.tif
clouds-decoded cloud-properties scene.SAFE --height-path height.tif --properties-method shading
```

| Option | Default | Description |
|--------|---------|-------------|
| `scene_path` (arg) | required | Path to `.SAFE` directory |
| `--height-path` | required | Path to cloud height raster |
| `--output-path` | `properties_output.tif` | Output file path |
| `--config-path` | None | Config YAML |
| `--properties-method` | `standard` | `standard` or `shading` |
| `--crop-window` | None | Spatial crop |

### `full-workflow`

Run the complete pipeline in sequence: Cloud Mask, Cloud Height, Albedo, Refocus, Cloud Properties.

```bash
clouds-decoded full-workflow scene.SAFE
clouds-decoded full-workflow scene.SAFE --output-dir results/ --config pipeline.yaml
clouds-decoded full-workflow scene.SAFE --mask-method threshold --no-use-emulator
```

| Option | Default | Description |
|--------|---------|-------------|
| `scene_path` (arg) | required | Path to `.SAFE` directory |
| `--output-dir` | `output` | Directory for all outputs |
| `--crop-window` | None | Spatial crop |
| `--mask-method` | `senseiv2` | Cloud mask method |
| `--use-emulator / --no-use-emulator` | `True` | Use DL emulator for height |
| `--config` | None | Pipeline config YAML with per-module sections |

The pipeline config YAML can contain sections for each module:

```yaml
cloud_mask:
  method: senseiv2
cloud_height:
  output_resolution: 60
albedo:
  method: idw
  output_resolution: 300
refocus:
  output_resolution: 60
cloud_properties:
  method: standard
```

---

## Project Commands

Project commands are accessed via `clouds-decoded project <subcommand>`.

### `project init`

Create a new project directory with default configs.

```bash
clouds-decoded project init ./my_analysis
clouds-decoded project init ./my_analysis --pipeline cloud-height-comparison
clouds-decoded project init ./new_analysis --clone ./existing_analysis
```

| Option | Default | Description |
|--------|---------|-------------|
| `project_dir` (arg) | required | Directory for the new project |
| `--name` | None | Project name (defaults to directory name) |
| `--pipeline` | `full-workflow` | Recipe name |
| `--clone` | None | Clone configs from an existing project |

### `project run`

Batch process scenes through the pipeline. Scenes can be passed directly (auto-registered) or pre-staged.

```bash
clouds-decoded project run ./analysis scene1.SAFE scene2.SAFE
clouds-decoded project run ./analysis --force --verbose
clouds-decoded project run ./analysis --parallel -j cloud_height=2 -j albedo=2
clouds-decoded project run ./analysis --parallel -j 4
```

| Option | Default | Description |
|--------|---------|-------------|
| `project_dir` (arg) | required | Path to project directory |
| `scenes` (arg) | None | Scene `.SAFE` paths to process |
| `--force` | False | Reprocess all steps |
| `--unsafe` | False | Skip file provenance validation |
| `--crop-window` | None | Spatial crop |
| `--parallel` | False | Enable concurrent scene/stage processing |
| `--workers`, `-j` | None | Per-stage worker count (e.g. `-j cloud_height=2` or `-j 4`) |
| `--queue-depth` | `2` | Max scenes buffered between stages |
| `--verbose`, `-v` | False | Print INFO-level logs to terminal |
| `--no-progress` | False | Disable live progress display |
| `--no-stats` | False | Skip automatic stats computation |
| `--force-overwrite` | False | Reset and re-run scenes with changed configs |
| `--ignore-integrity` | False | Skip config integrity check |

### `project status`

Show processing status for all scenes in the project.

```bash
clouds-decoded project status ./analysis
```

### `project stage`

Register scenes in the project without processing them. Accepts `.SAFE` paths or parent directories (scanned for `*.SAFE`).

```bash
clouds-decoded project stage ./analysis ./scene.SAFE
clouds-decoded project stage ./analysis /data/sentinel2/
```

### `project list`

List all registered runs and their status.

```bash
clouds-decoded project list ./analysis
clouds-decoded project list ./analysis --status staged
clouds-decoded project list ./analysis --crop-window 0,0,512,512
```

### `project stats`

Compute and store statistics for all completed runs. Results are written to DuckDB tables in `project.db`.

```bash
clouds-decoded project stats ./analysis
clouds-decoded project stats ./analysis --force
clouds-decoded project stats ./analysis --method cloud_mask::class_fractions
clouds-decoded project stats ./analysis --run-id abc123def456abcd
```

---

## Utility Commands

### `setup`

Configure the directory used to store model weights and other large binary assets. See [Installation](installation.md) for details.

```bash
clouds-decoded setup
```

### `download`

Download managed binary assets (model weights, GEBCO bathymetry).

```bash
clouds-decoded download emulator
clouds-decoded download all --force
```

### `view`

Launch a 3D point-cloud viewer (viser) for a project's cloud height outputs.

```bash
clouds-decoded view ./analysis
clouds-decoded view ./analysis --port 8080 --max-grid-dim 800
```

For remote servers, port-forward with: `ssh -L 8080:localhost:8080 user@server`

### `serve`

Launch a web-based viewer (Panel + Bokeh) for a processed scene.

```bash
clouds-decoded serve ./analysis
clouds-decoded serve ./analysis --scene-id S2A_MSIL1C_... --port 5006
```

For remote servers, port-forward with: `ssh -L 5006:localhost:5006 user@server`
