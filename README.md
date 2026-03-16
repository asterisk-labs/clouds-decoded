# clouds-decoded

Retrieve physical cloud properties from Sentinel-2 satellite imagery. Given a Level-1C scene (`.SAFE` directory), the toolkit produces:

- **Cloud mask** (binary cloud/clear)
- **Cloud top height** (meters, via stereo parallax)
- **Surface albedo** (inverse-distance weighted interpolation of clear-sky pixels, with MLP fallback)
- **Cloud optical/microphysical properties** (optical thickness, effective radii, ice-liquid ratio)

All outputs are georeferenced GeoTIFFs that open directly in QGIS, Python, or any GDAL-compatible tool.

## Installation

Requires Python >= 3.10. A conda environment is recommended:

```bash
git clone <repo-url>
cd clouds-decoded

conda create -n clouds-decoded python=3.12
conda activate clouds-decoded

pip install -e .
```

After installation the `clouds-decoded` command is available in your terminal.

## Quick Start

### Run Everything on a Scene

The simplest way to get all outputs is the `full-workflow` command. It runs
cloud masking, height retrieval, albedo estimation, parallax correction (refocusing),
and cloud property inversion in sequence:

```bash
clouds-decoded full-workflow /data/S2A_MSIL1C_20210801.SAFE --output-dir results/
```

This creates `results/` containing `cloud_mask.tif`, `cloud_height.tif`,
`albedo.tif`, and `properties.tif`.

### Run Individual Steps

Each processing step is also available as a standalone command:

```bash
# Cloud mask (threshold method, fast)
clouds-decoded cloud-mask scene.SAFE --method threshold --output-path mask.tif

# Cloud mask (SEnSeIv2 deep learning, more accurate)
clouds-decoded cloud-mask scene.SAFE --method senseiv2 --output-path mask.tif

# Cloud height
clouds-decoded cloud-height scene.SAFE --output-path height.tif

# Surface albedo (needs a cloud mask for clear-sky fitting)
clouds-decoded albedo scene.SAFE --mask-path mask.tif --output-path albedo.tif

# Cloud properties (needs cloud height; model_path set via config YAML or defaults)
clouds-decoded cloud-properties scene.SAFE --height-path height.tif --output-path props.tif

# Parallax correction (refocus)
clouds-decoded refocus scene.SAFE --height-path height.tif --output-dir refocused/
```

### Spatial Cropping

All commands accept `--crop-window` to process a sub-region instead of the full
110 km x 110 km scene. The format is `col_off,row_off,width,height` in 10 m pixels:

```bash
clouds-decoded full-workflow scene.SAFE --crop-window 2000,3000,1000,1000
```

## Configuration

Every processing step has a Pydantic config class with documented defaults. You
can override settings via YAML files:

```bash
clouds-decoded cloud-height scene.SAFE --config-path my_height_config.yaml
```

To see all available options, run any command with `--help`:

```bash
clouds-decoded cloud-height --help
clouds-decoded full-workflow --help
```

## Projects

For batch processing of multiple scenes with consistent settings, use the
**project** system. A project is a directory that holds editable config YAMLs
and organizes outputs per scene with automatic resumability.

```bash
# Create a project with default configs
clouds-decoded project init ./my_analysis

# Edit configs to your liking
nano ./my_analysis/configs/cloud_height.yaml

# Process one or more scenes (auto-registered)
clouds-decoded project run ./my_analysis /data/scene1.SAFE /data/scene2.SAFE

# Check progress
clouds-decoded project status ./my_analysis
```

Re-running `project run` skips steps that already completed with the same config.
If you edit a config, downstream steps are automatically re-run. Use `--force` to
reprocess everything from scratch.

## Development

```bash
# Run tests
pytest tests/ -v
```
