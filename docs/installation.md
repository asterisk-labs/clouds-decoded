# Installation

## Requirements

- Python >= 3.10
- conda (recommended for environment management)
- Git

## Setup

Clone the repository and create a conda environment:

```bash
git clone https://github.com/asterisk-labs/clouds-decoded.git
cd clouds-decoded
conda create -n cd-build-0 python=3.10
conda activate cd-build-0
pip install -e .
```

After installation, the `clouds-decoded` CLI is available:

```bash
clouds-decoded --help
```

## Asset Setup

clouds-decoded uses large binary assets (model weights) that are hosted on HuggingFace and not bundled with the package. Two commands manage these assets.

### Configure the assets directory

Run `setup` to choose where assets are stored on disk. The path is saved to a user config file and persists across sessions. You can override it at any time with the `CLOUDS_DECODED_ASSETS_DIR` environment variable.

```bash
clouds-decoded setup
```

The command lists the assets and their approximate sizes:

| Asset | Size |
|-------|------|
| Cloud mask SegFormer model weights | ~105 MB |
| Height emulator model weights | ~100 MB |
| Refl2prop model weights | ~1 MB |
| Data-driven albedo model weights | ~1 MB |
| GEBCO bathymetry (optional) | ~2.7 GB |

### Download model weights

Use the `download` command to fetch individual assets or all of them:

```bash
clouds-decoded download emulator      # height emulator weights
clouds-decoded download refl2prop     # cloud property inversion weights
clouds-decoded download cloud_mask    # cloud mask segmentation weights
clouds-decoded download sample_scene  # sample Sentinel-2 scene (~705 MB)
clouds-decoded download gebco         # GEBCO bathymetry (optional)
clouds-decoded download all           # everything
```

Use `--force` to re-download an asset that already exists locally. Use `--yes` to skip confirmation prompts.
