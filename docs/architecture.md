# Architecture Overview

This document covers the key design patterns and abstractions in clouds-decoded for developers working on or extending the codebase.

## Pipeline Order

The processing pipeline runs in a fixed order, where each step depends on outputs from previous steps:

```
cloud_mask --> cloud_height --> albedo --> refocus --> cloud_properties
```

1. **Cloud Mask** -- binary cloud mask via SegFormer-B2 (4-class inference internally, then binarized) or simple thresholding.
2. **Cloud Height** -- Stereo parallax or DL emulator. Uses cloud mask to focus on cloud pixels.
3. **Albedo** -- Surface reflectance from clear-sky pixels (identified by cloud mask).
4. **Refocus** -- Parallax correction using cloud height. Returns a new [`Sentinel2Scene`][clouds_decoded.data.sentinel.Sentinel2Scene] with `is_refocused=True`. No output file -- pure in-memory transformation.
5. **Cloud Properties** -- Neural inversion on refocused scene, using height and albedo.

The `full-workflow` CLI command and the project system both execute these steps in sequence.

---

## Processor Pattern

Every processing module follows the same interface. Source: `src/modules/<name>/processor.py`.

```python
class SomeProcessor:
    def __init__(self, config: SomeConfig):
        # Store config, load model weights, etc.
        self.config = config

    def process(self, scene: Sentinel2Scene, **kwargs) -> TypedOutput:
        # Run processing, return typed data object
        ...
```

Key rules:

- **Config in `__init__`**: all parameters come from the config object. No processing happens during construction.
- **Processing in `process()`**: the scene (and optional dependencies like cloud mask, height data) are passed here.
- **Typed output**: each processor returns a specific data class ([`CloudMaskData`][clouds_decoded.data.cloud_mask.CloudMaskData], [`CloudHeightGridData`][clouds_decoded.data.cloud_height.CloudHeightGridData], etc.).
- **Stateless**: processors hold no mutable state between `process()` calls beyond their config and loaded model weights.

---

## Config Pattern

All configs inherit from [`BaseProcessorConfig`][clouds_decoded.config.BaseProcessorConfig] (Pydantic `BaseModel`). Source: `src/shared_utils/config.py`.

```python
class MyConfig(BaseProcessorConfig):
    model_config = ConfigDict(extra='forbid')  # inherited, but explicit is fine

    some_param: int = Field(default=10, ge=1, le=100, description="...")
    method: Literal["a", "b"] = Field(default="a", description="...")

    @field_validator('some_param')
    @classmethod
    def validate_some_param(cls, v):
        ...
        return v

    @computed_field
    @property
    def derived_value(self) -> int:
        return self.some_param * 2
```

Conventions:

- `extra='forbid'` -- unknown YAML fields raise errors.
- `Field(description=...)` on every field.
- Numeric constraints with `ge=`, `le=`.
- `@field_validator` for custom validation.
- `@computed_field` for derived values (excluded from YAML serialization).
- `from_yaml(path)` / `to_yaml(path)` for serialization.
- Model paths default to `None` and are resolved from managed assets via `@model_validator`.

---

## Data Pattern

Processing outputs inherit from [`GeoRasterData`][clouds_decoded.data.base.GeoRasterData]. Source: `src/shared_utils/data/base.py`.

```python
class MyOutputData(GeoRasterData):
    metadata: MyMetadata = Field(default_factory=MyMetadata)

    def validate(self) -> bool:
        # Custom validation
        ...
```

GeoTIFF outputs embed provenance metadata in tags (under the key defined by `METADATA_TAG` in `constants.py`). This includes the project name, codebase version, git hash, step config snapshot, and scene path. The project system uses this for integrity checking on resume.

See [Data Classes](data-classes.md) for the full reference.

---

## Lazy Imports in CLI

The CLI entry point (`src/cli/entry.py`) must keep `--help` and shell autocomplete fast. Heavy dependencies (torch, scipy, rasterio, transformers) are imported **inside function bodies**, never at module level.

```python
# CORRECT -- lazy import
@app.command()
def cloud_mask(...):
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
    ...

# WRONG -- module-level import slows --help
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
```

Type annotations use `TYPE_CHECKING` guards for IDE support without runtime cost:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
```

---

## [`NormalizationWrapper`][clouds_decoded.normalization.NormalizationWrapper]

Neural network modules wrap their core model in a [`NormalizationWrapper`][clouds_decoded.normalization.NormalizationWrapper] subclass. Source: `src/shared_utils/normalization.py`.

The pattern:

1. Register input/output normalization statistics as PyTorch buffers (survive `save`/`load`).
2. `forward()` normalizes inputs, runs the core model, then denormalizes outputs.
3. The base class provides min-max [-1, 1] defaults. Subclasses override for different conventions (e.g. sigmoid [0, 1] output, linear scaling).

Examples:

- `src/modules/refl2prop/model.py`
- `src/modules/albedo_estimator/datadriven/model.py`
- `src/modules/cloud_height_emulator/processor.py`

---

## Sliding Window Inference

Shared infrastructure for tiled inference on large images. Source: `src/shared_utils/sliding_window.py`.

[`SlidingWindowInference`][clouds_decoded.sliding_window.SlidingWindowInference] handles:

- Splitting an image into overlapping tiles.
- Batched GPU/CPU inference.
- Blending overlapping predictions back into a full-resolution output.

Used by:

- **Cloud mask processor** -- SegFormer-B2 inference at 512px patches.
- **Cloud height emulator** -- ResUNet inference at 1024px windows.

The `skip_mask` parameter (bool H x W) allows skipping windows with no relevant pixels (e.g. no cloud pixels), saving computation.

---

## Module Directory Structure

Each processing module follows a consistent layout:

```
src/modules/<name>/
  config.py       # Pydantic config class
  processor.py    # Processor class with process() method
  model.py        # Neural network architecture (if applicable)
  train.py        # Training code (if applicable)
  dataset.py      # Dataset class (if applicable)
  physics.py      # Physical models (e.g. cloud_height)
  data.py         # Module-specific data helpers
```

At minimum, every module has `config.py` and `processor.py`.

---

## Key Source Files

| File | Role |
|------|------|
| `src/cli/entry.py` | CLI entry point (Typer) |
| `src/shared_utils/config.py` | [`BaseProcessorConfig`][clouds_decoded.config.BaseProcessorConfig] base class |
| `src/shared_utils/project.py` | Project system, workflow execution, DuckDB |
| `src/shared_utils/data/base.py` | [`Data`][clouds_decoded.data.base.Data], [`GeoRasterData`][clouds_decoded.data.base.GeoRasterData] base classes |
| `src/shared_utils/data/sentinel.py` | [`Sentinel2Scene`][clouds_decoded.data.sentinel.Sentinel2Scene] -- reads `.SAFE` directories |
| `src/shared_utils/data/band.py` | [`Sentinel2Band`][clouds_decoded.data.band.Sentinel2Band], [`BandDict`][clouds_decoded.data.band.BandDict] |
| `src/shared_utils/constants.py` | Band metadata, physical constants |
| `src/shared_utils/normalization.py` | [`NormalizationWrapper`][clouds_decoded.normalization.NormalizationWrapper] base class |
| `src/shared_utils/sliding_window.py` | [`SlidingWindowInference`][clouds_decoded.sliding_window.SlidingWindowInference] |
| `src/stats/__init__.py` | [`StatsCaller`][clouds_decoded.stats.StatsCaller], [`resolve_stats_fn()`][clouds_decoded.stats.resolve_stats_fn] |
