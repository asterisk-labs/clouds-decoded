# Clouds Decoded: Architecture & Developer Guide

**Version**: 1.0
**Date**: 2026-02-04
**Purpose**: Definitive guide to codebase structure, design principles, and development standards

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [Directory Structure](#directory-structure)
4. [Core Design Patterns](#core-design-patterns)
5. [Data Model System](#data-model-system)
6. [Processor Pattern](#processor-pattern)
7. [Configuration System](#configuration-system)
8. [Import Conventions](#import-conventions)
9. [Module Development Guide](#module-development-guide)
10. [CLI Design](#cli-design)
11. [Testing Standards](#testing-standards)
12. [Common Pitfalls](#common-pitfalls)

---

## Overview

### What Is This Project?

**Clouds Decoded** is a unified toolkit for processing Sentinel-2 satellite imagery to extract cloud properties. It brings together multiple specialized algorithms under a single, user-friendly CLI.

### Core Capabilities

1. **Cloud Masking**: Identify cloudy vs. clear pixels (ML-based or simple thresholding)
2. **Cloud Height**: Stereo-matching using parallax between S2 bands
3. **Cloud Properties**: Neural inversion to retrieve optical thickness, particle size, phase
4. **Albedo Estimation**: Surface reflectance estimation for radiative transfer

### Design Philosophy

> **"Consistent interfaces, strict types, clear data flow"**

We prioritize:
- **Type safety** via Pydantic data models
- **Reproducibility** via configuration files
- **Composability** via standardized processor interfaces
- **Usability** via progressive disclosure (simple defaults, advanced options available)

---

## Architectural Principles

### 1. **Separation of Concerns**

```
┌─────────────┐
│     CLI     │  ← User interaction, argument parsing, orchestration
└──────┬──────┘
       │
┌──────▼────────┐
│  Processors   │  ← Algorithm logic, computation
└──────┬────────┘
       │
┌──────▼────────┐
│  Data Models  │  ← Type-safe containers, I/O operations
└───────────────┘
```

**Rules**:
- CLI code never contains algorithm logic
- Processors never parse command-line arguments
- Data models never contain business logic

### 2. **Explicit Over Implicit**

```python
# ✅ GOOD: Clear, explicit
processor = CloudHeightProcessor(config)
result = processor.process(scene)

# ❌ BAD: Magic defaults, hidden behavior
result = auto_process(scene)  # What did it do? What config?
```

### 3. **Fail Fast, Fail Clearly**

```python
# ✅ GOOD: Validate early
if scene.sun_zenith is None:
    raise ValueError("Scene missing sun_zenith angle. Call scene.read() first.")

# ❌ BAD: Let it crash deep in processing
height = calculate_something(scene.sun_zenith)  # AttributeError 500 lines later
```

### 4. **Configuration Is Code**

All parameters must be:
- Documented (Field descriptions)
- Validated (Pydantic types)
- Versioned (saved with outputs)
- Reproducible (YAML serializable)

### 5. **One Source of Truth**

- Constants defined once in `shared_utils/constants.py`
- Base classes defined once in `shared_utils/`
- No duplication between modules

---

## Directory Structure

### Current Structure

```
clouds-decoded/
├── src/
│   ├── cli/                    # Command-line interface
│   │   ├── __init__.py
│   │   └── entry.py           # Typer app, commands
│   │
│   ├── shared_utils/          # Shared across all modules
│   │   ├── __init__.py
│   │   ├── config.py          # BaseProcessorConfig
│   │   ├── constants.py       # S2 bands, physics constants
│   │   ├── processors.py      # Unified imports
│   │   └── data/              # Data model definitions
│   │       ├── __init__.py
│   │       ├── base.py        # Abstract base classes
│   │       ├── sentinel.py    # Sentinel2Scene
│   │       ├── cloud_height.py
│   │       ├── cloud_mask.py
│   │       └── refl2prop.py
│   │
│   └── modules/               # Processing modules
│       ├── __init__.py
│       ├── cloud_height/      # Stereo height retrieval
│       ├── cloud_mask/        # Cloud detection
│       ├── refl2prop/         # Property inversion
│       └── albedo_estimator/  # Surface albedo
│
├── tests/                     # All tests mirror src/
├── docs/                      # User documentation
├── pyproject.toml             # Single source of dependencies
└── README.md
```

### Module Structure (Standard)

Every processing module follows this structure:

```
modules/my_module/
├── __init__.py          # Exports: MyModuleProcessor, MyModuleConfig
├── config.py            # Pydantic config class
├── processor.py         # Main Processor class
├── internal.py          # Helper functions (optional)
└── README.md            # Module-specific documentation
```

**Do NOT include**:
- ❌ Standalone `setup.py` (use root `pyproject.toml`)
- ❌ Module-specific `config.yaml` files (examples go in `docs/examples/`)
- ❌ Docker files (use root-level containerization)

---

## Core Design Patterns

### The Processing Pipeline

All algorithms follow this flow:

```
Input Data → Processor.process() → Output Data
     ↓              ↓                    ↓
Sentinel2Scene   Algorithm          Typed Result
  (validated)    (configured)        (georeferenced)
```

### Type Flow

```python
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from clouds_decoded.modules.cloud_height import CloudHeightProcessor, CloudHeightConfig

# 1. Load input (type: Sentinel2Scene)
scene = Sentinel2Scene()
scene.read("path/to/scene.SAFE")

# 2. Configure (type: CloudHeightConfig)
config = CloudHeightConfig.from_yaml("config.yaml")

# 3. Process (type: CloudHeightProcessor)
processor = CloudHeightProcessor(config)

# 4. Get output (type: CloudHeightGridData)
result = processor.process(scene)

# 5. Save output
result.write("output.tif")
```

**Every step is typed. No raw dictionaries crossing module boundaries.**

### Flexible Input Pattern

Processors should accept either in-memory objects OR file paths for inputs:

```python
from typing import Union
from pathlib import Path

def process(
    self,
    scene: Union[Sentinel2Scene, str, Path],
    mask: Optional[Union[CloudMaskData, str, Path]] = None
) -> OutputData:
    # Normalize inputs
    scene = self._load_if_needed(scene, Sentinel2Scene)
    if mask:
        mask = self._load_if_needed(mask, CloudMaskData)
    # Process...
```

**Benefits**: Enables pipeline mode (fast, in-memory) and standalone mode (resumable, checkpointed).

### Data Transformation Pattern

Some data products need format adaptation for different downstream uses. Use a single flexible `transform()` method:

```python
class CloudMaskData(GeoRasterData):
    def transform(
        self,
        positive_classes: Optional[List[int]] = None,
        confidence_threshold: float = 0.5,
        dilation_meters: float = 0.0,
        output_resolution: Optional[int] = None
    ) -> 'CloudMaskData':
        """Transform mask for downstream processing."""
        # Implementation handles categorical/binary automatically
```

**Usage**: `clear_mask = cloud_mask.transform(positive_classes=[0])`

---

## Data Model System

### Hierarchy

```
Data (ABC)
├── GeoRasterData
│   ├── CloudHeightGridData
│   ├── CloudMaskData
│   ├── CloudPropertiesData
│   └── AlbedoData
├── PointCloudData
│   └── CloudHeightPointsData
└── Sentinel2Scene (special case)
```

### Rules for Data Models

#### ✅ DO:

```python
from pydantic import Field
from clouds_decoded.data import GeoRasterData, Metadata

class MyOutputMetadata(Metadata):
    """Metadata specific to my algorithm output."""
    algorithm_version: str = "1.0"
    processing_time_seconds: float = Field(..., description="Total runtime")

class MyOutputData(GeoRasterData):
    """My algorithm output as a georeferenced raster."""
    metadata: MyOutputMetadata = Field(default_factory=MyOutputMetadata)

    def validate(self) -> bool:
        """Custom validation for my output."""
        if self.data is None:
            return True
        # Add specific checks
        return super().validate()
```

#### ❌ DON'T:

```python
# BAD: Raw dictionary for metadata
output = GeoRasterData(
    data=array,
    metadata={"some": "dict"}  # ← Type unsafe!
)

# BAD: No validation
output = GeoRasterData(data=array)
# What if data has wrong shape? Wrong dtype?
```

### The `Sentinel2Scene` Contract

`Sentinel2Scene` is the **universal input** type. After calling `scene.read()`, it MUST have:

**Required attributes**:
- `scene_directory: Path`
- `bands: Dict[str, np.ndarray]` (at minimum: B02, B03, B04, B08)
- `transform: Affine`
- `crs: CRS`
- `sun_zenith: float`
- `sun_azimuth: float`

**Optional but recommended**:
- `view_zenith: float`
- `view_azimuth: float`
- `latitude: float`
- `longitude: float`

**Processors MUST validate** what they need:

```python
def process(self, scene: Sentinel2Scene) -> MyOutputData:
    # Validate requirements
    required_bands = ['B02', 'B04', 'B08']
    missing = [b for b in required_bands if b not in scene.bands]
    if missing:
        raise ValueError(f"Scene missing required bands: {missing}")

    if scene.sun_zenith is None:
        raise ValueError("Scene missing sun_zenith angle")

    # Proceed with processing...
```

---

## Processor Pattern

### The Standard Interface

Every processor MUST implement this pattern:

```python
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.data import Sentinel2Scene

class MyModuleConfig(BaseProcessorConfig):
    """Configuration for MyModule."""
    param1: float = Field(default=1.0, description="...")
    param2: int = Field(default=10, description="...")

class MyModuleProcessor:
    """Processes Sentinel-2 scenes to produce X."""

    def __init__(self, config: MyModuleConfig):
        """Initialize with configuration."""
        self.config = config
        # Initialize models, load weights, etc.

    def process(self, scene: Sentinel2Scene) -> MyOutputData:
        """
        Process a scene.

        Args:
            scene: Validated Sentinel2Scene object

        Returns:
            MyOutputData: Georeferenced output matching scene extent
        """
        # 1. Validate inputs
        self._validate_scene(scene)

        # 2. Run algorithm
        result_array = self._run_algorithm(scene)

        # 3. Package output
        output = MyOutputData(
            data=result_array,
            transform=scene.transform,
            crs=scene.crs,
            metadata=MyOutputMetadata(...)
        )

        return output

    def _validate_scene(self, scene: Sentinel2Scene):
        """Validate scene has required data."""
        # Raise ValueError if missing requirements
        pass

    def _run_algorithm(self, scene: Sentinel2Scene) -> np.ndarray:
        """Internal computation logic."""
        pass
```

### Interface Rules

#### ✅ Required:
1. **Typed inputs**: All inputs use `Union[DataType, str, Path]` for flexibility
2. **Single output type**: A subclass of `Data`
3. **Configuration required**: Constructor takes a config object
4. **Type hints**: All public methods fully typed
5. **Docstrings**: Google-style docstrings for class and `process()`

#### ❌ Prohibited:
1. **Multiple return types**: Don't return `Union[X, Y]` or `Optional[X]`
2. **Raw dictionaries**: Don't return `Dict[str, np.ndarray]` (use typed Data classes)
3. **Magic defaults**: Don't instantiate config internally (require explicit config)

### Multi-Input Processors

Fusion processors often need multiple data products. This is expected for algorithms that combine sources:

```python
def process(
    self,
    scene: Union[Sentinel2Scene, str, Path],
    heights: Union[CloudHeightGridData, str, Path],
    albedo: Optional[Union[AlbedoData, str, Path]] = None
) -> CloudPropertiesData:
    """
    Retrieve cloud properties from multiple inputs.

    Args:
        scene: Sentinel-2 scene (required)
        heights: Cloud top heights (required dependency)
        albedo: Surface albedo (optional, estimated if missing)
    """
```

**Maximum 3-4 inputs.** All must be typed. Document dependencies clearly.

---

## Configuration System

### The BaseProcessorConfig

All configs inherit from `BaseProcessorConfig`:

```python
from clouds_decoded.config import BaseProcessorConfig
from pydantic import Field

class MyConfig(BaseProcessorConfig):
    # Algorithm parameters
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection threshold [0-1]. Higher = more conservative."
    )

    # Resource parameters (inherited: output_dir, n_workers, debug_mode)
    # Add more as needed
```

### Configuration Guidelines

#### Field Definitions

```python
# ✅ GOOD: Clear description, validation, sensible default
resolution: int = Field(
    default=10,
    ge=10,
    le=60,
    description="Processing resolution in meters. Must match S2 native resolutions (10, 20, or 60)."
)

# ❌ BAD: No description, no validation
resolution: int = 10
```

#### Default Values Philosophy

**Essential parameters** (affect output correctness):
- ✅ Provide sensible, scientifically validated defaults
- ✅ Document why this is the default
- Example: `cloudy_thresh=1600` based on S2 NIR reflectance of typical clouds

**Optional parameters** (tuning, performance):
- ✅ Use `Optional[T] = None` to indicate "algorithm will auto-select"
- Example: `batch_size: Optional[int] = None  # Auto-detected from GPU memory`

**Required parameters** (no sensible default):
- ✅ Use `Field(..., description="...")` (ellipsis = required)
- Example: `model_path: str = Field(..., description="Path to trained model checkpoint")`

#### YAML Files

**Purpose**: Example configurations, not defaults.

Place in `docs/examples/`:

```yaml
# docs/examples/cloud_height_high_quality.yaml
# High-quality cloud height retrieval (slow but accurate)

stride: 100                    # Dense sampling (default: 300)
max_height: 20000              # Include very high clouds
correlation_weighting: true
spatial_smoothing_sigma: 150  # Fine detail
```

**Never** put `config.yaml` files inside module directories. Configs are data, not code.

---

## Import Conventions

### Absolute Imports (Preferred)

```python
# ✅ In any file, use absolute imports for cross-module references
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import BAND_RESOLUTIONS
from clouds_decoded.modules.cloud_height import CloudHeightProcessor
```

### Relative Imports (Within Module)

```python
# ✅ Within a module, use relative imports for internal files
# In modules/cloud_height/processor.py:
from .config import CloudHeightConfig
from .physics import heightsToOffsets
from .data import ColumnExtractor
```

### Import Organization

Standard order:

```python
# 1. Standard library
import os
from pathlib import Path
from typing import Optional, Dict

# 2. Third-party
import numpy as np
import rasterio as rio
from pydantic import Field

# 3. Clouds Decoded shared
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData
from clouds_decoded.config import BaseProcessorConfig
from clouds_decoded.constants import BAND_RESOLUTIONS

# 4. Relative imports (within module)
from .config import MyConfig
from .internal import my_helper_function
```

### The `clouds_decoded` Namespace

The package is named `clouds_decoded` (underscore, not hyphen).

**Directory structure**:
```
src/
├── cli/               → clouds_decoded.cli
├── shared_utils/      → clouds_decoded.data, clouds_decoded.config, etc.
└── modules/
    ├── cloud_height/  → clouds_decoded.modules.cloud_height
    └── cloud_mask/    → clouds_decoded.modules.cloud_mask
```

**Never** use hyphens in directory names. Python can't import `cloud-height`.

---

## Module Development Guide

### Creating a New Module

**Step 1**: Create directory structure

```bash
mkdir -p src/modules/my_module
cd src/modules/my_module
touch __init__.py config.py processor.py README.md
```

**Step 2**: Define data model (in `shared_utils/data/my_module.py`)

```python
from pydantic import Field
from .base import GeoRasterData, Metadata

class MyModuleMetadata(Metadata):
    """Metadata for MyModule outputs."""
    algorithm_version: str = "1.0"
    processing_params: Dict[str, Any] = Field(default_factory=dict)

class MyModuleData(GeoRasterData):
    """Output from MyModule."""
    metadata: MyModuleMetadata = Field(default_factory=MyModuleMetadata)
```

**Step 3**: Create config (`config.py`)

```python
from clouds_decoded.config import BaseProcessorConfig
from pydantic import Field

class MyModuleConfig(BaseProcessorConfig):
    """Configuration for MyModule."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection threshold"
    )
```

**Step 4**: Implement processor (`processor.py`)

```python
from clouds_decoded.data import Sentinel2Scene, MyModuleData
from .config import MyModuleConfig
import logging

logger = logging.getLogger(__name__)

class MyModuleProcessor:
    """Processes S2 scenes using MyAlgorithm."""

    def __init__(self, config: MyModuleConfig):
        self.config = config
        logger.info(f"Initialized MyModuleProcessor with threshold={config.threshold}")

    def process(self, scene: Sentinel2Scene) -> MyModuleData:
        """Process a scene."""
        # Implementation...
        pass
```

**Step 5**: Export in `__init__.py`

```python
from .processor import MyModuleProcessor
from .config import MyModuleConfig

__all__ = ['MyModuleProcessor', 'MyModuleConfig']
```

**Step 6**: Register in shared exports (`shared_utils/processors.py`)

```python
try:
    from clouds_decoded.modules.my_module import MyModuleProcessor, MyModuleConfig
except ImportError:
    logger.debug("MyModule not available")
    MyModuleProcessor = None
    MyModuleConfig = None
```

**Step 7**: Add CLI command (`cli/entry.py`)

```python
@app.command()
def my_module(
    scene_path: str = typer.Argument(..., help="Path to S2 .SAFE"),
    output_path: str = typer.Option("output.tif", help="Output path"),
    threshold: float = typer.Option(0.5, help="Detection threshold"),
):
    """Run MyModule processing."""
    scene = Sentinel2Scene()
    scene.read(scene_path)

    config = MyModuleConfig(threshold=threshold)
    processor = MyModuleProcessor(config)
    result = processor.process(scene)

    result.write(output_path)
    logger.info(f"Output saved to {output_path}")
```

---

## CLI Design

### Principles

1. **Progressive Disclosure**: Simple commands should be simple
2. **Composability**: Users can chain commands via files
3. **Reproducibility**: All parameters logged and saveable

### Command Structure

```
clouds-decoded <command> <scene> [options]
```

**Standard pattern**:

```python
@app.command()
def my_command(
    # 1. Required positional: input path
    scene_path: str = typer.Argument(..., help="Path to Sentinel-2 .SAFE directory"),

    # 2. Common options: output path, config
    output_path: str = typer.Option("output.tif", help="Output path"),
    config_path: Optional[str] = typer.Option(None, help="Path to config YAML (optional)"),

    # 3. Key algorithm parameters (3-5 max)
    param1: float = typer.Option(default_value, help="..."),
    param2: int = typer.Option(default_value, help="..."),
):
    """
    Brief description of what this command does.

    More detailed explanation can go here.
    """
    # Implementation
```

### CLI Rules

#### ✅ DO:

- Load scene **once** at the start
- Use the `run_X()` helper functions for logic
- Log key steps at INFO level
- Return early on errors with clear messages

```python
@app.command()
def process(scene_path: str, output_path: str):
    """Process a scene."""
    logger.info(f"Loading scene: {scene_path}")

    try:
        scene = Sentinel2Scene()
        scene.read(scene_path)
    except Exception as e:
        logger.error(f"Failed to load scene: {e}")
        raise typer.Exit(1)

    run_my_processing(scene, output_path)
```

#### ❌ DON'T:

- Put algorithm logic in CLI commands
- Duplicate scene loading across commands
- Use `print()` instead of `logger`
- Catch all exceptions silently

### Workflow Commands

For multi-step pipelines, create a workflow command:

```python
@app.command()
def full_pipeline(
    scene_path: str,
    model_path: str,
    output_dir: str = "outputs",
):
    """
    Run complete pipeline: Mask → Height → Properties.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load scene once
    scene = Sentinel2Scene()
    scene.read(scene_path)

    # Step 1: Mask
    logger.info("Step 1/3: Cloud Masking")
    mask = run_cloud_mask(scene, output_path=out_dir / "mask.tif")

    # Step 2: Height
    logger.info("Step 2/3: Cloud Height")
    heights = run_cloud_height(scene, output_path=out_dir / "heights.tif")

    # Step 3: Properties
    logger.info("Step 3/3: Cloud Properties")
    props = run_cloud_properties(scene, heights, model_path, output_path=out_dir / "properties.nc")

    logger.info(f"Pipeline complete. Outputs in {output_dir}/")
```

---

## Testing Standards

### Test Organization

Mirror the `src/` structure:

```
tests/
├── conftest.py              # Shared fixtures
├── test_data/               # Sample data for testing
│   └── small_scene.SAFE/
├── shared_utils/
│   ├── test_data_models.py
│   └── test_config.py
└── modules/
    ├── cloud_height/
    │   ├── test_config.py
    │   ├── test_processor.py
    │   └── test_physics.py
    └── cloud_mask/
        └── test_processor.py
```

### Test Fixtures

Create reusable fixtures in `conftest.py`:

```python
import pytest
from pathlib import Path
from clouds_decoded.data import Sentinel2Scene

@pytest.fixture
def sample_scene(tmp_path):
    """Load a small test scene."""
    scene = Sentinel2Scene()
    scene.read("tests/test_data/small_scene.SAFE")
    return scene

@pytest.fixture
def temp_output(tmp_path):
    """Provide temporary output path."""
    return tmp_path / "output.tif"
```

### Test Categories

**Unit Tests**: Test individual functions in isolation

```python
def test_heights_to_offsets():
    from clouds_decoded.modules.cloud_height.physics import heightsToOffsets

    heights = [1000, 5000, 10000]  # meters
    bands = ['B02', 'B03', 'B04']
    pixel_size = 10  # meters

    offsets = heightsToOffsets(heights, bands, pixel_size)

    assert len(offsets) == 3
    assert all(offsets >= 0)  # Offsets should be positive
```

**Integration Tests**: Test processor end-to-end

```python
def test_cloud_height_processor(sample_scene, temp_output):
    from clouds_decoded.modules.cloud_height import CloudHeightProcessor, CloudHeightConfig

    config = CloudHeightConfig(
        stride=500,  # Fast for testing
        max_height=5000
    )

    processor = CloudHeightProcessor(config)
    result = processor.process(sample_scene)

    # Validate output
    assert result.data is not None
    assert result.data.ndim == 2
    assert result.transform is not None
    assert result.crs is not None

    # Test I/O
    result.write(temp_output)
    assert temp_output.exists()
```

**Property Tests**: Validate data model contracts

```python
def test_cloud_height_data_validation():
    from clouds_decoded.data import CloudHeightGridData
    import numpy as np

    # Valid data
    valid_heights = CloudHeightGridData(data=np.array([[100, 200], [300, 400]]))
    assert valid_heights.validate()

    # Invalid data (negative heights)
    invalid_heights = CloudHeightGridData(data=np.array([[100, -50], [300, 400]]))
    assert not invalid_heights.validate()
```

### Coverage Goals

- **Shared utilities**: 90%+ coverage
- **Processor core logic**: 80%+ coverage
- **CLI commands**: 60%+ coverage (integration tests)

---

## Common Pitfalls

### ❌ Pitfall 1: Raw Dict Metadata

```python
# BAD
output = CloudMaskData(
    data=mask,
    metadata={"method": "threshold", "value": 1600}
)

# GOOD
from clouds_decoded.data import CloudMaskMetadata
meta = CloudMaskMetadata(method="threshold", threshold_value=1600)
output = CloudMaskData(data=mask, metadata=meta)
```

### ❌ Pitfall 2: Implicit Config Creation

```python
# BAD: Processor creates its own config
class MyProcessor:
    def __init__(self):
        self.config = MyConfig()  # User can't customize!

# GOOD: Config is explicit
class MyProcessor:
    def __init__(self, config: MyConfig):
        self.config = config
```

### ❌ Pitfall 3: Multiple Return Types

```python
# BAD: Return type varies
def process(self, scene) -> Union[MyData, None]:
    if something:
        return None
    return result

# GOOD: Always return typed object, use metadata for status
def process(self, scene) -> MyData:
    result = MyData()
    if not something:
        result.data = None
        result.metadata.status = "failed"
    else:
        result.data = computed_data
        result.metadata.status = "success"
    return result
```

### ❌ Pitfall 4: Hardcoded Paths

```python
# BAD
def save_debug_output(self):
    np.save("/tmp/debug.npy", self.data)

# GOOD
def save_debug_output(self, path: Path):
    if self.config.debug_mode:
        np.save(path, self.data)
        logger.debug(f"Debug output saved to {path}")
```

### ❌ Pitfall 5: Silent Failures

```python
# BAD
try:
    result = compute_something()
except Exception:
    result = None  # User has no idea what went wrong

# GOOD
try:
    result = compute_something()
except ValueError as e:
    raise ValueError(f"Computation failed: {e}. Check input data quality.") from e
except Exception as e:
    logger.error(f"Unexpected error in computation: {e}")
    raise
```

---

## Development Workflow

### 1. Before You Code

- [ ] Read this document
- [ ] Understand the Processor Pattern
- [ ] Check if a similar module exists (copy its structure)
- [ ] Define your data model **first**

### 2. During Development

- [ ] Write docstrings as you go
- [ ] Add type hints to all functions
- [ ] Use `logger` not `print()`
- [ ] Create config parameters for all "magic numbers"

### 3. Before You Commit

- [ ] Run tests: `pytest tests/`
- [ ] Check types: `mypy src/`
- [ ] Format code: `black src/`
- [ ] Remove debug prints
- [ ] Update docstrings if interface changed

### 4. Code Review Checklist

Reviewer should verify:

- [ ] Follows Processor Pattern
- [ ] Uses typed Data models (no raw dicts)
- [ ] Config inherits from BaseProcessorConfig
- [ ] All public methods have docstrings
- [ ] No hardcoded paths or constants
- [ ] Tests cover main functionality
- [ ] No `try/except` silencing errors

---

## Example: Complete Module (Reference Implementation)

See `modules/cloud_mask/` as the **reference implementation**:

- ✅ Clean config with BaseProcessorConfig inheritance
- ✅ Multiple processor classes (ThresholdCloudMaskProcessor, CloudMaskProcessor)
- ✅ Separate postprocessing with PostProcessParams
- ✅ Proper metadata usage
- ✅ Type hints throughout
- ✅ Good logging

Use this as a template when creating new modules.

---

## Questions?

**For architecture decisions**: Open an issue tagged `architecture`
**For implementation help**: Check the reference implementation (`cloud_mask`)
**For data model questions**: See `shared_utils/data/base.py` docstrings

---

**Remember**: When in doubt, favor explicitness over cleverness. This codebase prioritizes clarity and reproducibility over brevity.
