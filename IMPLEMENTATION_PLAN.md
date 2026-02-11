# Implementation Plan: Codebase Refactoring

**Based on**: CODEBASE_ANALYSIS.md
**Target**: ARCHITECTURE.md compliance
**Timeline**: Phased approach, critical path first

---

## Overview

This plan addresses the issues identified in the codebase analysis and brings the project into alignment with the architectural principles defined in ARCHITECTURE.md.

**Strategy**: Fix foundational issues first (imports, naming), then enforce patterns, then polish.

---

## Phase 0: Pre-Flight Checks ✈️

**Duration**: 30 minutes
**Risk**: Low
**Goal**: Ensure we can safely refactor

### Tasks

- [ ] **Create feature branch**: `refactor/architecture-alignment`
- [ ] **Commit current state**: Tag as `pre-refactor-baseline`
- [ ] **Run existing tests** (if any): Document what passes/fails
- [ ] **Create test data**: Small sample scene for integration tests
- [ ] **Backup critical files**: config.yaml files, model checkpoints

### Verification

```bash
git checkout -b refactor/architecture-alignment
git tag pre-refactor-baseline
pytest tests/ || echo "Tests status documented"
```

---

## Phase 1: Critical Path - Module Naming & Imports 🚨

**Duration**: 2 hours
**Risk**: High (breaks all imports)
**Goal**: Fix module naming and establish import sanity

### 1.1 Rename Module Directories

**Current → Target**:

```
modules/cloud-height/      → modules/cloud_height/
modules/albedo-estimator/  → modules/albedo_estimator/
modules/cloud-mask/        → modules/cloud_mask/
modules/refl2prop/         → modules/refl2prop/  (OK)
```

**Commands**:

```bash
cd src/modules
git mv cloud-height cloud_height
git mv albedo-estimator albedo_estimator
git mv cloud-mask cloud_mask
```

**Files to update immediately after rename**:
- `cli/entry.py` - Update all import paths
- `shared_utils/processors.py` - Update import paths
- All `__init__.py` files in modules

### 1.2 Fix Import Paths Throughout

**Search and replace** (confirm each):

```bash
# In all Python files
cloud-height → cloud_height
albedo-estimator → albedo_estimator
cloud-mask → cloud_mask
```

**Verify**:

```bash
# No more try/except for local modules
grep -r "except ImportError" src/shared_utils/processors.py
# Should only remain for optional dependencies (torch, senseiv2)
```

### 1.3 Standardize Import Strategy

**Create**: `src/clouds_decoded/__init__.py` (if missing)

```python
"""Clouds Decoded: Sentinel-2 Cloud Property Retrieval."""
__version__ = "0.1.0"

# Expose key types at package level
from clouds_decoded.data import (
    Sentinel2Scene,
    CloudHeightGridData,
    CloudMaskData,
    CloudPropertiesData,
    AlbedoData,
)

__all__ = [
    "Sentinel2Scene",
    "CloudHeightGridData",
    "CloudMaskData",
    "CloudPropertiesData",
    "AlbedoData",
]
```

**Update all modules** to use absolute imports:

**Before** (in `cloud_height/processor.py`):
```python
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata
from .data import ColumnExtractor, ColumnIterator, RetrievalCube
from .physics import heightsToOffsets
from .config import CloudHeightConfig
```

**After**: Same (this is correct!)

**Before** (in `cli/entry.py`):
```python
from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
```

**After**: Same (correct)

### 1.4 Verification Tests

Create `tests/test_imports.py`:

```python
def test_core_imports():
    """Verify all core imports work."""
    from clouds_decoded.data import Sentinel2Scene
    from clouds_decoded.config import BaseProcessorConfig
    from clouds_decoded.constants import BAND_RESOLUTIONS

def test_module_imports():
    """Verify all module processors can be imported."""
    from clouds_decoded.modules.cloud_height import CloudHeightProcessor
    from clouds_decoded.modules.cloud_mask import CloudMaskProcessor
    from clouds_decoded.modules.refl2prop import CloudPropertyInverter
    from clouds_decoded.modules.albedo_estimator import AlbedoEstimator

def test_cli_imports():
    """Verify CLI can import."""
    from clouds_decoded.cli.entry import app
```

**Run**:
```bash
pytest tests/test_imports.py -v
```

All must pass before proceeding to Phase 2.

---

## Phase 2: Constants & Configuration Unification 📋

**Duration**: 2 hours
**Risk**: Medium
**Goal**: Single source of truth for constants and config loading

### 2.1 Eliminate Duplicate Constants

**Delete**: `src/modules/cloud_height/constants.py`

**Update** `src/modules/cloud_height/processor.py`, `physics.py`, etc.:

```python
# Change:
from .constants import BAND_TIME_DELAYS, ORBITAL_VELOCITY

# To:
from clouds_decoded.constants import BAND_TIME_DELAYS, ORBITAL_VELOCITY
```

**Verify no duplicates remain**:

```bash
grep -r "BAND_TIME_DELAYS = {" src/
# Should only appear in shared_utils/constants.py
```

### 2.2 Standardize Config Loading

**Remove branching logic** from `cli/entry.py`:

**Before** (lines 36-46):
```python
if hasattr(CloudHeightConfig, 'from_yaml'):
    config = CloudHeightConfig.from_yaml(config_path)
elif hasattr(CloudHeightConfig, 'load_yaml'):
    # ...
else:
    config = CloudHeightConfig()
```

**After**:
```python
if config_path:
    config = CloudHeightConfig.from_yaml(config_path)
else:
    config = CloudHeightConfig()
```

**Ensure** all config classes have `from_yaml` (from BaseProcessorConfig):

```python
# Verify each config inherits properly
assert hasattr(CloudHeightConfig, 'from_yaml')
assert hasattr(CloudMaskConfig, 'from_yaml')
assert hasattr(Refl2PropConfig, 'from_yaml')
```

### 2.3 Move Example Configs

**Create**: `docs/examples/` directory

**Move**:
```bash
mkdir -p docs/examples
git mv src/modules/cloud_height/config.yaml docs/examples/cloud_height_default.yaml
git mv src/modules/cloud_mask/config.yaml docs/examples/cloud_mask_default.yaml 2>/dev/null || true
git mv src/modules/refl2prop/config.yaml docs/examples/refl2prop_default.yaml
```

**Update** each YAML file with a header comment:

```yaml
# Cloud Height Configuration Example
# Copy and modify this file to customize processing parameters
# Usage: clouds-decoded cloud-height scene.SAFE --config my_config.yaml

stride: 300
max_height: 18000
# ... rest of config
```

---

## Phase 3: Enforce Processor Pattern 🔧

**Duration**: 3 hours
**Risk**: Medium
**Goal**: All processors follow standard interface

### 3.1 Fix AlbedoEstimator

**Current issues**:
1. Returns `Dict[str, AlbedoData]` instead of single object
2. Has `percentile` parameter instead of config
3. Contains unreachable code (lines 75-119)

**Strategy**: Two options

#### Option A: Keep Dict Return (Break Pattern)
**Justification**: Albedo inherently has one value per band, dict is clearer
**Action**: Document this as an exception in ARCHITECTURE.md

#### Option B: Conform to Pattern (Recommended)
**Justification**: Consistency matters more than minor convenience

**Refactored** `albedo_estimator/config.py`:

```python
from clouds_decoded.config import BaseProcessorConfig
from pydantic import Field

class AlbedoEstimatorConfig(BaseProcessorConfig):
    """Configuration for surface albedo estimation."""

    percentile: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Percentile for dark object subtraction [0-100]"
    )

    method: str = Field(
        default="percentile",
        description="Estimation method (currently only 'percentile')"
    )
```

**Refactored** `albedo_estimator/processor.py`:

```python
from clouds_decoded.data import Sentinel2Scene, GeoRasterData, Metadata
from .config import AlbedoEstimatorConfig
import numpy as np

class AlbedoData(GeoRasterData):
    """Multi-band albedo raster (one band per S2 band)."""
    pass  # Inherits from GeoRasterData

class AlbedoEstimator:
    """Estimates surface albedo from Sentinel-2 scenes."""

    def __init__(self, config: AlbedoEstimatorConfig):
        self.config = config

    def process(self, scene: Sentinel2Scene) -> AlbedoData:
        """
        Estimate surface albedo for all bands in scene.

        Returns:
            AlbedoData with shape (n_bands, height, width)
        """
        # Get list of bands
        band_names = list(scene.bands.keys())
        n_bands = len(band_names)

        # Get reference shape from B02
        ref_shape = scene.bands['B02'].shape
        h, w = ref_shape if len(ref_shape) == 2 else ref_shape[1:]

        # Initialize output array
        albedo_array = np.zeros((n_bands, h, w), dtype=np.float32)

        # Compute percentile for each band
        for idx, band_name in enumerate(band_names):
            band_data = scene.bands[band_name]
            if band_data.ndim == 3:
                band_data = band_data[0]

            # Resize to reference shape if needed
            if band_data.shape != (h, w):
                from skimage.transform import resize
                band_data = resize(band_data, (h, w), order=1, preserve_range=True)

            # Calculate percentile
            valid_data = band_data[np.isfinite(band_data)]
            if valid_data.size > 0:
                albedo_value = float(np.percentile(valid_data, self.config.percentile))
            else:
                albedo_value = 0.05  # Default fallback

            # Fill band with constant value
            albedo_array[idx, :, :] = albedo_value

        # Package output
        output = AlbedoData(
            data=albedo_array,
            transform=scene.transform,
            crs=scene.crs,
            metadata=Metadata(
                method=self.config.method,
                percentile=self.config.percentile
            )
        )

        return output
```

**Delete** lines 75-119 (unreachable code)

**Update** callers (refl2prop/processor.py):

```python
# Before:
albedo_dict = self.albedo_estimator.process(scene)
for b in required_bands:
    if b in albedo_dict and albedo_dict[b].data is not None:
        alb_arr = albedo_dict[b].data

# After:
albedo_data = self.albedo_estimator.process(scene)  # Now returns AlbedoData, not dict
# albedo_data.data has shape (n_bands, h, w)
# Need to map band names to indices
band_names = list(scene.bands.keys())
for b in required_bands:
    band_idx = band_names.index(b)
    alb_arr = albedo_data.data[band_idx]
```

### 3.2 Standardize Metadata Usage

**Create proper metadata classes** for all processors:

Already exist:
- ✅ CloudHeightMetadata
- ✅ CloudMaskMetadata
- ✅ CloudPropertiesMetadata

**Fix usages**:

`cloud_mask/processor.py` line 52:
```python
# Before:
metadata={"method": "simple_threshold", ...}

# After:
from clouds_decoded.data import CloudMaskMetadata
metadata = CloudMaskMetadata(
    method="simple_threshold",
    threshold_band=threshold_band,
    threshold_value=threshold_value
)
```

### 3.3 Standardize Post-Processing

**Decision**: Make post-processing **explicit** (user calls it if needed)

**Update** CloudHeightProcessor:

```python
# Current: postprocess() called automatically in process()
# Change: Make it a separate method users can call

def process(self, scene: Sentinel2Scene, apply_smoothing: bool = True) -> CloudHeightGridData:
    """
    Process scene to retrieve cloud heights.

    Args:
        scene: Input scene
        apply_smoothing: If True, applies spatial smoothing (default: True)
    """
    # ... correlation logic ...

    if apply_smoothing:
        return self.smooth(raw_result)
    else:
        return raw_result

def smooth(self, height_data: CloudHeightGridData) -> CloudHeightGridData:
    """Apply spatial smoothing to height retrievals."""
    # Current postprocess() logic
```

**Keep** CloudMaskProcessor.postprocess() as-is (good pattern)

---

## Phase 4: Data Model Compliance 📊

**Duration**: 1 hour
**Risk**: Low
**Goal**: Enforce validation and metadata typing

### 4.1 Add Validation Calls

**Update base class** `shared_utils/data/base.py`:

```python
class GeoRasterData(Data):
    def write(self, filepath: Union[str, Path]):
        """Write data to file."""
        if self.data is None:
            raise ValueError("No data to write")

        # Validate before writing
        if hasattr(self, 'validate') and not self.validate():
            logger.warning(f"Data validation failed for {type(self).__name__}")

        # ... existing write logic
```

### 4.2 Improve Validation Methods

**CloudMaskData**:

```python
def validate(self) -> bool:
    """Validate mask data."""
    if self.data is None:
        return True

    if self.metadata.categorical:
        # Check that all values are in classes
        unique_vals = np.unique(self.data[~np.isnan(self.data)])
        valid_classes = set(self.metadata.classes.keys())
        invalid = set(unique_vals) - valid_classes

        if invalid:
            logger.error(f"Mask contains invalid classes: {invalid}")
            return False

    return True
```

### 4.3 Implement CloudMaskData.transform()

**Add to** `shared_utils/data/cloud_mask.py`:

```python
def transform(
    self,
    positive_classes: Optional[List[int]] = None,
    confidence_threshold: float = 0.5,
    dilation_meters: float = 0.0,
    output_resolution: Optional[int] = None
) -> 'CloudMaskData':
    """Transform mask for downstream use."""

    if positive_classes is None:
        return self  # No transformation

    # Handle both binary and categorical
    if self.metadata.categorical:
        # Multi-class: extract specified classes
        binary = np.isin(self.data, positive_classes).astype(np.uint8)
    else:
        # Already binary, check if we need to invert
        if 0 in positive_classes and 1 not in positive_classes:
            binary = (self.data == 0).astype(np.uint8)
        else:
            binary = self.data.astype(np.uint8)

    # Apply dilation if requested
    if dilation_meters > 0:
        from skimage.morphology import disk, dilation
        res = abs(self.transform[0])
        pixels = int(dilation_meters / res)
        if pixels > 0:
            binary = dilation(binary, disk(pixels))

    # Resample if requested
    # ... implementation ...

    return CloudMaskData(
        data=binary,
        transform=self.transform,
        crs=self.crs,
        metadata=CloudMaskMetadata(
            categorical=True,
            classes={0: "Negative", 1: "Positive"},
            method=f"transformed_from_{self.metadata.method}"
        )
    )
```

### 4.4 Add Scene Validation Helper

**Create** `shared_utils/data/validation.py`:

```python
"""Validation utilities for data models."""
from typing import List
from .sentinel import Sentinel2Scene

def validate_scene_for_processing(
    scene: Sentinel2Scene,
    required_bands: List[str],
    require_angles: bool = True
) -> None:
    """
    Validate that scene has all required attributes for processing.

    Args:
        scene: Scene to validate
        required_bands: List of band names (e.g. ['B02', 'B04'])
        require_angles: If True, sun/view angles must be present

    Raises:
        ValueError: If scene is missing required data
    """
    # Check bands loaded
    if not scene.bands:
        raise ValueError("Scene has no bands loaded. Call scene.read() first.")

    missing_bands = [b for b in required_bands if b not in scene.bands]
    if missing_bands:
        raise ValueError(f"Scene missing required bands: {missing_bands}")

    # Check georeferencing
    if scene.transform is None:
        raise ValueError("Scene missing geospatial transform")

    if scene.crs is None:
        raise ValueError("Scene missing CRS")

    # Check angles
    if require_angles:
        if scene.sun_zenith is None:
            raise ValueError("Scene missing sun_zenith angle")

        if scene.sun_azimuth is None:
            raise ValueError("Scene missing sun_azimuth angle")
```

**Use in processors**:

```python
def process(self, scene: Sentinel2Scene) -> CloudHeightGridData:
    """Process scene."""
    from clouds_decoded.data.validation import validate_scene_for_processing

    validate_scene_for_processing(
        scene,
        required_bands=self.config.bands,
        require_angles=False  # Cloud height doesn't need sun angles
    )

    # Proceed with processing...
```

---

## Phase 5: Configuration Polish ⚙️

**Duration**: 1.5 hours
**Risk**: Low
**Goal**: Better defaults, validation, documentation

### 5.1 Add Field Validation

**Example** in `cloud_height/config.py`:

```python
from pydantic import Field, field_validator

class CloudHeightConfig(BaseProcessorConfig):
    stride: int = Field(
        default=300,
        ge=10,  # Minimum 10m
        le=1000,  # Maximum 1km
        description="Stride between retrieval points in meters"
    )

    max_height: int = Field(
        default=18000,
        ge=0,
        le=30000,
        description="Maximum cloud height to search (meters)"
    )

    @field_validator('stride')
    @classmethod
    def stride_must_be_multiple_of_resolution(cls, v, values):
        """Stride should be divisible by processing resolution."""
        if 'along_track_resolution' in values.data:
            res = values.data['along_track_resolution']
            if v % res != 0:
                raise ValueError(f"stride ({v}) must be multiple of along_track_resolution ({res})")
        return v
```

### 5.2 Improve Descriptions

**Review all Field descriptions**:

```python
# Bad:
threshold: float = 0.5

# Better:
threshold: float = Field(default=0.5, description="Threshold value")

# Best:
threshold: float = Field(
    default=0.5,
    ge=0.0,
    le=1.0,
    description=(
        "Detection threshold [0-1]. Lower values = more sensitive detection "
        "(more false positives). Recommended: 0.5 for balanced results."
    )
)
```

### 5.3 Create Config Documentation

**Generate** from code:

```python
# scripts/generate_config_docs.py
from clouds_decoded.modules.cloud_height import CloudHeightConfig

config = CloudHeightConfig()
print("# Cloud Height Configuration\n")

for field_name, field_info in config.model_fields.items():
    print(f"## `{field_name}`")
    print(f"- **Type**: {field_info.annotation}")
    print(f"- **Default**: {field_info.default}")
    print(f"- **Description**: {field_info.description}")
    print()
```

**Run** for all modules, save to `docs/configuration.md`

---

## Phase 6: Clean Up & Polish 🧹

**Duration**: 2 hours
**Risk**: Low
**Goal**: Remove dead code, add logging, improve UX

### 6.1 Remove Dead Code

**Files to clean**:

1. **albedo_estimator/processor.py** lines 75-119 (done in Phase 3)

2. **cloud_height/producer.py** and **worker.py**:
   - Either integrate into CLI or move to `scripts/distributed/`
   - Document that they're for cluster processing

3. **refl2prop/example.py**:
   - Move to `examples/refl2prop_inference.py`
   - Update paths to be relative/configurable

### 6.2 Remove Debug Code

**Search for**:

```bash
grep -n "print(" src/**/*.py | grep -v "# OK"
grep -n "pprint" src/**/*.py
grep -n "import pdb" src/**/*.py
```

**Remove**:
- refl2prop/processor.py lines 284-292 (pprint, debug prints)

**Replace with logging**:

```python
# Before:
print(f"Input Feature {i}: min={np.nanmin(col_data)}, max={np.nanmax(col_data)}")

# After:
logger.debug(f"Input Feature {i}: min={np.nanmin(col_data):.4f}, max={np.nanmax(col_data):.4f}")
```

### 6.3 Standardize Logging

**Pattern**:

```python
import logging

logger = logging.getLogger(__name__)

class MyProcessor:
    def __init__(self, config):
        logger.info(f"Initialized {self.__class__.__name__}")
        logger.debug(f"Config: {config.model_dump()}")

    def process(self, scene):
        logger.info(f"Processing scene: {scene.scene_directory}")
        # ... processing ...
        logger.info("Processing complete")
```

**Levels**:
- `DEBUG`: Detailed info for debugging (parameter values, intermediate results)
- `INFO`: Key progress steps (Loading, Processing, Saving)
- `WARNING`: Non-fatal issues (Missing optional data, using defaults)
- `ERROR`: Errors that stop processing

### 6.4 Improve CLI Help Text

**Review all commands**:

```python
@app.command()
def cloud_height(
    scene_path: str = typer.Argument(
        ...,
        help="Path to Sentinel-2 .SAFE directory (e.g., S2A_MSIL1C_*.SAFE)"
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML config file (see docs/examples/cloud_height_default.yaml)"
    ),
    output_path: str = typer.Option(
        "height_output.tif",
        "--output", "-o",
        help="Output file path (.tif or .nc)"
    ),
):
    """
    Retrieve cloud top heights using stereo parallax method.

    Computes cloud heights by correlating spectral bands acquired at different
    times during the Sentinel-2 push-broom scan. Higher correlation at a given
    height offset indicates clouds at that altitude.

    Example:
        clouds-decoded cloud-height scene.SAFE --output heights.tif

    For advanced configuration:
        clouds-decoded cloud-height scene.SAFE --config my_config.yaml
    """
```

---

## Phase 7: Testing & Documentation 📚

**Duration**: 2 hours
**Risk**: Low
**Goal**: Ensure changes work, document for users

### 7.1 Create Integration Tests

**File**: `tests/test_integration.py`

```python
import pytest
from pathlib import Path
from clouds_decoded.data import Sentinel2Scene
from clouds_decoded.modules.cloud_height import CloudHeightProcessor, CloudHeightConfig
from clouds_decoded.modules.cloud_mask import CloudMaskProcessor, CloudMaskConfig

@pytest.fixture
def sample_scene():
    """Load a small test scene."""
    scene = Sentinel2Scene()
    scene.read("tests/test_data/small_scene.SAFE")
    return scene

def test_end_to_end_cloud_height(sample_scene, tmp_path):
    """Test complete cloud height workflow."""
    # Configure
    config = CloudHeightConfig(stride=500, max_height=5000)

    # Process
    processor = CloudHeightProcessor(config)
    result = processor.process(sample_scene)

    # Validate
    assert result.data is not None
    assert result.transform is not None
    assert result.metadata.processing_config is not None

    # I/O
    output_path = tmp_path / "heights.tif"
    result.write(output_path)
    assert output_path.exists()

def test_end_to_end_cloud_mask(sample_scene, tmp_path):
    """Test cloud mask workflow."""
    config = CloudMaskConfig(method="threshold", threshold_value=1600)
    processor = ThresholdCloudMaskProcessor(config)
    result = processor.process(sample_scene)

    assert result.data is not None
    assert result.validate()

    output_path = tmp_path / "mask.tif"
    result.write(output_path)
    assert output_path.exists()
```

### 7.2 Test CLI Commands

**File**: `tests/test_cli.py`

```python
from typer.testing import CliRunner
from clouds_decoded.cli.entry import app

runner = CliRunner()

def test_cli_help():
    """Test that help text works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Clouds Decoded" in result.stdout

def test_cloud_height_command_help():
    """Test cloud-height help."""
    result = runner.invoke(app, ["cloud-height", "--help"])
    assert result.exit_code == 0
    assert "stereo parallax" in result.stdout.lower()
```

### 7.3 Update README.md

**Add sections**:

```markdown
## Installation

```bash
pip install -e .
```

## Quick Start

### Cloud Height Retrieval

```bash
clouds-decoded cloud-height path/to/scene.SAFE --output heights.tif
```

### Cloud Masking

```bash
# Deep learning method (requires SEnSeIv2)
clouds-decoded cloud-mask path/to/scene.SAFE --method senseiv2

# Simple threshold
clouds-decoded cloud-mask path/to/scene.SAFE --method threshold --threshold-value 1600
```

### Full Pipeline

```bash
clouds-decoded workflow path/to/scene.SAFE --model-path model.pth --output-dir results/
```

## Configuration

All processors can be configured via YAML files. See `docs/examples/` for templates.

```bash
clouds-decoded cloud-height scene.SAFE --config my_config.yaml
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for design principles and developer guide.
```

### 7.4 Create Changelog

**File**: `CHANGELOG.md`

```markdown
# Changelog

## [0.2.0] - 2026-02-04 - Architecture Refactor

### Breaking Changes
- Renamed modules from hyphenated to underscored (cloud-height → cloud_height)
- AlbedoEstimator.process() now returns AlbedoData instead of Dict[str, AlbedoData]
- All configs must now inherit from BaseProcessorConfig
- Removed standalone module config.yaml files (moved to docs/examples/)

### Added
- Comprehensive ARCHITECTURE.md documenting design principles
- Validation helpers for Sentinel2Scene
- Proper metadata classes for all outputs
- Field validation on all configuration parameters
- Integration test suite

### Fixed
- Import reliability (removed all try/except for local modules)
- Constants duplication (single source of truth in shared_utils)
- Config loading consistency (all use from_yaml())
- Dead code removed from albedo_estimator

### Changed
- Standardized logging across all modules
- Improved CLI help text and examples
- Post-processing made explicit in CloudHeightProcessor
- Debug output removed from production code
```

---

## Phase 8: Final Verification ✅

**Duration**: 1 hour
**Risk**: Low
**Goal**: Confirm everything works

### 8.1 Run Full Test Suite

```bash
pytest tests/ -v --cov=src --cov-report=html
```

**Target coverage**: >70% overall

### 8.2 Manual Testing

**Test each command**:

```bash
# Cloud height
clouds-decoded cloud-height tests/test_data/small_scene.SAFE --output /tmp/height.tif

# Cloud mask (threshold)
clouds-decoded cloud-mask tests/test_data/small_scene.SAFE --method threshold --output /tmp/mask.tif

# Cloud mask (SEnSeIv2, if available)
clouds-decoded cloud-mask tests/test_data/small_scene.SAFE --method senseiv2 --output /tmp/mask_ml.tif

# Workflow (if model available)
clouds-decoded workflow tests/test_data/small_scene.SAFE --model-path model.pth --output-dir /tmp/results
```

### 8.3 Type Checking

```bash
mypy src/ --ignore-missing-imports
```

Fix critical issues, document known typing limitations.

### 8.4 Code Formatting

```bash
black src/ tests/
```

---

## Rollout Strategy

### Step 1: Merge to Main

```bash
git add -A
git commit -m "refactor: align codebase with architecture standards

- Rename modules (hyphens → underscores)
- Standardize processor pattern
- Enforce strict data models
- Add validation and documentation
- Remove dead code and debug output

See IMPLEMENTATION_PLAN.md for details.
"

git push origin refactor/architecture-alignment
```

**Create PR**, get review, merge.

### Step 2: Tag Release

```bash
git tag v0.2.0 -m "Architecture refactor release"
git push --tags
```

### Step 3: Update Documentation Sites

If you have external docs (ReadTheDocs, GitHub Pages):
- Regenerate from new docstrings
- Update examples with new import paths

---

## Rollback Plan

If critical issues emerge post-merge:

```bash
# Revert to baseline
git checkout pre-refactor-baseline

# Create hotfix branch
git checkout -b hotfix/post-refactor

# Fix issues, then merge
```

**OR** if issues are pervasive:

```bash
# Nuclear option: revert the merge
git revert -m 1 <merge-commit-sha>
```

---

## Success Criteria

### Must Have ✅
- [ ] All modules use underscore naming
- [ ] No try/except blocks for local imports
- [ ] All constants defined in single location
- [ ] All processors follow standard pattern
- [ ] All configs inherit from BaseProcessorConfig
- [ ] No raw dicts for metadata
- [ ] All tests pass
- [ ] CLI commands work for basic use cases

### Should Have 🎯
- [ ] >70% test coverage
- [ ] All Field parameters have descriptions
- [ ] Validation on all configs
- [ ] Scene validation helper used in processors
- [ ] Example configs in docs/examples/
- [ ] Updated README with examples

### Nice to Have 🌟
- [ ] Type checking passes (mypy)
- [ ] Sphinx documentation generated
- [ ] GitHub Actions CI configured
- [ ] Performance benchmarks documented

---

## Risk Mitigation

### High Risk: Import Breakage

**Mitigation**:
- Rename modules in single commit
- Immediately update all import paths
- Run import test after each change
- Keep old tag for quick rollback

### Medium Risk: Breaking User Code

**Mitigation**:
- Document all breaking changes in CHANGELOG
- Provide migration guide
- Keep old examples in docs/legacy/

### Low Risk: Configuration Incompatibility

**Mitigation**:
- Config classes are backward compatible (new fields have defaults)
- YAML files will still load (extra fields ignored by Pydantic)

---

## Timeline Summary

| Phase | Duration | Blocker? |
|-------|----------|----------|
| 0: Pre-flight | 0.5h | No |
| 1: Module naming | 2h | **YES** |
| 2: Constants/config | 2h | **YES** |
| 3: Processor pattern | 3h | No |
| 4: Data models | 1h | No |
| 5: Config polish | 1.5h | No |
| 6: Cleanup | 2h | No |
| 7: Testing/docs | 2h | No |
| 8: Verification | 1h | No |
| **Total** | **15h** | |

**Estimated**: 2 days of focused work, or 1 week calendar time with breaks.

---

## Next Steps After This Plan

Once architecture is solid:

1. **Add features**: New modules following the pattern
2. **Performance**: Profile and optimize hot paths
3. **Robustness**: Handle edge cases, improve error messages
4. **Usability**: GUI, Jupyter integration, cloud deployment

But **first**: Get the foundation right. This plan does that.

---

**Questions?** See ARCHITECTURE.md for design rationale, or open an issue.
