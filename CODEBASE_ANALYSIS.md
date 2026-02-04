# Clouds Decoded: Codebase Analysis Report

**Date**: 2026-02-04
**Analyzed**: 4,117 lines across 35 Python files in `src/`
**Purpose**: Comprehensive assessment of current state and identification of issues

---

## EXECUTIVE SUMMARY

While the individual modules show solid technical work, the project is suffering from **integration debt**. You have 4 separate processing modules that were likely developed independently and are now being forced together without a coherent architecture. The result: import chaos, configuration sprawl, and inconsistent interfaces.

**Critical Issues**: 5
**Major Issues**: 6
**Minor Issues**: Multiple

**Recommendation**: Pause feature development. Fix foundational issues (imports, naming, patterns) before proceeding.

---

## 🔴 CRITICAL ISSUES

### 1. MODULE NAMING CATASTROPHE

**Problem**: Your directories use hyphens (`cloud-height`, `albedo-estimator`) but Python imports require underscores.

**Evidence**:
```
Directory: /modules/cloud-height/
Import: from clouds_decoded.modules.cloud_height import X
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ - Must use underscore
```

**Files affected**:
- `modules/cloud-height/` → Cannot import as `cloud-height`
- `modules/albedo-estimator/` → Cannot import as `albedo-estimator`

**Current workaround** ([shared_utils/processors.py:12-43](src/shared_utils/processors.py#L12-L43)):
Every import wrapped in try/except:

```python
try:
    from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
except ImportError:
    CloudHeightProcessor = None
```

**Impact**:
- Installation will fail or be fragile
- IDE autocomplete broken
- Testing is unreliable
- Every new developer will hit this immediately

**Fix**: Rename directories to use underscores. See IMPLEMENTATION_PLAN.md Phase 1.

---

### 2. CONSTANTS DUPLICATION

**Critical scientific constants duplicated**:

**Location 1**: [shared_utils/constants.py:11-48](src/shared_utils/constants.py#L11-L48)
```python
BAND_TIME_DELAYS = { "B02": 0, "B03": 0.527, ... }
BAND_RESOLUTIONS = { "B02": 10, "B03": 10, ... }
ORBITAL_VELOCITY = np.sqrt(...)
```

**Location 2**: [modules/cloud-height/constants.py:6-43](src/modules/cloud-height/constants.py#L6-L43)
```python
# EXACT SAME VALUES
BAND_TIME_DELAYS = { "B02": 0, "B03": 0.527, ... }
BAND_RESOLUTIONS = { "B02": 10, "B03": 10, ... }
ORBITAL_VELOCITY = np.sqrt(...)
```

**Risk**: These WILL diverge. Someone updates one, forgets the other → wrong physics calculations in different modules.

**Fix**: Delete `cloud-height/constants.py`, import from `shared_utils.constants`.

---

### 3. CONFIG LOADING HELL

**Problem**: CLI doesn't trust that configs have standard interface.

[cli/entry.py:36-46](src/cli/entry.py#L36-L46):
```python
if hasattr(CloudHeightConfig, 'from_yaml'):
    config = CloudHeightConfig.from_yaml(config_path)
elif hasattr(CloudHeightConfig, 'load_yaml'):
     if config_path:
         config = CloudHeightConfig.load_yaml(config_path)
     else:
         config = CloudHeightConfig()
else:
     config = CloudHeightConfig()
```

**Why this exists**: Config classes don't consistently implement `from_yaml()`... except they DO (inherited from BaseProcessorConfig).

**Reality**:
- [shared_utils/config.py:20-37](src/shared_utils/config.py#L20-L37) defines `BaseProcessorConfig.from_yaml()`
- All configs inherit from it
- This branching logic is defensive programming against imaginary problems

**Impact**: Developer paranoia. Code doesn't trust itself.

**Fix**: Remove branching, just call `Config.from_yaml()` directly.

---

### 4. BROKEN PROCESSOR PATTERN

**Nearly consistent**, then AlbedoEstimator breaks everything:

| Processor | Input | Output | Pattern |
|-----------|-------|--------|---------|
| CloudHeightProcessor | `process(scene)` | `CloudHeightGridData` | ✅ |
| CloudMaskProcessor | `process(scene)` | `CloudMaskData` | ✅ |
| CloudPropertyInverter | `process(scene, heights)` | `CloudPropertiesData` | ⚠️ Extra arg, but typed |
| AlbedoEstimator | `process(scene, percentile=1.0)` | `Dict[str, AlbedoData]` | ❌ **BROKEN** |

**AlbedoEstimator problems**:
1. Returns `Dict[str, AlbedoData]` instead of single typed object
2. Has `percentile` parameter (should be in config)
3. No config class at all

**Why this matters**: Processors are supposed to be composable. You can't pass a dict to a function expecting `AlbedoData`.

**Fix**: Refactor to return single `AlbedoData` object with multi-band array. See IMPLEMENTATION_PLAN Phase 3.

---

### 5. DEAD CODE EVERYWHERE

[modules/albedo-estimator/processor.py:75-119](src/modules/albedo-estimator/processor.py#L75-L119):

**45 lines of unreachable code** after a `return` statement.

```python
def process(self, scene, percentile):
    # ... 70 lines of working code ...
    return albedo_rasters  # ← Function returns here (line 72)

    # ← Lines 75-119: NEVER EXECUTED
    h, w = ref_raster.data.shape
    band_names = list(scene.bands.keys())
    # ... 40+ more lines ...
```

**This is clearly refactored code where someone forgot to delete the old implementation.**

**Impact**: Confusion, maintenance burden, suggests poor code review.

**Fix**: Delete lines 75-119.

---

## 🟡 MAJOR STRUCTURAL PROBLEMS

### 6. METADATA INCONSISTENCY

**Good foundation built** ([shared_utils/data/base.py](src/shared_utils/data/base.py)):

```python
class Metadata(BaseModel):
    """Base model for metadata structures."""
    model_config = ConfigDict(extra='allow')

class CloudHeightMetadata(Metadata):
    processing_config: Dict[str, Any] = ...
```

**But processors undermine it** ([cloud-mask/processor.py:52-56](src/modules/cloud-mask/processor.py#L52-L56)):

```python
# BAD: Raw dict instead of CloudMaskMetadata
CloudMaskData(
    data=mask,
    metadata={  # ← Should be CloudMaskMetadata object!
        "method": "simple_threshold",
        "band": threshold_band,
        "value": threshold_value
    }
)
```

**Impact**:
- Type safety lost
- Can't validate metadata
- IDE can't autocomplete
- Defeats purpose of Pydantic models

**Occurrences**:
- cloud-mask processor: Line 52
- refl2prop processor: Uses metadata correctly ✅
- cloud-height processor: Uses metadata correctly ✅

**Fix**: Create `CloudMaskMetadata` class, use it in processor.

---

### 7. CONFIGURATION EXPLOSION

**60+ user-facing parameters** with no organization:

| Module | Config Parameters | Exposed in CLI | In YAML |
|--------|-------------------|----------------|---------|
| CloudHeight | 14 options | 0 (just config path) | Yes |
| CloudMask | 10 options | 8 directly in CLI | No |
| Refl2Prop | 12 options | 2 directly in CLI | Yes |
| Albedo | None (no config!) | 1 hardcoded | No |

**Problems**:

1. **No hierarchy**: Which params are essential vs. expert-only?
2. **Inconsistent exposure**: Sometimes CLI has individual params, sometimes just config file
3. **No validation**: Can I set `stride=0`? `max_height=-1000`? Pydantic would catch this but not set up
4. **No documentation**: Field descriptions missing or minimal

**Example of good** ([refl2prop/config.py:66-92](src/modules/refl2prop/config.py#L66-L92)):
```python
model_path: str = Field(..., description="Path to the .pth model checkpoint")
batch_size: int = Field(32768, description="Inference batch size (pixels)")
return_uncertainty: bool = Field(False, description="If True, calculates uncertainty")
```

**Example of bad** ([cloud-height/config.py:11-34](src/modules/cloud-height/config.py#L11-L34)):
```python
reference_band: str = 'B02'  # ← No Field(), no description, no validation
bands: List[str] = ['B02','B03','B04','B05','B07','B08']  # ← No constraint
stride: int = 300  # ← Can it be 1? 10000? Who knows
```

**Fix**: Add Field() with descriptions and constraints to all config parameters.

---

### 8. IMPORT PATH CHAOS

**Three different import strategies coexist**:

**Strategy 1: Absolute** (CLI):
```python
from clouds_decoded.data import Sentinel2Scene
from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
```

**Strategy 2: Relative** (within modules):
```python
from .config import CloudHeightConfig
from .physics import heightsToOffsets
```

**Strategy 3: Mixed** (refl2prop):
```python
from clouds_decoded.data import Sentinel2Scene  # absolute
from .model import InversionNet  # relative
```

**None are wrong individually**, but together create fragility:
- Rename a module → some imports break, others work
- Copy a file → imports may or may not resolve
- Package structure changes → unpredictable failures

**Best practice**:
- Use absolute for cross-module
- Use relative for intra-module
- Be consistent

**Current state**: Inconsistent.

---

### 9. POST-PROCESSING CHAOS

**Three different patterns** for post-processing:

**Pattern 1**: Automatic internal ([cloud-height/processor.py:104](src/modules/cloud-height/processor.py#L104))
```python
def process(self, scene):
    # ... correlation ...
    self.postprocess()  # ← Called automatically
    return result
```

**Pattern 2**: Explicit separate method ([cloud-mask/processor.py:236](src/modules/cloud-mask/processor.py#L236))
```python
raw_mask = processor.process(scene)
refined_mask = processor.postprocess(raw_mask, params)  # ← User calls explicitly
```

**Pattern 3**: Inline, no separate method (refl2prop)
```python
def process(self, scene, heights):
    # ... inference ...
    # Post-processing inline at lines 263-268
    if self.config.mask_invalid_height:
        results[invalid] = np.nan
    return result
```

**User experience**: Inconsistent and confusing.

**Fix**: Choose one pattern (recommend explicit separate method), standardize.

---

### 10. NORMALIZATION NIGHTMARE

**Three different normalization schemes** for Sentinel-2 reflectance:

**Scheme 1** ([refl2prop/config.py:78-85](src/modules/refl2prop/config.py#L78-L85)):
```python
norm_bands_offset: float = 1000.0
norm_bands_scale: float = 10000.0
# Formula: (value - 1000) / 10000
```

**Scheme 2** ([cloud-mask/processor.py:162-165](src/modules/cloud-mask/processor.py#L162-L165)):
```python
if offset > 0:
    arr_float = (arr_float - offset) / 10000.0
else:
    arr_float = arr_float / 10000.0
# Formula: Dynamic offset detection, then / 10000
```

**Scheme 3** ([refl2prop/example.py:20](src/modules/refl2prop/example.py#L20)):
```python
band_data = (src.read(1).astype(np.float32) - 1000.0) / 10000.0
# Formula: Hardcoded (x - 1000) / 10000
```

**Questions**:
- Which is correct?
- Are they compatible?
- What happens if scene is from different processing baseline?

**Impact**: Model trained on one normalization, inference uses another → wrong predictions.

**Fix**:
1. Document Sentinel-2 processing baseline detection
2. Centralize normalization logic
3. Make it explicit in configs

---

### 11. SENTINEL-2 LOADING INCONSISTENCY

**Sentinel2Scene is well-designed** ([data/sentinel.py](src/shared_utils/data/sentinel.py)):
- 337 lines of careful XML parsing
- Footprint handling
- Angle calculations
- Proper error handling

**But no validation that scene is "ready"**:

```python
# You can do this:
scene = Sentinel2Scene()
processor.process(scene)  # ← Will crash deep in processing

# Should require:
scene = Sentinel2Scene()
scene.read("path.SAFE")  # ← Now has bands, angles, etc.
processor.process(scene)
```

**Processors sometimes check** ([cloud-height/processor.py:41](src/modules/cloud-height/processor.py#L41)):
```python
logger.info(f"Processing scene: {scene.scene_directory}")
# ← Assumes scene_directory exists, but doesn't check
```

**Processors sometimes don't check** (refl2prop):
```python
for b in required_bands:
    if b not in scene.bands:  # ← Checks here
        raise ValueError(...)

# But doesn't check if sun_zenith is None before using it
sza = to_grid(scene.sun_zenith)  # ← Could be None
```

**Fix**: Create `validate_scene_for_processing()` helper, use in all processors.

---

## 🟢 WHAT'S ACTUALLY GOOD

### ✅ Data Model Foundation

[shared_utils/data/base.py](src/shared_utils/data/base.py) is **well-architected**:

```python
class Data(BaseModel, ABC):
    """Abstract base class for data models."""
    @abstractmethod
    def read(self, filepath: str): pass

    @abstractmethod
    def write(self, filepath: str): pass

class GeoRasterData(Data):
    """Geospatial raster with transform, CRS, metadata."""
    data: Optional[np.ndarray] = None
    transform: Optional[Any] = None
    crs: Optional[Any] = None
    metadata: Metadata = Field(default_factory=Metadata)
```

**Good patterns**:
- Abstract base classes enforce interface
- Pydantic provides validation
- Read/write methods handle GeoTIFF and NetCDF
- Metadata system is extensible
- `with_template()` helper for georeferencing

**Room for improvement**:
- Validation methods exist but not always called
- Metadata not always used properly (see issue #6)

---

### ✅ Individual Module Quality

Each processor shows **good scientific computing**:

**cloud-height**:
- Sophisticated parallax correlation
- Proper orbital physics ([physics.py](src/modules/cloud-height/physics.py))
- Efficient column extraction with rotation transforms
- Multiprocessing for performance

**refl2prop**:
- Modern neural network architecture (dual-head for physics + uncertainty)
- Strict feature ordering via Enums ([config.py:12-56](src/modules/refl2prop/config.py#L12-L56))
- Uncertainty quantification (OOD detection)
- Custom loss for phase-specific masking

**cloud-mask**:
- Clean SEnSeIv2 integration
- Separate threshold processor for simple cases
- Post-processing with morphological operations
- Proper resolution handling

**Algorithms are solid. Integration is the problem.**

---

### ✅ Type Hints

**Mostly good use** of type hints throughout:

```python
def process(self, scene: Sentinel2Scene) -> CloudHeightGridData:
    """Process scene to retrieve heights."""
```

**Missing in some places**:
- Helper functions sometimes untyped
- Some return types missing
- Dict vs typed objects inconsistent

**Score**: 7/10

---

### ✅ CLI Structure

[cli/entry.py](src/cli/entry.py) **separates logic from commands**:

```python
# Logic functions (lines 28-134)
def run_cloud_height(scene, config_path, output_path) -> CloudHeightGridData:
    # ... actual logic ...

def run_cloud_mask(scene, output_path, method, ...):
    # ... actual logic ...

# CLI commands (lines 137-264)
@app.command()
def cloud_height(scene_path, config_path, output_path):
    """CLI wrapper."""
    scene = Sentinel2Scene()
    scene.read(scene_path)
    run_cloud_height(scene, config_path, output_path)
```

**This is good architecture**: Logic is reusable, CLI is thin wrapper.

**Could be better**: Duplication of scene loading across commands.

---

## 📊 QUANTIFIED MESS METRICS

| Metric | Count | Status |
|--------|-------|--------|
| Total Python files | 35 | 📈 Growing |
| Total lines of code | 4,117 | 📈 |
| Empty `__init__.py` files | 5 | ❌ Underutilized |
| Duplicated constant definitions | 2 sets | ❌ DRY violation |
| Config loading strategies | 3 different | ❌ Inconsistent |
| Import error try/except blocks | 6 | ⚠️ Fragile imports |
| Unreachable code lines | 45+ | ❌ Technical debt |
| Debug print statements | 6+ | ⚠️ Left in production |
| Standalone setup.py files | 4 | ⚠️ Monorepo confusion |
| Normalization schemes | 3 | ❌ Scientific risk |
| Processor pattern violations | 1 major | ❌ |
| Metadata type violations | 2+ | ❌ |

---

## 🎯 FILE-BY-FILE BREAKDOWN

### cli/entry.py (266 lines)

**Good**:
- ✅ Clean command structure with Typer
- ✅ Separation of logic functions from CLI commands
- ✅ Workflow command for multi-step pipeline

**Issues**:
- ❌ Lines 38-46: Defensive config loading (unnecessary)
- ❌ Scene loading duplicated across commands
- ⚠️ Workflow is hardcoded, should be configurable

**Severity**: Medium

---

### shared_utils/processors.py (44 lines)

**Purpose**: Unified import point for all processors

**Issues**:
- ❌ Lines 12-43: Every import in try/except
- ❌ Empty exports when import fails
- ❌ Uses `albedo_estimator` (underscore) but module is `albedo-estimator` (hyphen)

**Why this exists**: Imports are broken, this papers over it.

**Severity**: Critical (symptom of deeper issue)

---

### shared_utils/constants.py (48 lines)

**Good**:
- ✅ Clean constant definitions
- ✅ Proper scientific values
- ✅ Good documentation

**Issues**:
- ❌ Duplicated in `cloud-height/constants.py`

**Severity**: Critical

---

### shared_utils/data/base.py (312 lines)

**Good**:
- ✅ Excellent ABC design
- ✅ GeoRasterData handles GeoTIFF and NetCDF
- ✅ PointCloudData with Parquet support
- ✅ Metadata serialization via JSON

**Issues**:
- ⚠️ Validation exists but not enforced in write()
- ⚠️ Could use more type specificity (Any for transform/crs)

**Severity**: Low

**Score**: 9/10

---

### shared_utils/data/sentinel.py (337 lines)

**Good**:
- ✅ Comprehensive S2 .SAFE parsing
- ✅ XML metadata extraction
- ✅ Proper angle calculations (circular mean for azimuth!)
- ✅ Footprint handling for detector IDs
- ✅ Good error messages

**Issues**:
- ⚠️ No validation that required attributes are set after read()
- ⚠️ write() raises NotImplementedError (could document why)

**Severity**: Low

**Score**: 9/10

---

### modules/cloud_height/ (7 files, ~800 lines)

**Good**:
- ✅ Sophisticated stereo matching
- ✅ Clean config using BaseProcessorConfig
- ✅ Good physics module (heightsToOffsets, etc.)
- ✅ Multiprocessing for performance

**Issues**:
- ❌ constants.py duplicates shared_utils
- ❌ Both config.py AND config.yaml (confusion)
- ❌ producer.py, worker.py use Redis/RQ but not in CLI
- ⚠️ Hardcodes /dev/shm (line 47) - breaks on Windows

**Severity**: Medium to High

---

### modules/cloud_mask/ (3 files, ~300 lines)

**Good**:
- ✅ Clean implementation
- ✅ Separate ThresholdCloudMaskProcessor
- ✅ Good post-processing pattern
- ✅ Resolution handling

**Issues**:
- ⚠️ PostProcessParams inherits from BaseProcessorConfig (is it a processor config?)
- ❌ Metadata as raw dict (line 52)
- ⚠️ No config.yaml (inconsistent with other modules)

**Severity**: Medium

---

### modules/refl2prop/ (9 files, ~1400 lines)

**Good**:
- ✅ Excellent neural network (dual-head, uncertainty)
- ✅ Strict feature ordering via Enums
- ✅ Good dataset with lazy loading
- ✅ Comprehensive loss functions

**Issues**:
- ❌ Lines 284-292 in processor.py: Debug print/pprint left in
- ❌ example.py hardcoded paths, not integrated
- ❌ setup.py suggests standalone package
- ⚠️ UNCERTAINTY_README.md in source tree (should be docs/)
- ⚠️ Multiple normalization schemes

**Severity**: Medium

---

### modules/albedo_estimator/ (2 files, ~120 lines)

**Issues**:
- ❌ Lines 75-119: Unreachable dead code
- ❌ No config class
- ❌ Returns Dict breaking pattern
- ❌ Two implementations in one function

**Severity**: High (needs complete refactor)

**Score**: 3/10

---

## 🔍 SPECIFIC CONCERNS FROM USER

> "imports not working"

**Root cause**: Module directory names use hyphens, Python imports use underscores. Try/except blocks mask failures.

**Fix**: Rename directories, remove try/except.

---

> "too many options for users to set"

**Count**: 60+ configurable parameters across 4 modules

**Issues**:
1. No hierarchy (essential vs advanced)
2. No validation constraints
3. Some exposed in CLI, some in YAML, inconsistent
4. No documentation of defaults or recommended values

**Fix**:
- Add Field() with descriptions and validation
- Categorize parameters (basic/advanced)
- Document recommended values

---

> "no central place where we can configure things"

**Current state**:
- BaseProcessorConfig exists
- Each module has config.py
- Some have config.yaml
- CLI has inline parameters
- Normalization scattered across modules

**No global settings** like:
- Default output directory
- Logging level
- Cache location
- Device selection (CPU/GPU)

**Fix**: Create `clouds_decoded.settings` with global defaults.

---

> "consistent, strict data models"

**Data models are good**, but:
- Metadata sometimes raw dicts
- Validation exists but not enforced
- Return types inconsistent (Dict vs typed object)

**Fix**: Enforce metadata typing, standardize processor pattern.

---

> "patterns of behaviour across the different modules"

**Processor pattern exists** (80% compliance):
- ✅ Constructor takes config
- ✅ process() method is main interface
- ❌ Post-processing inconsistent
- ❌ AlbedoEstimator breaks pattern

**Fix**: Standardize, document pattern, enforce in code reviews.

---

> "inputs and outputs to be clear and well defined"

**Inputs**: Mostly clear (Sentinel2Scene)

**Outputs**:
- ✅ Typed (CloudHeightGridData, etc.)
- ❌ AlbedoEstimator returns Dict
- ⚠️ Some return None on failure (should return empty object)

**Metadata**: Inconsistent usage undermines clarity.

**Fix**: Standardize return types, enforce metadata.

---

## 🚨 PRIORITY RANKING

### P0 (Critical - Blocks Development)
1. Module naming (hyphens → underscores)
2. Import reliability (remove try/except)
3. Constants duplication

### P1 (High - Affects Quality)
4. Processor pattern enforcement
5. Metadata type safety
6. Dead code removal

### P2 (Medium - Technical Debt)
7. Config parameter validation
8. Post-processing standardization
9. Normalization centralization
10. Scene validation

### P3 (Low - Polish)
11. Debug code removal
12. Documentation improvements
13. Type checking (mypy)
14. Test coverage

---

## 📈 RECOMMENDATIONS

### Immediate Actions (This Week)

1. **Rename modules** (hyphens → underscores)
2. **Delete duplicate constants** (keep only shared_utils)
3. **Remove try/except** from processors.py
4. **Delete dead code** (albedo_estimator lines 75-119)
5. **Remove debug code** (refl2prop processor lines 284-292)

**Estimated time**: 3-4 hours
**Risk**: Medium (breaks imports temporarily)
**Payoff**: Eliminates critical fragility

---

### Next Sprint (This Month)

6. **Refactor AlbedoEstimator** to match processor pattern
7. **Standardize metadata usage** (no raw dicts)
8. **Add config validation** (Field constraints)
9. **Create scene validation** helper
10. **Standardize post-processing** pattern

**Estimated time**: 2 days
**Risk**: Low (additive changes)
**Payoff**: Consistent, maintainable codebase

---

### Future Improvements

11. Add comprehensive test suite (>70% coverage)
12. Set up type checking (mypy)
13. Create developer documentation
14. CI/CD pipeline

---

## 📋 SUCCESS METRICS

After refactoring, you should have:

- [ ] Zero import errors in processors.py
- [ ] Zero duplicated constants
- [ ] All processors follow standard pattern
- [ ] All metadata uses typed classes
- [ ] All configs have Field() with validation
- [ ] No unreachable code
- [ ] No debug print() statements
- [ ] Test suite passes
- [ ] Documentation updated

---

## 🎯 CONCLUSION

**The good news**: Your algorithms are scientifically solid. The neural networks, physics, and processing logic are well-implemented.

**The bad news**: The codebase is suffering from lack of integration planning. Each module was developed in isolation, then shoved together.

**The path forward**:

1. **Fix critical issues** (naming, imports, duplication) - 1 day
2. **Enforce patterns** (processor interface, metadata) - 2 days
3. **Add validation and docs** - 1 day

**Total time investment**: ~1 week

**Payoff**: A maintainable, professional codebase that can scale.

---

**Next Steps**: See IMPLEMENTATION_PLAN.md for detailed refactoring roadmap.
