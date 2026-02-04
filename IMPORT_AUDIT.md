# Import Path Audit - Pre-Refactoring

**Date**: 2026-02-04
**Purpose**: Document all imports before Phase 1 module renaming

---

## Critical Issue Confirmed

**Directory names use HYPHENS, imports use UNDERSCORES**

```
Actual directories:          Import paths attempt:
├── cloud-height/       →    from clouds_decoded.modules.cloud_height
├── cloud-mask/         →    from clouds_decoded.modules.cloud_mask
├── albedo-estimator/   →    from clouds_decoded.modules.albedo_estimator
└── refl2prop/          →    from clouds_decoded.modules.refl2prop (OK)
```

Python internally converts hyphens to underscores in module names, but this creates import failures.

**Test result**:
```python
>>> from clouds_decoded.modules.cloud_height import CloudHeightProcessor
ImportError: cannot import name 'CloudHeightProcessor' from 'clouds_decoded.modules.cloud_height'
```

---

## Modules Requiring Rename

### 1. cloud-height → cloud_height
**Directory**: `src/modules/cloud-height/`
**Target**: `src/modules/cloud_height/`

**Current imports attempting to use it**:
- None found (all wrapped in try/except and failing silently)

**Files that will need updates after rename**:
- `src/shared_utils/processors.py` (line ~15, try/except block)
- `src/cli/entry.py` (will need import statement added)

---

### 2. cloud-mask → cloud_mask
**Directory**: `src/modules/cloud-mask/`
**Target**: `src/modules/cloud_mask/`

**Current imports attempting to use it**:
```python
# src/cli/entry.py:9-10
from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor, ThresholdCloudMaskProcessor
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig, PostProcessParams

# src/shared_utils/processors.py:24-26 (in try/except)
from clouds_decoded.modules.cloud_mask.processor import CloudMaskProcessor, ThresholdCloudMaskProcessor
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig, PostProcessParams

# src/shared_utils/data/__init__.py:7
from .cloud_mask import CloudMaskData, CloudMaskMetadata
```

**Status**: ⚠️ Imports exist but may be failing silently due to directory name mismatch

---

### 3. albedo-estimator → albedo_estimator
**Directory**: `src/modules/albedo-estimator/`
**Target**: `src/modules/albedo_estimator/`

**Current imports attempting to use it**:
```python
# src/shared_utils/processors.py:37 (in try/except)
from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator

# src/modules/refl2prop/processor.py:6 (in try/except)
from clouds_decoded.modules.albedo_estimator.processor import AlbedoEstimator
```

**Status**: ⚠️ Imports exist but wrapped in try/except, likely failing

---

### 4. refl2prop (NO CHANGE NEEDED)
**Directory**: `src/modules/refl2prop/`
**Status**: ✅ Already uses underscore, no rename needed

---

## Files Requiring Import Path Updates After Rename

### Critical Path Files (will break if not updated)
1. **src/cli/entry.py**
   - Lines with cloud_mask imports
   - Will need cloud_height imports added once working

2. **src/shared_utils/processors.py**
   - All try/except blocks (lines 12-43)
   - Should be simplified to direct imports after rename

3. **src/modules/refl2prop/processor.py**
   - AlbedoEstimator import (line 6)

---

## Try/Except Blocks to Remove

After renaming, these defensive try/except blocks can be removed:

```python
# src/shared_utils/processors.py:12-43
try:
    from clouds_decoded.modules.cloud_height.processor import CloudHeightProcessor
    from clouds_decoded.modules.cloud_height.config import CloudHeightConfig
except ImportError:
    logger.debug("Cloud height module not available")
    CloudHeightProcessor = None
    CloudHeightConfig = None

# Similar blocks for cloud_mask, refl2prop, albedo_estimator
```

**Action**: Replace with direct imports once directories are renamed.

---

## Verification Commands

After Phase 1 rename, verify imports work:

```bash
# Test each module individually
python3 -c "from clouds_decoded.modules.cloud_height import CloudHeightProcessor"
python3 -c "from clouds_decoded.modules.cloud_mask import CloudMaskProcessor"
python3 -c "from clouds_decoded.modules.albedo_estimator import AlbedoEstimator"
python3 -c "from clouds_decoded.modules.refl2prop import CloudPropertyInverter"

# Test unified import
python3 -c "from clouds_decoded.shared_utils.processors import *"

# Test CLI still works
clouds-decoded --help
```

---

## Risk Assessment

**High Risk**: Phase 1 will temporarily break ALL imports
- CLI commands will fail
- Tests (when created) will fail
- Any running processes will fail

**Mitigation**:
- Rename all 3 directories in single commit
- Update all import paths in same commit
- Verify with test imports before committing
- Have rollback tag ready (pre-refactor-baseline)

---

## Summary Statistics

- **Directories to rename**: 3 (cloud-height, cloud-mask, albedo-estimator)
- **Files with imports to update**: 3 (entry.py, processors.py, refl2prop/processor.py)
- **Try/except blocks to remove**: 4
- **Total import statements affected**: ~10-12

**Estimated time for Phase 1**: 2-3 hours including verification
