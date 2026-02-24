# TODOs

## Code quality and functionality
  - [x] Asset and download helpers with path variable on installation: GEBCO, model weights
  - [ ] (pre-release) Unify asset handling: host all model weights (refl2prop, emulator, datadriven albedo) on HuggingFace, remove bundled refl2prop and albedo weights from source tree, make all processors use the same managed-asset-only pattern with no silent bundled fallback
  - [x] Review and update all processors to use band-caching logic implemented in 0cd91c1
  - [x] Use raw weights files from sensei-v2 and stop using its sliding window inference feature. Deduplicate sliding window code so that height emulation window is reused for sensei-v2, and use asset and download helpers for the model weights instead of sensei-v2's own handling.
  - [ ] Improved manifest.json: hashing of IO files, crop-window tracking and support, intelligently assigning steps as "outdated" when a processor upstream of it is changed. More integrity checks between project.yaml and manifests, e.g. git repo commit hashes / release version numbers to cross-check.
  - [ ] Improve clarity and accessibility of configs: explore nested configs to avoid long lists of independent sets of parameters (e.g. the height emulator's parameters all sitting next to the original height algorithm's parameters), project recipes for different sensible configurations. How do we inject n_workers for band parallel ops across all processors - current n_workers in configs are useless.
  - [ ] Add improved write options for data classes: e.g. netcdf, zarr
  - [ ] Refactor height emulator to sit inside the cloud_height module, ala albedo and the datadriven alternative. Avoid config parameters like "use_emulator" in project.yaml
  - [ ] Remove Gaussian Process albedo method (IDW seems better and much faster)
  - [ ] (major bug) datadriven albedo currently has strong detector dependency, creating unwanted stripes
  - [ ] more flexibility and independence for _output_ resolutions of processors via the .write function, e.g. cloud height emulation processed @ 20m/pixel and saved at 60 m/pixel. Strict distinction between "working_resolution" and "output_resolution" in configs?

## Features 
  - [ ] (new processor) Using B09 and B10 for water vapor column retrieval above cloud: referencing LUT and looking at difference in reflectance for the two bands against expected
  - [ ] (new processor) Using B01 for aerosol retrieval above cloud: referencing LUT and looking at difference in reflectance for the two bands against expected (need to make refl2prop model independent of B01 to be valid)
  - [ ] (new processor) Cloud type with SSL: latent embeddings output for use with clustering and distance metrics between samples (e.g. similarity search)
  - [ ] Speed up refocus with parallelisation / vectorising / something because it can definitely be quicker (tbd)
  - [ ] Orchestration: how to handle big processing runs efficiently (i) parallelising at whole scene processing level vs. parallelising at individual processor level. Could be linked to another config file in project spaces with runtime/orchestration parameters
  - [ ] Standardised approach for output catalogue management: STAC/geoparquet/DuckDB/alternatives
  - [ ] Regridding tool to turn project run into ERA5/6-like products
  - [ ] (big) Analysis toolkit for statistical checks on regridded outputs and original outputs, database-style queries for certain conditions, integration with ERA5/6 layers for more complex analysis
  - [ ] (optional) Web dashboard homepage for project, linking to visualiser for individual scenes and project-wide stats and maps


## (potential) and confirmed bugs
  - [x] Dead identical if/else branches in `heightsToOffsets()` (`physics.py`) — removed
  - [x] Missing `np.radians()` on relative azimuth in `ShadingPropertyInverter` geometry (`refl2prop/processor.py`) — fixed
  - [x] `model_dump()` missing `mode='json'` in GeoTIFF/NetCDF metadata write paths (`base.py`) — fixed
  - [x] Bare `except Exception` swallowing mask load errors silently in `CloudHeightProcessor` — narrowed to specific exceptions
  - [x] Non-square `window_size` accepted but assumed square in `CloudHeightEmulatorConfig` — square validation added
  - [x] Hand-rolled TOML parser in `assets.py` broke on inline comments — fixed
  - [x] `debug_mode` field in `BaseProcessorConfig` unused everywhere — removed
  - [x] Inconsistent wind/bathymetry fallback values between datadriven `processor.py` and `sampler.py` — unified to `0.0` / `-1.0`
  - [x] Dead `reference_resolution` field in `AlbedoSamplerConfig` — removed
  - [x] `scene.orientation` field was buggy (wrong pyproj axis order) and unused — removed entirely along with `_get_scene_orientation()`
  - [ ] (potential) Crop window scaling uses nominal `BAND_RESOLUTIONS` constants (`sentinel.py`) — will produce misaligned crops with newer SAFE baselines where actual pixel dimensions differ from nominal
  - [ ] Zombie worker processes in `CloudHeightProcessor` when retrieval returns zero points; also unconditional `/dev/shm` usage (`cloud_height/processor.py`)
  - [ ] Missing nodata masking in `DataDrivenAlbedoEstimator` — all other albedo paths apply it, this one doesn't
  - [ ] `PostProcessParams` passed as a method argument to `postprocess()` rather than via `__init__` — breaks the stated processor interface pattern
  - [ ] `refl2prop/config.py` has a silent bundled-model fallback that contradicts the stated design intent (see pre-release unify asset handling above)
  - [ ] `ShadingPropertyInverter.__init__` duplicates model-loading logic without calling `super().__init__()` — changes to parent won't propagate
  - [ ] (potential) `cfgrib` is an optional install extra but called unconditionally in the datadriven albedo path — `ImportError` at runtime with no guidance if not installed
  - [ ] Non-standard multi-root `package-dir` layout in `pyproject.toml` causes IDE import resolution failures; `cloud_height_emulator` not explicitly listed
