# TODOs

## Code quality and functionality
  - [x] Asset and download helpers with path variable on installation: GEBCO, model weights
  - [ ] (pre-release) Unify asset handling: host all model weights (refl2prop, emulator, datadriven albedo) on HuggingFace, remove bundled refl2prop and albedo weights from source tree, make all processors use the same managed-asset-only pattern with no silent bundled fallback
  - [x] Review and update all processors to use band-caching logic implemented in 0cd91c1
  - [x] Use raw weights files from sensei-v2 and stop using its sliding window inference feature. Deduplicate sliding window code so that height emulation window is reused for sensei-v2, and use asset and download helpers for the model weights instead of sensei-v2's own handling.
  - [x] Improved manifest.json: hashing of IO files, crop-window tracking and support, intelligently assigning steps as "outdated" when a processor upstream of it is changed. More integrity checks between project.yaml and manifests, e.g. git repo commit hashes / release version numbers to cross-check.
  - [ ] Improve clarity and accessibility of configs: explore nested configs to avoid long lists of independent sets of parameters (e.g. the height emulator's parameters all sitting next to the original height algorithm's parameters), project recipes for different sensible configurations. How do we inject n_workers for band parallel ops across all processors - current n_workers in configs are useless.
  - [ ] Add improved write options for data classes: e.g. netcdf, zarr
  - [ ] Remove Gaussian Process albedo method (IDW seems better and much faster)
  - [ ] (major bug) datadriven albedo currently has strong detector dependency, creating unwanted stripes
  - [x] more flexibility and independence for _output_ resolutions of processors via the .write function, e.g. cloud height emulation processed @ 20m/pixel and saved at 60 m/pixel. Strict distinction between "working_resolution" and "output_resolution" in configs?
  - [ ] Decouple stride from output_resolution in original cloud_height algorithm, shouuld be two separate parameters. 
  - [x] Improve logging behaviour (log files per computation)
  - [x] Define computation graph in YAML spec, so that users can modify the pipeline without changing project.py or entry.py

## Features 
  - [ ] (new processor) Using B09 and B10 for water vapor column retrieval above cloud: referencing LUT and looking at difference in reflectance for the two bands against expected
  - [ ] (new processor) Using B01 for aerosol retrieval above cloud: referencing LUT and looking at difference in reflectance for the two bands against expected (need to make refl2prop model independent of B01 to be valid)
  - [ ] (new processor) Cloud type with SSL: latent embeddings output for use with clustering and distance metrics between samples (e.g. similarity search)
  - [x] Orchestration: how to handle big processing runs efficiently (i) parallelising at whole scene processing level vs. parallelising at individual processor level. Could be linked to another config file in project spaces with runtime/orchestration parameters
  - [x] Standardised approach for output catalogue management: STAC/geoparquet/DuckDB/alternatives
  - [ ] Regridding tool to turn project run into ERA5/6-like products
  - [x] Overhaul project scene management (currently a list in the project.yaml) with a database. Forms basis for next point
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
  - [x] (potential) Crop window scaling uses nominal `BAND_RESOLUTIONS` constants (`sentinel.py`) — `_get_bands` and `_get_footprints` now derive scale from actual B02 file dimensions via `_get_b02_dims()`
  - [ ] Zombie worker processes in `CloudHeightProcessor` when retrieval returns zero points; also unconditional `/dev/shm` usage (`cloud_height/processor.py`)
  - [x] Missing nodata masking in `DataDrivenAlbedoEstimator` — B02 DN loaded after inference, resized to output grid, NaN applied to DN=0 pixels
  - [x] `PostProcessParams` subclassed `BaseProcessorConfig` incorrectly — changed to plain `BaseModel` with `extra='forbid'`; no unused `output_dir`/`n_workers` fields
  - [x] `refl2prop/config.py` had a silent bundled-model fallback — removed; config now points directly at managed asset; `FileNotFoundError` at inference time gives actionable message
  - [x] `ShadingPropertyInverter.__init__` duplicated model-loading logic without calling `super().__init__()` — shared sequence extracted to `CloudPropertyInverter._load_and_init_model()` staticmethod, called from both
  - [x] (potential) `cfgrib` is an optional install extra but called unconditionally in the datadriven albedo path — `ImportError` now caught separately with a message pointing to `pip install 'clouds-decoded[sampling]'` in both `datadriven/processor.py` and `datadriven/sampler.py`
  - [ ] Non-standard multi-root `package-dir` layout in `pyproject.toml` causes IDE import resolution failures — `cloud_height_emulator` entry added; broader `src/clouds_decoded/` restructure still outstanding
