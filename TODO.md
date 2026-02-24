# TODOs

## Code quality and functionality
  - [x] Asset and download helpers with path variable on installation: GEBCO, model weights
  - [ ] (pre-release) Unify asset handling: host all model weights (refl2prop, emulator, datadriven albedo) on HuggingFace, remove bundled refl2prop and albedo weights from source tree, make all processors use the same managed-asset-only pattern with no silent bundled fallback
  - [x] Review and update all processors to use band-caching logic implemented in 0cd91c1
  - [x] Use raw weights files from sensei-v2 and stop using its sliding window inference feature. Deduplicate sliding window code so that height emulation window is reused for sensei-v2, and use asset and download helpers for the model weights instead of sensei-v2's own handling.
  - [ ] Improved manifest.json: hashing of IO files, crop-window tracking and support, intelligently assigning steps as "outdated" when a processor upstream of it is changed. More integrity checks between project.yaml and manifests, e.g. git repo commit hashes / release version numbers to cross-check.
  - [ ] Improve clarity and accessibility of configs: explore nested configs to avoid long lists of independent sets of parameters (e.g. the height emulator's parameters all sitting next to the original height algorithm's parameters), project recipes for different sensible configurations
  - [ ] Add improved write options for data classes: e.g. netcdf, zarr
  - [ ] Refactor height emulator to sit inside the cloud_height module, ala albedo and the datadriven alternative. Avoid config parameters like "use_emulator" in project.yaml
  - [ ] Remove Gaussian Process albedo method (IDW seems better and much faster)

## Features 
  - [ ] (new processor) Using B09 and B10 for water vapor column retrieval above cloud: referencing LUT and looking at difference in reflectance for the two bands against expected
  - [ ] (new processor) Using B01 for aerosol retrieval above cloud: referencing LUT and looking at difference in reflectance for the two bands against expected (need to make refl2prop model independent of B01 to be valid)
  - [ ] (new processor) Cloud type with SSL: latent embeddings output for use with clustering and distance metrics between samples (e.g. similarity search)
  - [ ] Speed up refocus with parallelisation / vectorising / something because it can definitely be quicker (tbd)
  - [ ] Orchestration: how to handle big processing runs efficiently (i) parallelising at whole scene processing level vs. parallelising at individual processor level
  - [ ] Standardised approach for output catalogue management: STAC/geoparquet/DuckDB/alternatives
  - [ ] Regridding tool to turn project run into ERA5/6-like products
  - [ ] (big) Analysis toolkit for statistical checks on regridded outputs and original outputs, database-style queries for certain conditions, integration with ERA5/6 layers for more complex analysis
  - [ ] (optional) Web dashboard homepage for project, linking to visualiser for individual scenes and project-wide stats and maps
