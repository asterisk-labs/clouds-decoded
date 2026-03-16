# clouds-decoded Documentation

**clouds-decoded** is an end-to-end toolkit for retrieving cloud physical properties from Sentinel-2 Level-1C satellite imagery. It reads `.SAFE` directories and produces georeferenced GeoTIFF outputs covering cloud masking, cloud height, surface albedo, parallax correction, and cloud optical/microphysical property inversion.

## Contents

- [Installation](installation.md) -- Setting up the environment, installing the package, and downloading model weights.
- [CLI Reference](cli-reference.md) -- Complete reference for all `clouds-decoded` commands and options.
- [Configuration](configuration.md) -- How the config system works, and a field-by-field reference for every module config.
- [Data Classes](data-classes.md) -- The data class hierarchy for reading, writing, and inspecting processing outputs in Python.
- [Projects](projects.md) -- The project system for batch processing: init, stage, run, status, and statistics.
- [Architecture](architecture.md) -- Architecture overview for developers: pipeline design, processor pattern, and key abstractions.
- [API Reference](api/index.md) -- Auto-generated reference for all public classes and functions.
