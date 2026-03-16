# Data Classes Reference

All data classes live in `src/shared_utils/data/`. They define how processing inputs and outputs are represented, read, and written.

## Class Hierarchy

```
Data (ABC)
  +-- GeoRasterData         # 2D/3D raster with CRS, transform, nodata
  |     +-- CloudMaskData
  |     +-- CloudHeightGridData
  |     +-- AlbedoData
  |     +-- CloudPropertiesData
  +-- PointCloudData         # 3D point cloud (sparse)

Sentinel2Scene               # Reads .SAFE directories, provides band access
Sentinel2Band                # Lazy numpy proxy for a single band
BandDict                     # Dict subclass mapping band names to Sentinel2Band
```

---

## [`Data`][clouds_decoded.data.base.Data] (ABC)

Base class for all data models. Source: `src/shared_utils/data/base.py`.

All subclasses must implement:

- `read(filepath)` -- Load data from a file.
- `write(filepath)` -- Write data to a file.
- `from_file(filepath)` -- Class method factory that creates an instance and calls `read()`.

---

## [`GeoRasterData`][clouds_decoded.data.base.GeoRasterData]

Georeferenced raster data stored as GeoTIFF. The primary base class for processing outputs.

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `Optional[np.ndarray]` | Raster array (2D or 3D with bands-first layout) |
| `transform` | Affine | Maps pixel coordinates to map coordinates |
| `crs` | CRS | Coordinate reference system |
| `nodata` | `Optional[float]` | Nodata sentinel value (default: `NaN` for float) |
| `metadata` | `Metadata` | Provenance and processing metadata |

Key methods:

- `read(filepath)` -- Read from GeoTIFF. Populates `data`, `transform`, `crs`, and `metadata` (from GeoTIFF tags).
- `write(filepath)` -- Write to GeoTIFF with compression and embedded metadata.
- `from_file(filepath)` -- Class method shortcut.
- [`resample()`][clouds_decoded.data.base.GeoRasterData.resample] -- Resample to a new grid.
- `validate()` -- Check that data is a valid 2D or 3D array.

### Loading an output in Python

```python
from clouds_decoded.data import CloudHeightGridData

height = CloudHeightGridData.from_file("cloud_height.tif")
print(height.data.shape)       # e.g. (183, 183)
print(height.crs)              # e.g. EPSG:32635
print(height.transform)        # Affine transform
print(height.metadata)         # Processing provenance
```

---

## [`CloudMaskData`][clouds_decoded.data.cloud_mask.CloudMaskData]

4-class cloud mask output. Source: `src/shared_utils/data/cloud_mask.py`.

Supports two modes:

- **Categorical** (default): `uint8` 2D array with values 0 (clear), 1 (thick cloud), 2 (thin cloud), 3 (cloud shadow). Nodata = 255.
- **Probability**: `float32` 4-band array `(4, H, W)` with per-class probabilities.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `nodata` | `float` | `255` | Nodata sentinel for uint8 masks |
| `metadata` | [`CloudMaskMetadata`][clouds_decoded.data.cloud_mask.CloudMaskMetadata] | -- | Includes `categorical`, `classes`, `method`, `model`, `resolution` |

Key methods:

- [`to_binary()`][clouds_decoded.data.cloud_mask.CloudMaskData.to_binary] -- Convert to a binary mask. `positive_classes` selects which class indices count as "positive" (e.g. `[1, 2, 3]` for any cloud/shadow). `threshold` applies to probability masks.

```python
from clouds_decoded.data import CloudMaskData

mask = CloudMaskData.from_file("cloud_mask.tif")
binary = mask.to_binary(positive_classes=[1, 2], threshold=0.5)
```

---

## [`CloudHeightGridData`][clouds_decoded.data.cloud_height.CloudHeightGridData]

Cloud top height on a raster grid. Source: `src/shared_utils/data/cloud_height.py`.

Values represent height in metres above ground level (AGL). Non-cloud pixels are `NaN`.

```python
from clouds_decoded.data import CloudHeightGridData
import numpy as np

height = CloudHeightGridData.from_file("cloud_height.tif")
valid = height.data[~np.isnan(height.data)]
print(f"Mean cloud height: {valid.mean():.0f} m")
```

---

## [`AlbedoData`][clouds_decoded.data.base.AlbedoData]

Per-band surface albedo at coarse resolution. Source: `src/shared_utils/data/base.py`.

Multi-band GeoTIFF where each band corresponds to a Sentinel-2 spectral band. Typically produced at ~300 m resolution.

---

## [`CloudPropertiesData`][clouds_decoded.data.refl2prop.CloudPropertiesData]

Cloud optical and microphysical properties. Source: `src/shared_utils/data/refl2prop.py`.

4-band (or 5-band with uncertainty) GeoTIFF:

| Band | Name | Description |
|------|------|-------------|
| 1 | `tau` | Cloud optical thickness |
| 2 | `ice_liq_ratio` | Ice-to-liquid ratio |
| 3 | `r_eff_liq` | Liquid effective radius |
| 4 | `r_eff_ice` | Ice effective radius |
| 5 | `uncertainty` | (optional) Prediction uncertainty |

```python
from clouds_decoded.data import CloudPropertiesData

props = CloudPropertiesData.from_file("properties.tif")
tau = props.data[0]  # optical thickness
```

---

## [`Sentinel2Scene`][clouds_decoded.data.sentinel.Sentinel2Scene]

Reads Sentinel-2 `.SAFE` directories and provides access to bands, geometry, and auxiliary data. Source: `src/shared_utils/data/sentinel.py`.

### Loading a scene

```python
from clouds_decoded.data import Sentinel2Scene

scene = Sentinel2Scene()
scene.read("/path/to/scene.SAFE")

# With a spatial crop (B02 pixel coordinates)
scene.read("/path/to/scene.SAFE", crop_window=(col_off, row_off, width, height))
```

### Key attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `bands` | [`BandDict`][clouds_decoded.data.band.BandDict] | Dict of [`Sentinel2Band`][clouds_decoded.data.band.Sentinel2Band] objects (lazy numpy proxies) |
| `footprints` | dict | Detector footprint IDs per band |
| `transform` | Affine | B02 reference transform |
| `crs` | CRS | Coordinate reference system |
| `quantification_value` | float | DN to reflectance denominator |
| `radio_add_offset` | dict | Per-band radiometric offset |
| `sun_zenith`, `sun_azimuth` | ndarray | Sun angle grids |
| `view_zenith`, `view_azimuth` | ndarray | View angle grids |
| `sensing_time` | datetime | Acquisition time |
| `latitude`, `longitude` | float | Scene centre coordinates |

### Key methods

- [`get_band()`][clouds_decoded.data.sentinel.Sentinel2Scene.get_band] -- Raw DN as `np.ndarray` (2D). With `reflectance=True`, returns calibrated reflectance (cached). With `resolution=20`, resamples to target resolution.
- [`get_bands()`][clouds_decoded.data.sentinel.Sentinel2Scene.get_bands] -- Batch retrieval returning `List[Sentinel2Band]`. Use `n_workers > 1` for parallel loading.
- [`get_wind_data()`][clouds_decoded.data.sentinel.Sentinel2Scene.get_wind_data] -- Read ECMWF wind data from AUX_ECMWFT GRIB file.
- [`get_angles_at_pixels()`][clouds_decoded.data.sentinel.Sentinel2Scene.get_angles_at_pixels] -- Interpolate sun/view angles from 5 km grids.

---

## [`Sentinel2Band`][clouds_decoded.data.band.Sentinel2Band]

Lazy numpy proxy for a single band. Source: `src/shared_utils/data/band.py`.

Acts as an ndarray via `__array__` protocol -- can be used directly in numpy operations. Access the underlying array with `.data`.

Key methods:

- `to_reflectance(quantification_value, radio_add_offset)` -- Convert DN to reflectance.
- `to_resolution(target_resolution)` -- Resample to a different pixel size.
