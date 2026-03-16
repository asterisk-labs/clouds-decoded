# Configuration Reference

## Config System Overview

All processor configs inherit from [`BaseProcessorConfig`][clouds_decoded.config.BaseProcessorConfig] (see `src/shared_utils/config.py`), which is a Pydantic `BaseModel` with:

- **Strict validation**: `extra='forbid'` -- unknown fields cause a validation error.
- **YAML serialization**: [`from_yaml()`][clouds_decoded.config.BaseProcessorConfig.from_yaml] to load, [`to_yaml()`][clouds_decoded.config.BaseProcessorConfig.to_yaml] to save. Computed fields are excluded from YAML output.
- **Type checking**: all fields are type-annotated with Pydantic `Field` descriptors including constraints (`ge=`, `le=`) and descriptions.

### Using config files

Pass a YAML config to any CLI command with `--config-path`:

```bash
clouds-decoded cloud-mask scene.SAFE --config-path my_mask_config.yaml
```

Generate a default config file by instantiating the config class in Python and calling `to_yaml()`:

```python
from clouds_decoded.modules.cloud_mask.config import CloudMaskConfig
CloudMaskConfig().to_yaml("my_mask_config.yaml")
```

Unknown fields in a YAML file will raise a validation error. This prevents silent typos from going unnoticed.

---

## [`BaseProcessorConfig`][clouds_decoded.config.BaseProcessorConfig]

Base class for all processor configs. Source: `src/shared_utils/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | `Optional[str]` | `None` | Directory to save outputs |
| `n_workers` | `int` | `1` | Number of parallel workers |
| `working_resolution` | `Optional[int]` | `None` | Resolution in metres for inference (None = processor default). Min 10. |
| `output_resolution` | `Optional[int]` | `None` | Output resolution in metres. Must be >= `working_resolution`. Min 10. |

---

## [`CloudMaskConfig`][clouds_decoded.modules.cloud_mask.config.CloudMaskConfig]

4-class cloud segmentation. Source: `src/modules/cloud_mask/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `Literal["senseiv2", "threshold"]` | `"senseiv2"` | Detection method |
| `model_path` | `Optional[str]` | None (resolved from managed assets) | Path to model weights (.pt) |
| `device` | `Optional[str]` | None (auto-detect) | Compute device: `"cuda"`, `"cpu"`, or None |
| `batch_size` | `int` | `8` | Batch size for inference (1--64) |
| `working_resolution` | `int` | `10` | Inference resolution in metres (10--60) |
| `stride` | `int` | `128` | Tiling stride in pixels (1--256) |
| `threshold_band` | `str` | `"B08"` | Band for threshold method |
| `threshold_value` | `float` | `0.06` | Reflectance threshold (0--1) |

---

## [`CloudHeightConfig`][clouds_decoded.modules.cloud_height.config.CloudHeightConfig]

Physics-based stereo parallax retrieval. Source: `src/modules/cloud_height/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reference_band` | `str` | `"B02"` | Reference band (zero parallax) |
| `bands` | `List[str]` | `["B01","B02","B03","B04","B05","B06","B07","B08"]` | Bands for correlation (min 2) |
| `cloudy_thresh` | `float` | `0.06` | Cloud detection reflectance threshold (0--1) |
| `threshold_band` | `str` | `"B08"` | Band for cloud thresholding |
| `along_track_resolution` | `int` | `3` | Along-track pixel size during convolution (metres) |
| `across_track_resolution` | `int` | `10` | Across-track pixel size (metres) |
| `stride` | `int` | `180` | Stride between retrieval points (metres) |
| `grid_resolution` | `Optional[int]` | None | Output grid pixel size in metres (None = stride) |
| `convolved_size_along_track` | `int` | `200` | Correlation window along track (metres) |
| `convolved_size_across_track` | `int` | `200` | Correlation window across track (metres) |
| `correlation_weighting` | `bool` | `True` | Weight by correlation strength |
| `spatial_smoothing_sigma` | `float` | `180.0` | Gaussian smoothing sigma (metres, 0 = none) |
| `max_height` | `int` | `18000` | Maximum search height (metres) |
| `height_step` | `int` | `100` | Height search step size (metres) |
| `use_emulator` | `bool` | `False` | Use DL emulator instead |
| `n_workers` | `int` | `96` | Parallel workers |
| `temp_dir` | `Optional[str]` | None | Temp directory (default: /dev/shm) |

---

## [`CloudHeightEmulatorConfig`][clouds_decoded.modules.cloud_height_emulator.config.CloudHeightEmulatorConfig]

Deep learning cloud height emulator (ResUNet). Source: `src/modules/cloud_height_emulator/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_emulator` | `bool` | `True` | Flag for emulator mode |
| `model_path` | `Optional[str]` | None (resolved from managed assets) | Path to model weights (.pth) |
| `bands` | `List[str]` | `["B02","B03","B04","B08","B11","B12","B09","B10"]` | Input bands |
| `window_size` | `Tuple[int, int]` | `(1024, 1024)` | Sliding window size (must be square) |
| `overlap` | `int` | `512` | Window overlap in pixels |
| `batch_size` | `int` | `4` | Inference batch size |
| `in_channels` | `int` | `8` | Number of input channels |
| `device` | `Optional[str]` | None (auto-detect) | Compute device |
| `working_resolution` | `int` | `10` | Inference resolution in metres |
| `output_resolution` | `Optional[int]` | `60` | Output resolution in metres |
| `cloud_mask_classes` | `List[int]` | `[1, 2]` | Cloud mask classes to treat as cloud |
| `cloud_mask_threshold` | `float` | `0.2` | Probability threshold for cloud detection |

---

## [`AlbedoEstimatorConfig`][clouds_decoded.modules.albedo_estimator.config.AlbedoEstimatorConfig]

Surface albedo estimation. Source: `src/modules/albedo_estimator/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `Literal["idw", "datadriven"]` | `"idw"` | Estimation method |
| `fallback` | `Literal["datadriven", "constant"]` | `"datadriven"` | Fallback when insufficient clear pixels |
| `min_clear_fraction` | `float` | `0.05` | Min clear-sky fraction before fallback |
| `cloud_mask_classes` | `List[int]` | `[1, 2, 3]` | Classes to treat as cloud |
| `cloud_mask_threshold` | `float` | `0.5` | Probability threshold for cloud mask |
| `output_resolution` | `int` | `300` | Output grid resolution in metres (10--1000) |
| `max_samples` | `int` | `1000` | Max clear-sky pixels for IDW fitting |
| `window_m` | `float` | `180.0` | Spatial averaging window in metres |
| `idw_cloud_mask_dilation_m` | `float` | `100.0` | Cloud mask dilation distance in metres |
| `idw_k_neighbours` | `int` | `8` | Nearest neighbours per output pixel |
| `idw_smoothing_m` | `float` | `2000.0` | IDW regularisation distance in metres |
| `model_path` | `Optional[str]` | None (resolved from managed assets) | Path to trained MLP checkpoint |
| `default_albedo` | `Dict[str, float]` | per-band defaults | Constant fallback albedo per band |

---

## [`RefocusConfig`][clouds_decoded.modules.refocus.config.RefocusConfig]

Parallax correction. Source: `src/modules/refocus/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reference_band` | `str` | `"B02"` | Reference band (zero parallax) |
| `bands` | `Optional[List[str]]` | 11 bands (excludes B09, B10) | Bands to refocus (None = all) |
| `output_resolution` | `Optional[int]` | `60` | Output resolution (None = native per band) |
| `save_refocused` | `bool` | `False` | Save refocused bands as GeoTIFFs |
| `interpolation_order` | `int` | `1` | Band warping interpolation (0=nearest, 1=bilinear, 3=cubic) |
| `height_interpolation_order` | `int` | `1` | Height map upsampling interpolation |

---

## [`Refl2PropConfig`][clouds_decoded.modules.refl2prop.config.Refl2PropConfig]

Cloud property inversion. Source: `src/modules/refl2prop/config.py`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `Literal["standard", "shading"]` | `"standard"` | Inversion method |
| `bands` | `List[str]` | 11 bands (B01--B12 excl. B09, B10) | Input bands |
| `model_path` | `Optional[str]` | None (resolved from managed assets) | Path to model checkpoint |
| `return_uncertainty` | `bool` | `True` | Append uncertainty channel |
| `mask_invalid_height` | `bool` | `True` | Mask pixels with height <= 0 |
| `batch_size` | `int` | `32768` | Inference batch size (pixels) |
| `working_resolution` | `int` | `60` | Inference resolution in metres |
| `output_resolution` | `int` | `60` | Output resolution in metres |
| `output_size` | `int` | `4` | Number of output properties |
| `output_feature_names` | `List[str]` | `["tau", "ice_liq_ratio", "r_eff_liq", "r_eff_ice"]` | Output band names |
| `default_albedo` | `Dict[str, float]` | per-band defaults | Fallback albedo values |
| `default_shading_ratio` | `float` | `0.5` | Fallback shading ratio |

**Computed fields** (derived from `bands`, not editable in YAML):

- `num_bands` -- number of spectral bands
- `input_size` -- total NN input size (2 * num_bands + 5 geometry features)
- `noise_output_size` -- noise channels (= num_bands)
- `input_feature_names` -- ordered list of all input feature names

### [`ShadingRefl2PropConfig`][clouds_decoded.modules.refl2prop.config.ShadingRefl2PropConfig]

Extends [`Refl2PropConfig`][clouds_decoded.modules.refl2prop.config.Refl2PropConfig] with additional fields for bag-based shading-aware inversion:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | `int` | `24` | Square processing window in pixels |
| `stride` | `int` | `8` | Step size between windows |
| `hidden_dim` | `int` | `256` | Hidden dimension for shading model |
| `n_heads` | `int` | `4` | Number of attention heads |
| `n_attention_layers` | `int` | `2` | Number of self-attention layers |
