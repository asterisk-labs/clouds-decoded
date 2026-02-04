# Uncertainty Quantification & OOD Detection

This module now includes uncertainty quantification capabilities using noise reconstruction for out-of-distribution (OOD) detection.

## Overview

The model is trained with a dual-head architecture:
1. **Physics Head**: Predicts the 4 physical parameters (tau, ice_liq_ratio, r_eff_liq, r_eff_ice)
2. **Noise Head**: Reconstructs the noise that was added to inputs during training

During inference on real data (with no added noise), the noise head's output indicates how "noisy" or out-of-distribution the input is compared to the idealized LUT distribution.

## Why This Works

### The Problem
- Training data comes from idealized forward model (LUT)
- Real satellite data has:
  - 3D shading effects
  - Forward model errors/simplifications
  - Sensor noise and calibration errors
  - Atmospheric effects not in the LUT

### The Solution
- During training: Add controlled noise to inputs (reflectances + geometry)
- Model learns to separate signal from noise on the LUT distribution
- At inference: Model tries to "denoise" the input
  - If input is like LUT → reconstructed noise ≈ 0
  - If input is OOD → reconstructed noise is large

## Training

### Basic Training
```bash
python -m src.modules.refl2prop.train \
    --lut path/to/lut.nc \
    --output model_with_uncertainty.pth \
    --epochs 100 \
    --batch_size 128
```

### Advanced Options
```bash
python -m src.modules.refl2prop.train \
    --lut path/to/lut.nc \
    --output model.pth \
    --noise_min 0.001 \        # Min noise magnitude (fraction of variable range)
    --noise_max 0.2 \          # Max noise magnitude (fraction of variable range)
    --physics_weight 1.0 \     # Weight for physics loss
    --noise_weight 1.0         # Weight for noise reconstruction loss
```

### What Gets Noised?

Noise is applied to 11 input features:
- **6 Reflectance bands** (B01, B02, B04, B08, B11, B12)
- **5 Geometry features** (incidence_angle, shading_ratio, cloud_top_height, mu, phi)

**No noise** is added to:
- **6 Albedo features** (these are considered known priors)

Rationale: For low optical depth, reflectance → albedo (degeneracy). Since albedo space already covers all possibilities, OOD detection on albedo is not meaningful.

## Inference

### Basic Inference with Uncertainty
```python
import torch
from src.modules.refl2prop.model import InversionNet, NormalizationWrapper

# Load model
model = torch.load('model_with_uncertainty.pth')
model.eval()

# Prepare input (batch_size, 17)
# [6 reflectances, 6 albedos, 5 geometry]
inputs = torch.randn(100, 17)  # Example

# Get predictions with uncertainty
predictions, uncertainty = model(inputs, return_uncertainty=True)

# predictions: (batch_size, 4) - [tau, ice_liq_ratio, r_eff_liq, r_eff_ice]
# uncertainty: (batch_size,) - L2 norm of reconstructed noise
```

### Calibrating Uncertainty Threshold

Use validation data to calibrate what "high uncertainty" means:

```python
from src.modules.refl2prop.uncertainty import (
    calibrate_uncertainty_threshold,
    get_uncertainty_statistics
)

# Run inference on validation data
val_predictions, val_uncertainties = model(val_inputs, return_uncertainty=True)
val_uncertainties = val_uncertainties.cpu().numpy()

# Get statistics
stats = get_uncertainty_statistics(val_uncertainties)
print(f"Validation uncertainty: mean={stats['mean']:.4f}, p95={stats['p95']:.4f}")

# Set threshold at 95th percentile
threshold = calibrate_uncertainty_threshold(val_uncertainties, percentile=95.0)
print(f"Uncertainty threshold: {threshold:.4f}")
```

### Flagging OOD Samples

```python
from src.modules.refl2prop.uncertainty import flag_ood_samples

# Run inference on real data
real_predictions, real_uncertainties = model(real_inputs, return_uncertainty=True)
real_uncertainties = real_uncertainties.cpu().numpy()

# Flag OOD samples
ood_indices, ood_mask = flag_ood_samples(
    real_uncertainties,
    threshold,
    return_mask=True
)

print(f"Flagged {len(ood_indices)} / {len(real_uncertainties)} as OOD")

# Filter predictions based on uncertainty
filtered_predictions = real_predictions.cpu().numpy()
filtered_predictions[ood_mask] = np.nan  # Mask out high-uncertainty predictions
```

### Alternative Uncertainty Metrics

```python
from src.modules.refl2prop.uncertainty import compute_uncertainty_score

# Get raw noise output
norm_inputs = model.normalize_input(inputs)
physics_pred, noise_output = model.model(norm_inputs)

# Try different uncertainty metrics
unc_l2 = compute_uncertainty_score(noise_output, method='l2')      # L2 norm (default)
unc_l1 = compute_uncertainty_score(noise_output, method='l1')      # L1 norm
unc_linf = compute_uncertainty_score(noise_output, method='linf')  # Max absolute value
unc_mean = compute_uncertainty_score(noise_output, method='mean_abs')  # Mean absolute
```

## Interpreting Uncertainty Scores

### Typical Ranges (after calibration on validation data)
- **Low uncertainty** (< p50): Input very similar to LUT, high confidence
- **Medium uncertainty** (p50-p90): Input somewhat different from LUT, moderate confidence
- **High uncertainty** (p90-p95): Input differs from LUT, lower confidence
- **Very high uncertainty** (> p95): Likely OOD, consider flagging/masking

### What Causes High Uncertainty?
1. **3D radiative effects**: Shadows, illumination geometry not in 1D LUT
2. **Broken cloud fields**: Mixed clear/cloudy scenes
3. **Multi-layer clouds**: LUT assumes single layer
4. **Extreme geometry**: Solar/viewing angles outside training range
5. **Surface effects**: Terrain, coastlines, unusual albedos
6. **Sensor issues**: Calibration errors, saturation, noise

## Complete Example

```python
import torch
import numpy as np
from src.modules.refl2prop.model import NormalizationWrapper
from src.modules.refl2prop.uncertainty import (
    calibrate_uncertainty_threshold,
    flag_ood_samples,
    uncertainty_based_filtering
)

# 1. Load model
model = torch.load('model_with_uncertainty.pth')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. Calibrate on validation data
val_inputs = torch.randn(1000, 17).to(device)  # Your validation data
with torch.no_grad():
    _, val_unc = model(val_inputs, return_uncertainty=True)
val_unc_np = val_unc.cpu().numpy()

threshold = calibrate_uncertainty_threshold(val_unc_np, percentile=95.0)
print(f"Uncertainty threshold (95th percentile): {threshold:.4f}")

# 3. Inference on real data
real_inputs = torch.randn(10000, 17).to(device)  # Your real satellite data
with torch.no_grad():
    predictions, uncertainties = model(real_inputs, return_uncertainty=True)

pred_np = predictions.cpu().numpy()
unc_np = uncertainties.cpu().numpy()

# 4. Filter based on uncertainty
filtered_preds = uncertainty_based_filtering(
    pred_np, unc_np, threshold, fill_value=np.nan
)

ood_frac = np.sum(unc_np > threshold) / len(unc_np)
print(f"Fraction flagged as OOD: {ood_frac:.2%}")
```

## Model Architecture Details

### Input (17 features)
```
[0:6]   - Reflectances (B01, B02, B04, B08, B11, B12)
[6:12]  - Surface Albedos (same 6 bands)
[12:17] - Geometry (incidence_angle, shading_ratio, cloud_top_height, mu, phi)
```

### Outputs
```
Physics Head: 4 outputs (tau, ice_liq_ratio, r_eff_liq, r_eff_ice)
Noise Head:   11 outputs (noise for reflectances + geometry, not albedos)
```

### Architecture
```
Input (17)
   ↓
Shared Encoder:
   Linear(17 → 512) + LeakyReLU + BatchNorm
   Linear(512 → 256) + LeakyReLU + BatchNorm
   Linear(256 → 128) + LeakyReLU + BatchNorm
   ↓
   ├─→ Physics Head:
   │      Linear(128 → 64) + LeakyReLU + BatchNorm
   │      Linear(64 → 64) + LeakyReLU + BatchNorm
   │      Linear(64 → 4) + Tanh → [tau, ice_liq, r_eff_liq, r_eff_ice]
   │
   └─→ Noise Head:
          Linear(128 → 64) + LeakyReLU + BatchNorm
          Linear(64 → 64) + LeakyReLU + BatchNorm
          Linear(64 → 11) + Tanh → [noise components]
```

### Loss Function
```
L_total = w_phys * L_physics + w_noise * L_noise

L_physics = physics_masked_MSE(pred, target)  # Masks invalid phase retrievals
L_noise = MSE(noise_pred, noise_target)
```

## Tips & Best Practices

1. **Start with balanced weights** (`--physics_weight 1.0 --noise_weight 1.0`)
2. **Tune noise range** based on your data:
   - Higher noise (0.1-0.3) → more robust to OOD, but may hurt in-distribution performance
   - Lower noise (0.001-0.1) → better in-distribution, less OOD detection
3. **Always calibrate** thresholds on representative validation data
4. **Monitor both losses** during training to ensure neither dominates
5. **Visualize** uncertainty spatially to identify systematic issues (e.g., coastlines)
6. **Consider adaptive thresholds** based on scene characteristics (e.g., lower threshold for complex scenes)
