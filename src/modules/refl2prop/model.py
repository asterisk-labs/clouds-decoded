# refl2prop/model.py
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

from clouds_decoded.normalization import NormalizationWrapper as _BaseNormalizationWrapper

class InversionNet(nn.Module):
    def __init__(self, input_size: int, output_size: int = 4, noise_output_size: int = 11, hidden_dim: int = 512):
        """
        Neural network with dual heads for physics prediction and noise reconstruction.

        Args:
            input_size: Number of input features (computed from config.input_size)
            output_size: Number of physical outputs (default 4)
            noise_output_size: Number of noise components to reconstruct (from config.noise_output_size)
            hidden_dim: Hidden layer dimension (default 512)
        """
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim // 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim // 4),
        )

        # Physics prediction head (specialized layers)
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),

            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),

            nn.Linear(64, output_size),
            nn.Tanh()  # Normalizes outputs to [-1, 1] range
        )

        # Noise reconstruction head (specialized layers)
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),

            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),

            nn.Linear(64, noise_output_size),
            nn.Tanh()  # Normalizes noise outputs to [-1, 1] range
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both physics predictions and noise reconstruction.

        Returns:
            physics: Predicted physical values (batch_size, 4)
            noise: Reconstructed noise components (batch_size, 11)
        """
        features = self.encoder(x)
        physics = self.physics_head(features)
        noise = self.noise_head(features)
        return physics, noise


class NoiseGenerator:
    """
    Generates structured noise for OOD detection training that simulates
    realistic differences between forward model outputs and measured reflectances.

    Noise model: R_noisy = R_clean * (gain_global * gain_band) + offset_global + offset_band

    Components:
        - Global gain: Scene brightness mismatch (affects all bands similarly)
        - Per-band gain: Calibration/atmospheric variations per wavelength
        - Global offset: Atmospheric path radiance (correlated across bands)
        - Per-band offset: Independent measurement noise

    Noise is applied to reflectance bands only (indices [0:num_bands]).
    No noise is applied to albedos or geometry features.
    """

    def __init__(
        self,
        input_stats: Dict[str, list],
        noise_indices: list,
        noise_min: float = 0.001,
        noise_max: float = 0.02,
        # Global gain: scene brightness mismatch
        gain_global_mean: float = 1.0,
        gain_global_std: float = 0.05,
        # Per-band gain: calibration/spectral variations
        gain_band_std: float = 0.02,
        # Global offset: atmospheric path radiance (as fraction of mean reflectance)
        offset_global_std: float = 0.01,
        # Per-band offset: independent noise (legacy behavior, scaled by range)
        offset_band_min: float = 0.001,
        offset_band_max: float = 0.01,
    ):
        """
        Args:
            input_stats: Dictionary with 'min' and 'max' lists for input normalization
            noise_indices: List of input indices where noise should be applied
            noise_min: (deprecated) Min noise magnitude - use offset_band_min instead
            noise_max: (deprecated) Max noise magnitude - use offset_band_max instead
            gain_global_mean: Mean of global gain factor (default 1.0)
            gain_global_std: Std of global gain factor (default 0.05 = 5% brightness variation)
            gain_band_std: Std of per-band gain factors around 1.0 (default 0.02 = 2%)
            offset_global_std: Std of global offset as fraction of typical reflectance (default 0.01)
            offset_band_min: Min per-band offset noise magnitude (fraction of range)
            offset_band_max: Max per-band offset noise magnitude (fraction of range)
        """
        self.in_min = torch.tensor(input_stats['min'], dtype=torch.float32)
        self.in_max = torch.tensor(input_stats['max'], dtype=torch.float32)
        self.ranges = self.in_max - self.in_min

        # Avoid division by zero
        self.ranges[self.ranges == 0] = 1.0

        self.noise_indices = noise_indices
        self.noise_size = len(noise_indices)

        # Structured noise parameters
        self.gain_global_mean = gain_global_mean
        self.gain_global_std = gain_global_std
        self.gain_band_std = gain_band_std
        self.offset_global_std = offset_global_std
        self.offset_band_min = offset_band_min
        self.offset_band_max = offset_band_max

    def generate(self, batch_size: int, device: torch.device, inputs: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate structured noise and corresponding target vector.

        Args:
            batch_size: Number of samples in the batch
            device: Device to create tensors on
            inputs: Optional input tensor for input-dependent noise scaling

        Returns:
            noise_to_add: Full-size noise vector (additive form for compatibility)
            target_noise: Ground truth total perturbation in normalized space [-1, 1]
        """
        # Move stats to device
        ranges_device = self.ranges.to(device)
        in_min_device = self.in_min.to(device)
        noise_ranges = ranges_device[self.noise_indices]

        # Extract reflectance values if inputs provided (for input-dependent noise)
        if inputs is not None:
            refl_values = inputs[:, self.noise_indices]
        else:
            # Use midpoint of range as typical value
            refl_values = (in_min_device[self.noise_indices] + ranges_device[self.noise_indices] / 2).unsqueeze(0).expand(batch_size, -1)

        # 1. Global gain: single scalar per sample, affects all bands similarly
        #    Simulates scene brightness mismatch
        gain_global = torch.randn(batch_size, 1, device=device) * self.gain_global_std + self.gain_global_mean

        # 2. Per-band gain: different for each band, simulates calibration/atmospheric variations
        gain_band = torch.randn(batch_size, self.noise_size, device=device) * self.gain_band_std + 1.0

        # Combined multiplicative factor
        gain_total = gain_global * gain_band  # (batch, n_bands)

        # 3. Global offset: single value per sample, simulates atmospheric path radiance
        #    Scale by typical reflectance magnitude
        typical_refl = refl_values.mean(dim=1, keepdim=True)
        offset_global = torch.randn(batch_size, 1, device=device) * self.offset_global_std * typical_refl

        # 4. Per-band offset: independent noise for each band (original behavior)
        offset_scales = torch.rand(batch_size, self.noise_size, device=device) * (self.offset_band_max - self.offset_band_min) + self.offset_band_min
        offset_band = torch.randn(batch_size, self.noise_size, device=device) * offset_scales * noise_ranges.unsqueeze(0)

        # Compute noisy reflectances: R_noisy = R_clean * gain_total + offset_global + offset_band
        noisy_refl = refl_values * gain_total + offset_global + offset_band

        # Total additive perturbation (for compatibility with existing training loop)
        total_perturbation = noisy_refl - refl_values  # (batch, n_bands)

        # Create full-size noise tensor
        noise_to_add = torch.zeros(batch_size, len(self.in_min), device=device)
        noise_to_add[:, self.noise_indices] = total_perturbation

        # Normalize target to [-1, 1] range per band
        target_noise = 2 * total_perturbation / noise_ranges.unsqueeze(0)
        target_noise = torch.clamp(target_noise, -1, 1)

        return noise_to_add, target_noise

class NormalizationWrapper(_BaseNormalizationWrapper):
    """Wraps InversionNet with normalization and optional uncertainty output.

    Inherits min-max input normalization and tanh output denormalization
    from the shared base class.  Overrides ``forward`` to handle the
    dual-head model (physics + noise) and optional uncertainty.
    """

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """Raw physical inputs -> physical outputs, optionally with uncertainty.

        Args:
            x: Input tensor (batch_size, input_size).
            return_uncertainty: If True, returns (physics, uncertainty_score).

        Returns:
            If return_uncertainty=False: physics predictions (denormalized).
            If return_uncertainty=True: (physics predictions, uncertainty score).
        """
        x_norm = self.normalize_input(x)
        physics_norm, noise_norm = self.model(x_norm)
        physics = self.denormalize_output(physics_norm)

        if return_uncertainty:
            uncertainty = torch.norm(noise_norm, dim=1)
            return physics, uncertainty

        return physics