# refl2prop/model.py
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

class InversionNet(nn.Module):
    def __init__(self, input_size: int = 17, output_size: int = 4, noise_output_size: int = 11, hidden_dim: int = 512):
        """
        Neural network with dual heads for physics prediction and noise reconstruction.

        Args:
            input_size: Number of input features (default 17)
            output_size: Number of physical outputs (default 4)
            noise_output_size: Number of noise components to reconstruct (default 11)
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
    Generates normalized noise for OOD detection training.

    Noise is applied only to:
    - Indices [0:6]: Reflectance bands
    - Indices [12:17]: Geometry (incidence_angle, shading_ratio, cloud_top_height, mu, phi)

    No noise on indices [6:12]: Albedos
    """
    # Indices where noise is applied
    NOISE_INDICES = list(range(6))  # [0,1,2,3,4,5,12,13,14,15,16]
    NOISE_SIZE = 6  # Total number of noise components

    def __init__(self, input_stats: Dict[str, list], noise_min: float = 0.001, noise_max: float = 0.2):
        """
        Args:
            input_stats: Dictionary with 'min' and 'max' lists for input normalization
            noise_min: Minimum noise magnitude as fraction of variable range
            noise_max: Maximum noise magnitude as fraction of variable range
        """
        self.in_min = torch.tensor(input_stats['min'], dtype=torch.float32)
        self.in_max = torch.tensor(input_stats['max'], dtype=torch.float32)
        self.ranges = self.in_max - self.in_min

        # Avoid division by zero
        self.ranges[self.ranges == 0] = 1.0

        self.noise_min = noise_min
        self.noise_max = noise_max

    def generate(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate noise to add to inputs and the corresponding target noise vector.

        Args:
            batch_size: Number of samples in the batch
            device: Device to create tensors on

        Returns:
            noise_to_add: Full-size noise vector (batch_size, 17) with zeros for albedo indices
            target_noise: Ground truth noise in normalized space (batch_size, 11)
        """
        # Create full-size noise tensor (all zeros initially)
        noise_to_add = torch.zeros(batch_size, len(self.in_min), device=device)

        # Sample noise magnitude for each component (uniform between noise_min and noise_max)
        noise_scales = torch.rand(batch_size, self.NOISE_SIZE, device=device) * (self.noise_max - self.noise_min) + self.noise_min

        # Generate Gaussian noise
        raw_noise = torch.randn(batch_size, self.NOISE_SIZE, device=device)

        # Move stats to device if needed
        ranges_device = self.ranges.to(device)

        # Scale noise by variable ranges and noise magnitude
        # Extract ranges for noise indices
        noise_ranges = ranges_device[self.NOISE_INDICES]
        scaled_noise = raw_noise * noise_scales * noise_ranges.unsqueeze(0)

        # Place scaled noise into the appropriate indices
        noise_to_add[:, self.NOISE_INDICES] = scaled_noise

        # Create target noise in normalized space [-1, 1]
        # Normalize the scaled noise by the ranges
        target_noise = 2 * scaled_noise / noise_ranges.unsqueeze(0)
        target_noise = torch.clamp(target_noise, -1, 1)  # Ensure within bounds

        return noise_to_add, target_noise

class NormalizationWrapper(nn.Module):
    """
    Wraps the InversionNet to handle normalization internally.
    Stats are registered as buffers, so they save/load automatically with torch.save().
    """
    def __init__(self, model: nn.Module, input_stats: Dict[str, list], output_stats: Dict[str, list]):
        super().__init__()
        self.model = model

        # Register constants as buffers (non-trainable tensors)
        self.register_buffer('in_min', torch.tensor(input_stats['min'], dtype=torch.float32))
        self.register_buffer('in_max', torch.tensor(input_stats['max'], dtype=torch.float32))

        self.register_buffer('out_min', torch.tensor(output_stats['min'], dtype=torch.float32))
        self.register_buffer('out_max', torch.tensor(output_stats['max'], dtype=torch.float32))

    def normalize_input(self, x):
        # Scale to [-1, 1]
        # Epsilon to avoid div by zero if max == min
        denom = (self.in_max - self.in_min)
        denom[denom == 0] = 1.0
        return 2 * (x - self.in_min) / denom - 1

    def denormalize_output(self, y_norm):
        # Scale back from [-1, 1] to physical units
        return (y_norm + 1) / 2 * (self.out_max - self.out_min) + self.out_min

    def forward(self, x, return_uncertainty: bool = False):
        """
        Inference Mode: Takes raw physical inputs -> Returns raw physical outputs and optionally uncertainty.

        Args:
            x: Input tensor (batch_size, input_size)
            return_uncertainty: If True, returns (physics, uncertainty_score)

        Returns:
            If return_uncertainty=False: physics predictions (denormalized)
            If return_uncertainty=True: (physics predictions, uncertainty score)
        """
        x_norm = self.normalize_input(x)
        physics_norm, noise_norm = self.model(x_norm)
        physics = self.denormalize_output(physics_norm)

        if return_uncertainty:
            # Compute uncertainty as L2 norm of reconstructed noise
            # Since noise is in [-1, 1] range, this gives a measure of OOD distance
            uncertainty = torch.norm(noise_norm, dim=1)
            return physics, uncertainty

        return physics

    @property
    def ranges(self):
        # Return dictionary of input/output ranges
        in_ranges = (list(zip(self.in_min.cpu().numpy(), self.in_max.cpu().numpy())))
        out_ranges = (list(zip(self.out_min.cpu().numpy(), self.out_max.cpu().numpy())))
        return {'input_ranges': in_ranges, 'output_ranges': out_ranges}