# refl2prop/model_shading.py
"""Shading-aware neural network for cloud property inversion from pixel bags."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ShadingAwareInversionNet(nn.Module):
    """
    Neural network for shading-aware cloud property inversion.

    Takes a bag of shaded pixel observations and predicts:
    - Per-pixel tau_effective_shading
    - Global (shared) physical properties

    Architecture:
    1. Per-pixel encoder (shared weights)
    2. Self-attention layers for inter-pixel context
    3. Per-pixel shading head
    4. Global pooling + global properties head
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_attention_layers: int = 2,
        output_size: int = 4,
    ):
        """
        Args:
            input_size: Per-pixel input dimension (reflectances + albedos + geometry)
            hidden_dim: Hidden layer dimension
            n_heads: Number of attention heads
            n_attention_layers: Number of self-attention layers
            output_size: Number of global physical properties (default 4)
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Per-pixel encoder (shared across bag)
        self.pixel_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim),
        )

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(n_attention_layers)
        ])
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_attention_layers)
        ])
        self.attention_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(n_attention_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_attention_layers)
        ])

        # Per-pixel shading head
        # Outputs log(tau_shading) - exponentiated in wrapper for positive values
        self.shading_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Global aggregation and properties head
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim // 2),
        )

        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, output_size),
            nn.Tanh()  # Normalized output [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch_size, bag_size, input_size) - bag of pixel observations

        Returns:
            shading_pred: (batch_size, bag_size) - per-pixel tau_shading (normalized)
            global_pred: (batch_size, output_size) - global physical properties (normalized)
        """
        batch_size, bag_size, _ = x.shape

        # Encode each pixel
        encoded = self.pixel_encoder(x)  # (batch, bag_size, hidden_dim)

        # Apply self-attention with residual connections
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.attention_layers,
            self.attention_norms,
            self.attention_ffn,
            self.ffn_norms
        ):
            # Self-attention + residual
            attended, _ = attn(encoded, encoded, encoded)
            encoded = attn_norm(encoded + attended)

            # Feed-forward + residual
            ff_out = ffn(encoded)
            encoded = ffn_norm(encoded + ff_out)

        # Per-pixel shading prediction
        shading_pred = self.shading_head(encoded).squeeze(-1)  # (batch, bag_size)

        # Global properties via mean pooling + head
        pooled = encoded.mean(dim=1)  # (batch, hidden_dim)
        pooled = self.global_pool(pooled)
        global_pred = self.global_head(pooled)  # (batch, 4)

        return shading_pred, global_pred


class ShadingNormalizationWrapper(nn.Module):
    """
    Wraps ShadingAwareInversionNet with input/output normalization.

    Stats are registered as buffers, so they save/load automatically with torch.save().
    """

    def __init__(
        self,
        model: nn.Module,
        input_stats: Dict[str, list],
        output_stats: Dict[str, list],
        shading_stats: Dict[str, float],
    ):
        """
        Args:
            model: The ShadingAwareInversionNet model
            input_stats: Dictionary with 'min' and 'max' lists for input normalization
            output_stats: Dictionary with 'min' and 'max' lists for global output normalization
            shading_stats: Dictionary with 'min' and 'max' for tau_shading normalization
        """
        super().__init__()
        self.model = model

        # Input normalization buffers
        self.register_buffer('in_min', torch.tensor(input_stats['min'], dtype=torch.float32))
        self.register_buffer('in_max', torch.tensor(input_stats['max'], dtype=torch.float32))

        # Global output normalization buffers
        self.register_buffer('out_min', torch.tensor(output_stats['min'], dtype=torch.float32))
        self.register_buffer('out_max', torch.tensor(output_stats['max'], dtype=torch.float32))

        # Shading normalization buffers
        self.register_buffer('shading_min', torch.tensor(shading_stats['min'], dtype=torch.float32))
        self.register_buffer('shading_max', torch.tensor(shading_stats['max'], dtype=torch.float32))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input to [-1, 1]."""
        denom = (self.in_max - self.in_min)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        return 2 * (x - self.in_min) / denom - 1

    def denormalize_shading(self, log_shading: torch.Tensor) -> torch.Tensor:
        """
        Convert log-space shading prediction to physical tau_shading.

        Model outputs log(tau_shading), we exponentiate to get positive values.
        Clipped to [eps, shading_max] for stability.
        """
        eps = 1e-3
        tau_shading = torch.exp(log_shading)
        return torch.clamp(tau_shading, eps, self.shading_max)

    def normalize_shading(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert physical tau_shading to log-space for loss computation.

        Takes physical tau_shading values and returns log(tau_shading).
        """
        eps = 1e-3
        return torch.log(y + eps)

    def denormalize_global(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Scale global properties from [-1, 1] to physical units."""
        return (y_norm + 1) / 2 * (self.out_max - self.out_min) + self.out_min

    def normalize_global(self, y: torch.Tensor) -> torch.Tensor:
        """Scale global properties from physical units to [-1, 1]."""
        denom = (self.out_max - self.out_min)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        return 2 * (y - self.out_min) / denom - 1

    def forward(
        self,
        x: torch.Tensor,
        return_denormalized: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional denormalization.

        Args:
            x: (batch, bag_size, input_dim) raw inputs
            return_denormalized: If True, return physical values; if False, return normalized

        Returns:
            shading: (batch, bag_size) tau_shading values
            global_props: (batch, 4) physical properties
        """
        x_norm = self.normalize_input(x)
        shading_norm, global_norm = self.model(x_norm)

        if return_denormalized:
            shading = self.denormalize_shading(shading_norm)
            global_props = self.denormalize_global(global_norm)
            return shading, global_props

        return shading_norm, global_norm

    @property
    def ranges(self):
        """Return dictionary of input/output ranges."""
        in_ranges = list(zip(self.in_min.cpu().numpy(), self.in_max.cpu().numpy()))
        out_ranges = list(zip(self.out_min.cpu().numpy(), self.out_max.cpu().numpy()))
        shading_range = (self.shading_min.cpu().item(), self.shading_max.cpu().item())
        return {
            'input_ranges': in_ranges,
            'output_ranges': out_ranges,
            'shading_range': shading_range,
        }
