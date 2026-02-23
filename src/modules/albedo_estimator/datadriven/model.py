"""AlbedoNet MLP and NormalizationWrapper for data-driven albedo estimation."""
from typing import Dict

import torch
import torch.nn as nn

from clouds_decoded.normalization import NormalizationWrapper as _BaseNormalizationWrapper


class AlbedoNet(nn.Module):
    """MLP predicting per-band ocean TOA reflectance from physical conditions.

    Architecture::

        Input(14) -> [Linear + LeakyReLU + BN] x N -> Linear(13) -> Sigmoid

    The first half of hidden layers hold width; the second half narrows
    progressively (//2 each layer, floored at ``output_size``).

    Args:
        input_size: Number of engineered input features (default 14).
        output_size: Number of output bands (default 13).
        hidden_dim: Width of the first hidden layer.
        num_hidden_layers: Number of hidden blocks.
        dropout: Dropout rate after each hidden block.
    """

    def __init__(
        self,
        input_size: int = 14,
        output_size: int = 13,
        hidden_dim: int = 256,
        num_hidden_layers: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: list = []

        # Input projection
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers with progressive narrowing in the second half
        current_dim = hidden_dim
        for i in range(num_hidden_layers):
            if i >= num_hidden_layers // 2:
                next_dim = max(current_dim // 2, output_size)
            else:
                next_dim = current_dim
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.BatchNorm1d(next_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = next_dim

        self.encoder = nn.Sequential(*layers)
        self.output_head = nn.Sequential(
            nn.Linear(current_dim, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: normalised input → [0, 1] reflectance."""
        return self.output_head(self.encoder(x))


class NormalizationWrapper(_BaseNormalizationWrapper):
    """Wraps AlbedoNet with input/output min-max normalization.

    Inherits min-max input normalization from the shared base class.
    Overrides ``denormalize_output`` for Sigmoid [0, 1] model outputs.
    """

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Scale Sigmoid [0, 1] output to physical reflectance range."""
        return self.out_min + y_norm * (self.out_max - self.out_min)
