"""Base normalization wrapper for neural network models.

Provides a shared ``NormalizationWrapper`` that registers input/output
min-max statistics as buffers and exposes ``normalize_input`` /
``denormalize_output`` methods.  Subclasses override these when their
model uses a different convention (sigmoid, linear scaling, etc.).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn


class NormalizationWrapper(nn.Module):
    """Wraps a model with input/output min-max normalization.

    Statistics are registered as non-trainable buffers so they persist
    in checkpoints created with ``torch.save(wrapper.state_dict(), ...)``.

    Default behaviour (suitable for tanh-activated outputs):

    * **Input**:  min-max scale to [-1, 1].
    * **Output**: inverse min-max from [-1, 1] back to physical units.

    Subclasses may override ``normalize_input``, ``denormalize_output``,
    and ``forward`` for different activation conventions.

    Args:
        model: The core ``nn.Module`` to wrap.
        input_stats: Dict with ``'min'`` and ``'max'`` lists for input
            normalization.
        output_stats: Dict with ``'min'`` and ``'max'`` lists for output
            denormalization.
    """

    def __init__(
        self,
        model: nn.Module,
        input_stats: Dict[str, list],
        output_stats: Dict[str, list],
    ) -> None:
        super().__init__()
        self.model = model

        self.register_buffer(
            "in_min", torch.tensor(input_stats["min"], dtype=torch.float32),
        )
        self.register_buffer(
            "in_max", torch.tensor(input_stats["max"], dtype=torch.float32),
        )
        self.register_buffer(
            "out_min", torch.tensor(output_stats["min"], dtype=torch.float32),
        )
        self.register_buffer(
            "out_max", torch.tensor(output_stats["max"], dtype=torch.float32),
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Scale raw inputs to [-1, 1] via min-max normalization."""
        denom = (self.in_max - self.in_min).clamp(min=1e-8)
        return 2.0 * (x - self.in_min) / denom - 1.0

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Scale tanh [-1, 1] outputs back to physical units."""
        return (y_norm + 1) / 2 * (self.out_max - self.out_min) + self.out_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input, run model, denormalize output."""
        return self.denormalize_output(self.model(self.normalize_input(x)))

    @property
    def ranges(self) -> Dict[str, List[Tuple[float, float]]]:
        """Return input/output min-max ranges as lists of (min, max) tuples."""
        in_ranges = list(zip(
            self.in_min.cpu().numpy().tolist(),
            self.in_max.cpu().numpy().tolist(),
        ))
        out_ranges = list(zip(
            self.out_min.cpu().numpy().tolist(),
            self.out_max.cpu().numpy().tolist(),
        ))
        return {"input_ranges": in_ranges, "output_ranges": out_ranges}


class CloudHeightNormalizationWrapper(NormalizationWrapper):
    """NormalizationWrapper subclass for cloud height inversion models.

    Overrides ``forward`` to return both physical (denormalized) and
    normalized regression outputs.  When the wrapped model returns a dict
    containing a ``'regression'`` key, the normalized prediction is kept
    under ``'regression_norm'`` so that the training loss can operate in
    normalized space while downstream consumers receive metres AGL.
    """

    def forward(self, x):
        """Normalize input, run model, and return physical + normalized outputs."""
        x_norm = self.normalize_input(x)
        output = self.model(x_norm)
        
        if isinstance(output, dict):
            if "regression" in output:
                output["regression_norm"] = output["regression"]
                output["regression"] = self.denormalize_output(output["regression"])
        else:
            output = self.denormalize_output(output)
            
        return output
