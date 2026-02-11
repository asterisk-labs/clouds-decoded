# refl2prop/loss_shading.py
"""Loss functions for shading-aware cloud property inversion."""

import torch
import torch.nn.functional as F
from typing import Tuple


def shading_loss(
    pred_log_shading: torch.Tensor,
    target_log_shading: torch.Tensor,
) -> torch.Tensor:
    """
    Per-pixel shading loss in log-space.

    Both inputs should already be in log-space (log(tau_shading)).
    Using log-space handles the wide range of tau values and
    prioritizes accuracy at low tau (more common in real imagery).

    Args:
        pred_log_shading: Predicted log(tau_shading) (batch, bag_size)
        target_log_shading: Target log(tau_shading) (batch, bag_size)

    Returns:
        Scalar loss value
    """
    return F.mse_loss(pred_log_shading, target_log_shading)


def global_physics_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_denorm: torch.Tensor
) -> torch.Tensor:
    """
    Global physical properties loss with phase-aware masking.

    Ignores effective radius errors for phases that don't exist:
    - Pure ice clouds: ignore r_eff_liq error
    - Pure liquid clouds: ignore r_eff_ice error

    Args:
        pred_norm: Predicted physics values (normalized [-1, 1]), shape (batch, 4)
        target_norm: Target physics values (normalized [-1, 1]), shape (batch, 4)
        target_denorm: Target physics values (physical units), shape (batch, 4)
                       Order: [tau, ice_liq_ratio, r_eff_liq, r_eff_ice]

    Returns:
        Scalar loss value
    """
    squared_error = (pred_norm - target_norm) ** 2

    # Index 1 is ice/liquid ratio (0=liquid, 1=ice)
    ice_liq_ratio = target_denorm[:, 1]

    # Define thresholds for "pure" phase
    is_liquid_cloud = (ice_liq_ratio <= 0.01).float()
    is_ice_cloud = (ice_liq_ratio >= 0.99).float()

    # Index 2 = r_eff_liq, Index 3 = r_eff_ice
    # If pure ice, zero out r_eff_liq error
    squared_error[:, 2] = squared_error[:, 2] * (1 - is_ice_cloud)
    # If pure liquid, zero out r_eff_ice error
    squared_error[:, 3] = squared_error[:, 3] * (1 - is_liquid_cloud)

    return torch.mean(squared_error)


def combined_shading_loss(
    pred_log_shading: torch.Tensor,
    target_log_shading: torch.Tensor,
    pred_global_norm: torch.Tensor,
    target_global_norm: torch.Tensor,
    target_global_denorm: torch.Tensor,
    shading_weight: float = 1.0,
    global_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss for shading model training.

    Args:
        pred_log_shading: Predicted log(tau_shading) (batch, bag_size)
        target_log_shading: Target log(tau_shading) (batch, bag_size)
        pred_global_norm: Predicted global physics (normalized), shape (batch, 4)
        target_global_norm: Target global physics (normalized), shape (batch, 4)
        target_global_denorm: Target global physics (physical units), shape (batch, 4)
        shading_weight: Weight for shading loss
        global_weight: Weight for global physics loss

    Returns:
        total_loss: Weighted sum of both losses
        loss_dict: Dictionary with individual loss components for logging
    """
    s_loss = shading_loss(pred_log_shading, target_log_shading)
    g_loss = global_physics_loss(pred_global_norm, target_global_norm, target_global_denorm)

    total_loss = shading_weight * s_loss + global_weight * g_loss

    loss_dict = {
        'total': total_loss.item(),
        'shading': s_loss.item(),
        'global': g_loss.item()
    }

    return total_loss, loss_dict
