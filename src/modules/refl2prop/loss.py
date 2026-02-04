# refl2prop/loss.py
import torch
from typing import Tuple

def physics_masked_loss(predicted_norm, target_norm, original_target_denorm):
    """
    Custom loss that ignores effective radius errors for phases that don't exist.

    Args:
        predicted_norm: Model output (normalized [-1, 1]).
        target_norm: Ground truth (normalized [-1, 1]).
        original_target_denorm: Ground truth (physical units) to check phase presence.
                                Expected order: [Tau, IceLiq, ReffLiq, ReffIce]
    """
    # 1. Calculate MSE on everything
    squared_error = (predicted_norm - target_norm) ** 2

    # 2. Get Phase Masks
    # Index 1 is Ice/Liq Ratio (0=Liquid, 1=Ice)
    ice_liq_ratio = original_target_denorm[:, 1]

    # Define thresholds for "pure" phase
    is_liquid_cloud = (ice_liq_ratio <= 0.01).float()
    is_ice_cloud = (ice_liq_ratio >= 0.99).float()

    # Index 2 = Reff Liq, Index 3 = Reff Ice
    # If it's pure ICE, multiply Reff_Liq error by 0
    squared_error[:, 2] = squared_error[:, 2] * (1 - is_ice_cloud)

    # If it's pure LIQUID, multiply Reff_Ice error by 0
    squared_error[:, 3] = squared_error[:, 3] * (1 - is_liquid_cloud)

    return torch.mean(squared_error)


def noise_reconstruction_loss(predicted_noise, target_noise):
    """
    MSE loss for noise reconstruction.

    Args:
        predicted_noise: Model's reconstructed noise (normalized [-1, 1]), shape (batch, 11)
        target_noise: Ground truth noise (normalized [-1, 1]), shape (batch, 11)

    Returns:
        MSE loss
    """
    return torch.mean((predicted_noise - target_noise) ** 2)


def combined_loss(physics_pred, physics_target, original_target_denorm,
                  noise_pred, noise_target, physics_weight: float = 1.0,
                  noise_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss for physics prediction and noise reconstruction.

    Args:
        physics_pred: Predicted physics values (normalized)
        physics_target: Target physics values (normalized)
        original_target_denorm: Original physics targets (denormalized) for masking
        noise_pred: Predicted noise components (normalized)
        noise_target: Target noise components (normalized)
        physics_weight: Weight for physics loss
        noise_weight: Weight for noise reconstruction loss

    Returns:
        total_loss: Weighted sum of both losses
        loss_dict: Dictionary with individual loss components for logging
    """
    phys_loss = physics_masked_loss(physics_pred, physics_target, original_target_denorm)
    noise_loss = noise_reconstruction_loss(noise_pred, noise_target)

    total_loss = physics_weight * phys_loss + noise_weight * noise_loss

    loss_dict = {
        'total': total_loss.item(),
        'physics': phys_loss.item(),
        'noise': noise_loss.item()
    }

    return total_loss, loss_dict