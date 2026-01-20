# refl2prop/loss.py
import torch

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