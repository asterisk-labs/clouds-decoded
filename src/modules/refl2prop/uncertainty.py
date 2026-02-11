# refl2prop/uncertainty.py
"""
Utilities for uncertainty quantification and out-of-distribution detection.
"""
import logging
import torch
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def compute_uncertainty_score(noise_output: torch.Tensor, method: str = 'l2') -> torch.Tensor:
    """
    Compute uncertainty score from the model's noise reconstruction output.

    Args:
        noise_output: Reconstructed noise tensor (batch_size, 11) in range [-1, 1]
        method: Method for computing uncertainty
            - 'l2': L2 norm of noise vector (default)
            - 'l1': L1 norm of noise vector
            - 'linf': L-infinity norm (max absolute value)
            - 'mean_abs': Mean absolute value

    Returns:
        Uncertainty scores (batch_size,)
    """
    if method == 'l2':
        return torch.norm(noise_output, p=2, dim=1)
    elif method == 'l1':
        return torch.norm(noise_output, p=1, dim=1)
    elif method == 'linf':
        return torch.norm(noise_output, p=float('inf'), dim=1)
    elif method == 'mean_abs':
        return torch.mean(torch.abs(noise_output), dim=1)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


def calibrate_uncertainty_threshold(uncertainties: np.ndarray,
                                     percentile: float = 95.0) -> float:
    """
    Calibrate an uncertainty threshold based on a percentile of validation data.

    This assumes you have uncertainties from in-distribution validation data.
    Samples with uncertainty above this threshold can be flagged as potentially OOD.

    Args:
        uncertainties: Array of uncertainty scores from validation data
        percentile: Percentile for threshold (default 95.0)

    Returns:
        Threshold value
    """
    return np.percentile(uncertainties, percentile)


def flag_ood_samples(uncertainties: np.ndarray,
                     threshold: float,
                     return_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Flag samples as out-of-distribution based on uncertainty threshold.

    Args:
        uncertainties: Array of uncertainty scores
        threshold: Threshold value (from calibrate_uncertainty_threshold)
        return_mask: If True, also return boolean mask

    Returns:
        indices: Indices of samples flagged as OOD
        mask: (Optional) Boolean mask with True for OOD samples
    """
    mask = uncertainties > threshold
    indices = np.where(mask)[0]

    if return_mask:
        return indices, mask
    return indices, None


def get_uncertainty_statistics(uncertainties: np.ndarray) -> dict:
    """
    Compute statistics for uncertainty scores.

    Args:
        uncertainties: Array of uncertainty scores

    Returns:
        Dictionary with statistics (mean, std, min, max, median, percentiles)
    """
    return {
        'mean': float(np.mean(uncertainties)),
        'std': float(np.std(uncertainties)),
        'min': float(np.min(uncertainties)),
        'max': float(np.max(uncertainties)),
        'median': float(np.median(uncertainties)),
        'p25': float(np.percentile(uncertainties, 25)),
        'p75': float(np.percentile(uncertainties, 75)),
        'p90': float(np.percentile(uncertainties, 90)),
        'p95': float(np.percentile(uncertainties, 95)),
        'p99': float(np.percentile(uncertainties, 99))
    }


def uncertainty_based_filtering(predictions: np.ndarray,
                                 uncertainties: np.ndarray,
                                 threshold: float,
                                 fill_value: float = np.nan) -> np.ndarray:
    """
    Filter predictions based on uncertainty, replacing high-uncertainty values.

    Args:
        predictions: Array of predictions (batch_size, n_outputs)
        uncertainties: Array of uncertainty scores (batch_size,)
        threshold: Uncertainty threshold
        fill_value: Value to use for high-uncertainty predictions (default: NaN)

    Returns:
        Filtered predictions with high-uncertainty values replaced
    """
    filtered = predictions.copy()
    ood_mask = uncertainties > threshold
    filtered[ood_mask] = fill_value
    return filtered


# Example usage function
def example_inference_with_uncertainty(model, input_data: torch.Tensor,
                                       uncertainty_threshold: Optional[float] = None):
    """
    Example of how to use the model with uncertainty quantification during inference.

    Args:
        model: Trained NormalizationWrapper model
        input_data: Input tensor (batch_size, 17)
        uncertainty_threshold: Optional threshold for flagging OOD samples

    Returns:
        predictions: Physics predictions
        uncertainties: Uncertainty scores
        is_ood: Boolean mask (if threshold provided)
    """
    model.eval()
    with torch.no_grad():
        # Get predictions and uncertainty
        predictions, uncertainties = model(input_data, return_uncertainty=True)

    # Convert to numpy
    predictions_np = predictions.cpu().numpy()
    uncertainties_np = uncertainties.cpu().numpy()

    # Get statistics
    stats = get_uncertainty_statistics(uncertainties_np)
    logger.info(f"Uncertainty statistics: {stats}")

    # Flag OOD samples if threshold provided
    if uncertainty_threshold is not None:
        ood_indices, ood_mask = flag_ood_samples(
            uncertainties_np, uncertainty_threshold, return_mask=True
        )
        logger.info(f"Flagged {len(ood_indices)} / {len(uncertainties_np)} samples as OOD "
                    f"({100 * len(ood_indices) / len(uncertainties_np):.1f}%)")
        return predictions_np, uncertainties_np, ood_mask

    return predictions_np, uncertainties_np, None
