"""Shared spatial-sampling utilities for albedo estimation.

Used by the IDW fitter (smooth training targets) and the data-driven
training sampler (extract windowed reflectance).
"""
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


def clear_pixel_count(
    clear_mask: np.ndarray,
    window_px: int,
) -> np.ndarray:
    """Pre-compute the windowed clear-pixel count.

    Returns the denominator used by :func:`windowed_clear_mean`.
    Caching this across bands at the same resolution avoids one
    ``uniform_filter`` call per band.

    Args:
        clear_mask: 2-D array (H, W), boolean or 0/1 float.
        window_px: Side-length of the averaging square in pixels.

    Returns:
        Float32 array of the same shape; each pixel holds the
        fraction of clear neighbours in its window.
    """
    if window_px <= 1:
        return clear_mask.astype(np.float32)
    valid = clear_mask.astype(np.float32)
    return uniform_filter(valid, size=window_px, mode="constant", cval=0).astype(np.float32)


def windowed_clear_mean(
    band_data: np.ndarray,
    clear_mask: np.ndarray,
    window_px: int,
    *,
    precomputed_count=None,
) -> np.ndarray:
    """Compute windowed mean of *band_data*, averaging only clear pixels.

    Each output pixel is the mean of clear-sky values within a
    ``window_px × window_px`` neighbourhood.  Pixels with no clear
    neighbours in their window are set to NaN.

    Args:
        band_data: 2-D array (H, W) of reflectance / DN values.
        clear_mask: 2-D boolean array (H, W), True where clear.
        window_px: Side-length of the averaging square in pixels.
            If ≤ 1 the input is returned unchanged (cast to float32).
        precomputed_count: Optional float32 array from
            :func:`clear_pixel_count`.  When provided the denominator
            ``uniform_filter`` is skipped, saving one convolution.

    Returns:
        Smoothed 2-D float32 array of the same shape.
    """
    if window_px <= 1:
        return band_data.astype(np.float32)

    valid = clear_mask.astype(np.float32)
    band_f = band_data.astype(np.float32) * valid

    val_sum = uniform_filter(band_f, size=window_px, mode="constant", cval=0)
    val_count = (
        precomputed_count
        if precomputed_count is not None
        else uniform_filter(valid, size=window_px, mode="constant", cval=0)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(val_count > 0, val_sum / val_count, np.nan)
    return result.astype(np.float32)


def farthest_point_sample(
    clear_mask: np.ndarray,
    n_samples: int,
    seed: int = 42,
    edge_margin: int = 0,
    cloud_dilation_px: int = 0,
) -> tuple:
    """Greedily sample clear-sky locations that are maximally spread out.

    Starts from a random seed point and iteratively selects the candidate
    farthest from all previously selected points.  This produces a more
    spatially uniform set of samples than random selection, which is
    beneficial for spatial interpolation methods like IDW.

    Complexity is O(n_candidates * n_samples) which is fast for typical
    scene sizes.

    Args:
        clear_mask: 2-D boolean array (H, W).
        n_samples: Maximum number of locations to return.
        seed: RNG seed for choosing the initial point.
        edge_margin: Exclude this many pixels from each image edge.
        cloud_dilation_px: Dilate cloud regions by this many pixels
            before sampling, keeping samples away from cloud edges.

    Returns:
        ``(rows, cols)`` integer arrays of sampled coordinates.
    """
    mask = clear_mask.copy()

    if cloud_dilation_px > 0:
        cloud = ~clear_mask
        dilated = maximum_filter(
            cloud.view(np.uint8), size=2 * cloud_dilation_px + 1,
        ).astype(bool)
        mask &= ~dilated

    if edge_margin > 0:
        mask[:edge_margin, :] = False
        mask[-edge_margin:, :] = False
        mask[:, :edge_margin] = False
        mask[:, -edge_margin:] = False

    rows, cols = np.where(mask)
    n_available = len(rows)
    if n_available == 0:
        return rows, cols

    if n_available <= n_samples:
        return rows, cols

    # Pre-subsample the candidate pool when it is much larger than
    # n_samples.  FPS quality depends on having a diverse pool, not an
    # exhaustive one, so 50× oversampling is ample.
    rng = np.random.default_rng(seed)
    max_candidates = max(n_samples * 50, 10_000)
    if n_available > max_candidates:
        idx = rng.choice(n_available, max_candidates, replace=False)
        rows, cols = rows[idx], cols[idx]
        n_available = max_candidates

    # Greedy farthest-point selection
    candidates = np.column_stack([rows, cols]).astype(np.float64)

    first_idx = rng.integers(n_available)
    selected = [first_idx]
    # min_dist[i] = distance from candidate i to the nearest selected point
    min_dist = np.full(n_available, np.inf)

    for _ in range(n_samples - 1):
        last = candidates[selected[-1]]
        d = np.sum((candidates - last) ** 2, axis=1)
        np.minimum(min_dist, d, out=min_dist)
        # Already-selected points get distance 0 so won't be picked again
        min_dist[selected[-1]] = 0.0
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)

    selected = np.array(selected)
    return rows[selected], cols[selected]
