"""Cluster + gated cross-year temporal kernel model.

Port of the validated research pipeline in ``demo/albedo_multitemporal/``
(``_pipeline.py``): pixels are clustered on their sparse (time x band) clear
signal with a missing-data-aware k-means, each cluster gets a robust gated
cross-year temporal fit of its high-SNR mean signal, and each pixel keeps a
per-band offset from its cluster. Reconstruction at any date is
``mu_k(t, b) + offset_p(b)``.

All pixel-dimension linear algebra is chunked so a full 610x610 tile with
thousands of scenes stays within memory (the research code materialised
``P x (T*B)`` float32 arrays, which does not scale past ROI-sized fits).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CHUNK = 32768  # pixels per chunk for P-dimension matmuls


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def time_kernel(t: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel over observation time differences (days)."""
    d = t[:, None] - t[None, :]
    return np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)


def doy_kernel_otheryear(doy: np.ndarray, year: np.ndarray,
                         sigma: float) -> np.ndarray:
    """Circular day-of-year kernel, zeroed for same-year pairs."""
    d = np.abs(doy[:, None] - doy[None, :])
    d = np.minimum(d, 365 - d)
    K = np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)
    K[year[:, None] == year[None, :]] = 0
    return K


def _time_kernel_row(t_star: float, t: np.ndarray, sigma: float) -> np.ndarray:
    d = t_star - t
    return np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)


def _doy_kernel_row_otheryear(doy_star: float, year_star: int,
                              doy: np.ndarray, year: np.ndarray,
                              sigma: float) -> np.ndarray:
    d = np.abs(doy_star - doy)
    d = np.minimum(d, 365 - d)
    k = np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)
    k[year == year_star] = 0
    return k


# ---------------------------------------------------------------------------
# Missing-data-aware k-means (chunked over pixels)
# ---------------------------------------------------------------------------

def _expand_mask(M_pt_chunk: np.ndarray, n_bands: int) -> np.ndarray:
    """Expand a ``(c, T)`` observation mask to ``(c, T*B)`` float32 (band-major
    per date, matching ``R_pf`` layout)."""
    return np.repeat(M_pt_chunk.astype(np.float32), n_bands, axis=1)


def _cluster_stats(R_pf: np.ndarray, M_pt: np.ndarray, n_bands: int,
                   assign: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate per-cluster sums ``num = A^T (M*R)`` and ``den = A^T M``.

    Chunked over pixels; inputs may be float16 (cast per chunk).
    """
    F = R_pf.shape[1]
    num = np.zeros((K, F), np.float32)
    den = np.zeros((K, F), np.float32)
    for s in range(0, R_pf.shape[0], _CHUNK):
        e = min(s + _CHUNK, R_pf.shape[0])
        M_c = _expand_mask(M_pt[s:e], n_bands)
        MR_c = M_c * R_pf[s:e].astype(np.float32)
        A_c = np.zeros((e - s, K), np.float32)
        A_c[np.arange(e - s), assign[s:e]] = 1
        num += A_c.T @ MR_c
        den += A_c.T @ M_c
    return num, den


def kmeans_sparse(R_pf: np.ndarray, M_pt: np.ndarray, n_bands: int, K: int,
                  n_iter: int, seed: int = 0) -> np.ndarray:
    """K-means on sparse per-pixel (time x band) signals.

    Distances count only observed entries, normalised by each pixel's
    observation count. ``R_pf`` is ``(P, T*B)`` (may be float16); the
    observation mask is passed compactly as ``M_pt`` ``(P, T)`` and expanded
    per chunk to keep peak memory at ~1x the signal array.

    Returns:
        ``(P,)`` int32 cluster assignment.
    """
    P, F = R_pf.shape
    # Per-pixel constants: A_p = sum_f M*R^2, den_p = sum_f M.
    A = np.empty(P, np.float32)
    den_p = np.empty(P, np.float32)
    for s in range(0, P, _CHUNK):
        e = min(s + _CHUNK, P)
        M_c = _expand_mask(M_pt[s:e], n_bands)
        R_c = R_pf[s:e].astype(np.float32)
        A[s:e] = (M_c * R_c * R_c).sum(1)
        den_p[s:e] = M_c.sum(1)

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(P, K, replace=False)
    mu = R_pf[init_idx].astype(np.float32).copy()
    # Fill unobserved centroid entries with the column mean.
    col_num = np.zeros(F, np.float32)
    col_den = np.zeros(F, np.float32)
    for s in range(0, P, _CHUNK):
        e = min(s + _CHUNK, P)
        M_c = _expand_mask(M_pt[s:e], n_bands)
        col_num += (M_c * R_pf[s:e].astype(np.float32)).sum(0)
        col_den += M_c.sum(0)
    col_mean = col_num / np.maximum(col_den, 1.0)
    init_M = _expand_mask(M_pt[init_idx], n_bands)
    mu = np.where(init_M > 0, mu, col_mean[None, :])

    assign = np.zeros(P, np.int32)
    prev: Optional[np.ndarray] = None
    for it in range(n_iter):
        mu_sq = (mu ** 2).T  # (F, K)
        for s in range(0, P, _CHUNK):
            e = min(s + _CHUNK, P)
            M_c = _expand_mask(M_pt[s:e], n_bands)
            MR_c = M_c * R_pf[s:e].astype(np.float32)
            D = (A[s:e, None] - 2 * (MR_c @ mu.T) + M_c @ mu_sq)
            D /= np.maximum(den_p[s:e, None], 1)
            assign[s:e] = D.argmin(1)
        if prev is not None and (assign != prev).sum() < 0.001 * P:
            logger.info("k-means converged at iter %d", it + 1)
            break
        prev = assign.copy()
        num, den = _cluster_stats(R_pf, M_pt, n_bands, assign, K)
        mu = num / np.maximum(den, 1.0)
    return assign


def hierarchical_kmeans(R_pf: np.ndarray, M_pt: np.ndarray, n_bands: int,
                        K_init: int = 20, n_km_iter: int = 15,
                        mse_threshold: float = 0.0008,
                        min_postsplit_obs: int = 30, max_K: int = 600,
                        splits_per_round: int = 8,
                        min_pixels_to_split: int = 80,
                        min_child_pixels: int = 30,
                        seed: int = 0) -> Tuple[np.ndarray, list]:
    """Adaptive-split clustering: split high-MSE clusters that keep enough
    post-split temporal density. Returns ``(assign, history)``."""
    P, F = R_pf.shape
    logger.info("initial k-means K=%d...", K_init)
    assign = kmeans_sparse(R_pf, M_pt, n_bands, K_init, n_km_iter, seed)
    current_K = int(K_init)
    next_id = current_K
    history: list = []
    rounds = 0
    mse_K = np.zeros(current_K, np.float32)
    # Clusters whose split attempt produced an unbalanced (rejected) split
    # are excluded from future candidacy — retrying them would loop forever.
    blocked = np.zeros(max_K, bool)

    while current_K < max_K:
        num, den_F = _cluster_stats(R_pf, M_pt, n_bands, assign, current_K)
        mu_K_F = num / np.maximum(den_F, 1.0)
        # Per-cluster MSE and per-date observation density, chunked.
        sse_K = np.zeros(current_K, np.float32)
        n_K = np.zeros(current_K, np.float32)
        den_KT = np.zeros((current_K, M_pt.shape[1]), np.float32)
        size_K = np.bincount(assign, minlength=current_K).astype(np.int32)
        for s in range(0, P, _CHUNK):
            e = min(s + _CHUNK, P)
            M_c = _expand_mask(M_pt[s:e], n_bands)
            R_c = R_pf[s:e].astype(np.float32)
            resid = (R_c - mu_K_F[assign[s:e]])
            sse_c = (M_c * resid * resid).sum(1)
            np.add.at(sse_K, assign[s:e], sse_c)
            np.add.at(n_K, assign[s:e], M_c.sum(1))
            A_c = np.zeros((e - s, current_K), np.float32)
            A_c[np.arange(e - s), assign[s:e]] = 1
            den_KT += A_c.T @ M_pt[s:e].astype(np.float32)
        mse_K = sse_K / np.maximum(n_K, 1.0)
        masked = np.where(den_KT > 0, den_KT, np.nan)
        sparsity_K = np.nan_to_num(np.nanmedian(masked, axis=1), nan=0.0)

        history.append({"round": rounds, "K": current_K,
                        "max_mse": float(mse_K.max()),
                        "median_mse": float(np.median(mse_K))})

        eligible = ((mse_K > mse_threshold)
                    & (sparsity_K / 2.0 >= min_postsplit_obs)
                    & (size_K >= min_pixels_to_split)
                    & ~blocked[:current_K])
        if not eligible.any():
            logger.info("no more split candidates after %d rounds (K=%d)",
                        rounds, current_K)
            break

        cand = np.where(eligible)[0]
        cand = cand[np.argsort(-mse_K[cand])][:splits_per_round]
        logger.info("round %d K=%d -> splitting %d clusters",
                    rounds, current_K, len(cand))
        for k in cand:
            if current_K >= max_K:
                break
            in_k = np.where(assign == k)[0]
            sub = kmeans_sparse(R_pf[in_k], M_pt[in_k], n_bands, K=2,
                                n_iter=n_km_iter, seed=seed + current_K)
            n0 = int((sub == 0).sum())
            n1 = int((sub == 1).sum())
            if min(n0, n1) < min_child_pixels:
                logger.info("skipped split of cluster %d (%d/%d unbalanced)",
                            int(k), n0, n1)
                blocked[k] = True
                continue
            assign[in_k] = np.where(sub == 1, next_id, k)
            next_id += 1
            current_K += 1
        rounds += 1

    history.append({"round": rounds, "K": current_K,
                    "max_mse": float(mse_K.max()),
                    "median_mse": float(np.median(mse_K))})
    logger.info("hierarchical clustering done: K=%d after %d rounds",
                current_K, rounds)
    return assign, history


# ---------------------------------------------------------------------------
# Gated cross-year robust temporal fit on cluster signals
# ---------------------------------------------------------------------------

def fit_cluster_temporal(signal_num: np.ndarray, signal_den: np.ndarray,
                         Kt: np.ndarray, Kd: np.ndarray,
                         robust_iters: int = 3, lam: float = 1.0,
                         d0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``mu_k(t, b)`` for every cluster with IRLS bright-outlier
    down-weighting and a gated cross-year term.

    Args:
        signal_num: ``(K, T, B)`` sum of clear reflectance per cluster/date.
        signal_den: ``(K, T)`` clear-pixel counts per cluster/date.
        Kt: ``(T, T)`` time kernel.
        Kd: ``(T, T)`` other-year day-of-year kernel.

    Returns:
        ``(mu_KTB, rw)`` — the fit and the ``(T, K)`` robust weights used for
        the final iteration (needed to reproduce predictions at new dates).
    """
    K, T, B = signal_num.shape
    obs_mask = (signal_den > 0).T.astype(np.float32)  # (T, K)
    rw = obs_mask.copy()
    mu_TKB = np.zeros((T, K, B), np.float32)
    for it in range(robust_iters):
        weighted_num = rw[:, :, None] * signal_num.transpose(1, 0, 2)
        weighted_den = rw * signal_den.T
        num_t = (Kt @ weighted_num.reshape(T, K * B)).reshape(T, K, B)
        den_t = Kt @ weighted_den
        num_d = (Kd @ weighted_num.reshape(T, K * B)).reshape(T, K, B)
        den_d = Kd @ weighted_den
        g = lam / (1.0 + den_t / d0)
        num = num_t + g[:, :, None] * num_d
        den = den_t + g * den_d
        mu_TKB = num / np.maximum(den, 1e-6)[:, :, None]
        if it < robust_iters - 1:
            cm_TKB = (signal_num
                      / np.maximum(signal_den[:, :, None], 1e-6)
                      ).transpose(1, 0, 2)
            resid = (cm_TKB - mu_TKB) * obs_mask[:, :, None]
            bright = np.clip(resid, 0.0, None).mean(-1)
            s = 1.4826 * np.median(np.abs(resid)) + 1e-4
            rw = obs_mask * (1.0 / (1.0 + (bright / (2 * s)) ** 2))
    return mu_TKB.transpose(1, 0, 2), rw


# ---------------------------------------------------------------------------
# Full fit + arbitrary-date prediction
# ---------------------------------------------------------------------------

def fit_cluster_model(refl: np.ndarray, train: np.ndarray, times: np.ndarray,
                      doy: np.ndarray, year: np.ndarray,
                      params) -> Dict[str, np.ndarray]:
    """Fit the full cluster model on a stack.

    Args:
        refl: ``(T, B, H, W)`` reflectance (float16 ok, NaN where invalid).
        train: ``(T, H, W)`` bool — clear observations to train on.
        times: ``(T,)`` days since first observation.
        doy: ``(T,)`` day of year per observation.
        year: ``(T,)`` year per observation.
        params: :class:`MultitemporalAlbedoParams`.

    Returns:
        Model dict with everything needed by :func:`predict_reflectance`.
    """
    T, B, H, W = refl.shape
    P = H * W
    train_pt = train.transpose(1, 2, 0).reshape(P, T)
    cnt_p = train_pt.sum(1)
    valid = cnt_p >= params.min_clear_obs
    P_v = int(valid.sum())
    logger.info("fit: P=%d valid=%d (%.0f%%), T=%d", P, P_v, 100 * P_v / P, T)
    if P_v < params.n_clusters:
        raise ValueError(
            f"Only {P_v} pixels have >= {params.min_clear_obs} clear "
            f"observations — cannot fit {params.n_clusters} clusters."
        )

    # (P_v, T, B) float16 pixel-major copy of the training signal, NaN -> 0.
    # Gathered in chunks: reshaping the transposed stack would materialise
    # the full (P, T, B) array twice.
    valid_idx = np.where(valid)[0]
    R_ptb_v = np.empty((P_v, T, B), np.float16)
    for s in range(0, P_v, _CHUNK):
        e = min(s + _CHUNK, P_v)
        rows, cols = np.divmod(valid_idx[s:e], W)
        R_ptb_v[s:e] = np.nan_to_num(
            refl[:, :, rows, cols], nan=0.0).transpose(2, 0, 1)
    M_pt_v = train_pt[valid].astype(np.float16)
    F = T * B
    R_pf_v = R_ptb_v.reshape(P_v, F)

    if params.clustering == "hierarchical":
        assign, _history = hierarchical_kmeans(
            R_pf_v, M_pt_v, B,
            K_init=params.n_clusters, n_km_iter=params.n_kmeans_iter,
            mse_threshold=params.split_mse_threshold,
            min_postsplit_obs=params.min_postsplit_obs,
            max_K=params.max_clusters, seed=params.seed,
        )
        K = int(assign.max()) + 1
    else:
        K = params.n_clusters
        logger.info("k-means K=%d over %d pixels...", K, P_v)
        assign = kmeans_sparse(R_pf_v, M_pt_v, B, K, params.n_kmeans_iter,
                               params.seed)

    logger.info("building cluster signals (K=%d)...", K)
    signal_num = np.zeros((K, T, B), np.float32)
    signal_den = np.zeros((K, T), np.float32)
    for s in range(0, P_v, _CHUNK):
        e = min(s + _CHUNK, P_v)
        M_c = M_pt_v[s:e].astype(np.float32)
        MR_c = R_ptb_v[s:e].astype(np.float32) * M_c[:, :, None]
        A_c = np.zeros((e - s, K), np.float32)
        A_c[np.arange(e - s), assign[s:e]] = 1
        signal_num += (A_c.T @ MR_c.reshape(e - s, F)).reshape(K, T, B)
        signal_den += A_c.T @ M_c

    logger.info("fitting per-cluster gated temporal model...")
    Kt = time_kernel(times, params.sigma_t)
    Kd = doy_kernel_otheryear(doy, year, params.sigma_doy)
    mu_KTB, rw = fit_cluster_temporal(
        signal_num, signal_den, Kt, Kd,
        robust_iters=params.robust_iters,
        lam=params.cross_year_lam, d0=params.cross_year_d0,
    )

    logger.info("per-pixel offsets...")
    offset_v = np.zeros((P_v, B), np.float32)
    for s in range(0, P_v, _CHUNK):
        e = min(s + _CHUNK, P_v)
        M_c = M_pt_v[s:e].astype(np.float32)          # (c, T)
        cnt_c = np.maximum(M_c.sum(1), 1.0)
        mean_R = (R_ptb_v[s:e].astype(np.float32)
                  * M_c[:, :, None]).sum(1) / cnt_c[:, None]
        # mean of the cluster fit over each pixel's observed dates
        mean_mu = np.einsum("ct,ctb->cb", M_c,
                            mu_KTB[assign[s:e]]) / cnt_c[:, None]
        offset_v[s:e] = mean_R - mean_mu

    return {
        "assign": assign.astype(np.int32),
        "valid_mask": valid,
        "offset_v": offset_v,
        "signal_num": signal_num,
        "signal_den": signal_den,
        "rw": rw.astype(np.float32),
        "times": times.astype(np.float64),
        "doy": doy.astype(np.float64),
        "year": year.astype(np.int16),
        "H": np.int32(H), "W": np.int32(W),
        "K": np.int32(K), "B": np.int32(B), "T": np.int32(T),
        "sigma_t": np.float64(params.sigma_t),
        "sigma_doy": np.float64(params.sigma_doy),
        "lam": np.float64(params.cross_year_lam),
        "d0": np.float64(params.cross_year_d0),
    }


def predict_cluster_means(model: Dict[str, np.ndarray], t_star: float,
                          doy_star: float, year_star: int) -> np.ndarray:
    """Evaluate ``mu_k(t*, b)`` for every cluster at an arbitrary date.

    Uses the stored robust weights, so for dates in the training series this
    reproduces the fit's own reconstruction exactly.

    Returns:
        ``(K, B)`` float32 cluster mean reflectance.
    """
    signal_num = model["signal_num"]          # (K, T, B)
    signal_den = model["signal_den"]          # (K, T)
    rw = model["rw"]                          # (T, K)
    K, T, B = signal_num.shape
    kt = _time_kernel_row(t_star, model["times"], float(model["sigma_t"]))
    kd = _doy_kernel_row_otheryear(doy_star, year_star, model["doy"],
                                   model["year"], float(model["sigma_doy"]))
    weighted_num = rw.T[:, :, None] * signal_num      # (K, T, B)
    weighted_den = rw.T * signal_den                  # (K, T)
    num_t = np.einsum("t,ktb->kb", kt, weighted_num)
    den_t = weighted_den @ kt                          # (K,)
    num_d = np.einsum("t,ktb->kb", kd, weighted_num)
    den_d = weighted_den @ kd
    g = float(model["lam"]) / (1.0 + den_t / float(model["d0"]))
    num = num_t + g[:, None] * num_d
    den = den_t + g * den_d
    return num / np.maximum(den, 1e-6)[:, None]


def predict_reflectance(model: Dict[str, np.ndarray], t_star: float,
                        doy_star: float, year_star: int) -> np.ndarray:
    """Reconstruct clear-sky surface reflectance for the whole grid at a date.

    Returns:
        ``(B, H, W)`` float32; NaN where the pixel had too few clear
        observations to join the fit.
    """
    H, W, B = int(model["H"]), int(model["W"]), int(model["B"])
    mu_KB = predict_cluster_means(model, t_star, doy_star, year_star)
    pred_v = mu_KB[model["assign"]] + model["offset_v"]   # (P_v, B)
    out = np.full((H * W, B), np.nan, np.float32)
    out[model["valid_mask"]] = pred_v
    return out.reshape(H, W, B).transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# QC baseline (within-year robust kernel regression, held dates only)
# ---------------------------------------------------------------------------

def predict_within_year(Rf: np.ndarray, train_f: np.ndarray, Kt: np.ndarray,
                        robust_iters: int = 3) -> np.ndarray:
    """Per-pixel within-year robust temporal baseline (QC comparison only)."""
    T, B, P = Rf.shape
    M = train_f.astype(np.float32)
    Mw = M.copy()
    Rhat = np.zeros_like(Rf)
    for it in range(robust_iters):
        num = (Kt @ (Mw[:, None, :] * Rf).reshape(T, B * P)).reshape(T, B, P)
        den = Kt @ Mw
        Rhat = num / np.maximum(den, 1e-6)[:, None, :]
        if it < robust_iters - 1:
            resid = (Rf - Rhat) * (M[:, None, :] > 0)
            bright = np.clip(resid, 0, None).mean(1)
            s = 1.4826 * np.median(np.abs(resid)) + 1e-4
            Mw = M * (1.0 / (1.0 + (bright / (2 * s)) ** 2))
    return Rhat
