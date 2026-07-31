"""Configuration for the multitemporal albedo extension.

``MultitemporalAlbedoParams`` is embedded in ``AlbedoEstimatorConfig`` as the
``multitemporal`` field so the albedo step's config hash — and therefore the
provenance / resume machinery — covers the fit hyperparameters: changing any
of them correctly invalidates pre-populated ``albedo.tif`` outputs.
"""
from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MultitemporalAlbedoParams(BaseModel):
    """Hyperparameters for the tile-level cluster + temporal kernel model."""

    model_config = ConfigDict(extra="forbid")

    # ---- stack -----------------------------------------------------------
    grid_res: int = Field(
        default=180, ge=60, le=1080,
        description="Resolution (m) of the full-tile analysis grid the time "
                    "series is built on. 180 m gives a 610x610 grid.",
    )

    # ---- time-series admissibility --------------------------------------
    min_scenes: int = Field(
        default=50, ge=2,
        description="Minimum number of staged scenes required to fit.",
    )
    min_years: int = Field(
        default=2, ge=1,
        description="Minimum number of distinct years the series must span "
                    "(the cross-year kernel needs at least 2 to help).",
    )
    min_date_span_days: int = Field(
        default=365, ge=1,
        description="Minimum span in days between first and last scene.",
    )
    min_clear_obs: int = Field(
        default=20, ge=1,
        description="Minimum clear observations for a pixel to join the "
                    "cluster fit; sparser pixels fall back to the constant "
                    "surface albedo defaults.",
    )

    # ---- clustering ------------------------------------------------------
    clustering: Literal["flat", "hierarchical"] = Field(
        default="flat",
        description="'flat' k-means with K=n_clusters (the validated "
                    "default), or adaptive-split 'hierarchical' clustering.",
    )
    n_clusters: int = Field(
        default=100, ge=2,
        description="Number of clusters for flat clustering (initial K for "
                    "hierarchical).",
    )
    n_kmeans_iter: int = Field(default=25, ge=1,
                               description="Max k-means iterations.")
    max_clusters: int = Field(
        default=600, ge=2,
        description="(hierarchical) Hard ceiling on cluster count.",
    )
    split_mse_threshold: float = Field(
        default=0.0008, gt=0,
        description="(hierarchical) Split clusters whose within-cluster MSE "
                    "exceeds this.",
    )
    min_postsplit_obs: int = Field(
        default=30, ge=1,
        description="(hierarchical) Minimum median per-date observation "
                    "density a cluster must retain after splitting.",
    )
    seed: int = Field(default=0, description="RNG seed for k-means init.")

    # ---- temporal kernels ------------------------------------------------
    sigma_t: float = Field(
        default=12.0, gt=0,
        description="Std-dev (days) of the within-series Gaussian time kernel.",
    )
    sigma_doy: float = Field(
        default=12.0, gt=0,
        description="Std-dev (days) of the circular day-of-year kernel used "
                    "to borrow from other years.",
    )
    cross_year_lam: float = Field(
        default=1.0, ge=0,
        description="Gain of the gated cross-year term.",
    )
    cross_year_d0: float = Field(
        default=1.0, gt=0,
        description="Same-year support scale that gates the cross-year term "
                    "off: gate = lam / (1 + support/d0).",
    )
    robust_iters: int = Field(
        default=3, ge=1,
        description="IRLS iterations down-weighting bright (residual thin "
                    "cloud) outliers.",
    )

    # ---- QC holdout ------------------------------------------------------
    qc_holdout: bool = Field(
        default=False,
        description="Also fit with a held-out validation block and log MAE "
                    "vs the within-year baseline before the real fit.",
    )
    qc_holdout_year: int = Field(
        default=0,
        description="Year of the QC holdout block (0 = second-to-last year "
                    "in the series).",
    )
    qc_holdout_doy_window: Tuple[int, int] = Field(
        default=(1, 90),
        description="Day-of-year window of the QC holdout block.",
    )
