# Multitemporal Albedo

Most of clouds-decoded processes **one scene at a time**. The multitemporal
albedo extension is different: it fits a single model to a whole *time
series* of scenes over one tile, then uses that model to produce a
cloud-free surface albedo estimate for every scene — including scenes that
were fully overcast when acquired.

If you are processing scenes independently, you don't need this page: the
default `idw` and `datadriven` albedo methods work per scene with no extra
setup. Read on when you have a long archive over one tile and want albedo
interpolated intelligently through the cloudy gaps.

---

## How it works

1. Pixels are **clustered** on their sparse clear-sky (time x band) signal
   with a missing-data-aware k-means — pixels that behave alike through the
   seasons share a cluster.
2. Each cluster's high-SNR mean signal gets a **robust temporal fit**: a
   Gaussian kernel over observation time, plus a day-of-year kernel that
   borrows from *other years* only where same-year support is weak (the
   "gated cross-year" term). Iteratively-reweighted fitting down-weights
   bright outliers (residual thin cloud the mask missed).
3. Each pixel keeps a **per-band offset** from its cluster mean.

Reconstruction at any date is then `mu_cluster(t, band) + offset_pixel` —
smooth in time, sharp in space.

The fit runs on a coarse full-tile analysis grid (default 180 m), which is
the appropriate scale for surface albedo as consumed by the cloud-properties
inversion.

## How it integrates with the project system

Selecting the method inserts a **pre-run stage** ahead of the ordinary
per-scene run:

```
validate time series          (is this archive fit-worthy? fail fast)
  --> cloud masks             (restricted project run: cloud_mask step only)
  --> build 180 m stack       (per-scene cache, resumable)
  --> fit cluster model       (cached; refits only if data or params change)
  --> pre-populate outputs    (albedo.tif + manifest entry per scene)
  --> normal per-scene run    (resumes at cloud_height; albedo is honoured)
```

The pre-populated `albedo.tif` files carry the same provenance metadata and
manifest entries as orchestrator-written outputs, so the per-scene run
treats them exactly like cached results: it skips `cloud_mask` and
`albedo` and continues with `cloud_height`, `refocus`, and
`cloud_properties`. Every stage is cached and idempotent — re-running is
cheap.

!!! note "Recipe ordering"
    Use the `full-workflow-multitemporal` recipe (albedo ordered directly
    after cloud_mask). Resume restarts at the *first* incomplete step, so
    with the default recipe order the pre-populated albedo would be
    recomputed when cloud_height runs.

## Usage

```bash
# 1. Create the project with the multitemporal recipe
clouds-decoded project init ./tile_analysis --pipeline full-workflow-multitemporal

# 2. Select the method in configs/albedo.yaml
#      method: multitemporal
#    (a default `multitemporal:` parameter block is implied; add one to tune)

# 3. Stage the tile's archive and run
clouds-decoded project stage ./tile_analysis /data/sentinel2/T37VCC/
clouds-decoded project run ./tile_analysis
```

`project run` performs the pre-run stage automatically. To run the heavy
stage separately (e.g. overnight, or on a GPU box before a CPU batch run):

```bash
clouds-decoded project prefit ./tile_analysis --parallel
clouds-decoded project run ./tile_analysis        # stage is now a cached no-op
```

After the stage, the project contains an extra directory:

```
tile_analysis/
  multitemporal/
    model.npz             # fitted model (cached by data + params signature)
    scene_cache_<hash>/   # per-scene 180 m extractions, keyed to the
                          # cloud_mask config hash
```

## Data requirements

The stage validates the staged scenes before doing any heavy work, and
fails with an actionable message if the series is not fit-worthy. Defaults:

- **One tile per project.** Multi-tile projects are rejected — split them.
- At least **50 scenes**, spanning **2+ years** and **365+ days**
  (`min_scenes`, `min_years`, `min_date_span_days`).
- Pixels with fewer than `min_clear_obs` (default 20) clear observations
  are excluded from the fit and fall back to the constant per-band
  `default_albedo` values (recorded in the output metadata).

All thresholds live in the `multitemporal:` block of `configs/albedo.yaml`
— see the [configuration reference](configuration.md#multitemporalalbedoparams).
The block lives inside the albedo config deliberately: changing any fit
hyperparameter changes the albedo step's config hash, which invalidates the
pre-populated outputs through the ordinary resume machinery. If you tweak
the parameters and re-run, the refit and re-population happen
automatically.

## Quick validation runs with a crop window

A full-tile fit over thousands of scenes is GPU-hours of cloud masking. For
a sanity check, stage a small series with a shared crop window and the
whole stage runs in minutes:

```bash
clouds-decoded project init ./validation --pipeline full-workflow-multitemporal
# set method: multitemporal, and relax the series thresholds in albedo.yaml:
#   multitemporal:
#     n_clusters: 8
#     min_clear_obs: 5
#     min_scenes: 15
#     min_years: 1
#     min_date_span_days: 30
clouds-decoded project stage ./validation /data/T37VCC/some_scenes/   # then re-stage with crop:
clouds-decoded project run ./validation --crop-window "4978,4978,1024,1024"
```

Notes on crops:

- Every scene in the fit must share the **same** crop window; mixed crops
  are rejected.
- The crop is floored to whole analysis cells (18 B02 pixels per 180 m
  cell), so up to 17 edge pixels are dropped.
- Cropped and full-tile fits coexist in one project: the model and scene
  cache get a crop tag (`model_crop<...>.npz`), and outputs land in the
  standard `outputs/<scene_id>/crops/<window>/` directories.

## Adding scenes later

Staging more scenes and re-running `project run` works incrementally: only
the new scenes are masked, the stack gains their extractions (existing
ones are cached), the model is **refit** on the extended series, and only
the new scenes get `albedo.tif` + downstream processing.

Existing scenes deliberately keep their outputs from the earlier fit —
with a long series, one more scene changes the model negligibly, and
re-populating would cascade into re-running every downstream step for
every scene. This is never silent: each `albedo.tif` is stamped with the
signature of the model that produced it, and the stage **warns on every
run** listing the scenes whose outputs come from a superseded fit. When
you *do* want everything refreshed from the latest fit (e.g. after a
large batch of new scenes), run:

```bash
clouds-decoded project run ./tile_analysis --force
```

which re-computes the masks, refits, rewrites every scene's albedo from
the fresh model, and re-runs the remaining per-scene steps.

## Behaviour notes

- Per-scene `albedo.tif` outputs are **masked to each scene's swath
  footprint** — a scene whose orbit misses half the tile gets nodata there,
  like any other per-scene product. The fitted model itself covers the
  whole grid; use it directly (`multitemporal/model.npz`) if you want
  footprint-free reconstructions.
- The method is **project-only**. The single-scene `clouds-decoded albedo`
  command refuses `method: multitemporal` — there is no single-scene
  equivalent of a time-series fit.
- Scenes whose cloud mask failed are excluded from the fit and not
  pre-populated; they fail in the per-scene run as usual.
