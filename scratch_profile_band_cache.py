"""Profile Sentinel2Band caching vs the old uncached get_band() pattern.

Scenario: three processors each need all 12 bands as reflectance at 10m.
This is the worst case the band cache was designed for.
"""
import time
import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS

from clouds_decoded.data import Sentinel2Scene
from clouds_decoded.constants import BAND_RESOLUTIONS, BANDS

# --------------------------------------------------------------------------- #
# Build a realistic-sized scene (full-tile dimensions at native resolutions)
# --------------------------------------------------------------------------- #
FULL_10M = (10980, 10980)
BAND_NAMES = [b for b in BANDS if b != "B10"]  # B10 excluded (cirrus)

def make_scene():
    scene = Sentinel2Scene()
    scene.quantification_value = 10000.0
    scene.radio_add_offset = {b: -1000.0 for b in BAND_NAMES}
    scene.transform = Affine.translation(0.0, 0.0) * Affine.scale(10.0, -10.0)
    scene.crs = CRS.from_epsg(32633)

    np.random.seed(0)
    for band in BAND_NAMES:
        res = BAND_RESOLUTIONS[band]
        scale = 10 / res
        h, w = int(FULL_10M[0] * scale), int(FULL_10M[1] * scale)
        scene.bands[band] = np.random.randint(100, 8000, (h, w), dtype=np.uint16)
    return scene


# --------------------------------------------------------------------------- #
# Old pattern: every call recomputes reflectance + resize from scratch
# --------------------------------------------------------------------------- #
def old_pattern(scene: Sentinel2Scene, n_processors: int = 3):
    """Simulate n processors each calling get_band(reflectance=True) for all bands,
    then resizing 20m/60m bands to 10m — all without caching."""
    from skimage.transform import resize

    arrays = []
    for _ in range(n_processors):
        for band in BAND_NAMES:
            # Reflectance conversion (uncached — recomputes every time)
            data = scene.get_band(band, reflectance=True, cache=False)
            # Resize to 10m if needed
            res = BAND_RESOLUTIONS[band]
            if res != 10:
                data = resize(data, FULL_10M, order=3,
                              preserve_range=True, anti_aliasing=True
                              ).astype(np.float32)
            arrays.append(data.mean())  # consume the array
    return arrays


# --------------------------------------------------------------------------- #
# New pattern: cache=True + resolution kwarg — reuse across processors
# --------------------------------------------------------------------------- #
def new_pattern_sequential(scene: Sentinel2Scene, n_processors: int = 3):
    """Same work but using get_band(cache=True, resolution=10)."""
    arrays = []
    for _ in range(n_processors):
        for band in BAND_NAMES:
            data = scene.get_band(band, reflectance=True, resolution=10, cache=True)
            arrays.append(data.mean())
    return arrays


# --------------------------------------------------------------------------- #
# New pattern + parallel: first call uses threaded get_bands, rest hit cache
# --------------------------------------------------------------------------- #
def new_pattern_parallel(scene: Sentinel2Scene, n_processors: int = 3, n_workers: int = 4):
    """get_bands(n_workers=N) parallelises the first (cold) call."""
    arrays = []
    for _ in range(n_processors):
        bands = scene.get_bands(BAND_NAMES, reflectance=True, resolution=10,
                                cache=True, n_workers=n_workers)
        arrays.extend(np.asarray(b).mean() for b in bands)
    return arrays


# --------------------------------------------------------------------------- #
# Run and compare
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    N_PROCESSORS = 3
    N_WORKERS = min(os.cpu_count() or 4, len(BAND_NAMES))
    print(f"Scene: {len(BAND_NAMES)} bands, {FULL_10M[0]}x{FULL_10M[1]} @ 10m")
    print(f"Simulating {N_PROCESSORS} processors each requesting all bands")
    print(f"Thread pool: {N_WORKERS} workers\n")

    print("Building scene...", end=" ", flush=True)
    scene = make_scene()
    print("done\n")

    # --- Old pattern ---
    print(f"Old pattern (no caching, manual resize)...")
    t0 = time.perf_counter()
    old_result = old_pattern(scene, N_PROCESSORS)
    t_old = time.perf_counter() - t0
    print(f"  {t_old:.2f}s  ({N_PROCESSORS} x {len(BAND_NAMES)} bands "
          f"= {N_PROCESSORS * len(BAND_NAMES)} conversions + resizes)\n")

    # --- New pattern (sequential) ---
    scene._band_cache.clear()
    print(f"New pattern — sequential (cached get_band)...")
    t0 = time.perf_counter()
    seq_result = new_pattern_sequential(scene, N_PROCESSORS)
    t_seq = time.perf_counter() - t0
    print(f"  {t_seq:.2f}s  ({len(BAND_NAMES)} computations, rest cache hits)\n")

    # --- New pattern (parallel) ---
    scene._band_cache.clear()
    print(f"New pattern — parallel  (get_bands n_workers={N_WORKERS})...")
    t0 = time.perf_counter()
    par_result = new_pattern_parallel(scene, N_PROCESSORS, N_WORKERS)
    t_par = time.perf_counter() - t0
    print(f"  {t_par:.2f}s  ({len(BAND_NAMES)} computations across {N_WORKERS} threads, "
          f"rest cache hits)\n")

    # --- Summary ---
    print("=" * 60)
    print(f"{'Old (uncached)':30s}  {t_old:6.1f}s")
    print(f"{'New sequential':30s}  {t_seq:6.1f}s   ({t_old / t_seq:.1f}x vs old)")
    print(f"{'New parallel (n_workers=' + str(N_WORKERS) + ')':30s}  {t_par:6.1f}s   ({t_old / t_par:.1f}x vs old, "
          f"{t_seq / t_par:.1f}x vs sequential)")

    # Sanity: all results should match
    np.testing.assert_allclose(old_result, seq_result, rtol=1e-5)
    np.testing.assert_allclose(old_result, par_result, rtol=1e-5)
    print("\nResults match across all three methods.")
