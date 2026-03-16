# Sentinel2Scene

Reads Sentinel-2 `.SAFE` directories and provides access to spectral bands, sun/view geometry, and auxiliary data.

::: clouds_decoded.data.sentinel.Sentinel2Scene
    options:
      members:
        - read
        - get_band
        - get_bands
        - get_band_at_shape
        - prefetch_at_shape
        - get_scene_size_meters
        - get_angles_at_pixels
        - get_wind_data
