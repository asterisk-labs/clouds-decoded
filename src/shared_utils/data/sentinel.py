import logging
from datetime import datetime
from pathlib import Path
from pydantic import Field, PrivateAttr
import pyproj
import numpy as np
import xml.etree.ElementTree as ET
import rasterio as rio
from rasterio.windows import Window
from typing import Optional, Any, Dict, List, Tuple, Union, ClassVar

from .base import Data
from .band import Sentinel2Band, BandDict, BandUnit

logger = logging.getLogger(__name__)
from ..constants import BANDS, BAND_RESOLUTIONS

class Sentinel2Scene(Data):
    # Data model for a Sentinel-2 scene. Slightly confusingly, we don't use a metadata class to hold
    # the metadata here, instead storing it directly in the scene object. This is because we don't have
    # a write method for Sentinel-2 scenes, so the metadata is only ever read, not written, unlike other
    # data types.

    scene_directory: Optional[Path] = None
    bands: Dict[str, Any] = Field(default_factory=BandDict)
    footprints: Dict[str, Any] = Field(default_factory=dict)

    _band_cache: Dict[tuple, Sentinel2Band] = PrivateAttr(default_factory=dict)
    _resized_band_cache: Dict[tuple, np.ndarray] = PrivateAttr(default_factory=dict)
    sun_zenith: Optional[float] = None
    sun_azimuth: Optional[float] = None
    view_zenith: Optional[float] = None
    view_azimuth: Optional[float] = None
    image_azimuth: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    orbit_type: Optional[str] = None
    sensing_time: Optional[datetime] = None
    is_refocused: bool = False
    crs: Optional[Any] = None
    transform: Optional[Any] = None

    # Product identification (from MTD_MSIL1C.xml)
    product_uri: Optional[str] = Field(
        default=None,
        description="ESA product URI, e.g. 'S2B_MSIL1C_20250104T185019_N0511_R127_T09KVQ_20250104T220125.SAFE'"
    )

    # Radiometric calibration (read from product metadata)
    quantification_value: float = Field(default=10000.0)
    radio_add_offset: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-band radiometric offset keyed by band name. "
                    "Empty dict means offset=0 for all bands (older products)."
    )

    # Constants for SAFE format
    GRANULE_DIR: ClassVar[str] = "GRANULE"
    IMG_DATA_DIR: ClassVar[str] = "IMG_DATA"
    QI_DATA_DIR: ClassVar[str] = "QI_DATA"
    METADATA_TL: ClassVar[str] = "MTD_TL.xml"
    METADATA_MTD: ClassVar[str] = "MTD_MSIL1C.xml"

    def model_post_init(self, __context: Any) -> None:
        """Ensure bands is always a BandDict for auto-wrapping."""
        if not isinstance(self.bands, BandDict):
            bd = BandDict()
            for k, v in self.bands.items():
                bd[k] = v
            object.__setattr__(self, 'bands', bd)

    def read(self, filepath: Union[str, Path], bands: List[str] = None, crop_window: Tuple[int, int, int, int] = None):
        """
        Read data from a Sentinel-2 scene directory.
        
        Args:
            filepath: Path to .SAFE directory
            bands: List of band names to load
            crop_window: Optional tuple (col_off, row_off, width, height) in 10m pixels (B02 frame).
                         If provided, effectively crops the scene to this window.
        """
        self.scene_directory = Path(filepath)
        if bands is None:
             bands = BANDS
             
        # Use object.__setattr__ to bypass Pydantic's Dict[str, Any] coercion
        # which would strip BandDict to plain dict
        object.__setattr__(self, 'bands', self._get_bands(self.scene_directory, bands, crop_window))
        self._band_cache.clear()
        self._resized_band_cache.clear()

        # Get Georeferencing from reference band (B02)
        scene_width, scene_height = 0, 0
        
        try:
             # Use B02 as reference for CRS and Transform (10m resolution)
             ref_paths = self._get_band_paths(self.scene_directory, ["B02"])
             if "B02" in ref_paths:
                  with rio.open(ref_paths["B02"]) as src:
                       self.crs = src.crs
                       base_transform = src.transform
                       full_width, full_height = src.width, src.height
                  
                  if crop_window:
                       col_off, row_off, width, height = crop_window
                       self.transform = rio.windows.transform(
                           Window(col_off, row_off, width, height),
                           base_transform
                       )
                       scene_width, scene_height = width, height
                  else:
                       self.transform = base_transform
                       scene_width, scene_height = full_width, full_height
                       
        except Exception as e:
             # Just warn, don't crash if we can't extract CRS
             logger.warning(f"Could not extract CRS/Transform: {e}")

        self.footprints = self._get_footprints(self.scene_directory, bands, crop_window)
        self.sun_zenith, self.sun_azimuth = self._get_sun_angle(self.scene_directory)
        self.view_zenith, self.view_azimuth = self._get_view_angle(self.scene_directory)
        self.image_azimuth = self._get_orbit_image_angle(self.scene_directory)
        
        # Calculate Lat/Lon center based on (potentially cropped) geometry
        if self.transform is not None:
             self.latitude, self.longitude = self._calculate_center_coords(scene_width, scene_height)
        else:
             # Fallback (though probably won't be accurate if cropped and transform failed)
             self.latitude = self._get_latitude(self.scene_directory)
             self.longitude = self._get_longitude(self.scene_directory)
             
        self.orbit_type = self._get_orbit_type(self.scene_directory)

        # Read product metadata
        self.product_uri = self._get_product_uri(self.scene_directory)
        self.quantification_value, self.radio_add_offset = self._get_radiometric_params(
            self.scene_directory
        )
        self.sensing_time = self._get_sensing_time(self.scene_directory)

    def _calculate_center_coords(self, width: int, height: int) -> Tuple[float, float]:
        """Calculate center latitude/longitude based on current transform and size."""
        # Center in pixel coords
        cx, cy = width / 2, height / 2
        # Center in projected coords (X, Y)
        px, py = self.transform * (cx, cy)
        
        # Reproject to EPSG:4326
        transformer = pyproj.transformer.Transformer.from_crs(
             self.crs, 'EPSG:4326', always_xy=True
        )
        lon, lat = transformer.transform(px, py)
        return lat, lon

    def get_scene_size_meters(self) -> Tuple[float, float]:
        """
        Calculates the scene dimensions in meters based on a loaded band and its resolution.
        Attempts to use 'B02' (10m) first, then falls back to any available band.
        
        Returns:
            Tuple[float, float]: (width_meters, height_meters)
        
        Raises:
            ValueError: If no bands are loaded or if resolution is unknown for available bands.
        """
        # Try finding a resolution from available bands
        band_name = None
        if "B02" in self.bands:
            band_name = "B02"
        elif self.bands:
            band_name = list(self.bands.keys())[0]
        else:
            raise ValueError("No bands loaded to determine scene size.")

        if band_name not in BAND_RESOLUTIONS:
             raise ValueError(f"Unknown resolution for band {band_name}")

        resolution = BAND_RESOLUTIONS[band_name]
        band_data = self.bands[band_name]
        
        # Band data might be (C, H, W) or (H, W)
        if band_data.ndim == 3:
             height, width = band_data.shape[1], band_data.shape[2]
        elif band_data.ndim == 2:
             height, width = band_data.shape[0], band_data.shape[1]
        else:
             raise ValueError(f"Invalid shape for band {band_name}: {band_data.shape}")
             
        return (float(width * resolution), float(height * resolution))

    def get_band(
        self,
        band_name: str,
        reflectance: bool = True,
        resolution: Optional[int] = None,
        cache: bool = True,
    ) -> np.ndarray:
        """Retrieve a band array, optionally converted to TOA reflectance.

        Derived arrays are cached on the scene so repeated calls with the
        same parameters return the cached result without recomputation.

        Args:
            band_name: Band identifier (e.g. ``'B02'``, ``'B8A'``).
            reflectance: If True, return TOA reflectance as float32.
                If False, return raw DN values as stored.
            resolution: Target pixel size in metres. If ``None`` (default),
                return at the band's native resolution.
            cache: If True (default), store the derived band in an internal
                cache for fast repeated access.  If False, skip storing but
                still return an already-cached result if one exists.

        Returns:
            2-D numpy array of the band data.
        """
        if band_name not in self.bands:
            raise KeyError(f"Band '{band_name}' not loaded. Available: {list(self.bands.keys())}")

        raw = self.bands[band_name]

        # Normalise to Sentinel2Band (defensive — should already be one via BandDict)
        if isinstance(raw, Sentinel2Band):
            root = raw
        else:
            root = Sentinel2Band(
                name=band_name,
                data=raw,
                native_resolution=BAND_RESOLUTIONS.get(band_name),
            )

        # Fast path: raw DN at native resolution — no derivation needed
        if not reflectance and resolution is None:
            return root.data

        # Build a cache key that includes calibration params for correctness
        offset = self.radio_add_offset.get(band_name, 0.0) if reflectance else 0.0
        cache_key = (band_name, reflectance, resolution, offset, self.quantification_value)

        # Always check the cache — no point recomputing what's already in memory
        if cache_key in self._band_cache:
            return self._band_cache[cache_key].data

        current = root
        if reflectance:
            current = current.to_reflectance(offset, self.quantification_value)
        if resolution is not None:
            current = current.to_resolution(resolution)

        if cache:
            self._band_cache[cache_key] = current

        return current.data

    def get_bands(
        self,
        band_names: Optional[List[str]] = None,
        reflectance: bool = True,
        resolution: Optional[int] = None,
        cache: bool = True,
        n_workers: int = 1,
    ) -> List[Sentinel2Band]:
        """Retrieve multiple bands as ``Sentinel2Band`` objects.

        Args:
            band_names: Band identifiers to retrieve. Defaults to all loaded bands.
            reflectance: If True, convert to TOA reflectance.
            resolution: Target pixel size in metres, or None for native.
            cache: If True (default), store derived bands in an internal
                cache.  If False, skip storing but still return
                already-cached results if they exist.
            n_workers: Number of threads for parallel band evaluation.
                When > 1, uncached bands are computed concurrently using a
                thread pool (numpy/skimage release the GIL, so threads give
                true parallelism for the heavy work).  Use ``-1`` to
                auto-size the pool to the number of uncached bands.
                Defaults to 1 (sequential).

        Returns:
            List of ``Sentinel2Band`` objects (with evaluated data).
        """
        if band_names is None:
            band_names = list(self.bands.keys())

        # First pass: resolve cache hits and build derivation chains for misses.
        # result[i] is filled immediately for cache hits; misses are collected
        # in to_compute for (possibly parallel) evaluation.
        result: List[Optional[Sentinel2Band]] = [None] * len(band_names)
        to_compute: List[Tuple[int, Optional[tuple], Sentinel2Band]] = []  # (index, cache_key, band)

        for i, name in enumerate(band_names):
            if name not in self.bands:
                raise KeyError(f"Band '{name}' not loaded. Available: {list(self.bands.keys())}")

            raw = self.bands[name]
            if isinstance(raw, Sentinel2Band):
                root = raw
            else:
                root = Sentinel2Band(
                    name=name,
                    data=raw,
                    native_resolution=BAND_RESOLUTIONS.get(name),
                )

            if not reflectance and resolution is None:
                result[i] = root
                continue

            offset = self.radio_add_offset.get(name, 0.0) if reflectance else 0.0
            cache_key = (name, reflectance, resolution, offset, self.quantification_value)

            if cache_key in self._band_cache:
                result[i] = self._band_cache[cache_key]
                continue

            current = root
            if reflectance:
                current = current.to_reflectance(offset, self.quantification_value)
            if resolution is not None:
                current = current.to_resolution(resolution)

            to_compute.append((i, cache_key, current))

        # Second pass: evaluate uncached bands (sequential or parallel).
        if to_compute:
            if n_workers == -1:
                import os
                n_workers = min(len(to_compute), os.cpu_count() or 1)

            if n_workers > 1:
                from concurrent.futures import ThreadPoolExecutor

                def _evaluate(band: Sentinel2Band) -> Sentinel2Band:
                    _ = band.data  # force lazy evaluation
                    return band

                bands_to_eval = [band for _, _, band in to_compute]
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    evaluated = list(pool.map(_evaluate, bands_to_eval))

                for (idx, cache_key, _), band in zip(to_compute, evaluated):
                    result[idx] = band
                    if cache:
                        self._band_cache[cache_key] = band
            else:
                for idx, cache_key, band in to_compute:
                    _ = band.data  # force lazy evaluation
                    result[idx] = band
                    if cache:
                        self._band_cache[cache_key] = band

        return result

    def get_band_at_shape(
        self,
        band_name: str,
        target_shape: Tuple[int, int],
        reflectance: bool = True,
    ) -> np.ndarray:
        """Return band data resized to *target_shape*, using a per-scene cache.

        The cache is keyed by ``(band_name, reflectance, target_shape)`` so
        multiple processors sharing a scene object automatically share resized
        arrays.  Thread-safe for concurrent reads and writes under CPython's GIL.

        Args:
            band_name: Sentinel-2 band identifier (e.g. ``"B02"``).
            target_shape: ``(height, width)`` in pixels.
            reflectance: If True, return top-of-atmosphere reflectance.

        Returns:
            ``float32`` array of shape ``target_shape``.
        """
        key = (band_name, reflectance, target_shape)
        cached = self._resized_band_cache.get(key)
        if cached is not None:
            return cached
        raw = self.get_band(band_name, reflectance=reflectance)
        if raw.shape == target_shape:
            arr = raw.astype(np.float32)
        else:
            from skimage.transform import resize
            arr = resize(raw, target_shape, preserve_range=True, order=1).astype(np.float32)
        self._resized_band_cache[key] = arr
        return arr

    def prefetch_at_shape(
        self,
        band_names: List[str],
        target_shape: Tuple[int, int],
        reflectance: bool = True,
        n_workers: int = -1,
    ) -> None:
        """Pre-compute and cache *band_names* resized to *target_shape* in parallel.

        Intended to be called from a background thread in the reader pipeline
        stage so that resized arrays are ready before a downstream processor
        requests them via :meth:`get_band_at_shape`.

        Args:
            band_names: Bands to prefetch.
            target_shape: Target ``(height, width)``.
            reflectance: If True, use top-of-atmosphere reflectance.
            n_workers: Number of threads.  -1 = one per band (I/O + GIL-releasing
                resize ops benefit from threading even in CPython).
        """
        import os
        from concurrent.futures import ThreadPoolExecutor
        n = len(band_names) if n_workers == -1 else max(1, n_workers)
        n = min(n, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(lambda b: self.get_band_at_shape(b, target_shape, reflectance), band_names))

    def write(self, filepath: str):

        """Writing Sentinel-2 scenes is not supported."""
        raise NotImplementedError("Writing Sentinel-2 scenes is not supported.")

    def _find_single_path(self, parent: Path, pattern: str) -> Path:
        """Find exactly one file/dir matching pattern in parent."""
        matches = list(parent.glob(pattern))
        if len(matches) == 0:
            raise FileNotFoundError(f"No matches for '{pattern}' in {parent}")
        if len(matches) > 1:
            raise ValueError(f"Multiple matches for '{pattern}' in {parent}: {matches}")
        return matches[0]

    def _get_granule_directory(self, scene_directory: Path) -> Path:
        granule_base = scene_directory / self.GRANULE_DIR
        # Find the single subdirectory that isn't a hidden file (like .DS_Store)
        # Using iterdir() logic manually since glob doesn't easily say "is directory and not hidden"
        subdirs = [p for p in granule_base.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if len(subdirs) != 1:
            raise ValueError(f"Expected exactly one granule directory in {granule_base}, found {len(subdirs)}")
        return subdirs[0]

    def _get_band_paths(self, scene_directory: Path, bands: List[str]) -> Dict[str, Path]:
        granule = self._get_granule_directory(scene_directory)
        img_data = granule / self.IMG_DATA_DIR
        
        paths = {}
        for band in bands:
            # Replaces: [f for f in files if f.endswith(f"{band}.jp2")]
            # Uses glob pattern matching
            paths[band] = self._find_single_path(img_data, f"*{band}.jp2")
        return paths

    def _get_footprint_paths(self, scene_directory: Path, bands: List[str]) -> Dict[str, Path]:
        granule = self._get_granule_directory(scene_directory)
        qi_data = granule / self.QI_DATA_DIR
        
        paths = {}
        for band in bands:
            # Replaces: [f for f in files if f.endswith(f"DETFOO_{band}.jp2")]
            paths[band] = self._find_single_path(qi_data, f"*DETFOO_{band}.jp2")
        return paths
        
    def _read_xml_root(self, xml_path: Path) -> ET.Element:
        if not xml_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {xml_path}")
        return ET.parse(xml_path).getroot()

    # --- Data Retrieval Methods ---

    def _get_b02_dims(self, scene_directory: Path, existing_paths: dict) -> tuple:
        """Return the actual (width, height) of the B02 band file.

        Uses an already-resolved path from *existing_paths* if available,
        otherwise resolves it fresh.  This gives a reliable reference for
        scaling crop windows without relying on nominal BAND_RESOLUTIONS
        constants, which may differ from actual pixel dimensions in newer
        SAFE processing baselines.
        """
        if 'B02' in existing_paths:
            b02_path = existing_paths['B02']
        else:
            b02_path = self._get_band_paths(scene_directory, ['B02'])['B02']
        with rio.open(b02_path) as src:
            return src.width, src.height

    @staticmethod
    def _scale_crop_window(
        crop_window: tuple,
        b02_w: int,
        b02_h: int,
        src_w: int,
        src_h: int,
    ) -> "Window":
        """Scale a B02-pixel crop window to the coordinate space of a band raster.

        Args:
            crop_window: ``(col_off, row_off, width, height)`` in B02 (10 m) pixels.
            b02_w: Full-scene B02 pixel width (from the actual file, not BAND_RESOLUTIONS).
            b02_h: Full-scene B02 pixel height.
            src_w: Pixel width of the target band raster.
            src_h: Pixel height of the target band raster.

        Returns:
            A rasterio ``Window`` in the target band's pixel coordinates.
        """
        col_off, row_off, width, height = crop_window
        scale_x = src_w / b02_w
        scale_y = src_h / b02_h
        return Window(
            col_off=int(col_off * scale_x),
            row_off=int(row_off * scale_y),
            width=int(width * scale_x),
            height=int(height * scale_y),
        )

    def _get_bands(self, scene_directory: Path, bands: List[str], crop_window=None):
        paths = self._get_band_paths(scene_directory, bands)
        bands_data = BandDict()
        if crop_window:
            b02_w, b02_h = self._get_b02_dims(scene_directory, paths)
        for band, path in paths.items():
            with rio.open(path) as src:
                if crop_window:
                    window = self._scale_crop_window(crop_window, b02_w, b02_h, src.width, src.height)
                    bands_data[band] = src.read(1, window=window)
                else:
                    bands_data[band] = src.read(1)
        return bands_data

    def _get_footprints(self, scene_directory: Path, bands: List[str], crop_window=None):
        # Logic mirrors _get_bands but for footprints
        paths = self._get_footprint_paths(scene_directory, bands)
        footprints = {}
        if crop_window:
            b02_w, b02_h = self._get_b02_dims(scene_directory, paths)
        for band, path in paths.items():
            with rio.open(path) as src:
                if crop_window:
                    window = self._scale_crop_window(crop_window, b02_w, b02_h, src.width, src.height)
                    footprints[band] = src.read(1, window=window)
                else:
                    footprints[band] = src.read(1)
        return footprints

    def _get_sun_angle(self, scene_directory: Path):
        granule = self._get_granule_directory(scene_directory)
        root = self._read_xml_root(granule / self.METADATA_TL)
        
        # Look in tree for "Mean_Sun_Angle"
        sun_angle_node = root.find(".//Mean_Sun_Angle")
        if sun_angle_node is None:
            raise ValueError("Mean_Sun_Angle not found in metadata")

        zenith = float(sun_angle_node.find("ZENITH_ANGLE").text)
        azimuth = float(sun_angle_node.find("AZIMUTH_ANGLE").text)
        return zenith, azimuth

    def _get_view_angle(self, scene_directory: Path):
        """
        Parses Mean_Viewing_Incidence_Angle_List from MTD_TL.xml and calculates
        the mean Zenith and Azimuth across all available bands.
        """
        granule = self._get_granule_directory(scene_directory)
        root = self._read_xml_root(granule / self.METADATA_TL)
        
        angle_list_node = root.find(".//Mean_Viewing_Incidence_Angle_List")
        if angle_list_node is None:
            # Maybe warning? Or raise? For now raise as it is critical for cloud properties
            # Some old products (pre-2016?) might have different structure.
            # But normally it exists.
            raise ValueError("Mean_Viewing_Incidence_Angle_List not found in metadata")

        zeniths = []
        azimuths = []

        for angle_node in angle_list_node.findall("Mean_Viewing_Incidence_Angle"):
            z = float(angle_node.find("ZENITH_ANGLE").text)
            a = float(angle_node.find("AZIMUTH_ANGLE").text)
            zeniths.append(z)
            azimuths.append(a)
            
        if not zeniths:
            return 0.0, 0.0 # Or None?
            
        mean_zenith = float(np.mean(zeniths))
        
        # Calculate circular mean for azimuth to correctly handle 0/360 boundary
        az_rad = np.deg2rad(azimuths)
        mean_azimuth = float(np.rad2deg(np.arctan2(
            np.sum(np.sin(az_rad)), 
            np.sum(np.cos(az_rad))
        )))
        # Normalize to [0, 360)
        mean_azimuth = (mean_azimuth + 360) % 360
        
        return mean_zenith, mean_azimuth

    def _get_orbit_image_angle(self, scene_directory: Path):
        # Relies on B02 footprint
        paths = self._get_footprint_paths(scene_directory, bands=['B02'])
        with rio.open(paths['B02']) as src:
            footprint = src.read(1)

        # First, let's get rid of any rows and columns that are entirely zero (no data)
        del_rows = np.where(np.all(footprint == 0, axis=1))[0]
        del_cols = np.where(np.all(footprint == 0, axis=0))[0]
        footprint = np.delete(footprint, del_rows, axis=0)
        footprint = np.delete(footprint, del_cols, axis=1)

        # Vectorized approach or column search? Keeping original logic but cleaned up variables
        rows, cols = footprint.shape
        lower_id, upper_id = None, None
        
        # 1. Find the first column dealing with a transition
        transition_col_idx = -1
        lower_point_y = -1
        
        for col_idx in range(cols):
            column = footprint[:, col_idx]
            # check if there is a change in values in this column (transition between detector footprints)
            diffs = np.diff(column)
            if np.any(diffs == 1): # specific transition type check from original code
                transition_idx = np.where(diffs == 1)[0]
                if len(transition_idx) > 0:
                    y = transition_idx[0]
                    l_id = column[y]
                    u_id = column[y + 1]
                    if l_id != u_id:
                        lower_point_y = y
                        lower_id = l_id
                        upper_id = u_id
                        transition_col_idx = col_idx
                        break
        
        if transition_col_idx == -1:
             return 0.0 # Default if no transition found (e.g. full scene)

        # 2. Find the end column for this transition pair
        last_col_idx = transition_col_idx
        for next_col_idx in range(transition_col_idx + 1, cols):
            next_column = footprint[:, next_col_idx]
            if lower_id in next_column and upper_id in next_column:
                last_col_idx = next_col_idx
            else:
                break
        
        # 3. Calculate angle
        horizontal_dist = last_col_idx - transition_col_idx
        if horizontal_dist == 0:
            return 0.0
            
        next_column = footprint[:, last_col_idx]
        # Find where the transition is now
        # argmax gives index of first True match
        upper_point_y = np.argmax(next_column == lower_id)
        
        vertical_dist = lower_point_y - upper_point_y
        angle = -np.arctan2(horizontal_dist, vertical_dist)
        return angle

    def _get_latitude(self, scene_directory: Path):
        paths = self._get_band_paths(scene_directory, bands=['B02'])
        with rio.open(paths['B02']) as src:
            bounds = src.bounds
            crs = src.crs
            
        transform = pyproj.transformer.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        # Lat is Y
        lat = (
            transform.transform(bounds.left, bounds.top)[1] + 
            transform.transform(bounds.right, bounds.bottom)[1]
        ) / 2
        return lat
        
    def _get_longitude(self, scene_directory: Path):
        paths = self._get_band_paths(scene_directory, bands=['B02'])
        with rio.open(paths['B02']) as src:
            bounds = src.bounds
            crs = src.crs
            
        transform = pyproj.transformer.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        # Lon is X
        lon = (
            transform.transform(bounds.left, bounds.top)[0] + 
            transform.transform(bounds.right, bounds.bottom)[0]
        ) / 2
        return lon

    def _get_orbit_type(self, scene_directory: Path):
        root = self._read_xml_root(scene_directory / self.METADATA_MTD)
        node = root.find(".//SENSING_ORBIT_DIRECTION")
        if node is None:
             raise ValueError("SENSING_ORBIT_DIRECTION not found")
        return node.text

    def _get_product_uri(self, scene_directory: Path) -> Optional[str]:
        """Read PRODUCT_URI from MTD_MSIL1C.xml."""
        try:
            root = self._read_xml_root(scene_directory / self.METADATA_MTD)
            node = root.find(".//PRODUCT_URI")
            return node.text if node is not None else None
        except Exception:
            return None

    def _get_radiometric_params(self, scene_directory: Path) -> Tuple[float, Dict[str, float]]:
        """Read QUANTIFICATION_VALUE and per-band RADIO_ADD_OFFSET from MTD_MSIL1C.xml.

        Returns:
            (quantification_value, radio_add_offset_dict). Falls back to
            (10000.0, {}) for older products that lack Radiometric_Offset_List.
        """
        try:
            root = self._read_xml_root(scene_directory / self.METADATA_MTD)

            quant_node = root.find(".//QUANTIFICATION_VALUE")
            quantification = float(quant_node.text) if quant_node is not None else 10000.0

            # Per-band offsets: <RADIO_ADD_OFFSET band_id="0">-1000</RADIO_ADD_OFFSET>
            # band_id is an integer index into the BANDS list
            offsets: Dict[str, float] = {}
            for node in root.findall(".//RADIO_ADD_OFFSET"):
                band_id = node.get("band_id")
                if band_id is not None:
                    idx = int(band_id)
                    if idx < len(BANDS):
                        offsets[BANDS[idx]] = float(node.text)

            return quantification, offsets
        except Exception as e:
            logger.warning(f"Could not read radiometric params: {e}. Using defaults.")
            return 10000.0, {}

    def _get_sensing_time(self, scene_directory: Path) -> Optional[datetime]:
        """Extract sensing time from tile or product metadata."""
        try:
            granule = self._get_granule_directory(scene_directory)
            root = self._read_xml_root(granule / self.METADATA_TL)
            node = root.find(".//SENSING_TIME")
            if node is not None:
                return datetime.fromisoformat(node.text.rstrip('Z'))
        except Exception:
            pass

        try:
            root = self._read_xml_root(scene_directory / self.METADATA_MTD)
            node = root.find(".//DATATAKE_SENSING_START")
            if node is not None:
                return datetime.fromisoformat(node.text.rstrip('Z'))
        except Exception:
            pass

        logger.warning("Could not find sensing time in metadata")
        return None

    def _parse_angle_grid(self, values_list: ET.Element) -> np.ndarray:
        """Parse a Values_List XML element into a 2D numpy array."""
        return np.array([
            [float(v) for v in row.text.split()]
            for row in values_list.findall("VALUES")
        ])

    def _get_sun_angle_grids(self, scene_directory: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse Sun_Angles_Grid from MTD_TL.xml.

        Returns:
            (zenith_grid, azimuth_grid) as 2D float64 arrays (typically 23x23).
        """
        granule = self._get_granule_directory(scene_directory)
        root = self._read_xml_root(granule / self.METADATA_TL)

        sun_grid = root.find(".//Sun_Angles_Grid")
        if sun_grid is None:
            raise ValueError("Sun_Angles_Grid not found in MTD_TL.xml")

        zenith_grid = self._parse_angle_grid(sun_grid.find("Zenith/Values_List"))
        azimuth_grid = self._parse_angle_grid(sun_grid.find("Azimuth/Values_List"))
        return zenith_grid, azimuth_grid

    def _get_view_angle_grids(self, scene_directory: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse Viewing_Incidence_Angles_Grid from MTD_TL.xml.

        Averages across all detectors and bands using nanmean to handle
        NaN values outside each detector's footprint.

        Returns:
            (zenith_grid, azimuth_grid) as 2D float64 arrays (typically 23x23).
        """
        granule = self._get_granule_directory(scene_directory)
        root = self._read_xml_root(granule / self.METADATA_TL)

        all_zenith = []
        all_azimuth = []

        for grid in root.findall(".//Viewing_Incidence_Angles_Grids"):
            z_values = grid.find("Zenith/Values_List")
            a_values = grid.find("Azimuth/Values_List")

            z_grid = np.array([
                [float(v) if v != 'NaN' else np.nan for v in row.text.split()]
                for row in z_values.findall("VALUES")
            ])
            a_grid = np.array([
                [float(v) if v != 'NaN' else np.nan for v in row.text.split()]
                for row in a_values.findall("VALUES")
            ])
            all_zenith.append(z_grid)
            all_azimuth.append(a_grid)

        if not all_zenith:
            raise ValueError("No Viewing_Incidence_Angles_Grids found in MTD_TL.xml")

        zenith_grid = np.nanmean(np.stack(all_zenith), axis=0)
        azimuth_grid = np.nanmean(np.stack(all_azimuth), axis=0)

        # Fill remaining NaN cells (grid corners not covered by any detector)
        # with nearest-neighbour so the pixel interpolator has no holes.
        zenith_grid = self._fill_nan_nearest(zenith_grid)
        azimuth_grid = self._fill_nan_nearest(azimuth_grid)

        return zenith_grid, azimuth_grid

    @staticmethod
    def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
        """Fill NaN cells in a 2D grid using nearest valid neighbour."""
        if not np.isnan(grid).any():
            return grid
        from scipy.ndimage import distance_transform_edt
        mask = np.isnan(grid)
        _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
        return grid[tuple(nearest_idx)]

    def get_angles_at_pixels(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        resolution: float = 10.0,
    ) -> Dict[str, np.ndarray]:
        """Interpolate sun and view angles at specific pixel locations.

        Uses the 5km-step angle grids from MTD_TL.xml and bilinear
        interpolation to estimate angles at arbitrary pixel coordinates.

        Args:
            rows: Row indices in the reference band pixel grid.
            cols: Column indices in the reference band pixel grid.
            resolution: Pixel resolution in meters (default 10m for B02).

        Returns:
            Dict with keys 'sun_zenith', 'sun_azimuth', 'view_zenith',
            'view_azimuth', each a 1D float32 array.
        """
        from scipy.interpolate import RegularGridInterpolator

        step = 5000.0
        sun_z, sun_a = self._get_sun_angle_grids(self.scene_directory)
        view_z, view_a = self._get_view_angle_grids(self.scene_directory)

        n_rows_grid, n_cols_grid = sun_z.shape
        grid_y = np.arange(n_rows_grid) * step
        grid_x = np.arange(n_cols_grid) * step

        pixel_y = rows.astype(np.float64) * resolution
        pixel_x = cols.astype(np.float64) * resolution
        points = np.column_stack([pixel_y, pixel_x])

        result = {}
        for name, grid_data in [
            ('sun_zenith', sun_z), ('sun_azimuth', sun_a),
            ('view_zenith', view_z), ('view_azimuth', view_a),
        ]:
            interp = RegularGridInterpolator(
                (grid_y, grid_x), grid_data,
                method='linear', bounds_error=False, fill_value=None,
            )
            result[name] = interp(points).astype(np.float32)

        return result

    def get_wind_data(self) -> Tuple[float, float]:
        """Read 10m wind speed and direction from AUX_ECMWFT GRIB file.

        Returns:
            (wind_speed_ms, wind_direction_deg) as scene-level scalars.

        Raises:
            ImportError: If cfgrib is not installed.
        """
        try:
            import cfgrib
        except ImportError:
            raise ImportError(
                "cfgrib is required for reading AUX_ECMWFT wind data. "
                "Install with: pip install cfgrib"
            )

        granule = self._get_granule_directory(self.scene_directory)
        aux_dir = granule / "AUX_DATA"
        grib_path = self._find_single_path(aux_dir, "AUX_ECMWFT")

        # Disable cfgrib index file creation to avoid PermissionError
        # on read-only scene directories.
        # Suppress eccodes "unable to represent the step in h" noise — eccodes
        # writes directly to C stderr, so suppress at the fd level.
        import os
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            datasets = cfgrib.open_datasets(
                str(grib_path),
                backend_kwargs={"indexpath": ""},
                decode_timedelta=False,
            )
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)

        u10 = None
        v10 = None
        for ds in datasets:
            if 'u10' in ds:
                u10 = float(ds['u10'].values.mean())
            if 'v10' in ds:
                v10 = float(ds['v10'].values.mean())

        if u10 is None or v10 is None:
            raise ValueError("Could not find u10/v10 wind components in AUX_ECMWFT")

        wind_speed = float(np.sqrt(u10**2 + v10**2))
        wind_direction = float((np.degrees(np.arctan2(-u10, -v10)) + 360) % 360)
        return wind_speed, wind_direction
