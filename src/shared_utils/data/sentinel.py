from pathlib import Path
from pydantic import Field
import pyproj
import numpy as np
import xml.etree.ElementTree as ET
import rasterio as rio
from rasterio.windows import Window
from typing import Optional, Any, Dict, List, Tuple, Union, ClassVar

from .base import Data
from ..constants import BANDS, BAND_RESOLUTIONS

class Sentinel2Scene(Data):
    # Data model for a Sentinel-2 scene. Slightly confusingly, we don't use a metadata class to hold
    # the metadata here, instead storing it directly in the scene object. This is because we don't have
    # a write method for Sentinel-2 scenes, so the metadata is only ever read, not written, unlike other
    # data types.

    scene_directory: Optional[Path] = None
    bands: Dict[str, Any] = Field(default_factory=dict)
    footprints: Dict[str, Any] = Field(default_factory=dict)
    sun_zenith: Optional[float] = None
    sun_azimuth: Optional[float] = None
    view_zenith: Optional[float] = None
    view_azimuth: Optional[float] = None
    image_azimuth: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    orientation: Optional[float] = None
    orbit_type: Optional[str] = None
    crs: Optional[Any] = None
    transform: Optional[Any] = None

    # Constants for SAFE format
    GRANULE_DIR: ClassVar[str] = "GRANULE"
    IMG_DATA_DIR: ClassVar[str] = "IMG_DATA"
    QI_DATA_DIR: ClassVar[str] = "QI_DATA"
    METADATA_TL: ClassVar[str] = "MTD_TL.xml"
    METADATA_MTD: ClassVar[str] = "MTD_MSIL1C.xml"

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
             
        self.bands = self._get_bands(self.scene_directory, bands, crop_window)

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
             print(f"Warning: Could not extract CRS/Transform: {e}")

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
             
        self.orientation = self._get_scene_orientation(self.scene_directory)
        self.orbit_type = self._get_orbit_type(self.scene_directory)

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

    def _get_bands(self, scene_directory: Path, bands: List[str], crop_window=None):
        paths = self._get_band_paths(scene_directory, bands)
        bands_data = {}
        for band, path in paths.items():
            with rio.open(path) as src:
                if crop_window:
                    scale = BAND_RESOLUTIONS.get('B02', 10) / BAND_RESOLUTIONS.get(band, 10)
                    col_off, row_off, width, height = crop_window
                    
                    window = Window(
                        col_off=int(col_off * scale),
                        row_off=int(row_off * scale),
                        width=int(width * scale),
                        height=int(height * scale)
                    )
                    bands_data[band] = src.read(1, window=window)
                else:
                    bands_data[band] = src.read(1)
        return bands_data

    def _get_footprints(self, scene_directory: Path, bands: List[str], crop_window=None):
        # Logic mirrors _get_bands but for footprints
        paths = self._get_footprint_paths(scene_directory, bands)
        footprints = {}
        for band, path in paths.items():
            with rio.open(path) as src:
                if crop_window:
                    # Assuming same resolution scaling applies to footprints as bands
                    # (This is true for Sentinel-2 typically)
                    scale = BAND_RESOLUTIONS.get('B02', 10) / BAND_RESOLUTIONS.get(band, 10)
                    col_off, row_off, width, height = crop_window
                    
                    window = Window(
                        col_off=int(col_off * scale),
                        row_off=int(row_off * scale),
                        width=int(width * scale),
                        height=int(height * scale)
                    )
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

    def _get_scene_orientation(self, scene_directory: Path):
        paths = self._get_band_paths(scene_directory, bands=['B02'])
        with rio.open(paths['B02']) as src:
            # Calculate UTM coords of top-left and top-right pixels
            # Affine transform: x' = a*x + b*y + c, y' = d*x + e*y + f
            # top-left pixel (0,0)
            tl_utm = src.transform * (0, 0)
            # top-right pixel (width, 0)
            tr_utm = src.transform * (src.width, 0)
            crs = src.crs
            
        transform = pyproj.transformer.Transformer.from_crs(crs, 'EPSG:4326')
        # Note: transform() expects (x, y) if the CRS is configured right, but pyproj matches CRS definition order.
        # EPSG:4326 is usually lat, lon in some versions, but 'always_xy=True' is safer if used. 
        # Here we used default previously.
        # Previous code: transform(x,y) -> (lat, lon) usually for 4326 without always_xy
        
        tl_geo = transform.transform(tl_utm[0], tl_utm[1])
        tr_geo = transform.transform(tr_utm[0], tr_utm[1])
        
        # Arctan2(delta_y, delta_x)
        # Y is Lat, X is Lon in the previous code's output implicitly
        orientation = np.pi/2 - np.arctan2(
            tr_geo[1] - tl_geo[1], 
            tr_geo[0] - tl_geo[0]
        )
        return orientation

    def _get_orbit_type(self, scene_directory: Path):
        root = self._read_xml_root(scene_directory / self.METADATA_MTD)
        node = root.find(".//SENSING_ORBIT_DIRECTION")
        if node is None:
             raise ValueError("SENSING_ORBIT_DIRECTION not found")
        return node.text
