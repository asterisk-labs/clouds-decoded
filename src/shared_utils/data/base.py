from abc import ABC, abstractmethod
import json
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np
from typing import Optional, Any, Dict, List, Tuple, Union
import rasterio as rio
import pyarrow as pa
import pyarrow.parquet as pq
from clouds_decoded.constants import METADATA_TAG

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types.

    Converts ``np.ndarray`` to nested Python lists and ``np.generic``
    scalars (e.g. ``np.float32``, ``np.int64``) to their native Python
    equivalents so they can be serialized by the standard ``json`` module.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

class Data(BaseModel, ABC):
    """Abstract base class for data models."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @classmethod
    def from_file(cls, filepath: str) -> "Data":
        """Factory method to create instance from file."""
        instance = cls()
        instance.read(filepath)
        return instance

    @abstractmethod
    def read(self, filepath: str):
        """Read data from a file."""
        pass

    @abstractmethod
    def write(self, filepath: str):
        """Write data to a file."""
        pass

class Metadata(BaseModel):
    """Base model for metadata structures.

    Uses ``extra='allow'`` intentionally so that GeoTIFF tags and future
    metadata keys round-trip without errors.  Subclasses should document
    their own fields; unknown keys are preserved silently.
    """
    model_config = ConfigDict(extra='allow')
    provenance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing provenance (project name, version, git hash, config snapshot)"
    )

class GeoRasterData(Data):
    """
    Data model for geospatial raster data (e.g. GeoTIFF).

    Attributes:
        data: The raster data as a numpy array.
        transform: The affine transform of the raster (maps pixel coordinates to map coordinates).
        crs: The coordinate reference system of the raster.
        nodata: The value used to indicate missing/invalid pixels. Default is NaN
            for float rasters. Subclasses may override (e.g. 255 for uint8 masks).
        metadata: Metadata object containing additional information.
    """
    data: Optional[np.ndarray] = None
    transform: Optional[Any] = None
    crs: Optional[Any] = None
    nodata: Optional[float] = Field(default=np.nan, description="Nodata value")
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("data")
    @classmethod
    def validate_array(cls, v: Any) -> Optional[np.ndarray]:
        if v is not None and not isinstance(v, np.ndarray):
            raise ValueError("Data must be a numpy array")
        return v

    def validate(self) -> bool:
        """Validate data integrity. Override in subclasses for specific checks.

        Returns:
            True if data is valid, False otherwise.
        """
        if self.data is None:
            return True
        # Base validation: check it's an array with valid shape
        return isinstance(self.data, np.ndarray) and self.data.ndim in (2, 3)

    def read(self, filepath: Union[str, Path]):
        """Read data from a file (GeoTIFF or NetCDF)."""
        filepath = Path(filepath)
        
        with rio.open(filepath) as src:
            self.data = src.read()
            self.transform = src.transform
            self.crs = src.crs
            
            # Read metadata from tags
            tags = src.tags()
            json_dict = {}
            
            # Try to recover preserved types if they were JSON encoded, else use strings
            for k, v in tags.items():
                if k == METADATA_TAG:
                    try:
                        loaded_meta = json.loads(v)
                        if isinstance(loaded_meta, dict):
                             json_dict.update(loaded_meta)
                    except json.JSONDecodeError:
                        pass # Should handle gracefully
                else:
                    # Preserve other tags if needed, or maybe just standard ones?
                    if k not in json_dict:
                        json_dict[k] = v

            if json_dict:
                # Dynamically instantiate the correct metadata class based on the instance type
                # Check if the subclass redefined the 'metadata' field to a specific type
                # For Pydantic v2, we check dynamic type of self.metadata since default factory instantiated it
                meta_type = type(self.metadata)
                try:
                    self.metadata = meta_type.model_validate(json_dict)
                except (ValueError, TypeError):
                    self.metadata = meta_type(**json_dict)

    def write(self, filepath: Union[str, Path], compression: str = 'lzw'):
        """Write data to a file (GeoTIFF or NetCDF)."""
        if self.data is None:
            raise ValueError("No data to write")

        # Validate data before writing
        if not self.validate():
            logger.warning(f"Data validation failed for {type(self).__name__}. Writing anyway.")

        filepath = Path(filepath)
        is_netcdf = filepath.suffix.lower() == '.nc'
        
        # Determine count, height, width from data shape
        if self.data.ndim == 2:
            count = 1
            height, width = self.data.shape
            out_data = self.data[np.newaxis, ...]
        elif self.data.ndim == 3:
            count, height, width = self.data.shape
            out_data = self.data
        else:
            raise ValueError(f"Data has invalid number of dimensions: {self.data.ndim}")

        if is_netcdf:
            # Fallback to xarray/netcdf4 as GDAL NetCDF driver support varies
            try:
                import xarray as xr
            except ImportError:
                 raise ImportError("xarray is required to write NetCDF files.")
            
            metadata = self.metadata.model_dump(exclude_defaults=True, mode='json')
            
            da = xr.DataArray(
                out_data,
                dims=("band", "y", "x"),
                coords={"band": np.arange(1, count + 1)},
                attrs=metadata
            )
            
            if self.crs:
                 da.attrs['crs'] = str(self.crs)
            if self.transform:
                 da.attrs['transform'] = str(self.transform) # Serializing Affine object
                 
            da.to_netcdf(filepath)
            return

        driver = 'GTiff'

        # Use tiled output with appropriate predictor for better compression
        is_float = np.issubdtype(self.data.dtype, np.floating)
        predictor = 3 if is_float else 2  # 3=float predictor, 2=horizontal differencing

        profile = {
            'driver': driver,
            'height': height,
            'width': width,
            'count': count,
            'dtype': self.data.dtype,
            'crs': self.crs,
            'transform': self.transform,
            'nodata': self.nodata,
            'compress': 'deflate',
            'predictor': predictor,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
        }

        with rio.open(filepath, 'w', **profile) as dst:
            dst.write(out_data)
            
            # Write metadata as tags
            # We bundle the metadata object into a JSON string to preserve structure
            meta_dict = self.metadata.model_dump(exclude_defaults=True, mode='json')
            if meta_dict:
                 json_meta = json.dumps(meta_dict, cls=NumpyEncoder)
                 dst.update_tags(**{METADATA_TAG: json_meta})
                 
                 # Also write flattened simple keys for easier access in other tools if simple types
                 simple_tags = {}
                 for k, v in meta_dict.items():
                     if isinstance(v, (str, int, float, bool)):
                         simple_tags[k] = str(v)
                 if simple_tags:
                     dst.update_tags(**simple_tags)

    def resample(self, target_resolution_m: Optional[int]) -> "GeoRasterData":
        """Resample to a target resolution in metres.

        Returns self unchanged if ``target_resolution_m`` is None, data is None,
        or the current resolution is already within 0.5 m of the target.

        Float data uses bilinear interpolation (order=1) with NaN-safe fill/restore.
        Integer/categorical data uses nearest-neighbour (order=0).
        The affine transform is updated to reflect the new pixel size.

        Args:
            target_resolution_m: Target pixel size in metres, or None for no-op.

        Returns:
            A new instance of the same subclass at the requested resolution,
            or ``self`` if no resampling is needed.
        """
        if target_resolution_m is None or self.data is None:
            return self
        current_res = abs(self.transform.a)
        if abs(current_res - target_resolution_m) < 0.5:
            return self

        from skimage.transform import resize as sk_resize
        from rasterio.transform import Affine

        scale = current_res / target_resolution_m  # <1 downsample, >1 upsample
        data = self.data
        squeeze = data.ndim == 2
        if squeeze:
            data = data[np.newaxis]
        C, H, W = data.shape
        new_H = max(1, round(H * scale))
        new_W = max(1, round(W * scale))

        is_float = np.issubdtype(data.dtype, np.floating)
        if is_float:
            nan_mask = np.isnan(data)
            filled = np.where(nan_mask, 0.0, data)
            resized = sk_resize(
                filled, (C, new_H, new_W), order=1,
                preserve_range=True, anti_aliasing=False,
            )
            nan_mask_r = sk_resize(
                nan_mask.astype(np.float32), (C, new_H, new_W),
                order=0, preserve_range=True,
            ) > 0.5
            resized[nan_mask_r] = np.nan
        else:
            resized = sk_resize(
                data, (C, new_H, new_W), order=0,
                preserve_range=True, anti_aliasing=False,
            ).astype(data.dtype)

        if squeeze:
            resized = resized[0]

        new_transform = self.transform * Affine.scale(1.0 / scale)
        return self.model_copy(update={"data": resized, "transform": new_transform})

    @classmethod
    def with_template(cls, data: np.ndarray, template: Union[Any, str, Path], metadata: Optional[Metadata] = None) -> "GeoRasterData":
        """Create a GeoRasterData instance with a given data array, using the extent/projection
        from a template (GeoRasterData object or file path).

        Calculates a new transform to fit the template's bounding box at the new resolution.

        Args:
            data: 2-D array for the new instance.
            template: Either a file path (data will be loaded to determine extents) or an
                existing ``GeoRasterData`` object.  When passing an object its ``data``
                attribute must not be ``None`` — the array shape is needed to derive the
                bounding box.
            metadata: Optional metadata for the new instance.
        """
        if isinstance(template, (str, Path)):
            # Create a temporary instance to read metadata
            template_obj = cls.from_file(template)
        elif isinstance(template, GeoRasterData):
            template_obj = template
        else:
             raise ValueError("Template must be a filepath or GeoRasterData object")

        if template_obj.transform is None or template_obj.crs is None:
            raise ValueError("Template has no transform or CRS")

        # Determine dimensions of new data
        if data.ndim == 2:
            height, width = data.shape
        else:
             raise ValueError("Data must be 2D")

        # Calculate template bounds
        # Rasterio transform: (col, row) -> (x, y)
        # Bounding box is defined by top-left (0,0) and bottom-right (width, height)
        # Note: template_obj.data might not be loaded, but transform should preserve 
        # width/height information implicitly or explicitly? 
        # Actually transform doesn't store width/height. We need it from the template.
        # If template is from file, we read data so we know shape.
        # If template is object, it must have data.
        if template_obj.data is None:
             raise ValueError("Template object must have data array to determine extents")
             
        if template_obj.data.ndim == 2:
             t_h, t_w = template_obj.data.shape
        else:
             _, t_h, t_w = template_obj.data.shape
        
        t_transform = template_obj.transform
        
        # Corners of template
        # Top Left
        minx, maxy = t_transform * (0, 0)
        # Bottom Right
        maxx, miny = t_transform * (t_w, t_h)
        
        # New resolution
        new_res_x = (maxx - minx) / width
        new_res_y = (miny - maxy) / height # Usually negative
        
        new_transform = rio.Affine(new_res_x, t_transform.b, minx,
                                   t_transform.d, new_res_y, maxy)
                                   
        instance = cls()
        instance.data = data
        instance.transform = new_transform
        instance.crs = template_obj.crs
        
        if metadata:
            instance.metadata = metadata
        
        return instance

class PointCloudData(Data):
    """3-D point cloud stored as a Parquet file with geospatial metadata.

    Points and their attributes are held in a ``pandas.DataFrame``.
    Custom metadata (provenance, processing config, etc.) is embedded in
    the Parquet schema metadata under the ``clouds_decoded`` key and
    round-trips through read/write.
    """
    data: pd.DataFrame = Field(default_factory=pd.DataFrame)
    metadata: Metadata = Field(default_factory=Metadata)
    
    @field_validator("data")
    @classmethod
    def validate_dataframe(cls, v: Any) -> pd.DataFrame:
        if not isinstance(v, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        return v

    def read(self, filepath: Union[str, Path]):
        """Read data from a parquet file, restoring metadata."""
        filepath = Path(filepath)
        if not filepath.suffix == ".parquet":
            raise ValueError(f"PointCloudData requires a .parquet file, got: {filepath}")
        
        # Read using pyarrow to access metadata
        table = pq.read_table(filepath)
        self.data = table.to_pandas()
        
        # Restore metadata
        if table.schema.metadata:
            # Metadata keys are bytes
            meta_bytes = table.schema.metadata.get(METADATA_TAG.encode())
            if meta_bytes:
                try:
                    json_dict = json.loads(meta_bytes.decode('utf-8'))
                    if isinstance(json_dict, dict):
                         # Dynamic metadata instantiation
                         meta_type = type(self.metadata)
                         try:
                             self.metadata = meta_type.model_validate(json_dict)
                         except (ValueError, TypeError):
                             self.metadata = meta_type(**json_dict)
                except json.JSONDecodeError:
                    pass

    def write(self, filepath: Union[str, Path]):
        """Write data to a parquet file, preserving metadata."""
        filepath = Path(filepath)
        
        table = pa.Table.from_pandas(self.data)
        
        # Inject metadata
        meta_dict = self.metadata.model_dump(exclude_defaults=True, mode='json')
        if meta_dict:
            # Get existing metadata (usually pandas info) or empty dict
            existing_meta = table.schema.metadata or {}
            # Update with our custom metadata as JSON string
            json_meta = json.dumps(meta_dict)
            existing_meta[METADATA_TAG.encode()] = json_meta.encode('utf-8')
            
            table = table.replace_schema_metadata(existing_meta)
            
        pq.write_table(table, filepath)

class AlbedoMetadata(Metadata):
    """Metadata for surface albedo estimates.

    Tracks the estimation method (IDW, data-driven, or constant fallback),
    the number of clear-sky training samples, scene clear fraction, and
    per-band fallback values when insufficient clear pixels are available.
    """
    band_names: List[str] = Field(default_factory=list)
    method: str = Field(default="idw", description="Estimation method used (idw, datadriven, constant)")
    n_training_samples: int = Field(default=0, description="Number of clear-sky samples used for fitting")
    clear_fraction: float = Field(default=0.0, description="Fraction of scene that was clear sky")
    fallback_used: bool = Field(default=False, description="True if insufficient clear pixels triggered fallback")
    fallback_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-band constant albedo values used in fallback mode"
    )


class AlbedoData(GeoRasterData):
    """Per-band surface albedo stored as a coarse-resolution GeoTIFF.

    Each band in the raster corresponds to the estimated clear-sky surface
    reflectance for one Sentinel-2 spectral band.  Band ordering is
    recorded in ``metadata.band_names``.
    """
    metadata: AlbedoMetadata = Field(default_factory=AlbedoMetadata)
