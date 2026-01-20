from abc import ABC, abstractmethod
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np
from typing import Optional, Any, Dict, List, Tuple, Union
import rasterio as rio
import pyarrow as pa
import pyarrow.parquet as pq

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
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
    """Base model for metadata structures."""
    model_config = ConfigDict(extra='allow')

class GeoRasterData(Data):
    """
    Data model for geospatial raster data (e.g. GeoTIFF).
    
    Attributes:
        data: The raster data as a numpy array.
        transform: The affine transform of the raster (maps pixel coordinates to map coordinates).
        crs: The coordinate reference system of the raster.
        metadata: Metadata object containing additional information.
    """
    data: Optional[np.ndarray] = None
    transform: Optional[Any] = None
    crs: Optional[Any] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("data")
    @classmethod
    def validate_array(cls, v: Any) -> Optional[np.ndarray]:
        if v is not None and not isinstance(v, np.ndarray):
            raise ValueError("Data must be a numpy array")
        return v

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
                if k == 'extra_metadata':
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
                except Exception:
                    # Fallback or partial update if strict validation fails?
                    # For now just try to update common fields if validation fails?
                    # Or just construct with what we have (BaseModel ignores extra by default if configured,
                    # but Metadata has extra='allow')
                    self.metadata = meta_type(**json_dict)

    def write(self, filepath: Union[str, Path]):
        """Write data to a file (GeoTIFF or NetCDF)."""
        if self.data is None:
            raise ValueError("No data to write")
        
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

        driver = 'NetCDF' if is_netcdf else 'GTiff'
        
        profile = {
            'driver': driver,
            'height': height,
            'width': width,
            'count': count,
            'dtype': self.data.dtype,
            'crs': self.crs,
            'transform': self.transform
        }

        with rio.open(filepath, 'w', **profile) as dst:
            dst.write(out_data)
            
            # Write metadata as tags
            # We bundle the metadata object into a JSON string to preserve structure
            meta_dict = self.metadata.model_dump(exclude_defaults=True)
            if meta_dict:
                 json_meta = json.dumps(meta_dict, cls=NumpyEncoder)
                 # rasterio accepts string tags
                 # Note: standard TIFF tags are restrictive, but 'extra_metadata' custom tag 
                 # works in GDAL workflow usually, definitely works for NetCDF attributes.
                 dst.update_tags(extra_metadata=json_meta)
                 
                 # Also write flattened simple keys for easier access in other tools if simple types
                 simple_tags = {}
                 for k, v in meta_dict.items():
                     if isinstance(v, (str, int, float, bool)):
                         simple_tags[k] = str(v)
                 if simple_tags:
                     dst.update_tags(**simple_tags)

    @classmethod
    def with_template(cls, data: np.ndarray, template: Union[Any, str, Path], metadata: Optional[Metadata] = None) -> "GeoRasterData":
        """
        Create a GeoRasterData instance with a given data array, using the extent/projection
        from a template (GeoRasterData object or file path). 
        Calculates new transform to fit the template's bounding box at the new resolution.
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
    """
    Uses a list of n-D points with attributes, with r-tree indexing available for spatial queries.
    Stores data in Parquet format with custom metadata support.
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
            meta_bytes = table.schema.metadata.get(b'extra_metadata')
            if meta_bytes:
                try:
                    json_dict = json.loads(meta_bytes.decode('utf-8'))
                    if isinstance(json_dict, dict):
                         # Dynamic metadata instantiation
                         meta_type = type(self.metadata)
                         try:
                             self.metadata = meta_type.model_validate(json_dict)
                         except:
                             self.metadata = meta_type(**json_dict)
                except json.JSONDecodeError:
                    pass

    def write(self, filepath: Union[str, Path]):
        """Write data to a parquet file, preserving metadata."""
        filepath = Path(filepath)
        
        table = pa.Table.from_pandas(self.data)
        
        # Inject metadata
        meta_dict = self.metadata.model_dump(exclude_defaults=True)
        if meta_dict:
            # Get existing metadata (usually pandas info) or empty dict
            existing_meta = table.schema.metadata or {}
            # Update with our custom metadata as JSON string
            json_meta = json.dumps(meta_dict)
            existing_meta[b'extra_metadata'] = json_meta.encode('utf-8')
            
            table = table.replace_schema_metadata(existing_meta)
            
        pq.write_table(table, filepath)

class AlbedoData(GeoRasterData):
    pass

class RetrievalsData(GeoRasterData):
    pass