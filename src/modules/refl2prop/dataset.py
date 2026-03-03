import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import netCDF4 as nc
import zarr
from zarr.storage import ZipStore
from pathlib import Path
import random
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# We use the strict feature order from our local config
from .config import OutputFeature, DEFAULT_BANDS

# Local constants mapping to the LUT file keys
# Note: TAU_ALIASES supports both old ("tau") and new ("tau_ref") LUT naming conventions
TAU_ALIASES = ("tau", "tau_ref")

class P_LUT:
    TAU = "tau"  # Canonical name used internally
    ICE_LIQ_RATIO = "ice_liq_ratio"
    R_EFF_LIQ = "r_eff_liq"
    R_EFF_ICE = "r_eff_ice"
    CLOUD_TOP_HEIGHT = "cloud_top_height"
    SURFACE_ALBEDO = "surface_albedo"
    INCIDENCE_ANGLE = "incidence_angle"
    SHADING_RATIO = "shading_ratio"
    CLOUD_THICKNESS = "cloud_thickness"
    BAND = "band"
    MU = "mu"
    PHI = "phi"


def _resolve_tau_name(available_keys: set) -> str:
    """Return the tau dimension name present in the LUT (supports 'tau' or 'tau_ref')."""
    for alias in TAU_ALIASES:
        if alias in available_keys:
            return alias
    raise KeyError(f"No tau dimension found. Expected one of {TAU_ALIASES}, got {available_keys}")

class Refl2PropDataset(Dataset):
    """
    Generative Dataset that loads a high-dimensional LUT and samples training points
    on-the-fly using random interpolation. 
    Avoids 'stacking' or flattening the LUT, which saves massive amounts of RAM.
    Handles Lazy Loading to support Multiprocessing.
    """
    def __init__(self, lut_path: str, n_dims_interp: int = 2, spectral_mode: str = 'variable', selected_bands: List[str] = None):
        self.lut_path = Path(lut_path)
        if not self.lut_path.exists():
            raise FileNotFoundError(f"LUT not found at: {self.lut_path}")

        self.n_dims_interp = n_dims_interp
        self.spectral_mode = spectral_mode
        self.selected_bands = selected_bands if selected_bands else DEFAULT_BANDS

        # Initialize handles to None to ensure picklability
        self.dataset = None
        self.store = None
        self.root = None
        self.reflectance_cube = None

        # --- 1. Open File Temporarily to Read Metadata ---
        self._open_file()

        # Capture dimensions and shape while open
        if self.lut_path.suffix == '.nc':
            self.dims_in_array = list(self.reflectance_cube.dimensions)
            self._load_coords_nc()
        elif self.lut_path.suffix == '.zip':
            self.dims_in_array = self.root['reflectance'].attrs['_ARRAY_DIMENSIONS']
            self._load_coords_zarr()
        else:
            self._close_file()
            raise ValueError(f"Unsupported LUT format: {self.lut_path.suffix}")

        # Normalize tau_ref -> tau in dimension names for consistency
        self.dims_in_array = [P_LUT.TAU if d in TAU_ALIASES else d for d in self.dims_in_array]

        # Store shape for __len__ so we don't need file open
        self.refl_shape = self.reflectance_cube.shape

        # --- 2. Setup Interpolation Dimensions ---
        self.interp_dims = [
            P_LUT.TAU, P_LUT.ICE_LIQ_RATIO, P_LUT.R_EFF_LIQ, P_LUT.R_EFF_ICE,
            P_LUT.CLOUD_TOP_HEIGHT, P_LUT.SURFACE_ALBEDO,
            P_LUT.INCIDENCE_ANGLE, P_LUT.SHADING_RATIO
        ]
        
        # Handle coordinate ordering (exclude dims that are not physical params we loop over)
        self.coord_dims_ordered = [d for d in self.dims_in_array if d not in ['mu', 'phi', P_LUT.CLOUD_THICKNESS]]

        # Map selected bands to indices
        all_bands = list(self.coords[P_LUT.BAND])
        self.band_indices = [all_bands.index(b) for b in self.selected_bands if b in all_bands]
        # Update coordinate ref to only include selected
        self.coords[P_LUT.BAND] = self.selected_bands
        self.num_bands = len(self.selected_bands)

        # Pre-compute min/max stats for Normalization
        self.input_stats, self.output_stats = self._compute_stats()

        # --- 3. Close File Handles ---
        # Critical: We must close handles here so the object can be pickled by DataLoader
        self._close_file()

    def _open_file(self):
        """Opens the file handles. Called lazily."""
        if self.lut_path.suffix == '.nc':
            self.dataset = nc.Dataset(self.lut_path, 'r')
            self.reflectance_cube = self.dataset.variables['reflectance']
        elif self.lut_path.suffix == '.zip':
            self.store = ZipStore(self.lut_path, mode='r')
            self.root = zarr.group(store=self.store)
            self.reflectance_cube = self.root['reflectance']

    def _close_file(self):
        """Closes file handles and sets them to None."""
        if self.dataset:
            self.dataset.close()
            self.dataset = None
        if self.store:
            self.store.close()
            self.store = None
            self.root = None
        # We also clear reflectance_cube unless it's a numpy array (InMemory mode)
        if not isinstance(self.reflectance_cube, np.ndarray):
            self.reflectance_cube = None

    def _load_coords_nc(self):
        """Helper to load coordinates from NetCDF"""
        self.coords = {}
        for dim in self.dataset.variables:
            if dim not in ['reflectance', 'r_hemi']:
                val = self.dataset.variables[dim][:]
                if val.dtype.kind == 'S': # Decode bytes
                    val = [b.decode('utf-8') for b in val]
                # Normalize tau_ref -> tau for compatibility
                key = P_LUT.TAU if dim in TAU_ALIASES else dim
                self.coords[key] = val

        # Ensure Mu/Phi exist
        if P_LUT.MU not in self.coords:
            self.coords[P_LUT.MU] = np.linspace(0, 1, self.reflectance_cube.shape[-2])
        if P_LUT.PHI not in self.coords:
            self.coords[P_LUT.PHI] = np.linspace(0, 2 * np.pi, self.reflectance_cube.shape[-1])

    def _load_coords_zarr(self):
        """Helper to load coordinates from Zarr"""
        self.coords = {}
        for dim in self.root.keys():
            if dim not in ['reflectance', 'r_hemi']:
                val = self.root[dim][:]
                # Normalize tau_ref -> tau for compatibility
                key = P_LUT.TAU if dim in TAU_ALIASES else dim
                self.coords[key] = val

    def _compute_stats(self):
        """Computes min/max for all parameters to feed the NormalizationWrapper."""
        input_mins, input_maxs = [], []
        output_mins, output_maxs = [], []

        # 1. Inputs (Order: bands, albedos, geometry)
        # Bands (reflectance values typically in [0, 1.5] after normalization)
        input_mins.extend([0.0] * self.num_bands)
        input_maxs.extend([1.5] * self.num_bands)

        # Albedos (surface albedo in [0, 1])
        input_mins.extend([0.0] * self.num_bands)
        input_maxs.extend([1.0] * self.num_bands)
        
        # Geometry & Priors
        geo_keys = [P_LUT.INCIDENCE_ANGLE, P_LUT.SHADING_RATIO, P_LUT.CLOUD_TOP_HEIGHT, P_LUT.MU, P_LUT.PHI]
        for k in geo_keys:
            if k in self.coords:
                vals = self.coords[k]
                input_mins.append(float(np.min(vals)))
                input_maxs.append(float(np.max(vals)))
            else:
                input_mins.append(0.0)
                input_maxs.append(1.0 if k==P_LUT.SHADING_RATIO else 90.0)

        # 2. Outputs (Order from config.OutputFeature)
        out_keys = [P_LUT.TAU, P_LUT.ICE_LIQ_RATIO, P_LUT.R_EFF_LIQ, P_LUT.R_EFF_ICE]
        for k in out_keys:
            if k in self.coords:
                vals = self.coords[k]
                output_mins.append(float(np.min(vals)))
                output_maxs.append(float(np.max(vals)))
            else:
                output_mins.append(0.0)
                output_maxs.append(1.0)


        return {'min': input_mins, 'max': input_maxs}, {'min': output_mins, 'max': output_maxs}

    def random_selection(self):
        """Selects random dimensions to interpolate and fixed indices for others."""
        eligible = [d for d in self.interp_dims if len(self.coords.get(d, [])) > 1]
        
        if self.spectral_mode == 'variable' and P_LUT.SURFACE_ALBEDO in eligible:
            eligible.remove(P_LUT.SURFACE_ALBEDO)

        interp_dims = random.sample(eligible, min(len(eligible), self.n_dims_interp))
        
        fixed_coords = {}
        interp_indices = {}
        interp_details = {}

        all_dims = self.coord_dims_ordered + [P_LUT.MU, P_LUT.PHI]
        
        for dim in all_dims:
            if dim == P_LUT.CLOUD_THICKNESS: continue
            
            dim_coords = self.coords.get(dim)
            if dim_coords is None: continue
            
            n = len(dim_coords)
            
            if dim in interp_dims:
                idx = random.randint(0, n - 2)
                interp_indices[dim] = (idx, idx + 1)
                interp_details[dim] = (dim_coords[idx], dim_coords[idx+1])
            else:
                idx = random.randint(0, n - 1)
                interp_indices[dim] = (idx, idx)
                if dim in self.coords:
                    fixed_coords[dim] = dim_coords[idx]

        return interp_dims, fixed_coords, interp_indices, interp_details

    def __len__(self):
        # Return size proxy from cached shape, so no file open needed
        return int(np.prod(self.refl_shape[:-2]))

    def __getitem__(self, idx):
        # Lazy Open: Ensure file is open in the worker process
        if self.reflectance_cube is None:
            self._open_file()
        
        # Retry loop to handle NaNs in the LUT
        while True:
            interp_dims, fixed_coords, interp_indices, interp_details = self.random_selection()
            
            # --- 1. Build Slice and Track Active Dimensions ---
            slice_list = []
            
            # We need to track which dimensions are present in the resulting mini_cube
            # to map interpolation dimensions to axes correctly.
            current_dims_in_hypercube = []

            for dim in self.dims_in_array:
                if dim == P_LUT.CLOUD_THICKNESS:
                    slice_list.append(0) 
                    continue
                
                if self.spectral_mode == 'variable' and dim == P_LUT.SURFACE_ALBEDO:
                    slice_list.append(slice(None)) 
                    current_dims_in_hypercube.append(dim)
                    continue
                
                if dim == P_LUT.BAND:
                    slice_list.append(self.band_indices)
                    current_dims_in_hypercube.append(dim)
                    continue

                if dim in interp_indices:
                    idx_tuple = interp_indices[dim]
                    if idx_tuple[0] == idx_tuple[1]:
                        # Fixed dimension, reduced by integer indexing
                        slice_list.append(idx_tuple[0])
                    else:
                        # Interpolated dimension, sliced (size 2)
                        slice_list.append(slice(idx_tuple[0], idx_tuple[1] + 1))
                        current_dims_in_hypercube.append(dim)
                else:
                    # Dimensions not in interp_indices (should be covered, but safety fallback)
                    pass

            # --- 2. Extract Mini-Cube ---
            # mini_cube will have dimensions matching 'current_dims_in_hypercube'
            mini_cube = self.reflectance_cube[tuple(slice_list)]
            
            if np.isnan(mini_cube).any():
                continue 

            # --- 3. Perform Interpolation ---
            final_refl = mini_cube
            interp_factors = {dim: random.random() for dim in interp_dims}
            
            # We filter for dims that are actually in the array slice
            # (e.g. Band is not in interp_dims, Albedo might be special)
            ordered_interp_dims = [dim for dim in self.dims_in_array if dim in interp_dims]
            
            # Interpolate standard dimensions
            for dim in ordered_interp_dims:
                if dim not in current_dims_in_hypercube:
                    continue # Should have been fixed and sliced out
                
                factor = interp_factors[dim]
                axis = current_dims_in_hypercube.index(dim)
                
                # Take slice 0 and slice 1 along the interpolation axis
                # Because we sliced size 2, index 0 is lower bound, index 1 is upper bound
                # We use numpy/torch 'take' equivalent or slicing
                # final_refl = (1-f)*refl[0] + f*refl[1]
                
                # np.take is robust. 
                slice0 = np.take(final_refl, 0, axis=axis)
                slice1 = np.take(final_refl, 1, axis=axis)
                
                final_refl = (1 - factor) * slice0 + factor * slice1
                
                # The dimension is now collapsed
                current_dims_in_hypercube.pop(axis)

            # --- 4. Handle Variable Surface Albedo ---
            # If spectral_mode is 'variable', we have preserved the SURFACE_ALBEDO dimension
            # and the BAND dimension. We now create a random albedo vector and interpolate.
            surface_albedo_vector = None
            if self.spectral_mode == 'variable' and P_LUT.SURFACE_ALBEDO in current_dims_in_hypercube:
                
                # Find axes
                albedo_axis = current_dims_in_hypercube.index(P_LUT.SURFACE_ALBEDO)
                band_axis = current_dims_in_hypercube.index(P_LUT.BAND)
                
                num_bands = final_refl.shape[band_axis]
                albedo_coords = self.coords[P_LUT.SURFACE_ALBEDO]
                min_alb, max_alb = albedo_coords[0], albedo_coords[-1]
                
                # Generate random albedo per band
                surface_albedo_vector = np.random.uniform(min_alb, max_alb, num_bands)
                
                refl_per_band = []
                for i in range(num_bands):
                    target_alb = surface_albedo_vector[i]
                    
                    # Find indices in coordinate array
                    # np.searchsorted finds where to insert to maintain order. 
                    # index i means coords[i-1] <= target < coords[i]
                    idx = np.searchsorted(albedo_coords, target_alb)
                    # Clip to bounds
                    idx = np.clip(idx, 1, len(albedo_coords) - 1)
                    idx_lower = idx - 1
                    idx_upper = idx
                    
                    val_lower = albedo_coords[idx_lower]
                    val_upper = albedo_coords[idx_upper]
                    
                    factor = (target_alb - val_lower) / (val_upper - val_lower)
                    
                    # Extract the specific band i, and the albedo slices
                    # We need to slice 'final_refl' carefully. 
                    # It has dimensions [..., Albedo, ..., Band, ...] (or vice versa)
                    
                    # To use 'take' safely:
                    # Slice specific band i
                    # Note: taking from band axis reduces dims, so albedo axis index might shift?
                    # No, axis index assumes current shape. 
                    # Better to extract the full Albedo column for this Band first.
                    
                    # 1. Select Band i
                    band_slice = np.take(final_refl, i, axis=band_axis)
                    
                    # 2. Select Albedo slices from the result
                    # Note: Since we removed Band dim, Albedo axis might have shifted if Band was before Albedo.
                    eff_alb_axis = albedo_axis if albedo_axis < band_axis else albedo_axis - 1
                    
                    alb_slice0 = np.take(band_slice, idx_lower, axis=eff_alb_axis)
                    alb_slice1 = np.take(band_slice, idx_upper, axis=eff_alb_axis)
                    
                    refl_val = (1 - factor) * alb_slice0 + factor * alb_slice1
                    refl_per_band.append(refl_val)
                    
                # Stack results back into a vector (since we processed band by band)
                # If there are other dims (e.g. Mu/Phi not interpolated yet?), this logic assumes 
                # Albedo and Band were the main remaining dims. 
                # If Mu/Phi remain, refl_val is scalar? No, might be (Mu, Phi).
                final_refl = np.stack(refl_per_band, axis=0)
                # Now shape is (Bands, ...)
                
                # Remove Albedo from current dims list (effectively replaced by Band stack)
                # The stacking creates a Band dimension at axis 0.
                # Logic: We collapsed Albedo. We collapsed Band (by iterating). We stacked Band.
                # So resulting dim 0 is Band.
                current_dims_in_hypercube = [P_LUT.BAND] # + any others if they existed?
                
            elif self.spectral_mode == 'variable':
                # Fallback if logic didn't trigger (e.g. Albedo fixed unexpectedly?)
                # Generate vector for output consistency
                 surface_albedo_vector = np.full(self.num_bands, fixed_coords.get(P_LUT.SURFACE_ALBEDO, 0.5))

            # Final check to flatten
            if final_refl.ndim > 1:
                # Assuming (Band, ...) or just (Band)
                final_refl = final_refl.flatten()
            
            # --- 5. Construct Output Vectors ---
            inputs = []
            inputs.extend(final_refl)

            # Calculate interpolated values for non-array params (needed for albedo and geometry below)
            interp_vals = {}
            for dim in interp_dims:
                v1, v2 = interp_details[dim]
                f = interp_factors[dim]
                interp_vals[dim] = (1 - f) * v1 + f * v2

            # Albedos
            if surface_albedo_vector is not None:
                inputs.extend(surface_albedo_vector)
            else:
                # Constant albedo mode or fixed albedo
                alb_val = fixed_coords.get(P_LUT.SURFACE_ALBEDO,
                                         interp_details.get(P_LUT.SURFACE_ALBEDO, (0.5,))[0])
                # If interpolated, use the interpolated value
                if P_LUT.SURFACE_ALBEDO in interp_vals:
                    alb_val = interp_vals[P_LUT.SURFACE_ALBEDO]
                inputs.extend([alb_val] * self.num_bands)

            # Geometry & Priors
            geo_order = [P_LUT.INCIDENCE_ANGLE, P_LUT.SHADING_RATIO, P_LUT.CLOUD_TOP_HEIGHT, P_LUT.MU, P_LUT.PHI]

            for k in geo_order:
                if k in interp_vals:
                    val = interp_vals[k]
                else:
                    val = fixed_coords.get(k, 0.0)
                inputs.append(val)
                
            targets = []
            tgt_order = [P_LUT.TAU, P_LUT.ICE_LIQ_RATIO, P_LUT.R_EFF_LIQ, P_LUT.R_EFF_ICE]
            for k in tgt_order:
                if k in interp_vals:
                    val = interp_vals[k]
                else:
                    val = fixed_coords.get(k, 0.0)
                targets.append(val)

            return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

class InMemoryRefl2PropDataset(Refl2PropDataset):
    """Loads the entire cube into RAM for speed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Handles are closed by super().__init__. Reopen just to load into RAM.
        self._open_file()
        logger.info("Loading LUT into memory...")
        self.reflectance_cube = np.array(self.reflectance_cube[:])
        logger.info("LUT loaded.")
        # Close file handles to keep it picklable, but keep the numpy array
        self._close_file()
        # Ensure reflectance_cube is the numpy array
        # _close_file sets reflectance_cube to None unless it's ndarray, 
        # but since we assigned it above, it should persist if _close_file checks type.
        # But _close_file logic above sets it to None if not ndarray.
        # self.reflectance_cube is currently ndarray, so _close_file will skip it.
        # Just to be safe/explicit:
        if not isinstance(self.reflectance_cube, np.ndarray):
            # This shouldn't happen based on above lines, but logic:
            raise RuntimeError("Failed to load LUT into memory.")

def collate_fn(batch):
    inputs = torch.stack([x[0] for x in batch])
    targets = torch.stack([x[1] for x in batch])
    return inputs, targets