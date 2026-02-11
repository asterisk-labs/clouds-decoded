# refl2prop/dataset_shading.py
"""Dataset for shading-aware cloud property inversion training."""

import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .dataset import Refl2PropDataset, P_LUT
from .config import DEFAULT_BANDS

logger = logging.getLogger(__name__)


class TransmissionLUT:
    """
    Handles loading and interpolation of transmission lookup tables.

    Loads intermediate parquet files and builds KDTree interpolators for T_hemi lookup.
    Uses direct illumination at 0 degrees for simplified shading model.
    """

    def __init__(self, lut_dir: str, bands: List[str]):
        """
        Args:
            lut_dir: Directory containing band parquet files
            bands: List of band names to load (e.g., ['B01', 'B02', ...])
        """
        self.lut_dir = Path(lut_dir)
        self.bands = bands

        # Load and build interpolators for each band
        self.interpolators: Dict[str, dict] = {}  # {band: {'tree': cKDTree, 'values': array}}
        self.grid_coords: Dict[str, dict] = {}

        self._load_luts()

    def _load_luts(self):
        """Load transmission LUTs and build interpolators for each band."""
        logger.info(f"Loading transmission LUTs from {self.lut_dir}")

        for band in self.bands:
            pq_path = self.lut_dir / f"{band}.parquet"
            if not pq_path.exists():
                logger.warning(f"Transmission LUT not found for band {band}: {pq_path}")
                continue

            # Read parquet with fastparquet (avoids pyarrow issues with nested columns)
            # Use tau_ref (consistent across bands) instead of tau
            df = pd.read_parquet(
                pq_path,
                engine='fastparquet',
                columns=['tau_ref', 'ice_liq_ratio', 'r_eff_liq', 'r_eff_ice',
                         'incidence_angle', 'T_hemi', 'incidence_mode']
            )

            # Filter for direct illumination at 0 degrees only
            # This simplifies the lookup to 4D (tau_ref, ice_liq_ratio, r_eff_liq, r_eff_ice)
            df = df[(df['incidence_mode'] == 'direct') & (df['incidence_angle'] == 0.0)].copy()

            if len(df) == 0:
                logger.warning(f"No data for band {band} with direct mode and 0 degree angle")
                continue

            # Store coordinate ranges for this band
            self.grid_coords[band] = {
                'tau_ref': (df['tau_ref'].min(), df['tau_ref'].max()),
                'ice_liq_ratio': (df['ice_liq_ratio'].min(), df['ice_liq_ratio'].max()),
                'r_eff_liq': (df['r_eff_liq'].min(), df['r_eff_liq'].max()),
                'r_eff_ice': (df['r_eff_ice'].min(), df['r_eff_ice'].max()),
            }

            # Build KDTree for fast nearest-neighbor lookup (4D: tau_ref, ice_liq_ratio, r_eff_liq, r_eff_ice)
            points_raw = df[['tau_ref', 'ice_liq_ratio', 'r_eff_liq', 'r_eff_ice']].values
            values = df['T_hemi'].values

            # Compute normalization factors (range of each dimension)
            self._norm_min = points_raw.min(axis=0)
            self._norm_max = points_raw.max(axis=0)
            self._norm_range = self._norm_max - self._norm_min
            self._norm_range[self._norm_range == 0] = 1.0  # Avoid div by zero

            # Normalize points for KDTree
            points_norm = (points_raw - self._norm_min) / self._norm_range

            self.interpolators[band] = {
                'tree': cKDTree(points_norm),
                'values': values,
            }

            logger.info(f"  {band}: {len(df)} points, tau_ref range [{df['tau_ref'].min():.2f}, {df['tau_ref'].max():.2f}]")

        # Store global tau_ref range for shading generation
        all_tau_ranges = [self.grid_coords[b]['tau_ref'] for b in self.interpolators]
        self.tau_min = min(r[0] for r in all_tau_ranges)
        self.tau_max = max(r[1] for r in all_tau_ranges)

        logger.info(f"Loaded {len(self.interpolators)} band transmission LUTs")

    def lookup(
        self,
        tau_ref: np.ndarray,
        ice_liq_ratio: float,
        r_eff_liq: float,
        r_eff_ice: float,
    ) -> np.ndarray:
        """
        Look up transmission values for multiple tau_ref values.

        Uses direct illumination at 0 degrees (pre-filtered during loading).
        For tau values below the LUT minimum, uses linear extrapolation
        (T approaches 1 as tau approaches 0).

        Args:
            tau_ref: Array of tau_ref values to look up, shape (N,)
            ice_liq_ratio: Ice/liquid ratio (scalar, same for all)
            r_eff_liq: Effective radius liquid (scalar)
            r_eff_ice: Effective radius ice (scalar)

        Returns:
            transmissions: Array of shape (N, num_bands) with T_hemi values
        """
        N = len(tau_ref)
        transmissions = np.ones((N, len(self.bands)), dtype=np.float32)

        # Identify values below LUT minimum (need linear extrapolation)
        below_min_mask = tau_ref < self.tau_min

        # For values within LUT range, use KDTree lookup
        # Clamp tau values to LUT range for the lookup
        tau_clamped = np.clip(tau_ref, self.tau_min, self.tau_max)

        # Build query points: (N, 4) - tau_ref, ice_liq_ratio, r_eff_liq, r_eff_ice
        points = np.column_stack([
            tau_clamped,
            np.full(N, ice_liq_ratio),
            np.full(N, r_eff_liq),
            np.full(N, r_eff_ice),
        ])

        # Normalize query points
        points_norm = (points - self._norm_min) / self._norm_range

        for i, band in enumerate(self.bands):
            if band in self.interpolators:
                interp = self.interpolators[band]
                # Use KDTree for fast nearest-neighbor lookup
                _, indices = interp['tree'].query(points_norm)
                T_vals = interp['values'][indices].astype(np.float32)

                # For values below LUT minimum, extrapolate linearly to T=1 at tau=0
                # T(tau) = 1 - (1 - T_min) * (tau / tau_min) for tau < tau_min
                # This assumes linear behavior at small optical depths
                if below_min_mask.any():
                    # Get T value at tau_min (from the lookup)
                    T_at_min = T_vals[below_min_mask]
                    # Linear extrapolation: T = 1 - (1 - T_min) * (tau / tau_min)
                    tau_below = tau_ref[below_min_mask]
                    T_extrapolated = 1.0 - (1.0 - T_at_min) * (tau_below / self.tau_min)
                    T_vals[below_min_mask] = T_extrapolated

                transmissions[:, i] = np.clip(T_vals, 0.0, 1.0)

        return transmissions


class ShadingBagDataset(Dataset):
    """
    Dataset that generates bags of shaded pixel samples for training.

    Each bag shares the same physical properties (tau_ref, ice_liq_ratio,
    r_eff_liq, r_eff_ice) but has different tau_effective_shading per pixel.

    The shading modulates the clean reflectances via:
        R_shaded = R_clean * T(tau_effective_shading)
    """

    def __init__(
        self,
        lut_path: str,
        transmission_lut_dir: str,
        bag_size: int = 128,
        n_dims_interp: int = 4,
        spectral_mode: str = 'variable',
        selected_bands: Optional[List[str]] = None,
    ):
        """
        Args:
            lut_path: Path to main reflectance LUT (NetCDF/Zarr)
            transmission_lut_dir: Path to directory with intermediate parquets
            bag_size: Number of pixels per bag
            n_dims_interp: Number of dimensions to interpolate in base LUT
            spectral_mode: 'variable' for per-band albedos
            selected_bands: List of bands to use (default: all 11)
        """
        self.bag_size = bag_size
        self.selected_bands = selected_bands if selected_bands else DEFAULT_BANDS

        # Initialize base dataset for clean reflectances
        self.base_dataset = Refl2PropDataset(
            lut_path=lut_path,
            n_dims_interp=n_dims_interp,
            spectral_mode=spectral_mode,
            selected_bands=self.selected_bands,
        )

        # Load transmission LUTs
        self.transmission_lut = TransmissionLUT(
            lut_dir=transmission_lut_dir,
            bands=self.selected_bands,
        )

        # Copy stats from base dataset
        self.input_stats = self.base_dataset.input_stats
        self.output_stats = self.base_dataset.output_stats
        self.num_bands = self.base_dataset.num_bands

        # Shading stats (tau range from transmission LUT)
        self.shading_stats = {
            'min': 0.0,  # Include 0 for no shading
            'max': float(self.transmission_lut.tau_max),
        }

        logger.info(f"ShadingBagDataset initialized: bag_size={bag_size}, "
                    f"bands={len(self.selected_bands)}, "
                    f"tau_shading_max={self.shading_stats['max']:.1f}")

    def _generate_tau_shading_distribution(self) -> np.ndarray:
        """
        Generate tau_effective_shading values using multi-modal Gaussian mixture.

        Strategy:
        1. Most modes sample in log space (prioritize low-moderate values)
        2. Some modes sample in linear space near 0 (can reach tau=0)
        3. Use 1-5 Gaussian centers per bag
        4. Random std dev per center

        Returns:
            tau_shading: array of shape (bag_size,)
        """
        # Random number of modes (1-5)
        n_modes = np.random.randint(1, 6)

        tau_shading = np.zeros(self.bag_size, dtype=np.float32)

        # For each mode, decide if it's log-space or linear-space (near zero)
        # ~30% chance of a linear-space mode that can reach 0
        mode_is_linear = np.random.random(n_modes) < 0.3

        # Sample centers
        # Log-space: [-2, 4] corresponds to tau ~ [0.14, 55]
        # Linear-space: [0, 0.5] for very low tau values
        log_centers = np.random.uniform(-2, 4, n_modes)
        linear_centers = np.random.uniform(0, 0.5, n_modes)

        # Random std dev for each center
        log_stds = np.random.uniform(0.2, 1.5, n_modes)
        linear_stds = np.random.uniform(0.05, 0.3, n_modes)

        # Assign samples to modes (random assignment)
        mode_assignments = np.random.randint(0, n_modes, self.bag_size)

        # Generate samples
        for i in range(self.bag_size):
            mode = mode_assignments[i]

            if mode_is_linear[mode]:
                # Linear-space sampling: can reach 0
                tau = np.random.normal(linear_centers[mode], linear_stds[mode])
                tau_shading[i] = max(0.0, tau)  # Clip to non-negative
            else:
                # Log-space sampling: better for moderate-high tau
                log_tau = np.random.normal(log_centers[mode], log_stds[mode])
                tau_shading[i] = np.exp(log_tau)

        # Clip to valid range [0, tau_max]
        tau_shading = np.clip(tau_shading, 0.0, self.shading_stats['max'])

        return tau_shading

    def __len__(self):
        # Return base dataset length (effectively infinite sampling)
        return len(self.base_dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a bag of shaded samples.

        Returns:
            inputs: (bag_size, input_dim) - shaded reflectances + albedos + geometry
            tau_shading_targets: (bag_size,) - per-pixel shading tau targets
            global_targets: (4,) - shared physical properties [tau, ice_liq, reff_l, reff_i]
        """
        # Get clean sample from base dataset
        # This returns (inputs, targets) where inputs = [reflectances, albedos, geometry]
        clean_inputs, global_targets = self.base_dataset[idx]
        clean_inputs = clean_inputs.numpy()
        global_targets = global_targets.numpy()

        # Extract components from clean inputs
        refl_clean = clean_inputs[:self.num_bands]
        albedos = clean_inputs[self.num_bands:2*self.num_bands]
        geometry = clean_inputs[2*self.num_bands:]

        # Extract physical parameters for transmission lookup
        # Note: global_targets[0] is tau (from LUT), which equals tau_ref
        ice_liq_ratio = global_targets[1]
        r_eff_liq = global_targets[2]
        r_eff_ice = global_targets[3]

        # Generate tau_shading distribution for this bag
        tau_shading = self._generate_tau_shading_distribution()

        # Look up transmissions for all tau_shading values
        # Uses direct illumination at 0 degrees (simplified shading model)
        transmissions = self.transmission_lut.lookup(
            tau_ref=tau_shading,
            ice_liq_ratio=ice_liq_ratio,
            r_eff_liq=r_eff_liq,
            r_eff_ice=r_eff_ice,
        )  # Shape: (bag_size, num_bands)

        # Apply shading: R_shaded = R_clean * T(tau_shading)
        shaded_reflectances = refl_clean[np.newaxis, :] * transmissions

        # Build input vectors for the bag
        # Each pixel: [shaded_reflectances, albedos, geometry]
        inputs = np.zeros((self.bag_size, len(clean_inputs)), dtype=np.float32)
        inputs[:, :self.num_bands] = shaded_reflectances
        inputs[:, self.num_bands:2*self.num_bands] = albedos  # Same for all pixels
        inputs[:, 2*self.num_bands:] = geometry  # Same for all pixels

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(tau_shading),
            torch.from_numpy(global_targets),
        )


class InMemoryShadingBagDataset(ShadingBagDataset):
    """
    Shading bag dataset with base LUT loaded into memory for speed.
    """

    def __init__(self, *args, **kwargs):
        # Import here to avoid circular dependency
        from .dataset import InMemoryRefl2PropDataset

        # Extract lut_path before calling super().__init__
        lut_path = kwargs.get('lut_path') or args[0]
        transmission_lut_dir = kwargs.get('transmission_lut_dir') or args[1]
        bag_size = kwargs.get('bag_size', 128)
        n_dims_interp = kwargs.get('n_dims_interp', 4)
        spectral_mode = kwargs.get('spectral_mode', 'variable')
        selected_bands = kwargs.get('selected_bands', None)

        self.bag_size = bag_size
        self.selected_bands = selected_bands if selected_bands else DEFAULT_BANDS

        # Use in-memory version of base dataset
        self.base_dataset = InMemoryRefl2PropDataset(
            lut_path=lut_path,
            n_dims_interp=n_dims_interp,
            spectral_mode=spectral_mode,
            selected_bands=self.selected_bands,
        )

        # Load transmission LUTs
        self.transmission_lut = TransmissionLUT(
            lut_dir=transmission_lut_dir,
            bands=self.selected_bands,
        )

        # Copy stats from base dataset
        self.input_stats = self.base_dataset.input_stats
        self.output_stats = self.base_dataset.output_stats
        self.num_bands = self.base_dataset.num_bands

        # Shading stats
        self.shading_stats = {
            'min': 0.0,
            'max': float(self.transmission_lut.tau_max),
        }

        logger.info(f"InMemoryShadingBagDataset initialized: bag_size={bag_size}")


def shading_collate_fn(batch):
    """
    Custom collate function for shading bag batches.

    Args:
        batch: List of (inputs, shading_targets, global_targets) tuples

    Returns:
        inputs: (batch_size, bag_size, input_dim)
        shading_targets: (batch_size, bag_size)
        global_targets: (batch_size, 4)
    """
    inputs = torch.stack([x[0] for x in batch])
    shading_targets = torch.stack([x[1] for x in batch])
    global_targets = torch.stack([x[2] for x in batch])
    return inputs, shading_targets, global_targets
