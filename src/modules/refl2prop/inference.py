# refl2prop/inference.py
import torch
import numpy as np
from typing import Dict, Union

from refl2prop.model import InversionNet, NormalizationWrapper
from refl2prop.config import InputFeature

class CloudPropertyInverter:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to the .pth file saved by train.py
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load state
        state = torch.load(checkpoint_path, map_location=self.device)
        
        # Init Model
        core_model = InversionNet(input_size=17, output_size=4)
        
        # Reconstruct Normalization Wrapper using dummy stats (overwritten by load_state_dict)
        dummy = {'min': [0]*17, 'max': [1]*17}
        out_dummy = {'min': [0]*4, 'max': [1]*4}
        
        self.model = NormalizationWrapper(core_model, dummy, out_dummy)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_scene(self, 
                      bands: Dict[str, np.ndarray], 
                      surface_albedo: Dict[str, float],
                      geometry: Dict[str, Union[float, np.ndarray]],
                      cloud_top_height: np.ndarray,
                      shading_ratio: Union[float, np.ndarray] = 0.5) -> Dict[str, np.ndarray]:
        """
        Runs the inversion on a full scene.
        
        Args:
            bands: Dict {'B01': 2D_Array, ...} matches INVERSION_BANDS.
            surface_albedo: Dict {'B01': 0.1, ...} scalar values per band.
            geometry: Dict {'incidence_angle': float, 'mu': 2D, 'phi': 2D}.
            cloud_top_height: 2D Array (from Cloud Height Package).
            shading_ratio: Scalar or 2D Array.
        """
        
        # 1. Validation & Shape Check
        ref_band = 'B02'
        shape = bands[ref_band].shape
        
        # 2. Flatten Inputs (H*W, C)
        # We assume all bands are already resized to the same resolution
        # by the Cloud Height package's 'ColumnExtractor' or similar logic.
        
        # Helper to broadcast
        def _flat(v):
            if np.isscalar(v):
                return np.full(shape[0]*shape[1], v, dtype=np.float32)
            return v.reshape(-1).astype(np.float32)

        cols = []
        # Band Order must match InputFeature enum
        cols.append(_flat(bands['B01']))
        cols.append(_flat(bands['B02']))
        cols.append(_flat(bands['B04']))
        cols.append(_flat(bands['B08']))
        cols.append(_flat(bands['B11']))
        cols.append(_flat(bands['B12']))
        
        # Albedos
        cols.append(_flat(surface_albedo['B01']))
        cols.append(_flat(surface_albedo['B02']))
        cols.append(_flat(surface_albedo['B04']))
        cols.append(_flat(surface_albedo['B08']))
        cols.append(_flat(surface_albedo['B11']))
        cols.append(_flat(surface_albedo['B12']))
        
        # Geometry
        cols.append(_flat(geometry['incidence_angle']))
        cols.append(_flat(shading_ratio))
        cols.append(_flat(cloud_top_height))
        cols.append(_flat(geometry['mu']))
        cols.append(_flat(geometry['phi']))
        
        # Stack columns -> (N_pixels, 17)
        input_matrix = np.stack(cols, axis=1)
        
        # 3. Batch Inference
        batch_size = 32768 # Adjust based on VRAM
        n_pixels = input_matrix.shape[0]
        
        output_list = []
        
        with torch.no_grad():
            for i in range(0, n_pixels, batch_size):
                batch_np = input_matrix[i : i+batch_size]
                batch_t = torch.from_numpy(batch_np).to(self.device)
                
                # Model returns PHYSICAL values
                pred_t = self.model(batch_t)
                output_list.append(pred_t.cpu().numpy())
                
        results_flat = np.concatenate(output_list, axis=0)
        
        # 4. Reshape to Image
        # Output order: [Tau, IceLiq, ReffL, ReffI]
        res_img = results_flat.reshape(shape[0], shape[1], 4)
        
        return {
            'tau': res_img[..., 0],
            'ice_liq_ratio': res_img[..., 1],
            'r_eff_liq': res_img[..., 2],
            'r_eff_ice': res_img[..., 3]
        }