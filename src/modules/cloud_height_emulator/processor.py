import torch
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Dict
from skimage.transform import resize

from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata
from rasterio.transform import Affine

from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.modules.cloud_height_emulator.resunet import Res34_Unet


import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

class CloudHeightEmulatorProcessor:
    def __init__(self, config: Optional[CloudHeightEmulatorConfig] = None):
        if config is None:
            config = CloudHeightEmulatorConfig()
        self.config = config
        
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading Cloud Height Emulator model on {self.device}")
            self.model = Res34_Unet(
                in_channels=self.config.in_channels,
                out_channels=[1, 1],
                heads=["regression", "segmentation"],
                heads_hidden_channels=[48, 48],
                pretrained=False
            )
            

            if self.config.pth_path:
                pth_path = Path(self.config.pth_path)
                if pth_path.exists():
                    logger.info(f"Loading checkpoint from {pth_path}")
                    checkpoint = torch.load(pth_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint, strict=True)
                    # check if all keys are present
                    missing_keys = self.model.load_state_dict(checkpoint, strict=False)
                    if missing_keys.missing_keys:
                        raise ValueError(f"Missing keys: {missing_keys.missing_keys}, did you use the correct checkpoint?")
                    # check if all keys are present
                    if missing_keys.unexpected_keys:
                        raise ValueError(f"Unexpected keys: {missing_keys.unexpected_keys}, did you use the correct checkpoint?") 
                else:
                    logger.warning(f"Checkpoint {pth_path} not found. Using initialized weights.")

            # if self.config.ckpt_path:
            #     ckpt_path = Path(self.config.ckpt_path)
            #     if ckpt_path.exists():
            #         logger.info(f"Loading checkpoint from {ckpt_path}")
            #         checkpoint = torch.load(ckpt_path, map_location=self.device)
            #         state_dict = checkpoint.get("state_dict", checkpoint)
            #         if any(k.startswith("model.") for k in state_dict.keys()):
            #             state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            #         #logger.warning if some weights are missing
            #         missing_keys = self.model.load_state_dict(state_dict, strict=False)
            #         if missing_keys.missing_keys:
            #             raise ValueError(f"Missing keys: {missing_keys.missing_keys}, did you use the correct checkpoint?")
            #         # check if all keys are present
            #         if missing_keys.unexpected_keys:
            #             raise ValueError(f"Unexpected keys: {missing_keys.unexpected_keys}, did you use the correct checkpoint?") 
                    

            #     else:
            #         logger.warning(f"Checkpoint {ckpt_path} not found. Using initialized weights.")
            else:
                 logger.warning("No checkpoint path provided! Using initialized weights.")

            self.model.to(self.device)
            self.model.eval()

    def process(self, scene: Sentinel2Scene) -> CloudHeightGridData:
        """
        Runs cloud height emulation on a Sentinel-2 scene.
        """
        self._load_model()
     
        
        data_list = []
        target_shape = None
        
        # Find B02 (10m) for reference shape
        ref_band_name = "B02"
        ref_band = scene.get_band(ref_band_name)
        if ref_band is None:
             # Fallback
             ref_band_name = list(scene.bands.keys())[0]
             ref_band = scene.get_band(ref_band_name)
             
        target_shape = ref_band.shape
        band_order = self.config.bands
        
        
        for bname in band_order:
            band_data = scene.get_band(bname, reflectance=True)  
            if band_data.shape != target_shape:
                 band_data = resize(band_data, target_shape, preserve_range=True, order=1).astype(np.float32)
            
            data_list.append(band_data)
        input_stack = np.stack(data_list, axis=0) # (C, H, W)
        max_refl = 20_000 # TODO make in config or just train on the full range
        input_stack = input_stack * (max_refl / 40_000) # TODO change dataset logic and this
        
        if input_stack.shape[0] != self.config.in_channels:
             logger.warning(f"Input channel count {input_stack.shape[0]} != config.in_channels {self.config.in_channels}. Only using first {self.config.in_channels} or padding?")
             if input_stack.shape[0] > self.config.in_channels:
                  input_stack = input_stack[:self.config.in_channels]
        
        # 2. Run Inference
        logger.info(f"Running inference on shape {input_stack.shape}")
        input_tensor = torch.from_numpy(input_stack).float()
        output_data, output_cloud = self._sliding_window_inference(input_tensor)
        
        
        # 3. Wrap result
        # output_data is (1, H, W) or (H, W).
        if output_data.ndim == 3:
             output_data = output_data[0]
        if output_cloud.ndim == 3:
             output_cloud = output_cloud[0]
             
        # Create Metadata
        meta = CloudHeightMetadata(
             processing_config=self.config.model_dump()
        )
        
        return CloudHeightGridData(
             data=output_data,
             transform=scene.transform,
             crs=scene.crs,
             metadata=meta,
             cloud_mask=output_cloud
        )

    def _sliding_window_inference(self, input_bands: torch.Tensor):
        """
        Performs sliding window inference with symmetric context padding and cropping.
        """
        window_size = self.config.window_size
        overlap = window_size[0] // 2
        device = self.device

        C, H, W = input_bands.shape
        
        # Define padding for context (equal on all sides for 'center' crop)
        win_pad_h, win_pad_w = window_size[0] // 8, window_size[1] // 8
        
        # Define effective window size (original + 2 * context)
        h_win = window_size[0] + 2 * win_pad_h
        w_win = window_size[1] + 2 * win_pad_w

        # Calculate stride based on the *output* window size (which is window_size)
        stride_h = window_size[0] - overlap
        stride_w = window_size[1] - overlap

        h_steps = (H + stride_h - 1) // stride_h
        w_steps = (W + stride_w - 1) // stride_w
        
        # Calculate required padded size
        # The last window starts at (h_steps - 1) * stride_h
        # It has length h_win.
        # So total required size = (h_steps - 1) * stride_h + h_win
        
        required_H = (h_steps - 1) * stride_h + h_win
        required_W = (w_steps - 1) * stride_w + w_win
        
        # Current size with just context padding would be:
        # H + 2 * win_pad
        # But we pad specifically:
        # Left/Top: win_pad
        # Right/Bottom: remainder
        
        pad_top = win_pad_h
        pad_left = win_pad_w
        
        pad_bottom = required_H - (H + pad_top)
        pad_right = required_W - (W + pad_left)
        
        # Ensure non-negative (should be if logic is correct, but safer to clip)
        pad_bottom = max(pad_bottom, win_pad_h) # At least win_pad context
        pad_right = max(pad_right, win_pad_w)

        # input_tensor needs to be 4D (B, C, H, W)
        input_tensor = input_bands.unsqueeze(0).to(device)

        # Pad input
        input_padded = torch.nn.functional.pad(
            input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        _, _, H_pad, W_pad = input_padded.shape

        # Initialize output container
        # We'll make it large enough to hold all predictions, then crop
        output_padded = torch.zeros((1, H_pad, W_pad), device=device)
        output_padded_cloud = torch.zeros((1, H_pad, W_pad), device=device)
        count_map = torch.zeros((1, H_pad, W_pad), device=device)
        
        # Iterate
        with torch.no_grad():
            for i in range(h_steps):
                for j in range(w_steps):
                    h_start = i * stride_h
                    w_start = j * stride_w
                    h_end = h_start + h_win
                    w_end = w_start + w_win

                    # Check bounds (should be safe by calculation)
                    if h_end > H_pad or w_end > W_pad:
                        continue

                    window = input_padded[:, :, h_start:h_end, w_start:w_end]

                    # Inference
                    output = self.model(window)                    
                    # Handle output format
                    if isinstance(output, dict):
                        pred = output["regression"]
                        pred_cloud = output.get("segmentation")
                        if pred_cloud is not None:
                             cloud_mask = torch.sigmoid(pred_cloud)
                             pred[cloud_mask < 0.5] = 0
                    else:
                        pred = output

                    if pred.dim() == 3:
                        pred = pred.unsqueeze(1)
                    
                    output_window = pred[:, :, win_pad_h : win_pad_h + window_size[0], 
                                               win_pad_w : win_pad_w + window_size[1]]
                    output_window_cloud = cloud_mask[ :, win_pad_h : win_pad_h + window_size[0], 
                                               win_pad_w : win_pad_w + window_size[1]]
                    
                    h_end_out = h_start + window_size[0]
                    w_end_out = w_start + window_size[1]
                    
                    output_padded[:, h_start:h_end_out, w_start:w_end_out] += output_window[0]
                    output_padded_cloud[:, h_start:h_end_out, w_start:w_end_out] += output_window_cloud[0]
                    count_map[:, h_start:h_end_out, w_start:w_end_out] += 1.0

        # Average
        output_padded /= torch.clamp(count_map, min=1.0)
        output_padded_cloud /= torch.clamp(count_map, min=1.0)

        # Crop to original size
        # Since output_padded[0] <-> orig[0], we just take :H, :W
        output_full = output_padded[:, :H, :W]*20_000
        output_full_cloud = output_padded_cloud[:, :H, :W]

        return output_full.cpu().numpy(), output_full_cloud.cpu().numpy()


if __name__ == "__main__":
    scene = Sentinel2Scene()
    scene.read(
        "/data/CTH_emulator_dataset/Fiji/Sentinel-2/MSI/L1C/2024/01/24/S2B_MSIL1C_20240124T232839_N0510_R044_T58LDN_20240125T002811.SAFE/"
    )
    # config from yaml
    config = CloudHeightEmulatorConfig.from_yaml(
        "/home/paul/clouds-decoded/my_analysis/configs/cloud_height_emulator.yaml"
    )
    processor = CloudHeightEmulatorProcessor(config)
    result = processor.process(scene)
    print(result.data.shape)
    print(result.data.min())
    print(result.data.max())
    print(result.data.mean())
    



    fig,ax = plt.subplots(1,3,figsize=(10,5))

    input_bands = []
    target_shape = scene.get_band("B04").shape

    for bname in ["B12", "B11", "B04"]:
        band_data = scene.get_band(bname, reflectance=True)  
        if band_data.shape != target_shape:
            band_data = resize(band_data, target_shape, preserve_range=True, order=1).astype(np.float32)
        input_bands.append(band_data)
    ice_image = np.stack([input_bands[0]/0.18635, input_bands[1]/0.21484999, input_bands[2] / 0.6243],  # B12, B11, B4, hard coded max values
                             0)
    ice_image = np.moveaxis(ice_image, 0, 2)
    ax[0].imshow(ice_image)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(result.data, cmap="terrain", vmin=0, vmax=20000)
    ax[1].set_title("Cloud Height")
    ax[1].axis("off")
    im = ax[2].imshow(result.cloud_mask, cmap="gray")
    ax[2].set_title("Cloud Mask")
    ax[1].axis("off")
    plt.colorbar(im,ax=ax[1])
    plt.tight_layout()
    plt.savefig("cloud_height_emulator_test.png")
