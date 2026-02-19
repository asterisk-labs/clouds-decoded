from __future__ import annotations

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional
from skimage.transform import resize
from tqdm import tqdm
import time

from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata

from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.modules.cloud_height_emulator.resunet import Res34_Unet


logger = logging.getLogger(__name__)


class WindowDataset(Dataset):
    def __init__(self, input_padded, h_steps, w_steps, stride_h, stride_w, h_win, w_win):
        self.input_padded = input_padded
        self.h_steps = h_steps
        self.w_steps = w_steps
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.h_win = h_win
        self.w_win = w_win

    def __len__(self):
        return self.h_steps * self.w_steps

    def __getitem__(self, idx):
        i = idx // self.w_steps
        j = idx % self.w_steps
        h_start = i * self.stride_h
        w_start = j * self.stride_w
        h_end = h_start + self.h_win
        w_end = w_start + self.w_win

        window = self.input_padded[:, :, h_start:h_end, w_start:w_end]
        return window.squeeze(0), i, j


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
                else:
                    logger.warning(f"Checkpoint {pth_path} not found. Using initialized weights.")
            else:
                logger.warning("No checkpoint path provided! Using initialized weights.")

            self.model.to(self.device)
            self.model.eval()

    def process(self, scene: Sentinel2Scene) -> CloudHeightGridData:
        """Runs cloud height emulation on a Sentinel-2 scene."""
        starting_time = time.time()
        self._load_model()

        data_list = []

        # Find B02 (10m) for reference shape
        ref_band_name = "B02"
        ref_band = scene.get_band(ref_band_name)
        if ref_band is None:
            ref_band_name = list(scene.bands.keys())[0]
            ref_band = scene.get_band(ref_band_name)

        target_shape = ref_band.shape
        band_order = self.config.bands

        for bname in band_order:
            band_data = scene.get_band(bname, reflectance=True)
            if band_data.shape != target_shape:
                band_data = resize(band_data, target_shape, preserve_range=True, order=1).astype(np.float32)
            data_list.append(band_data)

        input_stack = np.stack(data_list, axis=0)  # (C, H, W)
        # Rescale from quantification range (0-40000) to training range (0-20000)
        input_stack = input_stack * (self.config.max_reflectance / 40_000)

        if input_stack.shape[0] != self.config.in_channels:
            logger.warning(f"Input channel count {input_stack.shape[0]} != config.in_channels {self.config.in_channels}.")
            if input_stack.shape[0] > self.config.in_channels:
                input_stack = input_stack[:self.config.in_channels]

        logger.info(f"Running inference on shape {input_stack.shape}")
        input_tensor = torch.from_numpy(input_stack).float()
        output_data = self._sliding_window_inference(input_tensor)

        # output_data is (1, H, W) or (H, W)
        if output_data.ndim == 3:
            output_data = output_data[0]

        # Mark invalid pixels (zero or negative height) as NaN to match
        # the original cloud height processor's convention.
        output_data[output_data <= 0] = np.nan

        meta = CloudHeightMetadata(
            processing_config=self.config.model_dump()
        )
        ending_time = time.time()
        logger.info(f"Inference completed in {ending_time - starting_time:.1f} seconds")
        return CloudHeightGridData(
            data=output_data,
            transform=scene.transform,
            crs=scene.crs,
            metadata=meta,
        )

    def _sliding_window_inference(self, input_bands: torch.Tensor) -> np.ndarray:
        """Performs sliding window inference with symmetric context padding and cropping.

        The segmentation head is used internally to zero out heights for pixels
        the model considers non-cloud (sigmoid < 0.5). These pixels get height=0
        in the output, following the convention that 0 means no valid height.
        """
        window_size = self.config.window_size
        overlap = self.config.overlap
        batch_size = self.config.batch_size
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

        pad_top = win_pad_h
        pad_left = win_pad_w

        required_H = (h_steps - 1) * stride_h + h_win
        required_W = (w_steps - 1) * stride_w + w_win

        pad_bottom = max(required_H - (H + pad_top), win_pad_h)
        pad_right = max(required_W - (W + pad_left), win_pad_w)

        input_tensor = input_bands.unsqueeze(0)  # Keep on CPU for Dataset
        input_padded = torch.nn.functional.pad(
            input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        _, _, H_pad, W_pad = input_padded.shape

        output_padded = torch.zeros((1, H_pad, W_pad), device=device)
        count_map = torch.zeros((1, H_pad, W_pad), device=device)

        dataset = WindowDataset(input_padded, h_steps, w_steps, stride_h, stride_w, h_win, w_win)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min(os.cpu_count(), self.config.batch_size + 4),
            shuffle=False,
            pin_memory=(device == "cuda")
        )

        with torch.no_grad():
            for batch_windows, batch_i, batch_j in tqdm(dataloader, desc="Inference Batches"):
                batch_windows = batch_windows.to(device)

                outputs = self.model(batch_windows)

                if isinstance(outputs, dict):
                    preds = outputs["regression"]
                    preds_cloud = outputs.get("segmentation")
                    if preds_cloud is not None:
                        cloud_masks = torch.sigmoid(preds_cloud)
                        if preds.dim() == 3:
                            preds = preds.unsqueeze(1)
                        if cloud_masks.dim() == 3:
                            cloud_masks = cloud_masks.unsqueeze(1)
                        # Zero out heights where model thinks it's not cloud
                        preds[cloud_masks < 0.5] = 0
                else:
                    preds = outputs

                if preds.dim() == 3:
                    preds = preds.unsqueeze(1)

                for idx in range(preds.shape[0]):
                    i = batch_i[idx].item()
                    j = batch_j[idx].item()
                    h_start = i * stride_h
                    w_start = j * stride_w

                    output_window = preds[idx, :, win_pad_h : win_pad_h + window_size[0],
                                                win_pad_w : win_pad_w + window_size[1]]

                    h_end_out = h_start + window_size[0]
                    w_end_out = w_start + window_size[1]

                    output_padded[:, h_start:h_end_out, w_start:w_end_out] += output_window
                    count_map[:, h_start:h_end_out, w_start:w_end_out] += 1.0

        # Average overlapping windows
        output_padded /= torch.clamp(count_map, min=1.0)

        # Crop to original size and rescale back to metres
        output_full = output_padded[:, :H, :W] * self.config.max_reflectance

        return output_full.cpu().numpy()
