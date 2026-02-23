from __future__ import annotations

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple
from skimage.transform import resize
from tqdm import tqdm
import time
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata, CloudMaskData

from clouds_decoded.normalization import CloudHeightNormalizationWrapper
from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.modules.cloud_height_emulator.resunet import Res34_Unet


logger = logging.getLogger(__name__)


class WindowDataset(Dataset):
    """Dataset that yields input windows for a list of (i, j) grid positions."""

    def __init__(
        self,
        input_padded: torch.Tensor,
        valid_indices: List[Tuple[int, int]],
        stride_h: int,
        stride_w: int,
        h_win: int,
        w_win: int,
    ):
        self.input_padded = input_padded
        self.valid_indices = valid_indices
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.h_win = h_win
        self.w_win = w_win

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i, j = self.valid_indices[idx]
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
            logger.info(
                f"Loading Cloud Height Emulator model on {self.device}")
            core = Res34_Unet(
                in_channels=self.config.in_channels,
                out_channels=[1, 1],
                heads=["regression", "segmentation"],
                heads_hidden_channels=[48, 48],
                pretrained=False
            )

            self.model = CloudHeightNormalizationWrapper(
                model=core,
                input_stats={"min": 0.0, "max": 1.0},
                output_stats={"min": 0.0, "max": 1.0},
            )  # default params, are loaded in the checkpoint.

            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Cloud height emulator weights not found at {model_path}.\n"
                    f"Run:  clouds-decoded download emulator\n"
                    f"or set CLOUDS_DECODED_ASSETS_DIR to a directory containing "
                    f"models/cloud_height_emulator/default.pth"
                )

            logger.info(f"Loading checkpoint from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            if any(k.startswith("model.") for k in state_dict.keys()):
                import re
                state_dict = {
                    re.sub(r"^model\.", "", k): v
                    for k, v in state_dict.items()
                }

            # Detect checkpoint format: wrapper (keys include
            # normalization buffers) vs raw Res34_Unet weights.
            has_wrapper_keys = any(k in state_dict for k in (
                "in_min", "in_max", "out_min", "out_max"))

            if has_wrapper_keys:
                # Full wrapper checkpoint — load into self.model
                self.model.load_state_dict(state_dict, strict=True)
            else:
                raise ValueError(
                    "Checkpoint does not contain wrapper keys.")

            self.model.to(self.device)
            self.model.eval()

    def process(
        self,
        scene: Sentinel2Scene,
        cloud_mask: Optional[Union[CloudMaskData,
                                   np.ndarray, str, Path]] = None,
    ) -> CloudHeightGridData:
        """Runs cloud height emulation on a Sentinel-2 scene.

        Args:
            scene: The Sentinel-2 scene to process.
            cloud_mask: Optional cloud mask. If provided, windows that are
                entirely clear sky are skipped during inference, and clear-sky
                pixels are set to NaN in the output.

        Returns:
            CloudHeightGridData with NaN for invalid/clear-sky pixels.
        """
        starting_time = time.time()
        self._load_model()

        # Resolve cloud mask early — it's used to skip windows during inference
        mask_array = self._resolve_cloud_mask(cloud_mask)

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
                band_data = resize(
                    band_data, target_shape, preserve_range=True, order=1).astype(np.float32)
            data_list.append(band_data)

        input_stack = np.stack(data_list, axis=0)  # (C, H, W)

        if input_stack.shape[0] != self.config.in_channels:
            logger.warning(
                f"Input channel count {input_stack.shape[0]} != config.in_channels {self.config.in_channels}.")
            if input_stack.shape[0] > self.config.in_channels:
                input_stack = input_stack[:self.config.in_channels]

        # Resize mask to match input spatial dims if needed
        if mask_array is not None and mask_array.shape != target_shape:
            mask_array = resize(
                mask_array.astype(np.float32), target_shape,
                preserve_range=True, order=0,
            ).astype(np.uint8)

        logger.info(f"Running inference on shape {input_stack.shape}")
        input_tensor = torch.from_numpy(input_stack).float()
        output_data = self._sliding_window_inference(input_tensor, mask_array)

        # output_data is (1, H, W) or (H, W)
        if output_data.ndim == 3:
            output_data = output_data[0]

        # Mark invalid pixels (zero or negative height) as NaN to match
        # the original cloud height processor's convention.
        output_data[output_data <= 0] = np.nan

        # Apply external cloud mask: set clear-sky pixels to NaN.
        # This catches partially-cloudy windows where some pixels are clear.
        if mask_array is not None:
            output_data[mask_array == 0] = np.nan

        meta = CloudHeightMetadata(
            processing_config=self.config.model_dump()
        )
        ending_time = time.time()
        logger.info(
            f"Inference completed in {ending_time - starting_time:.1f} seconds")
        return CloudHeightGridData(
            data=output_data,
            transform=scene.transform,
            crs=scene.crs,
            metadata=meta,
        )

    @staticmethod
    def _resolve_cloud_mask(
        cloud_mask: Optional[Union[CloudMaskData, np.ndarray, str, Path]],
    ) -> Optional[np.ndarray]:
        """Convert various cloud mask inputs to a 2-D numpy array (or None)."""
        if cloud_mask is None:
            return None

        if isinstance(cloud_mask, (str, Path)):
            try:
                cm_obj = CloudMaskData.from_file(str(cloud_mask))
                mask_array = cm_obj.data
            except Exception:
                logger.warning(
                    f"Could not load {cloud_mask} as CloudMaskData. Ignoring mask.")
                return None
        elif isinstance(cloud_mask, CloudMaskData):
            mask_array = cloud_mask.data
        elif isinstance(cloud_mask, np.ndarray):
            mask_array = cloud_mask
        else:
            logger.warning(
                f"Unsupported cloud_mask type: {type(cloud_mask)}. Ignoring mask.")
            return None

        if mask_array.ndim == 3:
            mask_array = mask_array[0]

        return mask_array

    def _sliding_window_inference(
        self,
        input_bands: torch.Tensor,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Performs sliding window inference with symmetric context padding.

        When a cloud mask is provided, windows whose output region is entirely
        clear sky (mask == 0) are skipped — no inference is run for them.
        The segmentation head additionally zeros out pixels the model itself
        considers non-cloud.
        """
        window_size = self.config.window_size
        overlap = self.config.overlap
        batch_size = self.config.batch_size
        device = self.device

        C, H, W = input_bands.shape

        # Context padding (equal on all sides for 'center' crop)
        win_pad_h, win_pad_w = window_size[0] // 8, window_size[1] // 8

        # Effective window size (output window + 2 * context)
        h_win = window_size[0] + 2 * win_pad_h
        w_win = window_size[1] + 2 * win_pad_w

        # Stride based on the output window size
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

        input_tensor = input_bands.unsqueeze(0)
        input_padded = torch.nn.functional.pad(
            input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        _, _, H_pad, W_pad = input_padded.shape

        # Determine which windows actually contain cloud pixels
        valid_indices = self._get_valid_window_indices(
            mask, h_steps, w_steps, stride_h, stride_w, window_size, H, W,
        )

        total_windows = h_steps * w_steps
        if len(valid_indices) < total_windows:
            logger.info(
                f"Mask skipping {total_windows - len(valid_indices)}/{total_windows} "
                f"clear-sky windows"
            )

        output_padded = torch.zeros((1, H_pad, W_pad), device=device)
        count_map = torch.zeros((1, H_pad, W_pad), device=device)

        dataset = WindowDataset(
            input_padded, valid_indices, stride_h, stride_w, h_win, w_win,
        )
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

                    output_window = preds[idx, :, win_pad_h: win_pad_h + window_size[0],
                                          win_pad_w: win_pad_w + window_size[1]]

                    h_end_out = h_start + window_size[0]
                    w_end_out = w_start + window_size[1]

                    output_padded[:, h_start:h_end_out,
                                  w_start:w_end_out] += output_window
                    count_map[:, h_start:h_end_out, w_start:w_end_out] += 1.0

        # Average overlapping windows
        output_padded /= torch.clamp(count_map, min=1.0)

        # Crop to original size (denormalization already handled by wrapper)
        output_full = output_padded[:, :H, :W]

        return output_full.cpu().numpy()

    @staticmethod
    def _get_valid_window_indices(
        mask: Optional[np.ndarray],
        h_steps: int,
        w_steps: int,
        stride_h: int,
        stride_w: int,
        window_size: Tuple[int, int],
        H: int,
        W: int,
    ) -> List[Tuple[int, int]]:
        """Return (i, j) pairs for windows that overlap with cloud pixels.

        If no mask is provided, all windows are valid.
        """
        all_indices = [
            (i, j) for i in range(h_steps) for j in range(w_steps)
        ]

        if mask is None:
            return all_indices

        valid = []
        for i, j in all_indices:
            # Output region in original (unpadded) coordinates
            h_start = i * stride_h
            w_start = j * stride_w
            h_end = min(h_start + window_size[0], H)
            w_end = min(w_start + window_size[1], W)

            if mask[h_start:h_end, w_start:w_end].any():
                valid.append((i, j))

        return valid
