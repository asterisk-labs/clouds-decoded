from __future__ import annotations

import torch
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Union
from skimage.transform import resize
from tqdm import tqdm
import time
from rasterio.transform import Affine
from clouds_decoded.data import Sentinel2Scene, CloudHeightGridData, CloudHeightMetadata, CloudMaskData
from clouds_decoded.base_processor import BaseProcessor
from clouds_decoded.normalization import CloudHeightNormalizationWrapper
from clouds_decoded.modules.cloud_height_emulator.config import CloudHeightEmulatorConfig
from clouds_decoded.modules.cloud_height_emulator.resunet import Res34_Unet
from clouds_decoded.sliding_window import SlidingWindowInference


logger = logging.getLogger(__name__)


class CloudHeightEmulatorProcessor(BaseProcessor):
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

    def _process(
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

        band_objects = scene.get_bands(
            self.config.bands, reflectance=True, n_workers=len(self.config.bands),
        )
        # Derive target shape from B02 actual dimensions scaled to working_resolution
        b02_idx = self.config.bands.index("B02") if "B02" in self.config.bands else 0
        b02_actual_shape = band_objects[b02_idx].data.shape
        b02_native_res = abs(scene.transform.a)  # typically 10.0 m
        w_scale = b02_native_res / self.config.working_resolution
        target_shape = (
            max(1, round(b02_actual_shape[0] * w_scale)),
            max(1, round(b02_actual_shape[1] * w_scale)),
        )
        working_transform = scene.transform * Affine.scale(1.0 / w_scale)

        data_list = []
        for band_obj in band_objects:
            band_arr = band_obj.data
            if band_arr.shape != target_shape:
                band_arr = resize(band_arr, target_shape, preserve_range=True, order=1).astype(np.float32)
            data_list.append(band_arr)
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

        w = self.config.window_size[0]  # window is always square

        def model_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                outputs = self.model(x)
            if isinstance(outputs, dict):
                preds = outputs["regression"]
                preds_cloud = outputs.get("segmentation")
                if preds_cloud is not None:
                    cloud_masks = torch.sigmoid(preds_cloud)
                    if preds.dim() == 3:
                        preds = preds.unsqueeze(1)
                    if cloud_masks.dim() == 3:
                        cloud_masks = cloud_masks.unsqueeze(1)
                    preds[cloud_masks < 0.5] = 0
            else:
                preds = outputs
            if preds.dim() == 3:
                preds = preds.unsqueeze(1)
            return preds

        swi = SlidingWindowInference(
            window_size=w,
            overlap=self.config.overlap,
            context_pad=w // 8,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        output_data = swi(
            input_stack, model_fn, n_output_channels=1, skip_mask=mask_array
        )  # (1, H, W)

        output_data = output_data[0]  # (H, W)

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
            transform=working_transform,
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
