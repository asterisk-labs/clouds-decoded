
import numpy as np
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Union, Optional
from skimage.transform import resize
from skimage.morphology import disk, dilation
import rasterio

from clouds_decoded.data import Sentinel2Scene, CloudMaskData, CloudMaskMetadata
from clouds_decoded.constants import BANDS as SENTINEL2_BAND_NAMES
from clouds_decoded.base_processor import BaseProcessor
from .config import CloudMaskConfig, PostProcessParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SegFormer-B2 architecture — expressed directly in code so that _load_model
# never needs to hit HuggingFace (no from_pretrained call).
#
# Weights in the checkpoint are saved under a "segmenter.*" prefix (from the
# senseiv2 FullModel wrapper), so _load_model strips that prefix and loads
# directly into a bare SegformerForSemanticSegmentation with strict=True.
# ---------------------------------------------------------------------------
_MODEL_NAME = "SegFormerB2-S2-unambiguous"

# SegFormer-B2 architecture params (verified against saved checkpoint shapes)
_SEGFORMER_B2 = dict(
    num_channels=13,
    num_labels=4,
    hidden_sizes=[64, 128, 320, 512],
    num_encoder_blocks=4,
    depths=[3, 4, 6, 3],
    sr_ratios=[8, 4, 2, 1],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    decoder_hidden_size=768,
)

_PATCH_SIZE = 512   # fixed SegFormerB2 patch size


class ThresholdCloudMaskProcessor(BaseProcessor):
    """Simple threshold-based cloud mask using a single band's reflectance.

    Pixels with reflectance greater than the configured threshold are classified as cloud.
    Produces a binary 2-class mask (0=clear, 1=cloud).
    """

    def __init__(self, config: Optional[CloudMaskConfig] = None):
        if config is None:
            config = CloudMaskConfig(method="threshold")
        self.config = config

    def _process(self, scene: Sentinel2Scene) -> CloudMaskData:
        """
        Creates a basic cloud mask by thresholding a specific band.
        Values GREATER than the threshold are considered cloud (1), others not (0).
        """
        threshold_band = self.config.threshold_band
        threshold_value = self.config.threshold_value

        # Get calibrated reflectance
        band_data = scene.get_band(threshold_band, reflectance=True)

        # Ensure 2D
        if isinstance(band_data, np.ndarray) and band_data.ndim == 3:
             band_data = band_data[0]

        logger.info(f"Generating cloud mask using: {threshold_band} > {threshold_value}")

        mask = (band_data > threshold_value).astype(np.uint8)

        # Construct CloudMaskData using Sentinel2Scene georeferencing
        return CloudMaskData(
             data=mask,
             transform=scene.transform,
             crs=scene.crs,
             metadata=CloudMaskMetadata(
                 categorical=True,
                 classes={0: 'Clear', 1: 'Cloud'},
                 method="simple_threshold",
                 threshold_band=threshold_band,
                 threshold_value=threshold_value
             )
        )

class CloudMaskProcessor(BaseProcessor):
    """Deep learning cloud mask using a SegFormer-B2 (SEnSeIv2) model.

    Produces a 4-class probability map (clear, thick cloud, thin cloud, shadow)
    via sliding-window inference over all 13 Sentinel-2 bands.
    """

    def __init__(self, config: Optional[CloudMaskConfig] = None):
        if config is None:
            config = CloudMaskConfig(method="senseiv2")
        self.config = config

        if self.config.device:
            self.device = self.config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None

    def _load_model(self):
        if self.model is None:
            weights_path = Path(self.config.model_path)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Cloud mask weights not found at {weights_path}.\n"
                    f"Run:  clouds-decoded download cloud_mask\n"
                    f"or set CLOUDS_DECODED_ASSETS_DIR to a directory containing "
                    f"models/cloud_mask/default.pt"
                )
            logger.info(f"Loading {_MODEL_NAME} cloud mask model on {self.device}")

            from transformers import SegformerConfig, SegformerForSemanticSegmentation
            cfg = SegformerConfig(**_SEGFORMER_B2)
            self.model = SegformerForSemanticSegmentation(cfg).to(self.device)

            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            # Checkpoint keys carry a "segmenter." prefix from the senseiv2
            # FullModel wrapper — strip it before loading.
            stripped = {k[len("segmenter."):]: v for k, v in state_dict.items()}
            self.model.load_state_dict(stripped, strict=True)
            self.model.eval()

    def _process(self, scene: Sentinel2Scene) -> CloudMaskData:
        """Generate a binary cloud mask using the SEnSeIv2 deep learning model.

        Runs SegFormer-B2 sliding-window inference to produce per-class
        probabilities, then binarizes using ``cloud_mask_classes`` and
        ``cloud_mask_threshold`` from config.

        Args:
            scene: Sentinel2Scene object with loaded bands.

        Returns:
            CloudMaskData: Binary cloud mask (0=clear, 1=cloud, uint8).
        """
        self._load_model()

        # 1. Prepare Data
        ref_arr = scene.bands.get("B02") or next(iter(scene.bands.values()))
        if ref_arr.ndim == 3:
            ref_arr = ref_arr[0]
        current_res = abs(scene.transform[0])
        target_res = float(self.config.working_resolution)
        scale_factor = current_res / target_res
        target_shape = (int(ref_arr.shape[0] * scale_factor), int(ref_arr.shape[1] * scale_factor))

        logger.info(
            f"Input Resolution Processing: Native {current_res:.2f}m -> "
            f"Target {target_res}m. Shape: {target_shape}"
        )

        input_data = np.stack(
            [scene.get_band_at_shape(name, target_shape) for name in SENTINEL2_BAND_NAMES],
            axis=0,
        )  # (13, H, W)

        # 2. Run Inference via shared sliding window
        from clouds_decoded.sliding_window import SlidingWindowInference

        overlap = _PATCH_SIZE - self.config.stride

        def model_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                logits = self.model(x).logits           # (B, 4, H/4, W/4)
                upsampled = F.interpolate(
                    logits, size=x.shape[-2:],
                    mode="bilinear", align_corners=False,
                )                                       # (B, 4, H, W)
                return F.softmax(upsampled, dim=1)

        swi = SlidingWindowInference(
            window_size=_PATCH_SIZE,
            overlap=overlap,
            context_pad=0,
            batch_size=self.config.batch_size,
            device=self.device,
        )
        logger.info("Running cloud mask inference...")
        probs = swi(input_data, model_fn, n_output_channels=4)  # (4, H, W)

        # 3. Build output transform for the (possibly resampled) resolution
        from rasterio.transform import Affine
        if isinstance(scene.transform, Affine):
            s = 1.0 / scale_factor
            new_affine = scene.transform * Affine.scale(s, s)
        else:
            new_affine = scene.transform

        # 4. Derive binary mask from probabilities.
        #    Shadow reclassification operates on the categorical argmax
        #    internally but only the binary mask is returned.
        categorical = np.argmax(probs, axis=0).astype(np.uint8)

        # 4a. Reclassify shadow pixels that are embedded within cloud.
        #     These are self-shading artefacts — shadow predicted in the
        #     middle of optically thick cloud.  Genuine ground shadows sit
        #     near clear-sky regions.
        reclassified_mask = None
        if self.config.reclassify_embedded_shadow:
            pre = categorical.copy()
            categorical = self._reclassify_embedded_shadow(categorical)
            reclassified_mask = (categorical != pre)

        cloud_prob = sum(
            probs[c] for c in self.config.cloud_mask_classes
            if c < probs.shape[0]
        )
        binary = (cloud_prob > self.config.cloud_mask_threshold).astype(np.uint8)

        # Apply reclassification to the binary mask — pixels that were
        # shadow-turned-cloud in the categorical must also be marked as
        # cloud in the binary, since the probability array still reflects
        # the original shadow prediction.
        if reclassified_mask is not None:
            binary[reclassified_mask] = 1

        return CloudMaskData(
            data=binary,
            transform=new_affine,
            crs=scene.crs,
            nodata=255,
            metadata=CloudMaskMetadata(
                categorical=True,
                classes={0: 'Clear', 1: 'Cloud'},
                method="senseiv2",
                model=_MODEL_NAME,
                resolution=target_res,
            )
        )

    def _reclassify_embedded_shadow(self, categorical: np.ndarray) -> np.ndarray:
        """Reclassify shadow pixels surrounded by cloud as thick cloud.

        For each shadow pixel (class 3), compares the local fraction of cloud
        (classes 1, 2) to clear (class 0) within a square window.  If cloud
        fraction exceeds clear fraction the pixel is reclassified as thick
        cloud (class 1), on the basis that genuine ground shadows are adjacent
        to clear sky while self-shading artefacts are embedded in cloud.

        Args:
            categorical: 2-D uint8 array with classes 0–3.

        Returns:
            Modified categorical array (mutated in-place and returned).
        """
        from scipy.ndimage import uniform_filter

        is_shadow = categorical == 3
        n_shadow = int(is_shadow.sum())
        if n_shadow == 0:
            return categorical

        window = 2 * self.config.shadow_reclassify_radius + 1

        cloud_frac = uniform_filter(
            np.isin(categorical, [1, 2]).astype(np.float32), size=window
        )
        clear_frac = uniform_filter(
            (categorical == 0).astype(np.float32), size=window
        )

        reclassify = is_shadow & (cloud_frac > clear_frac)
        n_reclassified = int(reclassify.sum())
        categorical[reclassify] = 1  # thick cloud

        logger.info(
            f"Shadow reclassification: {n_reclassified}/{n_shadow} shadow pixels "
            f"reclassified as thick cloud (window={window}px)"
        )
        return categorical

    def postprocess(self, mask_data: CloudMaskData, params: PostProcessParams) -> CloudMaskData:
        """
        Refines the cloud mask based on specific requirements (resolution, buffering, classes).
        """
        mask = mask_data.data.copy()
        current_transform = mask_data.transform

        # 1. Resize if needed
        if params.output_resolution is not None:
             current_res = abs(current_transform[0])
             target_res = float(params.output_resolution)

             if abs(current_res - target_res) > 0.1:
                  scale = current_res / target_res
                  new_h = int(mask.shape[0] * scale)
                  new_w = int(mask.shape[1] * scale)

                  logger.info(f"Post-processing: Resizing mask {mask.shape} -> {(new_h, new_w)}")
                  mask = resize(mask, (new_h, new_w), order=0, preserve_range=True).astype(np.uint8)

                  s = 1.0 / scale
                  from rasterio.transform import Affine
                  if isinstance(current_transform, Affine):
                      current_transform = current_transform * Affine.scale(s, s)

        # 2. Apply Class Selection & Binarization
        if params.binary_mask:
             binary = np.isin(mask, params.classes_to_mask).astype(np.uint8)
             mask = binary

        # 3. Buffering (Dilation)
        if params.buffer_size > 0:
             res = params.output_resolution if params.output_resolution else abs(current_transform[0])
             pixels = int(params.buffer_size / res)
             if pixels > 0:
                  logger.info(f"The mask will be buffered by {pixels} pixels ({params.buffer_size}m)")
                  selem = disk(pixels)
                  mask = dilation(mask, selem)

        meta_dict = mask_data.metadata.model_dump()
        meta_dict['postprocessed'] = True
        updated_metadata = CloudMaskMetadata(**meta_dict)

        return CloudMaskData(
             data=mask,
             transform=current_transform,
             crs=mask_data.crs,
             metadata=updated_metadata
        )
