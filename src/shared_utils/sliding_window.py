"""Model-agnostic sliding window inference.

Both the cloud-height emulator and the cloud-mask processor use this
shared utility so that there is a single implementation of the
pad → tile → batch-infer → accumulate → crop logic.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Set to True by project.run() before worker threads are created.
# All threads inherit the same module object, so no ContextVar needed.
suppress_inference_progress: bool = False


class _WindowDataset(Dataset):
    """Dataset that lazily extracts model-input windows from the original array.

    Windows are extracted on the fly rather than from a pre-padded copy of the
    full image. Interior windows are zero-copy numpy views; only boundary windows
    allocate a small padded copy. This eliminates the full ``(C, H_pad, W_pad)``
    upfront allocation (typically several GB for full Sentinel-2 scenes).
    """

    def __init__(
        self,
        input_arr: np.ndarray,              # (C, H, W) float32
        valid_indices: List[Tuple[int, int]],
        stride: int,
        h_win: int,
        w_win: int,
        ctx: int,                           # context_pad pixels on each side
    ):
        self.input_arr = input_arr
        self.valid_indices = valid_indices
        self.stride = stride
        self.h_win = h_win
        self.w_win = w_win
        self.ctx = ctx
        self.H = input_arr.shape[1]
        self.W = input_arr.shape[2]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        i, j = self.valid_indices[idx]
        # Model-input window origin in original (unpadded) coordinates.
        h_in = i * self.stride - self.ctx
        w_in = j * self.stride - self.ctx

        h_s = max(0, h_in)
        h_e = min(self.H, h_in + self.h_win)
        w_s = max(0, w_in)
        w_e = min(self.W, w_in + self.w_win)

        window = torch.from_numpy(self.input_arr[:, h_s:h_e, w_s:w_e]).float()

        pad_top    = h_s - h_in
        pad_bottom = (h_in + self.h_win) - h_e
        pad_left   = w_s - w_in
        pad_right  = (w_in + self.w_win) - w_e

        if pad_top or pad_bottom or pad_left or pad_right:
            _, h, w = window.shape
            # reflect requires padding < dimension; fall back to replicate (edge extension)
            # for extreme boundary windows where the slice is smaller than the padding.
            can_reflect = h > pad_top and h > pad_bottom and w > pad_left and w > pad_right
            window = F.pad(
                window.unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="reflect" if can_reflect else "replicate",
            ).squeeze(0)

        return window, i, j  # (C, h_win, w_win), int, int


class SlidingWindowInference:
    """Model-agnostic sliding window inference with optional context padding.

    Parameters
    ----------
    window_size:
        Side length (pixels) of the square *output* window.
    overlap:
        Number of pixels that adjacent output windows overlap.
    context_pad:
        Extra pixels fed to the model on each side beyond the output window
        but cropped from the result.  The actual model input is
        ``window_size + 2 * context_pad`` square.  Default 0.
    batch_size:
        Number of windows per model call.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        window_size: int,
        overlap: int,
        context_pad: int = 0,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> None:
        if overlap >= window_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than window_size ({window_size})"
            )
        self.window_size = window_size
        self.overlap = overlap
        self.context_pad = context_pad
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self,
        input_arr: np.ndarray,
        model_fn: Callable[[Tensor], Tensor],
        n_output_channels: int = 1,
        skip_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run sliding window inference over a spatial array.

        Parameters
        ----------
        input_arr:
            ``(C, H, W)`` float32 numpy array.
        model_fn:
            Callable ``(B, C, H_in, W_in) → (B, n_out, H_out, W_out)`` tensor.
            Should wrap its own ``torch.no_grad()`` if required.
        n_output_channels:
            Number of channels in the model output.
        skip_mask:
            Optional ``(H, W)`` boolean array.  Windows whose *output* region
            contains no True pixels are skipped entirely.

        Returns
        -------
        np.ndarray
            ``(n_output_channels, H, W)`` float32 result array.
        """
        C, H, W = input_arr.shape
        ws = self.window_size
        ctx = self.context_pad
        stride = ws - self.overlap

        # Full model-input window size (output window + context on each side)
        h_win = ws + 2 * ctx
        w_win = ws + 2 * ctx

        # Number of strides needed to cover the image
        h_steps = (H + stride - 1) // stride
        w_steps = (W + stride - 1) // stride

        # Accumulator dimensions: just enough to hold all output window placements
        H_acc = (h_steps - 1) * stride + ws
        W_acc = (w_steps - 1) * stride + ws

        # Ensure float32 (zero-copy if already float32)
        input_f32 = input_arr.astype(np.float32, copy=False)

        valid_indices = self._get_valid_indices(
            skip_mask, h_steps, w_steps, stride, ws, H, W
        )
        total = h_steps * w_steps
        skipped = total - len(valid_indices)
        if skipped:
            logger.info(
                "Sliding window: skipping %d/%d windows (no True pixels in skip_mask)",
                skipped, total,
            )

        output = torch.zeros((n_output_channels, H_acc, W_acc), device=self.device)
        # Single-channel count_map: overlap count is channel-independent, broadcast on divide
        count_map = torch.zeros((1, H_acc, W_acc), device=self.device)

        dataset = _WindowDataset(input_f32, valid_indices, stride, h_win, w_win, ctx)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,  # dataset is pure in-memory tensor slicing; workers add no throughput
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )

        for batch_windows, batch_i, batch_j in tqdm(
            dataloader, desc="Inference", disable=suppress_inference_progress
        ):
            batch_windows = batch_windows.to(self.device)
            preds = model_fn(batch_windows)  # (B, n_out, H_out, W_out)
            if preds.dim() == 3:
                preds = preds.unsqueeze(1)

            for k in range(preds.shape[0]):
                i = int(batch_i[k].item())
                j = int(batch_j[k].item())
                h_start = i * stride
                w_start = j * stride
                # Crop context border from model output to get the output window
                out_win = preds[k, :, ctx: ctx + ws, ctx: ctx + ws]
                output[:, h_start: h_start + ws, w_start: w_start + ws] += out_win
                count_map[:, h_start: h_start + ws, w_start: w_start + ws] += 1.0

        del dataset, dataloader

        output /= torch.clamp(count_map, min=1.0)
        return output[:, :H, :W].cpu().numpy()

    @staticmethod
    def _get_valid_indices(
        skip_mask: Optional[np.ndarray],
        h_steps: int,
        w_steps: int,
        stride: int,
        window_size: int,
        H: int,
        W: int,
    ) -> List[Tuple[int, int]]:
        """Return (i, j) pairs for windows that contain at least one True pixel."""
        all_indices = [(i, j) for i in range(h_steps) for j in range(w_steps)]
        if skip_mask is None:
            return all_indices
        valid = []
        for i, j in all_indices:
            h_s = i * stride
            w_s = j * stride
            h_e = min(h_s + window_size, H)
            w_e = min(w_s + window_size, W)
            if skip_mask[h_s:h_e, w_s:w_e].any():
                valid.append((i, j))
        return valid
