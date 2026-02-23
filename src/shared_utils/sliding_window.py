"""Model-agnostic sliding window inference.

Both the cloud-height emulator and the cloud-mask processor use this
shared utility so that there is a single implementation of the
pad → tile → batch-infer → accumulate → crop logic.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _WindowDataset(Dataset):
    """Dataset that yields batched input windows for a list of (i, j) grid positions."""

    def __init__(
        self,
        input_padded: torch.Tensor,             # (1, C, H_pad, W_pad)
        valid_indices: List[Tuple[int, int]],
        stride: int,
        h_win: int,   # full model-input window height (output_window + 2 * context_pad)
        w_win: int,
    ):
        self.input_padded = input_padded
        self.valid_indices = valid_indices
        self.stride = stride
        self.h_win = h_win
        self.w_win = w_win

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        i, j = self.valid_indices[idx]
        h_start = i * self.stride
        w_start = j * self.stride
        window = self.input_padded[
            :, :, h_start: h_start + self.h_win, w_start: w_start + self.w_win
        ]
        return window.squeeze(0), i, j  # (C, h_win, w_win), int, int


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

        # Reflect-pad input so every window position is fully covered
        pad_top = ctx
        pad_left = ctx
        required_H = (h_steps - 1) * stride + h_win
        required_W = (w_steps - 1) * stride + w_win
        pad_bottom = max(required_H - (H + pad_top), ctx)
        pad_right = max(required_W - (W + pad_left), ctx)

        input_tensor = torch.from_numpy(input_arr).float().unsqueeze(0)  # (1, C, H, W)
        input_padded = torch.nn.functional.pad(
            input_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect",
        )  # (1, C, H_pad, W_pad)

        _, _, H_pad, W_pad = input_padded.shape

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

        output = torch.zeros((n_output_channels, H_pad, W_pad), device=self.device)
        count_map = torch.zeros((n_output_channels, H_pad, W_pad), device=self.device)

        dataset = _WindowDataset(input_padded, valid_indices, stride, h_win, w_win)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=min(os.cpu_count() or 1, self.batch_size + 4),
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )

        for batch_windows, batch_i, batch_j in tqdm(dataloader, desc="Inference"):
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
