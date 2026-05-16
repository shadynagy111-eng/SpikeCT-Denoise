"""
src/utils/metrics.py
--------------------
PSNR, SSIM, and MSE computation for CT denoising evaluation.

All metrics operate on normalized [0, 1] float32 images.
SSIM uses skimage.metrics.structural_similarity with data_range=1.0
to match the implementation used in the encoding validation phase.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr


def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Compute PSNR, SSIM, and MSE for a batch of predictions.

    Args:
        pred:   (B, 1, H, W) float32 tensor, values in [0, 1]
        target: (B, 1, H, W) float32 tensor, values in [0, 1]

    Returns:
        dict with keys: psnr, ssim, mse (all floats, batch means)
    """
    pred_np   = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    psnr_vals, ssim_vals, mse_vals = [], [], []

    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0],   0.0, 1.0)  # (H, W)
        t = target_np[i, 0]                      # (H, W)

        mse = float(np.mean((p - t) ** 2))
        mse_vals.append(mse)
        psnr_vals.append(float(_psnr(t, p, data_range=1.0)))
        ssim_vals.append(float(_ssim(t, p, data_range=1.0)))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "mse":  float(np.mean(mse_vals)),
    }