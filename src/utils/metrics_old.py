"""
Evaluation Metrics: PSNR and SSIM
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor,
                   max_val: float = 1.0) -> float:
    """PSNR in dB. Inputs in [0, max_val]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10((max_val ** 2) / mse)).item()



def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """SSIM in [0, 1]. Inputs shape [1, H, W] or [B, 1, H, W]"""
    if img1.dim() == 4:
        img1 = img1[0, 0].cpu().numpy()
        img2 = img2[0, 0].cpu().numpy()
    else:
        img1 = img1[0].cpu().numpy()
        img2 = img2[0].cpu().numpy()

    return ssim(img1, img2, data_range=1.0)


def batch_psnr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Average PSNR across batch"""
    return sum(
        calculate_psnr(predictions[i], targets[i])
        for i in range(predictions.size(0))
    ) / predictions.size(0)


def batch_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Average SSIM across batch"""
    return sum(
        calculate_ssim(predictions[i:i+1], targets[i:i+1])
        for i in range(predictions.size(0))
    ) / predictions.size(0)