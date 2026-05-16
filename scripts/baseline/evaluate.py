"""
Evaluation Script — Visualize denoising results
Run from project root: python scripts/evaluate.py
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('src')
from data.dicom_dataset import PairedCTDataset
from models.cdae import CDAE
from utils.metrics import calculate_psnr, calculate_ssim

os.makedirs('results', exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────
CHECKPOINT  = 'results/checkpoints/cnn_best.pth'
FULL_DOSE   = 'data/dataset/C002/Full_Dose_Images'
LOW_DOSE    = 'data/dataset/C002/Low_Dose_Images'
NUM_SAMPLES = 5

# ── Load Checkpoint ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

VAL_SPLIT = checkpoint['config']['val_split']

model = CDAE(in_channels=1).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
print(f"Best PSNR: {checkpoint['metrics']['psnr']:.2f} dB")
print(f"Best SSIM: {checkpoint['metrics']['ssim']:.4f}")

# ── Load Dataset ──────────────────────────────────────────────────────
full_dataset = PairedCTDataset(
    full_dose_dir=FULL_DOSE,
    low_dose_dir=LOW_DOSE
)

# ── Reproduce Exact Same Split As Training ───────────────────────────
total = len(full_dataset)
split_idx = int((1 - VAL_SPLIT) * total)
val_indices = list(range(split_idx, total))

print(f"\nTotal slices: {total}")
print(f"Val slices:   {len(val_indices)} (indices {split_idx} → {total-1})")
print(f"Evaluating last {NUM_SAMPLES} validation slices")

sample_indices = val_indices[-NUM_SAMPLES:]

# ── Evaluation ────────────────────────────────────────────────────────
fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(12, 4 * NUM_SAMPLES))

print(f"\n{'Slice':<8} {'PSNR Input':>12} {'PSNR Output':>12} {'SSIM Input':>12} {'SSIM Output':>12}")
print("-" * 65)

all_psnr_input  = []
all_psnr_output = []
all_ssim_input  = []
all_ssim_output = []

with torch.no_grad():
    for plot_i, dataset_idx in enumerate(sample_indices):

        low, full = full_dataset[dataset_idx]
        low  = low.unsqueeze(0).to(device)
        full = full.unsqueeze(0).to(device)

        denoised = model(low)

        psnr_in  = calculate_psnr(low, full)
        psnr_out = calculate_psnr(denoised, full)
        ssim_in  = calculate_ssim(low, full)
        ssim_out = calculate_ssim(denoised, full)

        all_psnr_input.append(psnr_in)
        all_psnr_output.append(psnr_out)
        all_ssim_input.append(ssim_in)
        all_ssim_output.append(ssim_out)

        print(f"{dataset_idx:<8} {psnr_in:>12.2f} {psnr_out:>12.2f} "
              f"{ssim_in:>12.4f} {ssim_out:>12.4f}")

        # Convert for visualization
        low_np      = low[0, 0].cpu().numpy()
        denoised_np = denoised[0, 0].cpu().numpy()
        full_np     = full[0, 0].cpu().numpy()

        axes[plot_i, 0].imshow(low_np, cmap='gray', vmin=0, vmax=1)
        axes[plot_i, 0].set_title(f'Low Dose\nPSNR: {psnr_in:.2f} dB')
        axes[plot_i, 0].axis('off')

        axes[plot_i, 1].imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        axes[plot_i, 1].set_title(f'Denoised\nPSNR: {psnr_out:.2f} dB')
        axes[plot_i, 1].axis('off')

        axes[plot_i, 2].imshow(full_np, cmap='gray', vmin=0, vmax=1)
        axes[plot_i, 2].set_title(f'Full Dose\nSSIM: {ssim_out:.4f}')
        axes[plot_i, 2].axis('off')

# ── Summary ───────────────────────────────────────────────────────────
print("-" * 65)

avg_psnr_in  = np.mean(all_psnr_input)
avg_psnr_out = np.mean(all_psnr_output)
avg_ssim_in  = np.mean(all_ssim_input)
avg_ssim_out = np.mean(all_ssim_output)

print(f"{'Avg':<8} {avg_psnr_in:>12.2f} {avg_psnr_out:>12.2f} "
      f"{avg_ssim_in:>12.4f} {avg_ssim_out:>12.4f}")

print(f"\nAvg PSNR improvement: +{avg_psnr_out - avg_psnr_in:.2f} dB")
print(f"Avg SSIM improvement: +{avg_ssim_out - avg_ssim_in:.4f}")

# ── Save Figure ───────────────────────────────────────────────────────
plt.suptitle(
    f'CNN Baseline — Validation\n'
    f'Avg PSNR: {avg_psnr_in:.2f} → {avg_psnr_out:.2f} dB  |  '
    f'Avg SSIM: {avg_ssim_in:.4f} → {avg_ssim_out:.4f}',
    fontsize=11
)

plt.tight_layout()
plt.savefig('results/evaluation_results.png', dpi=150)
print("\nSaved results/evaluation_results.png")
