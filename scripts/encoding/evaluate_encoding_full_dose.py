"""
Dataset-Level Evaluation of Latency Encoding Fidelity (Full-Dose)
------------------------------------------------------------------
Evaluates pure quantization error on clean validation images.

Purpose:
- Characterize theoretical encoding limits
- Measure variance across slices
- Identify worst-case behavior

Metrics:
- Mean PSNR ± std
- Mean SSIM ± std  
- Worst-case slice
- Error vs intensity distribution
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('src')
from data.data_utils import HDF5Manager
from encoding.spike_encoders import LatencyEncoder


# Configuration
HDF5_FILE = Path("data/processed_h5/C002_processed.h5")
PATIENT_ID = "C002"
T = 32
VAL_SPLIT_START = 224  # Deterministic split
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_psnr(x, y):
    """Compute PSNR between two images"""
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0, device=x.device) / mse)


def compute_ssim(x, y):
    """Compute SSIM between two images"""
    from skimage.metrics import structural_similarity as ssim
    return ssim(x.cpu().numpy(), y.cpu().numpy(), data_range=1.0)


def main():
    print("=" * 70)
    print("FULL-DOSE ENCODING EVALUATION (Pure Quantization)")
    print("=" * 70)
    print(f"Timesteps: {T}")
    print(f"Device:    {DEVICE}")
    print(f"Seed:      {SEED}")
    print(f"Val start: {VAL_SPLIT_START}")
    print("=" * 70)
    
    # Load data
    print("\nLoading dataset...")
    low_dose, full_dose, metadata = HDF5Manager.load_patient_data(HDF5_FILE, PATIENT_ID)
    
    # Use full-dose validation slices
    val_slices = full_dose[VAL_SPLIT_START:]
    print(f"Validation slices: {len(val_slices)}")
    
    # Initialize encoder
    encoder = LatencyEncoder(T=T)
    
    # Storage
    psnr_values = []
    ssim_values = []
    mse_values = []
    worst_psnr = float("inf")
    worst_idx = -1
    
    # For intensity-error analysis
    all_intensity = []
    all_error = []
    
    # Evaluate each slice
    print("\nEvaluating encoding fidelity...")
    for i in tqdm(range(len(val_slices))):
        img = torch.from_numpy(val_slices[i]).float().to(DEVICE)
        
        # Encode → Decode
        spikes = encoder.encode(img)
        recon = encoder.decode(spikes)
        
        # Metrics
        psnr = compute_psnr(img, recon).item()
        ssim_val = compute_ssim(img, recon)
        mse = torch.mean((img - recon) ** 2).item()
        
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
        mse_values.append(mse)
        
        # Track worst case
        if psnr < worst_psnr:
            worst_psnr = psnr
            worst_idx = i + VAL_SPLIT_START
        
        # Collect for intensity analysis
        error = (img - recon).abs()
        all_intensity.append(img.flatten().cpu())
        all_error.append(error.flatten().cpu())
    
    # Convert to arrays
    psnr_values = np.array(psnr_values)
    ssim_values = np.array(ssim_values)
    mse_values = np.array(mse_values)
    
    all_intensity = torch.cat(all_intensity)
    all_error = torch.cat(all_error)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS: FULL-DOSE VALIDATION SET")
    print("=" * 70)
    
    print(f"\n📊 PSNR Statistics:")
    print(f"   Mean:       {psnr_values.mean():.2f} dB")
    print(f"   Std:        {psnr_values.std():.2f} dB")
    print(f"   Min:        {psnr_values.min():.2f} dB (worst case)")
    print(f"   Max:        {psnr_values.max():.2f} dB")
    print(f"   Worst slice: {worst_idx}")
    
    print(f"\n📊 SSIM Statistics:")
    print(f"   Mean:       {ssim_values.mean():.4f}")
    print(f"   Std:        {ssim_values.std():.4f}")
    print(f"   Min:        {ssim_values.min():.4f}")
    print(f"   Max:        {ssim_values.max():.4f}")
    
    print(f"\n📊 MSE Statistics:")
    print(f"   Mean:       {mse_values.mean():.6f}")
    print(f"   Std:        {mse_values.std():.6f}")
    
    # Theoretical comparison (CORRECTED)
    delta = 1 / (T - 1)
    theoretical_mse = (delta ** 2) / 12  # Standard quantization theory (uniform within-bin)
    
    print(f"\n📐 Theoretical Analysis:")
    print(f"   Quantization step (Δ):              {delta:.4f}")
    print(f"   Max absolute error:                 {delta/2:.4f}")
    print(f"   Reference MSE (Δ²/12, uniform):     {theoretical_mse:.6f}")
    print(f"   Empirical MSE:                      {mse_values.mean():.6f}")
    print(f"   Ratio (empirical/reference):        {mse_values.mean()/theoretical_mse:.2f}×")
    
    # CNN baseline comparison
    cnn_mse = 10**(-28.35/10)
    print(f"\n🔬 Comparison to CNN Baseline:")
    print(f"   CNN denoising MSE:                  {cnn_mse:.6f}")
    print(f"   Encoding MSE:                       {mse_values.mean():.6f}")
    print(f"   Ratio (encoding/denoising):         {mse_values.mean()/cnn_mse:.4f}")
    print(f"   Encoding is {cnn_mse/mse_values.mean():.1f}× smaller")
    
    # Error vs Intensity Analysis
    print("\n📈 Analyzing error vs intensity distribution...")
    bins = torch.linspace(0, 1, 21)
    bin_indices = torch.bucketize(all_intensity, bins)
    
    bin_means = []
    bin_centers = []
    bin_counts = []
    
    for b in range(1, len(bins)):
        mask = bin_indices == b
        count = mask.sum().item()
        bin_counts.append(count)
        
        if count > 0:
            bin_means.append(all_error[mask].mean().item())
        else:
            bin_means.append(0.0)
        
        bin_centers.append((bins[b] + bins[b-1]).item() / 2)
    
    # Plot 1: Error vs Intensity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(bin_centers, bin_means, linewidth=2, marker='o')
    axes[0].axhline(delta/2, color='r', linestyle='--', 
                    label=f'Max error (Δ/2 = {delta/2:.4f})')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Mean Absolute Encoding Error')
    axes[0].set_title(f'Latency Encoding Error vs Intensity (T={T})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Intensity Histogram
    axes[1].bar(bin_centers, bin_counts, width=0.04, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Pixel Count')
    axes[1].set_title('Full-Dose Intensity Distribution')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('encoding_full_dose_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: encoding_full_dose_analysis.png")
    
    # PSNR distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(psnr_values, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(psnr_values.mean(), color='r', linestyle='--', 
               label=f'Mean: {psnr_values.mean():.2f} dB')
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Number of Slices')
    ax.set_title('Full-Dose Encoding PSNR Distribution')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('encoding_full_dose_psnr_dist.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: encoding_full_dose_psnr_dist.png")
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()