"""
Compare results from different HU windows
Visual and quantitative comparison
"""

import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('src')
from data.dicom_dataset import PairedCTDataset
from models.cdae import CDAE
from utils.metrics import calculate_psnr, calculate_ssim


WINDOWS = [
    (-1000, 1000),
    (-1000, 600),
    (-1000, 400),
]

FULL_DOSE_DIR = 'data/dataset/C002/Full_Dose_Images'
LOW_DOSE_DIR = 'data/dataset/C002/Low_Dose_Images'


def load_model_and_config(window):
    """Load trained model and config for a window"""
    hu_min, hu_max = window
    result_dir = Path(f"results/window_{int(hu_min)}_{int(hu_max)}")
    
    model_path = result_dir / "best_model.pth"
    config_path = result_dir / "config.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = CDAE(in_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint, config, device


def evaluate_model(model, dataset, device, num_samples=20):
    """Evaluate model on dataset"""
    psnr_scores = []
    ssim_scores = []
    
    # Sample evenly across dataset
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    with torch.no_grad():
        for idx in indices:
            low, full = dataset[idx]
            low = low.unsqueeze(0).to(device)
            full = full.unsqueeze(0).to(device)
            
            denoised = model(low)
            
            psnr = calculate_psnr(denoised, full)
            ssim = calculate_ssim(denoised, full)
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
    
    return {
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
    }


def visualize_comparison(windows, datasets, models, device):
    """Create visual comparison of denoising results"""
    
    # Use middle slice
    slice_idx = len(datasets[windows[0]]) // 2
    
    fig, axes = plt.subplots(len(windows), 3, figsize=(12, 4*len(windows)))
    
    for i, window in enumerate(windows):
        hu_min, hu_max = window
        dataset = datasets[window]
        model = models[window]
        
        # Get data
        low, full = dataset[slice_idx]
        low_batch = low.unsqueeze(0).to(device)
        full_batch = full.unsqueeze(0).to(device)
        
        with torch.no_grad():
            denoised = model(low_batch)
        
        # Convert to numpy
        low_np = low[0].cpu().numpy()
        denoised_np = denoised[0, 0].cpu().numpy()
        full_np = full[0].cpu().numpy()
        
        # Calculate metrics
        psnr = calculate_psnr(denoised, full_batch)
        ssim = calculate_ssim(denoised, full_batch)
        
        # Plot
        axes[i, 0].imshow(low_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Low Dose\nWindow [{int(hu_min)}, {int(hu_max)}]')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Denoised\nPSNR: {psnr:.2f} dB')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(full_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Full Dose\nSSIM: {ssim:.4f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('window_comparison_visual.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visual comparison: window_comparison_visual.png")


def main():
    print("=" * 70)
    print("HU WINDOW COMPARISON")
    print("=" * 70)
    
    results = {}
    datasets = {}
    models = {}
    configs = {}
    device = None
    
    # Load all models
    for window in WINDOWS:
        hu_min, hu_max = window
        print(f"\nLoading window [{hu_min}, {hu_max}]...")
        
        model, checkpoint, config, dev = load_model_and_config(window)
        device = dev
        
        # Create dataset with same window
        dataset = PairedCTDataset(
            full_dose_dir=FULL_DOSE_DIR,
            low_dose_dir=LOW_DOSE_DIR,
            hu_min=hu_min,
            hu_max=hu_max
        )
        
        datasets[window] = dataset
        models[window] = model
        configs[window] = config
        
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
        print(f"  Val PSNR: {checkpoint['metrics']['psnr']:.2f} dB")
        print(f"  Val SSIM: {checkpoint['metrics']['ssim']:.4f}")
    
    # Evaluate all models
    print("\n" + "=" * 70)
    print("EVALUATING MODELS ON TEST SET")
    print("=" * 70)
    
    for window in WINDOWS:
        hu_min, hu_max = window
        print(f"\nEvaluating window [{hu_min}, {hu_max}]...")
        
        metrics = evaluate_model(models[window], datasets[window], device, num_samples=20)
        results[window] = metrics
        
        print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Window':<20} {'PSNR (dB)':<18} {'SSIM':<18}")
    print("-" * 56)
    for window in WINDOWS:
        hu_min, hu_max = window
        r = results[window]
        print(f"[{int(hu_min):5d}, {int(hu_max):5d}]      "
              f"{r['psnr_mean']:6.2f} ± {r['psnr_std']:4.2f}      "
              f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
    
    # Find best
    best_psnr_window = max(results.keys(), key=lambda w: results[w]['psnr_mean'])
    best_ssim_window = max(results.keys(), key=lambda w: results[w]['ssim_mean'])
    
    print(f"\n🏆 Best PSNR: [{int(best_psnr_window[0])}, {int(best_psnr_window[1])}] "
          f"→ {results[best_psnr_window]['psnr_mean']:.2f} dB")
    print(f"🏆 Best SSIM: [{int(best_ssim_window[0])}, {int(best_ssim_window[1])}] "
          f"→ {results[best_ssim_window]['ssim_mean']:.4f}")
    
    # Visual comparison
    print("\n" + "=" * 70)
    print("GENERATING VISUAL COMPARISON")
    print("=" * 70)
    visualize_comparison(WINDOWS, datasets, models, device)
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    window_labels = [f"[{int(w[0])},{int(w[1])}]" for w in WINDOWS]
    psnr_means = [results[w]['psnr_mean'] for w in WINDOWS]
    ssim_means = [results[w]['ssim_mean'] for w in WINDOWS]
    
    ax1.bar(window_labels, psnr_means, color=['orange', 'steelblue', 'green'])
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xlabel('HU Window')
    ax1.set_title('PSNR Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(window_labels, ssim_means, color=['orange', 'steelblue', 'green'])
    ax2.set_ylabel('SSIM')
    ax2.set_xlabel('HU Window')
    ax2.set_title('SSIM Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('window_comparison_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved metrics comparison: window_comparison_metrics.png")
    
    print("\n" + "=" * 70)
    print("✓ COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()