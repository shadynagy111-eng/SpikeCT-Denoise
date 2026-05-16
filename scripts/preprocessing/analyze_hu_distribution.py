"""
Analyze HU value distribution across the dataset
Helps determine optimal HU window for preprocessing
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append('src')
from data.data_utils import CTPreprocessor

# Paths
FULL_DOSE_DIR = Path("data/dataset/C002/Full_Dose_Images")
LOW_DOSE_DIR = Path("data/dataset/C002/Low_Dose_Images")


def analyze_hu_distribution(dicom_folder: Path, label: str, sample_size: int = 50):
    """
    Analyze HU value distribution across slices
    
    Args:
        dicom_folder: Path to DICOM folder
        label: Label for plots (e.g., "Full Dose")
        sample_size: Number of slices to sample (None = all)
    """
    dicom_files = sorted(dicom_folder.glob("*.dcm"))
    
    if sample_size:
        # Sample evenly across the volume
        indices = np.linspace(0, len(dicom_files)-1, sample_size, dtype=int)
        dicom_files = [dicom_files[i] for i in indices]
    
    print(f"\n{'='*70}")
    print(f"Analyzing {label}: {len(dicom_files)} slices")
    print(f"{'='*70}")
    
    preprocessor = CTPreprocessor(hu_min=-10000, hu_max=10000)  # No clipping
    
    all_values = []
    slice_stats = []
    
    for filepath in tqdm(dicom_files, desc=f"Processing {label}"):
        # Load without clipping
        image, _ = preprocessor.load_dicom(filepath)
        
        all_values.extend(image.flatten())
        
        slice_stats.append({
            'min': image.min(),
            'max': image.max(),
            'mean': image.mean(),
            'median': np.median(image),
            'p05': np.percentile(image, 5),
            'p95': np.percentile(image, 95),
        })
    
    all_values = np.array(all_values)
    
    # Global statistics
    print(f"\n📊 Global Statistics:")
    print(f"   Min HU:        {all_values.min():.1f}")
    print(f"   Max HU:        {all_values.max():.1f}")
    print(f"   Mean HU:       {all_values.mean():.1f}")
    print(f"   Median HU:     {np.median(all_values):.1f}")
    print(f"   Std HU:        {all_values.std():.1f}")
    
    # Percentiles
    print(f"\n📊 Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(all_values, p)
        print(f"   {p:2d}th percentile: {val:7.1f} HU")
    
    # Count values in different ranges
    print(f"\n📊 Value Distribution:")
    ranges = [
        ("Air region", -10000, -700),
        ("Lung tissue", -700, -300),
        ("Fat/soft tissue", -300, 100),
        ("Soft tissue/muscle", 100, 300),
        ("Bone", 300, 1000),
        ("Dense bone", 1000, 10000),
    ]
    
    for name, low, high in ranges:
        count = np.sum((all_values >= low) & (all_values < high))
        percent = 100 * count / len(all_values)
        print(f"   {name:20s} [{low:6.0f}, {high:6.0f}): {percent:5.2f}%")
    
    return all_values, slice_stats


def plot_distribution(full_values, low_values):
    """Plot HU distribution comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full histogram
    axes[0, 0].hist(full_values, bins=200, range=(-1500, 3000), 
                    color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Hounsfield Units')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Full-Dose HU Distribution')
    axes[0, 0].axvline(-1000, color='red', linestyle='--', label='Current window min')
    axes[0, 0].axvline(1000, color='red', linestyle='--', label='Current window max')
    axes[0, 0].axvline(400, color='orange', linestyle='--', label='Soft tissue max')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Low-dose histogram
    axes[0, 1].hist(low_values, bins=200, range=(-1500, 3000),
                    color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Hounsfield Units')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Low-Dose HU Distribution')
    axes[0, 1].axvline(-1000, color='red', linestyle='--', label='Current window min')
    axes[0, 1].axvline(1000, color='red', linestyle='--', label='Current window max')
    axes[0, 1].axvline(400, color='orange', linestyle='--', label='Soft tissue max')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Zoomed histogram (soft tissue region)
    axes[1, 0].hist(full_values, bins=100, range=(-1000, 1000),
                    color='steelblue', alpha=0.7, edgecolor='black', label='Full-dose')
    axes[1, 0].hist(low_values, bins=100, range=(-1000, 1000),
                    color='green', alpha=0.5, edgecolor='black', label='Low-dose')
    axes[1, 0].set_xlabel('Hounsfield Units')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Soft Tissue Range [-1000, 1000] (Current Window)')
    axes[1, 0].axvline(400, color='orange', linestyle='--', label='Proposed max')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Cumulative distribution
    full_sorted = np.sort(full_values)
    low_sorted = np.sort(low_values)
    
    full_cdf = np.arange(len(full_sorted)) / len(full_sorted) * 100
    low_cdf = np.arange(len(low_sorted)) / len(low_sorted) * 100
    
    axes[1, 1].plot(full_sorted, full_cdf, label='Full-dose', linewidth=2)
    axes[1, 1].plot(low_sorted, low_cdf, label='Low-dose', linewidth=2)
    axes[1, 1].set_xlabel('Hounsfield Units')
    axes[1, 1].set_ylabel('Cumulative Percentage (%)')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].axvline(-1000, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(400, color='orange', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(1000, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlim(-1500, 2000)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hu_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: hu_distribution_analysis.png")


def suggest_window(full_values, low_values):
    """Suggest optimal HU window based on data"""
    
    print(f"\n{'='*70}")
    print("WINDOW RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Find percentiles
    p95 = np.percentile(np.concatenate([full_values, low_values]), 95)
    p99 = np.percentile(np.concatenate([full_values, low_values]), 99)
    
    print(f"\n💡 Based on your data:")
    print(f"   95% of values are below: {p95:.0f} HU")
    print(f"   99% of values are below: {p99:.0f} HU")
    
    # Count values above different thresholds
    combined = np.concatenate([full_values, low_values])
    
    print(f"\n📊 Values outside different windows:")
    for threshold in [400, 600, 800, 1000, 1500]:
        above = np.sum(combined > threshold)
        percent = 100 * above / len(combined)
        print(f"   Above {threshold:4d} HU: {percent:5.2f}% ({above:,} pixels)")
    
    print(f"\n📋 Suggested Windows to Test:")
    print(f"\n   1. [-1000,  400]  — Soft tissue focused (clips {np.sum(combined > 400)/len(combined)*100:.2f}%)")
    print(f"   2. [-1000,  600]  — Balanced soft tissue + some bone")
    print(f"   3. [-1000,  800]  — Include moderate bone")
    print(f"   4. [-1000, 1000]  — Current (your baseline)")
    print(f"   5. [-1000, 1500]  — Wide range")
    
    print(f"\n⚠️  Recommendation:")
    if p95 < 500:
        print(f"   Most data is in soft tissue range.")
        print(f"   Consider [-1000, 400] or [-1000, 600] for better soft tissue resolution.")
    else:
        print(f"   Data includes significant bone content.")
        print(f"   Current window [-1000, 1000] is appropriate.")


def main():
    print("=" * 70)
    print("HU DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Analyze both series
    full_values, _ = analyze_hu_distribution(FULL_DOSE_DIR, "Full-Dose", sample_size=50)
    low_values, _ = analyze_hu_distribution(LOW_DOSE_DIR, "Low-Dose", sample_size=50)
    
    # Plot distributions
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")
    plot_distribution(full_values, low_values)
    
    # Make recommendation
    suggest_window(full_values, low_values)
    
    print(f"\n{'='*70}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()