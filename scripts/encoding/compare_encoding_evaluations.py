"""
Compare Full-Dose vs Low-Dose Encoding Results
----------------------------------------------
Side-by-side comparison of encoding behavior
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')


def main():
    print("=" * 70)
    print("ENCODING COMPARISON: FULL-DOSE vs LOW-DOSE")
    print("=" * 70)
    
    # These values will come from running the other two scripts
    # You'll need to fill them in manually after running both
    
    results = {
        'Full-Dose': {
            'mean_psnr': 35.17,  # Replace with actual
            'std_psnr': 0.5,     # Replace with actual
            'mean_mse': 3.0e-4,  # Replace with actual
        },
        'Low-Dose': {
            'mean_psnr': 0.0,    # Fill after running
            'std_psnr': 0.0,     # Fill after running
            'mean_mse': 0.0,     # Fill after running
        }
    }
    
    print("\n📊 Comparison Table:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Full-Dose':<20} {'Low-Dose':<20} {'Difference'}")
    print("-" * 70)
    
    for key in ['mean_psnr', 'std_psnr', 'mean_mse']:
        full = results['Full-Dose'][key]
        low = results['Low-Dose'][key]
        diff = low - full
        
        if 'psnr' in key:
            print(f"{key:<20} {full:>18.2f}   {low:>18.2f}   {diff:+.2f}")
        else:
            print(f"{key:<20} {full:>18.6f}   {low:>18.6f}   {diff:+.6f}")
    
    cnn_mse = 10**(-28.35/10)
    print(f"\n🔬 Safety Margin:")
    print(f"   CNN MSE:              {cnn_mse:.6f}")
    print(f"   Full-dose encoding:   {results['Full-Dose']['mean_mse']:.6f} ({cnn_mse/results['Full-Dose']['mean_mse']:.1f}× smaller)")
    print(f"   Low-dose encoding:    {results['Low-Dose']['mean_mse']:.6f} ({cnn_mse/results['Low-Dose']['mean_mse']:.1f}× smaller)")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("If both encoding MSE values are << CNN MSE (0.00146),")
    print("then T=32 is validated for the SNN pipeline.")


if __name__ == "__main__":
    main()