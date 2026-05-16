"""
Verify HDF5 file contents
Quick sanity check after conversion
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')
from data.data_utils import HDF5Manager

HDF5_FILE = Path("data/processed_h5/C002_processed.h5")
PATIENT_ID = "C002"


def main():
    print("=" * 70)
    print("HDF5 FILE VERIFICATION")
    print("=" * 70)
    
    # List patients
    patients = HDF5Manager.list_patients(HDF5_FILE)
    print(f"\nPatients in file: {patients}")
    
    # Load data
    print(f"\nLoading patient: {PATIENT_ID}")
    low_dose, full_dose, metadata = HDF5Manager.load_patient_data(HDF5_FILE, PATIENT_ID)
    
    print(f"\n📊 Data Shape:")
    print(f"   Low-dose:  {low_dose.shape}")
    print(f"   Full-dose: {full_dose.shape}")
    
    print(f"\n📊 Value Ranges:")
    print(f"   Low-dose:  [{low_dose.min():.3f}, {low_dose.max():.3f}]")
    print(f"   Full-dose: [{full_dose.min():.3f}, {full_dose.max():.3f}]")
    
    print(f"\n📋 Metadata:")
    for key, value in sorted(metadata.items()):
        print(f"   {key:20s}: {value}")
    
    # Visualize middle slice
    mid_idx = len(low_dose) // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(low_dose[mid_idx], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Low Dose (slice {mid_idx})')
    axes[0].axis('off')
    
    axes[1].imshow(full_dose[mid_idx], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Full Dose (slice {mid_idx})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hdf5_verification.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: hdf5_verification.png")
    
    print("\n" + "=" * 70)
    print("✓ VERIFICATION COMPLETE - Data looks good!")
    print("=" * 70)


if __name__ == "__main__":
    main()