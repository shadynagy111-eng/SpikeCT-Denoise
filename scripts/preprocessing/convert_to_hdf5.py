"""
Convert DICOM dataset to HDF5 format
One-time preprocessing of all CT slices
"""

import sys
from pathlib import Path

sys.path.append('src')
from data.data_utils import (
    CTPreprocessor, 
    HDF5Manager, 
    process_dicom_folder,
    verify_instance_alignment
)


# Configuration
FULL_DOSE_DIR = Path("data/dataset/C002/Full_Dose_Images")
LOW_DOSE_DIR = Path("data/dataset/C002/Low_Dose_Images")
OUTPUT_DIR = Path("data/processed_h5")
OUTPUT_FILE = OUTPUT_DIR / "C002_processed.h5"

PATIENT_ID = "C002"
HU_MIN = -1000.0
HU_MAX = 1000.0


def main():
    """Convert DICOM files to HDF5"""
    
    print("=" * 70)
    print("DICOM → HDF5 Conversion")
    print("=" * 70)
    
    # Verify input directories exist
    if not FULL_DOSE_DIR.exists():
        raise FileNotFoundError(f"Full-dose directory not found: {FULL_DOSE_DIR}")
    if not LOW_DOSE_DIR.exists():
        raise FileNotFoundError(f"Low-dose directory not found: {LOW_DOSE_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = CTPreprocessor(hu_min=HU_MIN, hu_max=HU_MAX)
    
    # Process full-dose images with metadata extraction
    print("\n📁 FULL-DOSE SERIES")
    full_dose_images, full_instances, full_metadata = process_dicom_folder(
        FULL_DOSE_DIR, 
        preprocessor,
        extract_metadata=True
    )
    
    # Process low-dose images
    print("\n📁 LOW-DOSE SERIES")
    low_dose_images, low_instances, low_metadata = process_dicom_folder(
        LOW_DOSE_DIR, 
        preprocessor,
        extract_metadata=True
    )
    
    # CRITICAL: Verify InstanceNumber alignment
    print("\n🔍 VERIFICATION")
    verify_instance_alignment(full_instances, low_instances)
    
    # Combine metadata from both series
    combined_metadata = {
        'patient_id': PATIENT_ID,
        'num_slices': len(full_dose_images),
        'image_height': full_dose_images.shape[1],
        'image_width': full_dose_images.shape[2],
        'hu_min': HU_MIN,
        'hu_max': HU_MAX,
        'normalization': '[0, 1]',
        # From DICOM metadata
        'slice_thickness_mm': full_metadata.get('slice_thickness', 0.0),
        'pixel_spacing': str(full_metadata.get('pixel_spacing', 'Unknown')),
        'study_description': full_metadata.get('study_description', 'Unknown'),
        'full_dose_series': full_metadata.get('series_description', 'Unknown'),
        'low_dose_series': low_metadata.get('series_description', 'Unknown'),
        'kvp': full_metadata.get('kvp', 0.0),
        'exposure': full_metadata.get('exposure', 0.0),
    }
    
    # Save to HDF5
    print(f"\n💾 SAVING TO HDF5")
    print(f"   Output: {OUTPUT_FILE}")
    HDF5Manager.save_patient_data(
        output_path=OUTPUT_FILE,
        patient_id=PATIENT_ID,
        low_dose_images=low_dose_images,
        full_dose_images=full_dose_images,
        metadata=combined_metadata,
        overwrite=True  # Safe for first run
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Patient ID:       {PATIENT_ID}")
    print(f"   Number of slices: {len(full_dose_images)}")
    print(f"   Image shape:      {full_dose_images.shape[1:]} (H×W)")
    print(f"   Slice thickness:  {combined_metadata['slice_thickness_mm']} mm")
    print(f"   Pixel spacing:    {combined_metadata['pixel_spacing']} mm")
    print(f"   HU window:        [{HU_MIN}, {HU_MAX}]")
    print(f"   Normalization:    {combined_metadata['normalization']}")
    print(f"   Output file:      {OUTPUT_FILE}")
    print(f"   File size:        {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()