"""
Extract DICOM metadata from dataset
Generates information for dataset report
"""

import sys
import pydicom
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append('src')

# Paths
FULL_DOSE_DIR = "data/dataset/C002/Full_Dose_Images"
LOW_DOSE_DIR = "data/dataset/C002/Low_Dose_Images"


def extract_metadata(dicom_file):
    """Extract key metadata from a DICOM file"""
    ds = pydicom.dcmread(dicom_file)
    
    metadata = {}
    
    # Basic info
    metadata['PatientID'] = getattr(ds, 'PatientID', 'Unknown')
    metadata['StudyDescription'] = getattr(ds, 'StudyDescription', 'Unknown')
    metadata['SeriesDescription'] = getattr(ds, 'SeriesDescription', 'Unknown')
    
    # Image properties
    metadata['Rows'] = getattr(ds, 'Rows', None)
    metadata['Columns'] = getattr(ds, 'Columns', None)
    metadata['PixelSpacing'] = getattr(ds, 'PixelSpacing', None)
    metadata['SliceThickness'] = getattr(ds, 'SliceThickness', None)
    
    # HU conversion
    metadata['RescaleSlope'] = getattr(ds, 'RescaleSlope', None)
    metadata['RescaleIntercept'] = getattr(ds, 'RescaleIntercept', None)
    
    # Acquisition parameters
    metadata['KVP'] = getattr(ds, 'KVP', None)  # X-ray tube voltage
    metadata['Exposure'] = getattr(ds, 'Exposure', None)  # mAs
    
    return metadata


def analyze_dataset():
    """Analyze full dataset and print report"""
    
    print("=" * 70)
    print("DATASET METADATA REPORT")
    print("=" * 70)
    
    # Get file lists
    full_files = sorted(Path(FULL_DOSE_DIR).glob("*.dcm"))
    low_files = sorted(Path(LOW_DOSE_DIR).glob("*.dcm"))
    
    print(f"\n📊 DATASET SIZE")
    print(f"   Full-dose slices: {len(full_files)}")
    print(f"   Low-dose slices:  {len(low_files)}")
    
    # Extract metadata from first file of each type
    print(f"\n📋 FULL-DOSE METADATA (from {full_files[0].name})")
    full_meta = extract_metadata(full_files[0])
    for key, value in full_meta.items():
        print(f"   {key:20s}: {value}")
    
    print(f"\n📋 LOW-DOSE METADATA (from {low_files[0].name})")
    low_meta = extract_metadata(low_files[0])
    for key, value in low_meta.items():
        print(f"   {key:20s}: {value}")
    
    # Check HU values across dataset
    print(f"\n📈 HOUNSFIELD UNIT ANALYSIS")
    print(f"   Sampling 10 slices from each series...")
    
    full_hu_stats = analyze_hu_range(full_files[:10])
    low_hu_stats = analyze_hu_range(low_files[:10])
    
    print(f"\n   Full-dose HU range:")
    print(f"      Min: {full_hu_stats['min']:.1f}")
    print(f"      Max: {full_hu_stats['max']:.1f}")
    print(f"      Mean: {full_hu_stats['mean']:.1f}")
    
    print(f"\n   Low-dose HU range:")
    print(f"      Min: {low_hu_stats['min']:.1f}")
    print(f"      Max: {low_hu_stats['max']:.1f}")
    print(f"      Mean: {low_hu_stats['mean']:.1f}")
    
    print("\n" + "=" * 70)
    print("Use this information to complete docs/dataset_report.md")
    print("=" * 70)


def analyze_hu_range(file_list):
    """Analyze HU value range across multiple files"""
    all_mins = []
    all_maxs = []
    all_means = []
    
    for filepath in file_list:
        ds = pydicom.dcmread(filepath)
        img = ds.pixel_array.astype(np.float32)
        
        # Convert to HU
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
        
        all_mins.append(img.min())
        all_maxs.append(img.max())
        all_means.append(img.mean())
    
    return {
        'min': min(all_mins),
        'max': max(all_maxs),
        'mean': np.mean(all_means)
    }


if __name__ == "__main__":
    analyze_dataset()