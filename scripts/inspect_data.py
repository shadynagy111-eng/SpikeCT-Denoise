import sys
sys.path.append('src')
from data.dicom_dataset import PairedCTDataset

dataset = PairedCTDataset(
    full_dose_dir="data/dataset/C002/Full_Dose_Images",
    low_dose_dir="data/dataset/C002/Low_Dose_Images",
    max_slices=5
)

low, full = dataset[0]
print(f"Low dose shape:  {low.shape}")
print(f"Full dose shape: {full.shape}")
print(f"Low range:  [{low.min():.3f}, {low.max():.3f}]")
print(f"Full range: [{full.min():.3f}, {full.max():.3f}]")
