"""
Paired CT Dataset Loader
Real Full-Dose / Low-Dose pairs from LDCT dataset
Input shape: (512, 512) Hounsfield Units
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from typing import Tuple, Optional


class PairedCTDataset(Dataset):
    """
    Loads matched Full-Dose / Low-Dose CT slice pairs.
    
    Returns:
        (low_dose_tensor, full_dose_tensor) — both [1, 512, 512]
    
    No synthetic noise needed — real pairs available.
    """

    # Standard CT window for normalization
    HU_MIN = -1000.0
    HU_MAX =  1000.0

    def __init__(
        self,
        full_dose_dir: str,
        low_dose_dir: str,
        max_slices: Optional[int] = None
    ):
        full_files = {f.stem: f for f in Path(full_dose_dir).glob("*.dcm")}
        low_files  = {f.stem: f for f in Path(low_dose_dir).glob("*.dcm")}

        common_keys = sorted(set(full_files.keys()) & set(low_files.keys()))

        assert len(common_keys) > 0, "No matching DICOM pairs found"

        self.full_dose_files = [full_files[k] for k in common_keys]
        self.low_dose_files  = [low_files[k]  for k in common_keys]


        # Validate pairing
        assert len(self.full_dose_files) == len(self.low_dose_files), \
            f"Mismatch: {len(self.full_dose_files)} full vs {len(self.low_dose_files)} low"

        if max_slices:
            self.full_dose_files = self.full_dose_files[:max_slices]
            self.low_dose_files  = self.low_dose_files[:max_slices]

        print(f"Dataset ready: {len(self.full_dose_files)} paired CT slices")

    def _load_dicom(self, filepath: Path) -> np.ndarray:
        """Load DICOM and convert to Hounsfield Units"""
        ds = pydicom.dcmread(filepath)
        image = ds.pixel_array.astype(np.float32)

        # Apply HU conversion
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            image = image * ds.RescaleSlope + ds.RescaleIntercept

        return image

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Clip to HU window and normalize to [0, 1]"""
        image = np.clip(image, self.HU_MIN, self.HU_MAX)
        image = (image - self.HU_MIN) / (self.HU_MAX - self.HU_MIN)
        return image

    def __len__(self) -> int:
        return len(self.full_dose_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            low_tensor:  [1, 512, 512] — noisy input
            full_tensor: [1, 512, 512] — clean target
        """
        full = self._normalize(self._load_dicom(self.full_dose_files[idx]))
        low  = self._normalize(self._load_dicom(self.low_dose_files[idx]))

        full_tensor = torch.from_numpy(full).unsqueeze(0).float()
        low_tensor  = torch.from_numpy(low).unsqueeze(0).float()

        return low_tensor, full_tensor