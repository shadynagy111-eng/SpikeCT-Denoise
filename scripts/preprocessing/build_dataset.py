"""
build_dataset.py
----------------
Converts Mayo Low-Dose CT DICOM files to a single HDF5 file.

Preprocessing spec (FROZEN):
  - HU conversion:   HU = pixel * RescaleSlope + RescaleIntercept
  - HU window:       [-1000, 1000]
  - Normalization:   (HU + 1000) / 2000  →  [0, 1], float32
  - Storage:         {patient_id}/full_dose  shape (N, 512, 512)
                     {patient_id}/low_dose   shape (N, 512, 512)

Patient split (seed=42, FROZEN):
  Train: C004, C027, C030, C050, C052, C107, C111, C121
  Val:   C224
  Test:  C249

Usage:
  python build_dataset.py

Output:
  data/processed_h5/mayo_10patients.h5
  data/processed_h5/patient_split.json
"""

import os
import sys
import json
import time
import numpy as np
import h5py
import pydicom
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT = Path("/mnt/c/Projects/SpikeCT-Denoise/data/LDCT-and-Projection-data")
OUTPUT_DIR = Path("/mnt/c/Projects/SpikeCT-Denoise/data/processed_h5")
OUTPUT_H5  = OUTPUT_DIR / "mayo_10patients.h5"
SPLIT_JSON = OUTPUT_DIR / "patient_split.json"

PATIENT_SPLIT = {
    "train": ["C004", "C027", "C030", "C050", "C052", "C107", "C111", "C121"],
    "val":   ["C224"],
    "test":  ["C249"],
}

HU_MIN = -1000.0
HU_MAX =  1000.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_dose_folder(patient_dir: Path, keyword: str) -> Path:
    """
    Recursively find the folder whose name contains `keyword`.
    Raises clearly if not found or ambiguous.
    """
    matches = [p for p in patient_dir.rglob("*") if p.is_dir() and keyword in p.name]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No folder containing '{keyword}' found under {patient_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple folders containing '{keyword}' found under {patient_dir}:\n"
            + "\n".join(str(m) for m in matches)
        )
    return matches[0]


def load_dicom_series(folder: Path) -> np.ndarray:
    """
    Load all .dcm files in `folder`, sorted by filename.
    Returns float32 array of shape (N, H, W) in normalized [0,1] range.
    Applies: HU conversion → clip [-1000,1000] → normalize to [0,1].
    """
    dcm_files = sorted(folder.glob("*.dcm"))
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No .dcm files found in {folder}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))

        # HU conversion
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept

        # Window and normalize
        hu = np.clip(hu, HU_MIN, HU_MAX)
        normalized = (hu - HU_MIN) / (HU_MAX - HU_MIN)  # (HU + 1000) / 2000

        slices.append(normalized.astype(np.float32))

    volume = np.stack(slices, axis=0)  # (N, H, W)
    return volume


def verify_alignment(full: np.ndarray, low: np.ndarray, patient_id: str):
    """Assert shapes match exactly."""
    if full.shape != low.shape:
        raise ValueError(
            f"Patient {patient_id}: shape mismatch. "
            f"Full={full.shape}, Low={low.shape}"
        )
    if full.ndim != 3:
        raise ValueError(
            f"Patient {patient_id}: expected 3D array (N,H,W), got {full.ndim}D"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_patients = (
        PATIENT_SPLIT["train"] +
        PATIENT_SPLIT["val"] +
        PATIENT_SPLIT["test"]
    )

    print(f"Building dataset: {len(all_patients)} patients")
    print(f"Output: {OUTPUT_H5}\n")

    total_start = time.time()

    with h5py.File(OUTPUT_H5, "w") as hf:
        for patient_id in all_patients:
            patient_dir = DATA_ROOT / patient_id

            if not patient_dir.exists():
                print(f"  [ERROR] {patient_id}: directory not found at {patient_dir}")
                sys.exit(1)

            print(f"  Processing {patient_id} ...", end=" ", flush=True)
            t0 = time.time()

            try:
                full_dir = find_dose_folder(patient_dir, "Full Dose Images")
                low_dir  = find_dose_folder(patient_dir, "Low Dose Images")
            except (FileNotFoundError, RuntimeError) as e:
                print(f"\n  [ERROR] {patient_id}: {e}")
                sys.exit(1)

            try:
                full_vol = load_dicom_series(full_dir)
                low_vol  = load_dicom_series(low_dir)
            except Exception as e:
                print(f"\n  [ERROR] {patient_id}: failed to load DICOM — {e}")
                sys.exit(1)

            try:
                verify_alignment(full_vol, low_vol, patient_id)
            except ValueError as e:
                print(f"\n  [ERROR] {e}")
                sys.exit(1)

            # Write to HDF5
            grp = hf.create_group(patient_id)
            grp.create_dataset("full_dose", data=full_vol, dtype=np.float32,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("low_dose",  data=low_vol,  dtype=np.float32,
                               compression="gzip", compression_opts=4)

            elapsed = time.time() - t0
            print(f"done. Slices: {full_vol.shape[0]}, "
                  f"Shape: {full_vol.shape[1]}x{full_vol.shape[2]}, "
                  f"Range: [{full_vol.min():.4f}, {full_vol.max():.4f}], "
                  f"Time: {elapsed:.1f}s")

    # Write split config
    with open(SPLIT_JSON, "w") as f:
        json.dump(PATIENT_SPLIT, f, indent=2)

    total_elapsed = time.time() - total_start
    h5_size_mb = OUTPUT_H5.stat().st_size / (1024 ** 2)

    print(f"\nDone.")
    print(f"  HDF5 file:    {OUTPUT_H5}")
    print(f"  File size:    {h5_size_mb:.1f} MB")
    print(f"  Split config: {SPLIT_JSON}")
    print(f"  Total time:   {total_elapsed:.1f}s")

    # Final verification pass
    print("\nVerification pass:")
    with h5py.File(OUTPUT_H5, "r") as hf:
        for patient_id in all_patients:
            if patient_id not in hf:
                print(f"  [FAIL] {patient_id} missing from HDF5")
                sys.exit(1)
            full_shape = hf[patient_id]["full_dose"].shape
            low_shape  = hf[patient_id]["low_dose"].shape
            assert full_shape == low_shape, f"{patient_id}: shape mismatch in HDF5"
            print(f"  {patient_id}: full={full_shape}, low={low_shape} ✓")

    print("\nDataset build complete. Ready for training.")


if __name__ == "__main__":
    main()