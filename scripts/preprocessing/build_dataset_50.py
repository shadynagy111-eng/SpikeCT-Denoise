"""
build_dataset_50.py
-------------------
Builds the full 50-patient Mayo CT dataset with uniform slice sampling.

Sampling strategy (FROZEN):
  - 200 slices per patient, selected at uniform intervals across full axial range
  - Indices: np.linspace(0, N-1, 200, dtype=int) where N = total slices
  - Ensures anatomically diverse coverage: apex, mid-chest, basal regions
  - Fully deterministic — no random sampling

Preprocessing spec (FROZEN, identical to 10-patient pipeline):
  - HU conversion:   HU = pixel * RescaleSlope + RescaleIntercept
  - HU window:       [-1000, 1000]
  - Normalization:   (HU + 1000) / 2000  →  [0, 1], float32

Patient split (seed=42, FROZEN):
  - 50 patients split 40 train / 5 val / 5 test at patient level
  - Split written to JSON before any training

Output:
  data/processed_h5/mayo_50patients.h5
  data/processed_h5/patient_split_50.json

Usage:
  python scripts/preprocessing/build_dataset_50.py
"""

import os
import sys
import json
import time
import random
import numpy as np
import h5py
import pydicom
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT  = Path("/mnt/c/Projects/SpikeCT-Denoise/data/LDCT-and-Projection-data")
OUTPUT_DIR = Path("/mnt/c/Projects/SpikeCT-Denoise/data/processed_h5")
OUTPUT_H5  = OUTPUT_DIR / "mayo_50patients.h5"
SPLIT_JSON = OUTPUT_DIR / "patient_split_50.json"

SLICES_PER_PATIENT = 200
HU_MIN = -1000.0
HU_MAX =  1000.0

# ── Patient split (seed=42) ───────────────────────────────────────────────────

ALL_PATIENTS = [
    'C002','C004','C012','C016','C021','C027','C030','C050',
    'C052','C067','C077','C081','C095','C099','C107','C111',
    'C120','C121','C124','C128','C130','C135','C158','C160',
    'C162','C166','C170','C179','C190','C193','C202','C203',
    'C218','C219','C224','C227','C232','C234','C241','C246',
    'C249','C252','C257','C258','C261','C267','C268','C280',
    'C295','C296',
]

def make_split(patients, seed=42):
    """
    80/10/10 patient-level split.
    40 train, 5 val, 5 test.
    Deterministic given seed.
    """
    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    train = sorted(shuffled[:40])
    val   = sorted(shuffled[40:45])
    test  = sorted(shuffled[45:50])
    return {"train": train, "val": val, "test": test}

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_dose_folder(patient_dir: Path, keyword: str) -> Path:
    matches = [p for p in patient_dir.rglob("*")
               if p.is_dir() and keyword in p.name]
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


def get_uniform_indices(n_total: int, n_select: int) -> np.ndarray:
    """
    Select n_select indices uniformly spaced across [0, n_total-1].
    Uses np.linspace for exact uniform coverage.
    Always includes first and last slice.
    """
    return np.linspace(0, n_total - 1, n_select, dtype=int)


def load_dicom_slice(filepath: Path) -> np.ndarray:
    """
    Load a single DICOM slice and return normalized float32 array (H, W).
    """
    ds = pydicom.dcmread(str(filepath))
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = ds.pixel_array.astype(np.float32) * slope + intercept
    hu = np.clip(hu, HU_MIN, HU_MAX)
    normalized = (hu - HU_MIN) / (HU_MAX - HU_MIN)
    return normalized.astype(np.float32)


def load_selected_slices(folder: Path, indices: np.ndarray) -> np.ndarray:
    """
    Load only the slices at `indices` from `folder`.
    Files are sorted by filename before indexing.
    Returns array of shape (len(indices), H, W).
    """
    dcm_files = sorted(folder.glob("*.dcm"))
    n_total = len(dcm_files)
    if n_total == 0:
        raise FileNotFoundError(f"No .dcm files found in {folder}")

    slices = []
    for idx in indices:
        slices.append(load_dicom_slice(dcm_files[idx]))

    return np.stack(slices, axis=0)  # (N, H, W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate and save split
    split = make_split(ALL_PATIENTS, seed=42)

    with open(SPLIT_JSON, "w") as f:
        json.dump(split, f, indent=2)

    print("=" * 60)
    print("Building 50-patient Mayo CT dataset")
    print("=" * 60)
    print(f"Slices per patient: {SLICES_PER_PATIENT} (uniform spacing)")
    print(f"Split: {len(split['train'])} train / "
          f"{len(split['val'])} val / {len(split['test'])} test patients")
    print(f"Expected train slices: {len(split['train']) * SLICES_PER_PATIENT:,}")
    print(f"Expected val slices:   {len(split['val'])   * SLICES_PER_PATIENT:,}")
    print(f"Expected test slices:  {len(split['test'])  * SLICES_PER_PATIENT:,}")
    print(f"Output: {OUTPUT_H5}")
    print()

    print(f"Train patients: {split['train']}")
    print(f"Val patients:   {split['val']}")
    print(f"Test patients:  {split['test']}")
    print()

    total_start = time.time()

    with h5py.File(OUTPUT_H5, "w") as hf:

        # Store split metadata inside the HDF5 for self-documentation
        hf.attrs["slices_per_patient"] = SLICES_PER_PATIENT
        hf.attrs["sampling_method"]    = "uniform_linspace"
        hf.attrs["hu_min"]             = HU_MIN
        hf.attrs["hu_max"]             = HU_MAX
        hf.attrs["train_patients"]     = json.dumps(split["train"])
        hf.attrs["val_patients"]       = json.dumps(split["val"])
        hf.attrs["test_patients"]      = json.dumps(split["test"])

        for phase in ["train", "val", "test"]:
            for patient_id in split[phase]:
                patient_dir = DATA_ROOT / patient_id

                if not patient_dir.exists():
                    print(f"  [ERROR] {patient_id}: directory not found")
                    sys.exit(1)

                print(f"  [{phase:5s}] {patient_id} ...", end=" ", flush=True)
                t0 = time.time()

                try:
                    full_dir = find_dose_folder(patient_dir, "Full Dose Images")
                    low_dir  = find_dose_folder(patient_dir, "Low Dose Images")
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"\n  [ERROR] {e}")
                    sys.exit(1)

                # Count available slices
                n_full = len(list(full_dir.glob("*.dcm")))
                n_low  = len(list(low_dir.glob("*.dcm")))

                if n_full != n_low:
                    print(f"\n  [ERROR] {patient_id}: "
                          f"full={n_full} vs low={n_low} slice count mismatch")
                    sys.exit(1)

                if n_full < SLICES_PER_PATIENT:
                    print(f"\n  [ERROR] {patient_id}: only {n_full} slices, "
                          f"need {SLICES_PER_PATIENT}")
                    sys.exit(1)

                # Uniform indices — same for full and low dose
                indices = get_uniform_indices(n_full, SLICES_PER_PATIENT)

                try:
                    full_vol = load_selected_slices(full_dir, indices)
                    low_vol  = load_selected_slices(low_dir,  indices)
                except Exception as e:
                    print(f"\n  [ERROR] {patient_id}: {e}")
                    sys.exit(1)

                # Verify shapes
                assert full_vol.shape == low_vol.shape == (SLICES_PER_PATIENT, 512, 512), \
                    f"{patient_id}: unexpected shape {full_vol.shape}"

                # Verify value ranges
                assert full_vol.min() >= 0.0 and full_vol.max() <= 1.0, \
                    f"{patient_id}: full_dose out of [0,1]"
                assert low_vol.min()  >= 0.0 and low_vol.max()  <= 1.0, \
                    f"{patient_id}: low_dose out of [0,1]"

                # Write to HDF5
                grp = hf.create_group(patient_id)
                grp.create_dataset("full_dose", data=full_vol, dtype=np.float32,
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("low_dose",  data=low_vol,  dtype=np.float32,
                                   compression="gzip", compression_opts=4)
                grp.attrs["total_available_slices"] = n_full
                grp.attrs["selected_indices_first"] = int(indices[0])
                grp.attrs["selected_indices_last"]  = int(indices[-1])
                grp.attrs["phase"] = phase

                elapsed = time.time() - t0
                print(f"done. Available: {n_full}, Selected: {SLICES_PER_PATIENT}, "
                      f"Range: [{full_vol.min():.3f},{full_vol.max():.3f}], "
                      f"Time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    h5_size_gb = OUTPUT_H5.stat().st_size / (1024 ** 3)

    print(f"\nBuild complete.")
    print(f"  HDF5:       {OUTPUT_H5}")
    print(f"  Size:       {h5_size_gb:.2f} GB")
    print(f"  Split JSON: {SPLIT_JSON}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")

    # Verification pass
    print("\nVerification pass:")
    with h5py.File(OUTPUT_H5, "r") as hf:
        all_patients = split["train"] + split["val"] + split["test"]
        for pid in all_patients:
            if pid not in hf:
                print(f"  [FAIL] {pid} missing")
                sys.exit(1)
            fs = hf[pid]["full_dose"].shape
            ls = hf[pid]["low_dose"].shape
            assert fs == ls == (SLICES_PER_PATIENT, 512, 512), \
                f"{pid}: shape error {fs}"
            print(f"  {pid}: {fs} ✓")

    print("\nDataset ready for training.")


if __name__ == "__main__":
    main()