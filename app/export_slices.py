"""
app/export_slices.py
--------------------
Exports a few test slices from the HDF5 file to .npy format
so you can upload them directly in the Gradio demo.

Run once:
    python app/export_slices.py

Creates files in app/demo_slices/:
    low_C004_s050.npy   — low-dose input
    full_C004_s050.npy  — full-dose reference
    low_C050_s100.npy
    full_C050_s100.npy
    ... etc
"""

import h5py
import numpy as np
from pathlib import Path

H5   = Path.home() / "SpikeCT_Data/mayo_50patients_fast.h5"
OUT  = Path(__file__).parent / "demo_slices"
OUT.mkdir(exist_ok=True)

# Patient, slice index pairs to export
CASES = [
    ("C004", 50),
    ("C004", 100),
    ("C050", 50),
    ("C050", 100),
    ("C111", 100),
    ("C121", 100),
    ("C249", 100),
]

print(f"Exporting {len(CASES)} slices to {OUT}/")

with h5py.File(H5, "r") as f:
    for pid, sidx in CASES:
        low  = f[pid]["low_dose"][sidx].astype(np.float32)
        full = f[pid]["full_dose"][sidx].astype(np.float32)

        low_path  = OUT / f"low_{pid}_s{sidx:03d}.npy"
        full_path = OUT / f"full_{pid}_s{sidx:03d}.npy"

        np.save(low_path,  low)
        np.save(full_path, full)

        print(f"  {pid} slice {sidx}: "
              f"low [{low.min():.3f}, {low.max():.3f}] → {low_path.name}")

print(f"\nDone. Upload these files in the Gradio demo.")
print(f"Upload the low_ file as input, full_ file as reference.")
