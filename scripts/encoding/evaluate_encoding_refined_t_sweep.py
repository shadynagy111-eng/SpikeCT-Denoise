"""
Phase 2.5: Fine-Grained Temporal Resolution Refinement

Goal:
Find minimum T that maintains safety margin:
Encoding MSE < 0.25 × CNN MSE

This refines Phase 2 selection (T=32).
"""

import sys
from pathlib import Path

# ------------------------------------------------------------
# Fix Python path so we can import from src/
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List

# Import correct LatencyEncoder (device-aware, vectorized)
from encoding.spike_encoders import LatencyEncoder


# ------------------------------------------------------------
# Metric computation (device-safe)
# ------------------------------------------------------------
def compute_metrics(original: torch.Tensor,
                    reconstructed: torch.Tensor) -> Dict[str, float]:

    mse = torch.mean((original - reconstructed) ** 2)

    if mse.item() == 0:
        psnr = float("inf")
    else:
        psnr = 10 * torch.log10(torch.tensor(1.0, device=original.device) / mse)

    return {
        "psnr": psnr.item(),
        "mse": mse.item()
    }


# ------------------------------------------------------------
# Evaluate one T value
# ------------------------------------------------------------
def evaluate_t_value(T: int,
                     val_low_dose: np.ndarray,
                     device: torch.device,
                     cnn_mse: float) -> Dict:

    print(f"\n{'='*60}")
    print(f"Evaluating T = {T}")
    print(f"{'='*60}")

    encoder = LatencyEncoder(T=T)

    psnrs = []
    mses = []
    sparsities = []

    for idx, slice_data in enumerate(val_low_dose):

        img = torch.from_numpy(slice_data).float().to(device)

        spikes = encoder.encode(img)
        reconstructed = encoder.decode(spikes)

        metrics = compute_metrics(img, reconstructed)

        psnrs.append(metrics["psnr"])
        mses.append(metrics["mse"])

        sparsity = 1.0 - (spikes.sum() / spikes.numel())
        sparsities.append(sparsity.item())

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(val_low_dose)} slices")

    mean_mse = float(np.mean(mses))

    return {
        "T": T,
        "mean_psnr": float(np.mean(psnrs)),
        "std_psnr": float(np.std(psnrs)),
        "mean_mse": mean_mse,
        "std_mse": float(np.std(mses)),
        "mean_sparsity": float(np.mean(sparsities)),
        "safety_ratio": mean_mse / cnn_mse,
        "quantization_step": 1.0 / (T - 1)
    }


# ------------------------------------------------------------
# Main Experiment
# ------------------------------------------------------------
def main():

    print("=" * 80)
    print("PHASE 2.5: FINE-GRAINED TEMPORAL RESOLUTION REFINEMENT")
    print("=" * 80)

    HDF5_PATH = PROJECT_ROOT / "data/processed_h5/C002_processed.h5"
    PATIENT_ID = "C002"

    CNN_MSE = 0.00146
    SAFETY_THRESHOLD = 0.25
    TARGET_MSE = CNN_MSE * SAFETY_THRESHOLD

    T_VALUES = [16, 18, 20, 22, 24, 26, 28, 30, 32]

    print(f"\nTarget Encoding MSE < {TARGET_MSE:.6f}")
    print(f"Testing T values: {T_VALUES}")

    # Load data
    with h5py.File(HDF5_PATH, "r") as f:
        patient_group = f[PATIENT_ID]
        low_dose = patient_group["low_dose"][:]

    val_low_dose = low_dose[224:]
    print(f"Validation slices: {len(val_low_dose)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []

    for T in T_VALUES:
        res = evaluate_t_value(T, val_low_dose, device, CNN_MSE)
        results.append(res)

        status = "PASS" if res["mean_mse"] < TARGET_MSE else "FAIL"

        print(f"\nT={T}")
        print(f"  PSNR: {res['mean_psnr']:.2f} ± {res['std_psnr']:.2f} dB")
        print(f"  MSE:  {res['mean_mse']:.6f}")
        print(f"  Safety ratio: {res['safety_ratio']:.3f}")
        print(f"  Sparsity: {res['mean_sparsity']*100:.2f}%")
        print(f"  Status: {status}")

    # --------------------------------------------------------
    # Find optimal T
    # --------------------------------------------------------
    viable = [r for r in results if r["mean_mse"] < TARGET_MSE]

    if viable:
        optimal = min(viable, key=lambda x: x["T"])
        print("\nOptimal T:", optimal["T"])
    else:
        optimal = None
        print("\nNo T meets safety criterion in tested range.")

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------
    output_dir = PROJECT_ROOT / "results/phase2_encoding/refined_t_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    T_vals = [r["T"] for r in results]
    mse_vals = [r["mean_mse"] for r in results]
    psnr_vals = [r["mean_psnr"] for r in results]

    plt.figure()
    plt.plot(T_vals, psnr_vals, marker="o")
    plt.axhline(y=28.35, linestyle="--")
    plt.xlabel("T")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Temporal Resolution")
    plt.tight_layout()
    plt.savefig(output_dir / "psnr_vs_T_refined.png", dpi=300)

    plt.figure()
    plt.plot(T_vals, mse_vals, marker="o")
    plt.axhline(y=TARGET_MSE, linestyle="--")
    plt.xlabel("T")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.title("MSE vs Temporal Resolution")
    plt.tight_layout()
    plt.savefig(output_dir / "mse_vs_T_refined.png", dpi=300)

    print("\nSaved plots to:", output_dir)

    plt.show()


if __name__ == "__main__":
    main()