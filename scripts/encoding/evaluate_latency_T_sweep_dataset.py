import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append('src')

from data.data_utils import HDF5Manager
from encoding.spike_encoders import LatencyEncoder

# -----------------------------
# Configuration
# -----------------------------
HDF5_FILE = Path("data/processed_h5/C002_processed.h5")
PATIENT_ID = "C002"
VAL_SPLIT_START = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

T_VALUES = [4, 8, 16, 32, 64]

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0, device=x.device) / mse)


def main():

    print("=" * 70)
    print("DATASET-LEVEL LATENCY T-SWEEP")
    print("=" * 70)

    # Load data
    low_dose, full_dose, metadata = HDF5Manager.load_patient_data(
        HDF5_FILE, PATIENT_ID
    )

    val_slices = full_dose[VAL_SPLIT_START:]
    print(f"Validation slices: {len(val_slices)}")

    mean_psnr_results = []
    mean_sparsity_results = []

    # -----------------------------
    # Loop over T values
    # -----------------------------
    for T in T_VALUES:

        print(f"\nEvaluating T = {T}")
        encoder = LatencyEncoder(T=T)

        psnr_values = []
        sparsity_values = []

        for i in tqdm(range(len(val_slices))):
            img = torch.from_numpy(val_slices[i]).float().to(DEVICE)

            spikes = encoder.encode(img)
            recon = encoder.decode(spikes)

            # PSNR
            psnr = compute_psnr(img, recon).item()
            psnr_values.append(psnr)

            # Sparsity
            total_spikes = spikes.sum().item()
            sparsity = 1 - (total_spikes / spikes.numel())
            sparsity_values.append(sparsity)

        mean_psnr = np.mean(psnr_values)
        mean_sparsity = np.mean(sparsity_values)

        mean_psnr_results.append(mean_psnr)
        mean_sparsity_results.append(mean_sparsity)

        print(f"Mean PSNR: {mean_psnr:.2f} dB")
        print(f"Mean Sparsity: {mean_sparsity:.4f}")

    # -----------------------------
    # Plot 1: PSNR vs T
    # -----------------------------
    plt.figure()
    plt.plot(T_VALUES, mean_psnr_results, marker='o')
    plt.axhline(y=28.35, linestyle='--')
    plt.xlabel("Timesteps (T)")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("Dataset-Level Encoding Fidelity vs Timesteps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dataset_psnr_vs_T.png", dpi=300)
    print("✓ Saved: dataset_psnr_vs_T.png")

    # -----------------------------
    # Plot 2: Sparsity vs T
    # -----------------------------
    plt.figure()
    plt.plot(T_VALUES, mean_sparsity_results, marker='o')
    plt.xlabel("Timesteps (T)")
    plt.ylabel("Mean Sparsity")
    plt.title("Dataset-Level Sparsity vs Timesteps (Latency)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dataset_sparsity_vs_T.png", dpi=300)
    print("✓ Saved: dataset_sparsity_vs_T.png")

    print("\n" + "=" * 70)
    print("✓ T-SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

