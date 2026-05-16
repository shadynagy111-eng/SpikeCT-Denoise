"""
Visualize Spike Encoding Methods
Demonstrates temporal conversion of static CT images

Generates:
1. spike_encoding_raster.png
2. spike_encoding_temporal.png
3. spike_encoding_comparison.png
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('src')

from data.data_utils import HDF5Manager
from encoding.spike_encoders import (
    DeterministicRateEncoder,
    BernoulliRateEncoder,
    LatencyEncoder
)

# -----------------------------
# Configuration
# -----------------------------
HDF5_FILE = Path("data/processed_h5/C002_processed.h5")
PATIENT_ID = "C002"
SLICE_IDX = 140
PATCH_SIZE = 32
T = 100
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utilities
# -----------------------------
def extract_patch(image: np.ndarray, size: int = 32, center: tuple = None):
    H, W = image.shape

    if center is None:
        center_h, center_w = H // 2, W // 2
    else:
        center_h, center_w = center

    h_start = center_h - size // 2
    w_start = center_w - size // 2

    return image[h_start:h_start+size, w_start:w_start+size]


def plot_raster(spikes: torch.Tensor, title: str, ax):
    T, H, W = spikes.shape
    spikes_flat = spikes.reshape(T, -1)

    spike_times = []
    neuron_ids = []

    for neuron_idx in range(spikes_flat.shape[1]):
        spike_t = torch.where(spikes_flat[:, neuron_idx] > 0)[0]
        if len(spike_t) > 0:
            spike_times.extend(spike_t.cpu().numpy())
            neuron_ids.extend([neuron_idx] * len(spike_t))

    ax.scatter(spike_times, neuron_ids, s=1, c='black', marker='|')
    ax.set_xlabel('Time (timestep)')
    ax.set_ylabel('Neuron ID')
    ax.set_title(title)
    ax.set_xlim(0, T)
    ax.set_ylim(0, H * W)
    ax.grid(alpha=0.3)

    total_spikes = len(spike_times)
    total_possible = T * H * W
    sparsity = 1 - (total_spikes / total_possible)

    ax.text(
        0.98, 0.98,
        f'Sparsity: {sparsity:.2%}\nTotal spikes: {total_spikes:.0f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )


def plot_temporal_activity(spikes: torch.Tensor, title: str, ax):
    activity = spikes.sum(dim=(1, 2)).cpu().numpy()
    ax.plot(activity, linewidth=1.5)
    ax.set_xlabel('Time (timestep)')
    ax.set_ylabel('Active Neurons')
    ax.set_title(f'{title} - Temporal Activity')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(activity))


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 70)
    print("SPIKE ENCODING VISUALIZATION")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    low_dose, full_dose, metadata = HDF5Manager.load_patient_data(
        HDF5_FILE, PATIENT_ID
    )

    image = full_dose[SLICE_IDX]
    patch = extract_patch(image, size=PATCH_SIZE, center=(256, 200))

    patch_tensor = torch.from_numpy(patch).float().to(DEVICE)

    # Initialize encoders
    rate_encoder = DeterministicRateEncoder(T=T)
    latency_encoder = LatencyEncoder(T=T)
    poisson_encoder = BernoulliRateEncoder(T=T)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    # Encode
    rate_spikes = rate_encoder.encode(patch_tensor)
    latency_spikes = latency_encoder.encode(patch_tensor)
    poisson_spikes = poisson_encoder.encode(patch_tensor, generator=generator)

    # -----------------------------
    # 1️⃣ Raster Plot
    # -----------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax_patch = fig.add_subplot(gs[0, 0])
    ax_patch.imshow(patch, cmap='gray', vmin=0, vmax=1)
    ax_patch.set_title(f'Original Patch\n{PATCH_SIZE}×{PATCH_SIZE}')
    ax_patch.axis('off')

    ax_rate = fig.add_subplot(gs[0, 1:])
    plot_raster(rate_spikes, 'Deterministic Rate Encoding', ax_rate)

    ax_latency = fig.add_subplot(gs[1, 1:])
    plot_raster(latency_spikes, 'Latency Encoding (Time-to-First-Spike)', ax_latency)

    ax_poisson = fig.add_subplot(gs[2, 1:])
    plot_raster(poisson_spikes, 'Bernoulli (Poisson-like) Encoding', ax_poisson)

    plt.savefig('spike_encoding_raster.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: spike_encoding_raster.png")

    # -----------------------------
    # 2️⃣ Temporal Activity
    # -----------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    plot_temporal_activity(rate_spikes, 'Deterministic Rate', axes[0])
    plot_temporal_activity(latency_spikes, 'Latency', axes[1])
    plot_temporal_activity(poisson_spikes, 'Bernoulli', axes[2])

    plt.tight_layout()
    plt.savefig('spike_encoding_temporal.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: spike_encoding_temporal.png")

    # -----------------------------
    # 3️⃣ Comparison Visualization
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(patch, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Patch')
    axes[0, 0].axis('off')

    rate_reconstruction = rate_encoder.decode(rate_spikes)
    axes[0, 1].imshow(rate_reconstruction.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Rate Reconstruction')
    axes[0, 1].axis('off')

    latency_times = torch.argmax(latency_spikes, dim=0)
    axes[0, 2].imshow(latency_times.cpu(), cmap='viridis')
    axes[0, 2].set_title('Latency Spike Times')
    axes[0, 2].axis('off')

    rate_counts = rate_spikes.sum(dim=0)
    latency_counts = latency_spikes.sum(dim=0)
    poisson_counts = poisson_spikes.sum(dim=0)

    im1 = axes[1, 0].imshow(rate_counts.cpu(), cmap='hot')
    axes[1, 0].set_title('Rate Spike Counts')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(latency_counts.cpu(), cmap='hot')
    axes[1, 1].set_title('Latency Spike Counts')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    im3 = axes[1, 2].imshow(poisson_counts.cpu(), cmap='hot')
    axes[1, 2].set_title('Bernoulli Spike Counts')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig('spike_encoding_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: spike_encoding_comparison.png")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()