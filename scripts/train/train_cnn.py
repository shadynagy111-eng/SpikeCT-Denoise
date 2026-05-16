"""
train_cnn.py
------------
Trains the final CNN baseline (4+4 symmetric CDAE, 314,401 parameters).

Architecture (FROZEN):
  Encoder: Conv2d 1->32->64->96->96 (stride=2, kernel=3)
  Decoder: ConvTranspose2d 96->96->64->32->1 (stride=2, kernel=3)
  No batch norm. No skip connections. No pooling. Linear output.

Training config:
  Loss:       MSE
  Optimizer:  Adam (lr=1e-3)
  Batch size: 8
  Epochs:     50 (with early stopping, patience=15)
  Seed:       42

Logging (per epoch):
  - train loss (MSE)
  - val loss (MSE)
  - val PSNR (dB)
  - val SSIM
  - epoch time (seconds)
  - peak VRAM (MB)

Output:
  checkpoints/cnn_final_best.pth      <- best val PSNR checkpoint
  checkpoints/cnn_final_last.pth      <- last epoch checkpoint
  logs/cnn_training_log.csv           <- full per-epoch log
  logs/cnn_training_summary.json      <- final summary
"""

import os
import time
import json
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path("/mnt/c/Projects/SpikeCT-Denoise")
H5_FILE       = PROJECT_ROOT / "data/processed_h5/mayo_10patients.h5"
SPLIT_JSON    = PROJECT_ROOT / "data/processed_h5/patient_split.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = CHECKPOINT_DIR / "cnn_final_best.pth"
LAST_CKPT = CHECKPOINT_DIR / "cnn_final_last.pth"
LOG_CSV   = LOG_DIR / "cnn_training_log.csv"
LOG_JSON  = LOG_DIR / "cnn_training_summary.json"

# ── Training config ───────────────────────────────────────────────────────────

BATCH_SIZE     = 8
EPOCHS         = 50
LR             = 1e-3
PATIENCE       = 15   # early stopping patience (epochs)
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────

class MayoCTDataset(Dataset):
    """
    Loads paired (low_dose, full_dose) slices from HDF5 for a list of patients.
    Returns tensors of shape (1, 512, 512).
    """
    def __init__(self, h5_path: Path, patient_ids: list):
        self.h5_path = h5_path
        self.index = []  # list of (patient_id, slice_idx)

        with h5py.File(h5_path, "r") as hf:
            for pid in patient_ids:
                n_slices = hf[pid]["full_dose"].shape[0]
                for i in range(n_slices):
                    self.index.append((pid, i))

        print(f"  Dataset: {len(patient_ids)} patients, {len(self.index)} slices total")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pid, slice_idx = self.index[idx]
        with h5py.File(self.h5_path, "r") as hf:
            low  = hf[pid]["low_dose"][slice_idx]   # (512, 512) float32
            full = hf[pid]["full_dose"][slice_idx]  # (512, 512) float32

        low  = torch.from_numpy(low).unsqueeze(0)   # (1, 512, 512)
        full = torch.from_numpy(full).unsqueeze(0)  # (1, 512, 512)
        return low, full


# ── Model ─────────────────────────────────────────────────────────────────────

class CNNFinal(nn.Module):
    """
    Symmetric CDAE: 4 encoder + 4 decoder.
    Parameters: 314,401
    Input/Output: (B, 1, 512, 512)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    Compute PSNR and SSIM on a batch.
    pred, target: (B, 1, H, W) float32 tensors on any device.
    Returns mean PSNR (dB) and mean SSIM over the batch.
    """
    pred_np   = pred.detach().cpu().numpy()    # (B, 1, H, W)
    target_np = target.detach().cpu().numpy()

    psnr_vals, ssim_vals = [], []
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0.0, 1.0)
        t = target_np[i, 0]
        psnr_vals.append(psnr_fn(t, p, data_range=1.0))
        ssim_vals.append(ssim_fn(t, p, data_range=1.0))

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for low, full in loader:
        low, full = low.to(DEVICE), full.to(DEVICE)
        optimizer.zero_grad()
        pred = model(low)
        loss = criterion(pred, full)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * low.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_psnr, all_ssim = [], []
    with torch.no_grad():
        for low, full in loader:
            low, full = low.to(DEVICE), full.to(DEVICE)
            pred = model(low)
            loss = criterion(pred, full)
            total_loss += loss.item() * low.size(0)
            psnr, ssim = compute_metrics(pred, full)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
    return (
        total_loss / len(loader.dataset),
        float(np.mean(all_psnr)),
        float(np.mean(all_ssim)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CNN-Final Training")
    print("=" * 60)
    print(f"Device:     {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs:     {EPOCHS} (patience={PATIENCE})")
    print(f"LR:         {LR}")
    print(f"Seed:       {SEED}")
    print()

    # Load split
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    train_patients = split["train"]
    val_patients   = split["val"]

    print("Building datasets...")
    train_ds = MayoCTDataset(H5_FILE, train_patients)
    val_ds   = MayoCTDataset(H5_FILE, val_patients)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
        worker_init_fn=lambda _: np.random.seed(SEED)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Model
    model = CNNFinal().to(DEVICE)
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")
    assert 280_000 <= n_params <= 330_000, \
        f"Parameter count {n_params} outside expected band [280k, 330k]"

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # CSV log header
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_psnr", "val_ssim",
            "epoch_time_s", "peak_vram_mb"
        ])

    best_val_psnr  = -float("inf")
    epochs_no_improve = 0
    training_start = time.time()

    print("\nStarting training...\n")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'ValPSNR':>9} {'ValSSIM':>8} {'Time(s)':>8} {'VRAM(MB)':>9}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start
        peak_vram  = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2  # MB

        print(f"{epoch:>6} {train_loss:>10.6f} {val_loss:>10.6f} "
              f"{val_psnr:>9.4f} {val_ssim:>8.4f} "
              f"{epoch_time:>8.1f} {peak_vram:>9.1f}")

        # CSV log
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_loss, val_psnr, val_ssim,
                round(epoch_time, 2), round(peak_vram, 1)
            ])

        # Checkpoint — best
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr":   val_psnr,
                "val_ssim":   val_ssim,
                "val_loss":   val_loss,
                "train_loss": train_loss,
            }, BEST_CKPT)
        else:
            epochs_no_improve += 1

        # Checkpoint — last
        torch.save({
            "epoch":      epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr":   val_psnr,
            "val_ssim":   val_ssim,
        }, LAST_CKPT)

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

    total_time = time.time() - training_start

    # Summary
    summary = {
        "model":           "CNN-Final",
        "parameters":      n_params,
        "architecture":    "4+4 symmetric CDAE, 1->32->64->96->96->96->64->32->1",
        "best_val_psnr":   round(best_val_psnr, 4),
        "best_epoch":      epoch - epochs_no_improve,
        "total_epochs_run": epoch,
        "total_train_time_s": round(total_time, 1),
        "train_patients":  train_patients,
        "val_patients":    val_patients,
        "batch_size":      BATCH_SIZE,
        "lr":              LR,
        "seed":            SEED,
        "device":          str(DEVICE),
    }

    with open(LOG_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete.")
    print(f"  Best val PSNR:  {best_val_psnr:.4f} dB")
    print(f"  Total time:     {total_time/3600:.2f} hours")
    print(f"  Best checkpoint: {BEST_CKPT}")
    print(f"  Log CSV:         {LOG_CSV}")
    print(f"  Summary JSON:    {LOG_JSON}")


if __name__ == "__main__":
    main()