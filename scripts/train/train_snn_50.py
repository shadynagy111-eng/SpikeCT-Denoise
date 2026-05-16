"""
scripts/train/train_snn_50.py
-----------------------------
Trains SNN-Final (IF or LIF variant) on the 50-patient Mayo CT dataset.

Usage:
  python scripts/train/train_snn_50.py --variant IF
  python scripts/train/train_snn_50.py --variant LIF

The variant argument selects the neuron model:
  IF:  Integrate-and-Fire (primary model, recommended)
  LIF: Leaky Integrate-and-Fire (ablation comparison)

Both use identical convolutional backbone as CNN-Final (314,401 params).
Input: latency-encoded spike tensors (T=30, one spike per pixel).
Output: continuous membrane readout, normalized by T.

Output:
  checkpoints/snn_{variant}_50_best.pth
  checkpoints/snn_{variant}_50_last.pth
  logs/snn_{variant}_50_training_log.csv
  logs/snn_{variant}_50_training_summary.json
"""

import sys
import time
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.metrics import compute_batch_metrics
from src.models.snn import SNNFinalIF, SNNFinalLIF, count_parameters, verify_snn
from src.data.dataset import MayoCTSpikeDatasetCached, load_split

# ── Config ────────────────────────────────────────────────────────────────────

SEED       = 42
T          = 30       # timesteps — frozen from encoding validation
BATCH_SIZE = 4        # reduced vs CNN — SNN processes T=30 frames per sample
GRAD_ACCUM = 4        # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5_FILE    = PROJECT_ROOT / "data/processed_h5/mayo_50patients.h5"
SPLIT_JSON = PROJECT_ROOT / "data/processed_h5/patient_split_50.json"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
LOG_DIR    = PROJECT_ROOT / "logs"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (spikes, full) in enumerate(loader):
        # spikes: (B, T, 1, 512, 512) from DataLoader collation
        # Need: (T, B, 1, 512, 512) for SNN forward
        spikes = spikes.permute(1, 0, 2, 3, 4).to(DEVICE)
        full   = full.to(DEVICE)

        pred = model(spikes)
        loss = criterion(pred, full) / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM * full.size(0)

        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining steps if dataset not divisible by GRAD_ACCUM
    if (len(loader)) % GRAD_ACCUM != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss, all_psnr, all_ssim = 0.0, [], []

    with torch.no_grad():
        for spikes, full in loader:
            spikes = spikes.permute(1, 0, 2, 3, 4).to(DEVICE)
            full   = full.to(DEVICE)
            pred   = model(spikes)
            total_loss += criterion(pred, full).item() * full.size(0)
            m = compute_batch_metrics(pred, full)
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])

    return (
        total_loss / len(loader.dataset),
        float(np.mean(all_psnr)),
        float(np.mean(all_ssim)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["IF", "LIF"],
                        default="IF", help="Neuron model variant")
    args = parser.parse_args()
    variant = args.variant

    set_seed(SEED)

    print("=" * 60)
    print(f"SNN-Final-{variant} Training (50 patients, T={T})")
    print("=" * 60)
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | "
          f"GradAccum: {GRAD_ACCUM} | EffBatch: {BATCH_SIZE*GRAD_ACCUM} | "
          f"LR: {LR} | Seed: {SEED}\n")

    split = load_split(SPLIT_JSON)

    print("Loading data into RAM...")
    train_ds = MayoCTSpikeDatasetCached(H5_FILE, split["train"], T=T, desc="train")
    val_ds   = MayoCTSpikeDatasetCached(H5_FILE, split["val"],   T=T, desc="val")
    print()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Build model
    if variant == "IF":
        model = SNNFinalIF(T=T).to(DEVICE)
    else:
        model = SNNFinalLIF(T=T).to(DEVICE)

    verify_snn(model, T=T, device=DEVICE)
    n_params  = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    log_csv   = LOG_DIR  / f"snn_{variant.lower()}_50_training_log.csv"
    log_json  = LOG_DIR  / f"snn_{variant.lower()}_50_training_summary.json"
    best_ckpt = CKPT_DIR / f"snn_{variant.lower()}_50_best.pth"
    last_ckpt = CKPT_DIR / f"snn_{variant.lower()}_50_last.pth"

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "val_psnr",
            "val_ssim", "epoch_time_s", "peak_vram_mb"
        ])

    best_psnr, best_epoch, no_improve = -float("inf"), 0, 0
    t_start = time.time()

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'ValPSNR':>9} {'ValSSIM':>8} {'Time(s)':>8} {'VRAM(MB)':>9}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion)

        epoch_time = time.time() - t0
        peak_vram  = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2

        print(f"{epoch:>6} {train_loss:>10.6f} {val_loss:>10.6f} "
              f"{val_psnr:>9.4f} {val_ssim:>8.4f} "
              f"{epoch_time:>8.1f} {peak_vram:>9.1f}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss, val_psnr,
                val_ssim, round(epoch_time, 2), round(peak_vram, 1)
            ])

        if val_psnr > best_psnr:
            best_psnr, best_epoch, no_improve = val_psnr, epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr, "val_ssim": val_ssim,
                "val_loss": val_loss, "parameters": n_params,
                "variant": variant, "T": T,
                "architecture": f"SNN-Final-{variant} 4+4 CDAE 1->32->64->96->96->96->64->32->1",
                "train_patients": split["train"], "val_patients": split["val"],
            }, best_ckpt)
        else:
            no_improve += 1

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "val_psnr": val_psnr}, last_ckpt)

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    summary = {
        "model": f"SNN-Final-{variant}-50",
        "variant": variant, "T": T,
        "parameters": n_params,
        "best_val_psnr": round(best_psnr, 4), "best_epoch": best_epoch,
        "total_epochs_run": epoch,
        "total_train_time_hr": round(total_time / 3600, 2),
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "lr": LR, "seed": SEED,
        "train_patients": split["train"], "val_patients": split["val"],
        "test_patients": split["test"],
    }
    with open(log_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete.")
    print(f"  Best val PSNR: {best_psnr:.4f} dB (epoch {best_epoch})")
    print(f"  Total time:    {total_time/3600:.2f} hours")
    print(f"  Checkpoint:    {best_ckpt}")


if __name__ == "__main__":
    main()