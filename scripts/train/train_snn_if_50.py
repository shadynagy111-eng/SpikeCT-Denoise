"""
scripts/train/train_snn_if_50.py
SNN-IF training on 50-patient dataset.
T=16 timesteps — matches SPIDEN (Castagnetti 2023) T=15 configuration.
AMP enabled. No torch.compile. Clean checkpoint saving.
"""

import sys, time, json, csv
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
from src.models.snn import SNNFinalIF, count_parameters, verify_snn
from src.data.dataset import MayoCTSpikeDatasetCached, load_split

SEED       = 42
T          = 16      # T=16: 8.7 min/epoch vs 111.5 min/epoch at T=30
BATCH_SIZE = 4       # increased from 4 — T=16 uses less VRAM
GRAD_ACCUM = 32      # effective batch = 8 × 16 = 128 (matches CNN)
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5_FILE    = Path.home() / "SpikeCT_Data/mayo_50patients_fast.h5"
SPLIT_JSON = Path.home() / "SpikeCT_Data/patient_split_50.json"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
LOG_DIR    = PROJECT_ROOT / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (spikes, full) in enumerate(loader):
        spikes = spikes.permute(1, 0, 2, 3, 4).to(DEVICE)
        full   = full.to(DEVICE)
        with torch.amp.autocast('cuda'):
            pred = model(spikes)
            loss = criterion(pred, full) / GRAD_ACCUM
        scaler.scale(loss).backward()
        total_loss += loss.item() * GRAD_ACCUM * full.size(0)
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    if len(loader) % GRAD_ACCUM != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss, all_psnr, all_ssim = 0.0, [], []
    with torch.no_grad():
        for spikes, full in loader:
            spikes = spikes.permute(1, 0, 2, 3, 4).to(DEVICE)
            full   = full.to(DEVICE)
            with torch.amp.autocast('cuda'):
                pred = model(spikes)
            total_loss += criterion(pred, full).item() * full.size(0)
            m = compute_batch_metrics(pred, full)
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])
    return total_loss/len(loader.dataset), float(np.mean(all_psnr)), float(np.mean(all_ssim))

def main():
    set_seed(SEED)
    print("=" * 60)
    print(f"SNN-IF Training (50 patients, T={T}, AMP, no compile)")
    print("=" * 60)
    print(f"Device:     {DEVICE}")
    print(f"Batch:      {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"LR:         {LR} | Seed: {SEED}")
    print(f"Note: T={T} follows SPIDEN (Castagnetti 2023) T=15 configuration\n")

    split = load_split(SPLIT_JSON)

    print("Loading data into RAM (FP16, spike encoding on-the-fly)...")
    train_ds = MayoCTSpikeDatasetCached(H5_FILE, split["train"], T=T, desc="train")
    val_ds   = MayoCTSpikeDatasetCached(H5_FILE, split["val"],   T=T, desc="val")
    print()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    model     = SNNFinalIF(T=T).to(DEVICE)
    verify_snn(model, T=T, device=DEVICE)
    n_params  = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler('cuda')

    log_csv   = LOG_DIR  / "snn_if_50_training_log.csv"
    log_json  = LOG_DIR  / "snn_if_50_training_summary.json"
    best_ckpt = CKPT_DIR / "snn_if_50_best.pth"
    last_ckpt = CKPT_DIR / "snn_if_50_last.pth"

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_psnr",
                                 "val_ssim","epoch_time_s","peak_vram_mb"])

    best_psnr, best_epoch, no_improve = -float("inf"), 0, 0
    t_start = time.time()

    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'ValPSNR':>9} {'ValSSIM':>8} {'Time(s)':>8} {'VRAM(MB)':>9}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion)

        epoch_time = time.time() - t0
        peak_vram  = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

        print(f"{epoch:>6} {train_loss:>10.6f} {val_loss:>10.6f} "
              f"{val_psnr:>9.4f} {val_ssim:>8.4f} "
              f"{epoch_time:>8.1f} {peak_vram:>9.1f}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_psnr,
                                    val_ssim, round(epoch_time,2), round(peak_vram,1)])

        if val_psnr > best_psnr:
            best_psnr, best_epoch, no_improve = val_psnr, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_psnr": val_psnr, "val_ssim": val_ssim,
                "val_loss": val_loss, "parameters": n_params,
                "T": T, "variant": "IF",
                "architecture": "SNN-IF 4+4 CDAE 1->32->64->96->96->96->64->32->1",
                "amp": True, "train_patients": split["train"],
                "val_patients": split["val"],
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
        "model": "SNN-IF-50", "variant": "IF", "T": T,
        "parameters": n_params,
        "best_val_psnr": round(best_psnr, 4), "best_epoch": best_epoch,
        "total_epochs_run": epoch,
        "total_train_time_hr": round(total_time/3600, 2),
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "lr": LR, "seed": SEED, "amp": True, "torch_compile": False,
        "train_patients": split["train"], "val_patients": split["val"],
        "test_patients": split["test"],
    }
    with open(log_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best val PSNR: {best_psnr:.4f} dB (epoch {best_epoch})")
    print(f"  Total time:    {total_time/3600:.2f} hours")
    print(f"  Checkpoint:    {best_ckpt}")

    print("\nVerifying checkpoint integrity...")
    ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    test_model = SNNFinalIF(T=T)
    test_model.load_state_dict(ckpt['model_state_dict'])
    print(f"Checkpoint verified. val_psnr={ckpt['val_psnr']:.4f} dB")

if __name__ == "__main__":
    main()
