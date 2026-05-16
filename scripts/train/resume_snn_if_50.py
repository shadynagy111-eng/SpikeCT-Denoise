"""
scripts/train/resume_snn_if_50.py
Resumes SNN-IF training from epoch 51 to 75.
Loads best checkpoint and last state.
Reduces LR to 5e-4 to stabilize after spike storm.
Appends to existing CSV log.
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

SEED          = 42
T             = 16
BATCH_SIZE    = 4
GRAD_ACCUM    = 32
EPOCHS        = 100        # FIXED: Set to 75 to allow training beyond epoch 50
LR_RESUME     = 5e-4      
PATIENCE      = 15
RESUME_EPOCH  = 76        
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H5_FILE    = Path.home() / "SpikeCT_Data/mayo_50patients_fast.h5"
SPLIT_JSON = Path.home() / "SpikeCT_Data/patient_split_50.json"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
LOG_DIR    = PROJECT_ROOT / "logs"

BEST_CKPT  = CKPT_DIR / "snn_if_50_best.pth"
LAST_CKPT  = CKPT_DIR / "snn_if_50_last.pth"
LOG_CSV    = LOG_DIR  / "snn_if_50_training_log.csv"
LOG_JSON   = LOG_DIR  / "snn_if_50_training_summary.json"

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
    print(f"SNN-IF Resume Training (epoch {RESUME_EPOCH} → {EPOCHS})")
    print("=" * 60)
    print(f"Device:     {DEVICE}")
    print(f"Batch:      {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"LR:         {LR_RESUME} (reduced from 1e-3 after spike storm)")
    print(f"Loading checkpoint: {BEST_CKPT}\n")

    if not BEST_CKPT.exists() or not LAST_CKPT.exists():
        print(f"ERROR: Missing checkpoints.")
        sys.exit(1)

    # Load best PSNR as the target to beat
    best_data  = torch.load(BEST_CKPT, map_location='cpu', weights_only=False)
    # Load state where training actually stopped
    last_data  = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
    
    print(f"Best checkpoint: epoch {best_data['epoch']}, val_psnr={best_data['val_psnr']:.4f} dB")
    print(f"Last checkpoint: epoch {last_data['epoch']}, val_psnr={last_data['val_psnr']:.4f} dB")

    split = load_split(SPLIT_JSON)

    print("\nLoading data into RAM...")
    train_ds = MayoCTSpikeDatasetCached(H5_FILE, split["train"], T=T, desc="train")
    val_ds   = MayoCTSpikeDatasetCached(H5_FILE, split["val"],   T=T, desc="val")
    print()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Load latest weights
    model = SNNFinalIF(T=T).to(DEVICE)
    model.load_state_dict(last_data['model_state_dict'])
    n_params = count_parameters(model)

    # Fresh optimizer at reduced LR
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RESUME)
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler('cuda')

    # Track best from previous run
    best_psnr  = best_data['val_psnr']   
    best_epoch = best_data['epoch']      
    no_improve = 0

    print(f"Weights loaded from epoch {last_data['epoch']} (last checkpoint).")
    print(f"Target to beat: {best_psnr:.4f} dB (epoch {best_epoch}).")
    print(f"Resuming epoch numbering from {RESUME_EPOCH}.\n")

    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>10} "
          f"{'ValPSNR':>9} {'ValSSIM':>8} {'Time(s)':>8} {'VRAM(MB)':>9}")
    print("-" * 70)

    t_start = time.time()

    for epoch in range(RESUME_EPOCH, EPOCHS + 1):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion)

        epoch_time = time.time() - t0
        peak_vram  = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

        print(f"{epoch:>6} {train_loss:>10.6f} {val_loss:>10.6f} "
              f"{val_psnr:>9.4f} {val_ssim:>8.4f} "
              f"{epoch_time:>8.1f} {peak_vram:>9.1f}")

        # Append to existing CSV
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_psnr,
                                    val_ssim, round(epoch_time,2),
                                    round(peak_vram,1)])

        if val_psnr > best_psnr:
            best_psnr, best_epoch, no_improve = val_psnr, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_psnr": val_psnr, "val_ssim": val_ssim,
                "val_loss": val_loss, "parameters": n_params,
                "T": T, "variant": "IF", "amp": True,
                "architecture": "SNN-IF 4+4 CDAE 1->32->64->96->96->96->64->32->1",
                "train_patients": split["train"],
                "val_patients": split["val"],
            }, BEST_CKPT)
            print(f"         ↑ New best checkpoint saved.")
        else:
            no_improve += 1

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "val_psnr": val_psnr}, LAST_CKPT)

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    total_time = time.time() - t_start
    summary = {
        "model": "SNN-IF-resumed", "variant": "IF", "T": T,
        "parameters": n_params,
        "best_val_psnr": round(best_psnr, 4),
        "best_epoch": best_epoch,
        "resumed_from_epoch": RESUME_EPOCH,
        "resumed_from_psnr": last_data['val_psnr'],
        "total_resume_time_hr": round(total_time/3600, 2),
        "lr_resume": LR_RESUME,
        "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "seed": SEED, "amp": True,
        "train_patients": split["train"],
        "val_patients": split["val"],
        "test_patients": split["test"],
    }
    with open(LOG_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Resume training complete.")
    print(f"  Best val PSNR: {best_psnr:.4f} dB (epoch {best_epoch})")
    print(f"  Resume time:   {total_time/3600:.2f} hours")
    print(f"  Checkpoint:    {BEST_CKPT}")

    print("\nVerifying checkpoint integrity...")
    verify_ckpt = torch.load(BEST_CKPT, map_location='cpu', weights_only=False)
    test_model = SNNFinalIF(T=T)
    test_model.load_state_dict(verify_ckpt['model_state_dict'])
    # FIXED: Use verify_ckpt instead of verify_last_data
    print(f"Checkpoint verified. val_psnr={verify_ckpt['val_psnr']:.4f} dB")

if __name__ == "__main__":
    main()