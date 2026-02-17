"""
Training Script — CNN Baseline (CDAE)
Run from project root: python scripts/train_cnn.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ──────────────────────────────────────────────────────────────────────

sys.path.append('src')
from data.dicom_dataset import PairedCTDataset
from models.cdae import CDAE
from utils.metrics import batch_psnr, batch_ssim


# ── Configuration ─────────────────────────────────────────────────────
CONFIG = {
    'full_dose_dir':            'data/dataset/C002/Full_Dose_Images',
    'low_dose_dir':             'data/dataset/C002/Low_Dose_Images',
    'batch_size':               8,
    'num_epochs':               50,
    'learning_rate':            1e-3,
    'weight_decay':             1e-5,
    'val_split':                0.2,
    'early_stopping_patience':  15,
    'num_workers':              4,
}


# ── Trainer ───────────────────────────────────────────────────────────
class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.model = CDAE(in_channels=1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': []
        }

        print(f"Model parameters: {self.model.get_num_params():,}")

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for low, full in tqdm(loader, desc='  Train'):
            low, full = low.to(self.device), full.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(low)
            loss = self.criterion(output, full)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for low, full in tqdm(loader, desc='  Val  '):
                low, full = low.to(self.device), full.to(self.device)
                output = self.model(low)

                total_loss += self.criterion(output, full).item()
                total_psnr += batch_psnr(output, full)
                total_ssim += batch_ssim(output, full)

        n = len(loader)
        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'ssim': total_ssim / n
        }

    def train(self, train_loader, val_loader):
        best_psnr = 0.0
        no_improve = 0

        for epoch in range(self.config['num_epochs']):
            print(f"\n{'='*55}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"{'='*55}")

            train_loss = self.train_epoch(train_loader)
            val = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val['loss'])
            self.history['val_psnr'].append(val['psnr'])
            self.history['val_ssim'].append(val['ssim'])

            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Val Loss   : {val['loss']:.4f}")
            print(f"  Val PSNR   : {val['psnr']:.2f} dB")
            print(f"  Val SSIM   : {val['ssim']:.4f}")

            self.scheduler.step(val['psnr'])

            if val['psnr'] > best_psnr:
                best_psnr = val['psnr']
                no_improve = 0
                os.makedirs('results/checkpoints', exist_ok=True)
                self._save('results/checkpoints/cnn_best.pth', epoch, val)
                print(f"  ✓ Best model saved — PSNR: {best_psnr:.2f} dB")
            else:
                no_improve += 1
                if no_improve >= self.config['early_stopping_patience']:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print(f"\n✓ Training done. Best PSNR: {best_psnr:.2f} dB")

    def _save(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history,
            'seed': SEED
        }, path)

    def plot_history(self, path='results/training_history.png'):
        os.makedirs('results', exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history['val_psnr'])
        axes[1].set_title('Val PSNR (dB)')
        axes[1].grid(True)

        axes[2].plot(self.history['val_ssim'])
        axes[2].set_title('Val SSIM')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    full_dataset = PairedCTDataset(
        full_dose_dir=CONFIG['full_dose_dir'],
        low_dose_dir=CONFIG['low_dose_dir']
    )

    split_idx = int((1 - CONFIG['val_split']) * len(full_dataset))
    indices = list(range(len(full_dataset)))

    train_ds = Subset(full_dataset, indices[:split_idx])
    val_ds   = Subset(full_dataset, indices[split_idx:])

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    trainer = Trainer(CONFIG)
    trainer.train(train_loader, val_loader)
    trainer.plot_history()

    os.makedirs('results', exist_ok=True)
    with open('results/config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)


if __name__ == "__main__":
    main()
