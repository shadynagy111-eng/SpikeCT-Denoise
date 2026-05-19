"""
scripts/eval/uncertainty_disagreement.py
-----------------------------------------
Inter-model disagreement heatmap as a proxy for reconstruction uncertainty.
Optimized for Thesis Defense Projectors & Print (Light Theme, Constrained Layout).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import h5py
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn
from spikingjelly.activation_based import functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import CNNFinal
from src.models.snn import SNNFinalIF
from src.models.snn_direct import SNNDirectIF, SNNDirectLIF, direct_encode
from src.data.dataset import latency_encode, load_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H5     = Path.home() / 'SpikeCT_Data/mayo_50patients_fast.h5'
SPLIT  = Path.home() / 'SpikeCT_Data/patient_split_50.json'
OUT    = PROJECT_ROOT / 'outputs/figures/new_uncertainity'
OUT.mkdir(parents=True, exist_ok=True)

# ── Clean Light Palette ────────────────────────────────────────────────────────
BG      = '#FFFFFF'
BG2     = '#F8F9FA'
C_CNN   = '#1F77B4'
C_DIF   = '#2CA02C'
C_DLF   = '#9467BD'
C_LAT   = '#FF7F0E'
C_INPUT = '#000000'
C_REF   = '#000000'
C_TEXT  = '#000000'
C_MUTED = '#444444'

# ── Load models & Inference ────────────────────────────────────────────────────
def load_models():
    ckpts = {
        'cnn': PROJECT_ROOT / 'checkpoints/cnn_50_best.pth',
        'dif': PROJECT_ROOT / 'checkpoints/snn_direct_if_best.pth',
        'dlf': PROJECT_ROOT / 'checkpoints/snn_direct_lif_best.pth',
        'lat': PROJECT_ROOT / 'checkpoints/snn_if_50_best.pth',
    }
    cnn = CNNFinal().to(DEVICE)
    cnn.load_state_dict(torch.load(ckpts['cnn'], map_location=DEVICE, weights_only=False)['model_state_dict'])
    cnn.eval()

    dif = SNNDirectIF(T=8).to(DEVICE)
    dif.load_state_dict(torch.load(ckpts['dif'], map_location=DEVICE, weights_only=False)['model_state_dict'])
    dif.eval()

    dlf = SNNDirectLIF(T=8).to(DEVICE)
    dlf.load_state_dict(torch.load(ckpts['dlf'], map_location=DEVICE, weights_only=False)['model_state_dict'])
    dlf.eval()

    lat = SNNFinalIF(T=16).to(DEVICE)
    lat.load_state_dict(torch.load(ckpts['lat'], map_location=DEVICE, weights_only=False)['model_state_dict'])
    lat.eval()

    return cnn, dif, dlf, lat

def infer_all(cnn, dif, dlf, lat, arr):
    def run_cnn(m, x):
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            return m(t).squeeze().cpu().float().numpy().clip(0, 1)

    def run_direct(m, x, T):
        img = torch.from_numpy(x).unsqueeze(0)
        di  = direct_encode(img, T=T).unsqueeze(1).to(DEVICE)
        functional.reset_net(m)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            return m(di).squeeze().cpu().float().numpy().clip(0, 1)

    def run_latency(m, x, T=16):
        img    = torch.from_numpy(x).unsqueeze(0)
        spikes = latency_encode(img, T=T).unsqueeze(1).to(DEVICE)
        functional.reset_net(m)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            return m(spikes).squeeze().cpu().float().numpy().clip(0, 1)

    return {
        'CNN-Final':           run_cnn(cnn, arr),
        'SNN-Direct-IF (T=8)': run_direct(dif, arr, T=8),
        'SNN-Direct-LIF (T=8)':run_direct(dlf, arr, T=8),
        'SNN-IF Latency (T=16)':run_latency(lat, arr, T=16),
    }

def compute_disagreement(preds):
    stack = np.stack(list(preds.values()), axis=0)
    return stack.std(axis=0)

# ── Full 7-panel figure ────────────────────────────────────────────────────────
def plot_full(pid, sidx, low, full, preds, disagree):
    model_names  = list(preds.keys())
    model_colors = [C_CNN, C_DIF, C_DLF, C_LAT]
    model_preds  = list(preds.values())

    fig = plt.figure(figsize=(24, 18), layout='constrained')
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # Row 1
    ax_low  = fig.add_subplot(gs[0, 0])
    ax_full = fig.add_subplot(gs[0, 1])
    ax_dis  = fig.add_subplot(gs[0, 2])
    ax_dis2 = fig.add_subplot(gs[0, 3])

    low_psnr = psnr_fn(full, np.clip(low, 0, 1), data_range=1.0)
    ax_low.imshow(low, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax_low.set_title(f'Low-Dose Input\nPSNR: {low_psnr:.2f} dB', color=C_INPUT, fontsize=14, fontweight='bold', pad=12)
    ax_low.axis('off')

    ax_full.imshow(full, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax_full.set_title('Full-Dose Reference\n(Ground Truth)', color=C_REF, fontsize=14, fontweight='bold', pad=12)
    ax_full.axis('off')

    im_dis = ax_dis.imshow(disagree, cmap='hot', vmin=0, vmax=disagree.max(), aspect='equal')
    
    mean_dis = disagree.mean()
    p95_dis  = np.percentile(disagree, 95)
    ax_dis.set_title(f'Inter-Model Disagreement\nMean \u03C3: {mean_dis:.4f}', color='#D9534F', fontsize=14, fontweight='bold', pad=12)
    ax_dis.axis('off')
    
    divider = make_axes_locatable(ax_dis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im_dis, cax=cax)
    cbar.set_label('Std Dev (pixel units)', color=C_TEXT, fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=C_TEXT)

    # Distribution
    ax_dis2.set_facecolor(BG2)
    ax_dis2.hist(disagree.ravel(), bins=80, color='#D9534F', alpha=0.8, edgecolor='none')
    ax_dis2.axvline(x=mean_dis, color='black', lw=2, ls='--', label=f'Mean: {mean_dis:.4f}')
    ax_dis2.axvline(x=p95_dis,  color='#5BC0DE', lw=2, ls='--', label=f'P95: {p95_dis:.4f}')
    ax_dis2.set_xlabel('Disagreement (std)', color=C_TEXT, fontsize=12)
    ax_dis2.set_ylabel('Pixel count', color=C_TEXT, fontsize=12)
    ax_dis2.set_title('Disagreement Distribution', color=C_TEXT, fontsize=14, fontweight='bold', pad=12)
    ax_dis2.legend(fontsize=12, facecolor=BG2)
    ax_dis2.tick_params(colors=C_TEXT)
    for spine in ax_dis2.spines.values(): spine.set_color('#CCCCCC')

    # Row 2 & 3
    errors = [np.abs(pred - full) for pred in model_preds]
    vmax_e = max(e.max() for e in errors)

    for ci, (name, pred, err, col) in enumerate(zip(model_names, model_preds, errors, model_colors)):
        # Model Outputs
        ax_pred = fig.add_subplot(gs[1, ci])
        ax_pred.imshow(pred, cmap='gray', vmin=0, vmax=1, aspect='equal')
        p = psnr_fn(full, pred, data_range=1.0)
        s = ssim_fn(full, pred, data_range=1.0)
        ax_pred.set_title(f'{name}\nPSNR: {p:.2f} | SSIM: {s:.3f}', color=col, fontsize=14, fontweight='bold', pad=12)
        ax_pred.axis('off')

        # Error Maps
        ax_err = fig.add_subplot(gs[2, ci])
        im_e = ax_err.imshow(err, cmap='hot', vmin=0, vmax=vmax_e, aspect='equal')
        mse = float(np.mean(err**2))
        ax_err.set_title(f'|Error Map|\nMSE: {mse:.5f}', color=col, fontsize=13, fontweight='bold', pad=12)
        ax_err.axis('off')

    # Shared colorbar for error maps (Constrained layout handles this cleanly)
    cbar_e = fig.colorbar(im_e, ax=[fig.axes[i] for i in range(5, 9)], orientation='horizontal', aspect=40, shrink=0.5)
    cbar_e.set_label('Absolute Error (Shared Scale)', color=C_TEXT, fontsize=14)

    fig.suptitle(f'Inter-Model Disagreement Analysis — Patient {pid}, Slice {sidx}', color=C_TEXT, fontsize=20, fontweight='bold')
    
    p = OUT / f'fig_uncertainty_{pid}_s{sidx}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ── Compact 3-panel thesis figure ─────────────────────────────────────────────
def plot_compact(pid, sidx, low, full, preds, disagree):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    # Low-dose
    axes[0].imshow(low, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axes[0].set_title('Low-Dose Input', color=C_INPUT, fontsize=16, fontweight='bold', pad=12)
    axes[0].axis('off')

    # Disagreement heatmap
    im = axes[1].imshow(disagree, cmap='hot', vmin=0, vmax=disagree.max(), aspect='equal')
    mean_d = disagree.mean()
    axes[1].set_title(f'Uncertainty Proxy (Disagreement)\nMean \u03C3: {mean_d:.4f}', color='#D9534F', fontsize=16, fontweight='bold', pad=12)
    axes[1].axis('off')

    # Aligned Colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Std Dev', color=C_TEXT, fontsize=12)

    # Full-dose reference
    axes[2].imshow(full, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axes[2].set_title('Full-Dose Reference', color=C_REF, fontsize=16, fontweight='bold', pad=12)
    axes[2].axis('off')

    fig.suptitle(f'Patient {pid}, Slice {sidx}', color=C_TEXT, fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    p = OUT / f'fig_uncertainty_compact_{pid}_s{sidx}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ── Multi-slice summary figure ─────────────────────────────────────────────────
def plot_multi_slice_summary(cases):
    n = len(cases)
    fig = plt.figure(figsize=(20, 6 * n), layout='constrained')
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(n, 4, figure=fig)

    col_titles  = ['Low-Dose Input', 'Uncertainty Proxy\n(Model Disagreement)', 'Best SNN Output\n(SNN-Direct-IF)', 'Full-Dose Reference']
    col_colors  = [C_INPUT, '#D9534F', C_DIF, C_REF]

    for ri, (pid, sidx, low, full, preds, disagree) in enumerate(cases):
        best_snn = preds['SNN-Direct-IF (T=8)']
        imgs     = [low, disagree, best_snn, full]
        cmaps    = ['gray', 'hot', 'gray', 'gray']
        vmins    = [0, 0, 0, 0]
        vmaxs    = [1, disagree.max(), 1, 1]

        for ci in range(4):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(imgs[ci], cmap=cmaps[ci], vmin=vmins[ci], vmax=vmaxs[ci], aspect='equal')
            
            # Hide ticks but keep axis active for ylabel
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)

            if ri == 0:
                ax.set_title(col_titles[ci], color=col_colors[ci], fontsize=16, fontweight='bold', pad=16)
            
            # Y-axis labels using set_ylabel to ensure bounding box includes it during save
            if ci == 0:
                ax.set_ylabel(f'{pid}\nSlice {sidx}', color=C_TEXT, fontsize=16, fontweight='bold', 
                              rotation=0, ha='right', va='center', labelpad=20)

            # Clean metric labels underneath the image
            if ci == 1:
                ax.text(0.5, -0.05, f'Mean \u03C3: {disagree.mean():.4f}', transform=ax.transAxes, color=C_TEXT, fontsize=14, ha='center', va='top')
            if ci == 2:
                p_val = psnr_fn(full, best_snn, data_range=1.0)
                s_val = ssim_fn(full, best_snn, data_range=1.0)
                ax.text(0.5, -0.05, f'PSNR: {p_val:.2f} dB | SSIM: {s_val:.3f}', transform=ax.transAxes, color=C_TEXT, fontsize=14, ha='center', va='top')

            # Aligned colorbar strictly for the heatmap column
            if ci == 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=10)

    p = OUT / 'fig_uncertainty_summary.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('='*60)
    print('Inter-Model Disagreement — Uncertainty Proxy')
    print('='*60)
    print(f'Device: {DEVICE}')

    cnn, dif, dlf, lat = load_models()
    split = load_split(SPLIT)

    CASES = [
        ('C004', 50),
        ('C004', 100),
        ('C050', 50),
        ('C050', 100),
        ('C111', 100),
    ]

    summary_cases = []

    with h5py.File(H5, 'r') as hf:
        for pid, sidx in CASES:
            low  = hf[pid]['low_dose'][sidx].astype(np.float32)
            full = hf[pid]['full_dose'][sidx].astype(np.float32)

            preds    = infer_all(cnn, dif, dlf, lat, low)
            disagree = compute_disagreement(preds)

            plot_full(pid, sidx, low, full, preds, disagree)
            plot_compact(pid, sidx, low, full, preds, disagree)

            summary_cases.append((pid, sidx, low, full, preds, disagree))

    plot_multi_slice_summary(summary_cases[:3])
    print('\nDone.')

if __name__ == '__main__':
    main()