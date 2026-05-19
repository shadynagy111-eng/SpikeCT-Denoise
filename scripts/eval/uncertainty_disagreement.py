"""
scripts/eval/uncertainty_disagreement.py
-----------------------------------------
Inter-model disagreement heatmap as a proxy for reconstruction uncertainty.

Method:
  Run all four trained models (CNN-Final, SNN-Direct-IF, SNN-Direct-LIF,
  SNN-IF Latency) on the same CT slice. Compute pixel-wise standard
  deviation across their four outputs. High std = models disagree =
  reconstruction is ambiguous.

Scientific justification:
  Since all models are deterministic (no dropout, no stochasticity),
  classical Monte Carlo uncertainty estimation is not applicable.
  Inter-model disagreement is a well-established alternative
  (Lakshminarayanan et al. 2017 — "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles"). The four models form
  an implicit ensemble that differs in: activation mechanism (ReLU vs
  IF vs LIF), input encoding (float vs latency vs direct), and temporal
  dynamics (1 step vs T=8 vs T=16). Regions of high disagreement
  identify areas where the reconstruction is sensitive to model choice.

Output figures (per selected slice):
  fig_uncertainty_{pid}_s{sidx}.png   — full 7-panel figure
  fig_uncertainty_compact_{pid}_s{sidx}.png — compact 3-panel for thesis

Run from project root:
  python scripts/eval/uncertainty_disagreement.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
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
OUT    = PROJECT_ROOT / 'outputs/figures/final'
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = '#0D1117'
BG2     = '#161B22'
C_CNN   = '#3B82F6'
C_DIF   = '#10B981'
C_DLF   = '#8B5CF6'
C_LAT   = '#F59E0B'
C_INPUT = '#EF4444'
C_REF   = '#D1D5DB'
C_TEXT  = '#E6EDF3'
C_MUTED = '#8B949E'

# ── Load models ────────────────────────────────────────────────────────────────
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

# ── Inference ──────────────────────────────────────────────────────────────────
def infer_all(cnn, dif, dlf, lat, arr):
    def run_cnn(m, x):
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                return m(t).squeeze().cpu().float().numpy().clip(0, 1)

    def run_direct(m, x, T):
        img = torch.from_numpy(x).unsqueeze(0)
        di  = direct_encode(img, T=T).unsqueeze(1).to(DEVICE)
        functional.reset_net(m)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                return m(di).squeeze().cpu().float().numpy().clip(0, 1)

    def run_latency(m, x, T=16):
        img    = torch.from_numpy(x).unsqueeze(0)
        spikes = latency_encode(img, T=T).unsqueeze(1).to(DEVICE)
        functional.reset_net(m)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                return m(spikes).squeeze().cpu().float().numpy().clip(0, 1)

    return {
        'CNN-Final':           run_cnn(cnn, arr),
        'SNN-Direct-IF (T=8)': run_direct(dif, arr, T=8),
        'SNN-Direct-LIF (T=8)':run_direct(dlf, arr, T=8),
        'SNN-IF Latency (T=16)':run_latency(lat, arr, T=16),
    }

# ── Disagreement computation ────────────────────────────────────────────────────
def compute_disagreement(preds):
    """
    Pixel-wise standard deviation across all model predictions.
    Higher value = models disagree more = higher reconstruction uncertainty.
    Returns: (H, W) float32 array
    """
    stack = np.stack(list(preds.values()), axis=0)  # (4, H, W)
    return stack.std(axis=0)                         # (H, W)

# ── Full 7-panel figure ────────────────────────────────────────────────────────
def plot_full(pid, sidx, low, full, preds, disagree):
    """
    Panel layout:
      Row 1: Low-dose input | Full-dose reference | Disagreement heatmap
      Row 2: CNN-Final | SNN-Direct-IF | SNN-Direct-LIF | SNN-IF Latency
      Row 3: CNN error | DIF error | DLF error | LAT error
    """
    model_names  = list(preds.keys())
    model_colors = [C_CNN, C_DIF, C_DLF, C_LAT]
    model_preds  = list(preds.values())

    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.06, wspace=0.04)

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    ax_low  = fig.add_subplot(gs[0, 0])
    ax_full = fig.add_subplot(gs[0, 1])
    ax_dis  = fig.add_subplot(gs[0, 2])
    ax_dis2 = fig.add_subplot(gs[0, 3])  # std distribution

    ax_low.imshow(low, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax_low.set_title('Low-Dose Input', color=C_INPUT, fontsize=12, fontweight='bold', pad=6)
    ax_low.axis('off')
    low_psnr = psnr_fn(full, np.clip(low, 0, 1), data_range=1.0)
    ax_low.text(0.02, 0.02, f'PSNR: {low_psnr:.2f} dB',
                transform=ax_low.transAxes, color='white', fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    ax_full.imshow(full, cmap='gray', vmin=0, vmax=1, aspect='equal')
    ax_full.set_title('Full-Dose Reference', color=C_REF, fontsize=12, fontweight='bold', pad=6)
    ax_full.axis('off')

    im_dis = ax_dis.imshow(disagree, cmap='hot', vmin=0, vmax=disagree.max(), aspect='equal')
    ax_dis.set_title('Inter-Model Disagreement\n(pixel-wise std across 4 models)',
                     color='#FF9F1C', fontsize=11, fontweight='bold', pad=6)
    ax_dis.axis('off')
    cbar = fig.colorbar(im_dis, ax=ax_dis, shrink=0.9, pad=0.02, aspect=20)
    cbar.set_label('Std Dev (pixel units)', color=C_TEXT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED, fontsize=7)

    # Disagreement statistics annotation
    mean_dis = disagree.mean()
    max_dis  = disagree.max()
    p95_dis  = np.percentile(disagree, 95)
    ax_dis.text(0.02, 0.02,
                f'Mean: {mean_dis:.4f}\nMax: {max_dis:.4f}\nP95: {p95_dis:.4f}',
                transform=ax_dis.transAxes, color='white', fontsize=8.5, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    # Distribution of disagreement values
    ax_dis2.set_facecolor(BG2)
    ax_dis2.hist(disagree.ravel(), bins=80, color='#FF9F1C', alpha=0.8, edgecolor='none')
    ax_dis2.axvline(x=mean_dis, color='white', lw=1.5, ls='--', label=f'Mean {mean_dis:.4f}')
    ax_dis2.axvline(x=p95_dis,  color='#FF6B6B', lw=1.5, ls='--', label=f'P95 {p95_dis:.4f}')
    ax_dis2.set_xlabel('Disagreement (std)', color=C_MUTED, fontsize=9)
    ax_dis2.set_ylabel('Pixel count', color=C_MUTED, fontsize=9)
    ax_dis2.set_title('Disagreement Distribution', color=C_TEXT, fontsize=11, fontweight='bold', pad=6)
    ax_dis2.legend(fontsize=8.5, facecolor=BG2, edgecolor=BG2)
    ax_dis2.tick_params(colors=C_MUTED)
    ax_dis2.spines[['top','right','left','bottom']].set_color('#21262D')

    # ── Row 2 — model outputs ──────────────────────────────────────────────────
    for ci, (name, pred, col) in enumerate(zip(model_names, model_preds, model_colors)):
        ax = fig.add_subplot(gs[1, ci])
        ax.imshow(pred, cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.set_title(name, color=col, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')
        p = psnr_fn(full, pred, data_range=1.0)
        s = ssim_fn(full, pred, data_range=1.0)
        ax.text(0.02, 0.02, f'PSNR: {p:.2f} dB\nSSIM: {s:.3f}',
                transform=ax.transAxes, color='white', fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    # ── Row 3 — error maps ─────────────────────────────────────────────────────
    errors = [np.abs(pred - full) for pred in model_preds]
    vmax_e = max(e.max() for e in errors)

    for ci, (name, err, col) in enumerate(zip(model_names, errors, model_colors)):
        ax = fig.add_subplot(gs[2, ci])
        im_e = ax.imshow(err, cmap='hot', vmin=0, vmax=vmax_e, aspect='equal')
        ax.set_title(f'{name}\n|Output − Reference|', color=col, fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')
        mse = float(np.mean(err**2))
        ax.text(0.02, 0.02, f'MSE: {mse:.5f}',
                transform=ax.transAxes, color='white', fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    # Shared colorbar for error maps
    cbar_e = fig.colorbar(im_e, ax=[fig.axes[ci+4] for ci in range(4)],
                          shrink=0.6, pad=0.01, aspect=20, location='bottom')
    cbar_e.set_label('Absolute Error (shared scale)', color=C_TEXT, fontsize=8)
    cbar_e.ax.xaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar_e.ax.xaxis.get_ticklabels(), color=C_MUTED, fontsize=7)

    fig.suptitle(
        f'Inter-Model Disagreement Analysis — Patient {pid}, Slice {sidx}\n'
        f'Pixel-wise std across 4 models used as reconstruction uncertainty proxy\n'
        f'High disagreement ↔ ambiguous regions where model choice affects output',
        color=C_TEXT, fontsize=12, fontweight='bold', y=1.005
    )

    p = OUT / f'fig_uncertainty_{pid}_s{sidx}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved: {p.name}')
    return p

# ── Compact 3-panel thesis figure ─────────────────────────────────────────────
def plot_compact(pid, sidx, low, full, preds, disagree):
    """
    Compact version for thesis inline figure:
      Low-dose | Disagreement heatmap | Full-dose reference
    With disagreement statistics annotation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor(BG)

    # Low-dose
    axes[0].imshow(low, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axes[0].set_title('Low-Dose Input', color=C_INPUT, fontsize=12, fontweight='bold', pad=6)
    axes[0].axis('off')

    # Disagreement heatmap
    im = axes[1].imshow(disagree, cmap='hot', vmin=0, vmax=disagree.max(), aspect='equal')
    axes[1].set_title('Inter-Model Disagreement\n(pixel-wise std, 4 models)',
                      color='#FF9F1C', fontsize=11, fontweight='bold', pad=6)
    axes[1].axis('off')

    mean_d = disagree.mean()
    max_d  = disagree.max()
    p95_d  = np.percentile(disagree, 95)
    axes[1].text(0.02, 0.02,
                 f'Mean σ: {mean_d:.4f}\nMax σ: {max_d:.4f}\nP95 σ: {p95_d:.4f}',
                 transform=axes[1].transAxes, color='white', fontsize=9.5, va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Full-dose reference
    axes[2].imshow(full, cmap='gray', vmin=0, vmax=1, aspect='equal')
    axes[2].set_title('Full-Dose Reference', color=C_REF, fontsize=12, fontweight='bold', pad=6)
    axes[2].axis('off')

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.02, aspect=25)
    cbar.set_label('Std Dev (pixel units)', color=C_TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED, fontsize=8)

    # High-disagreement regions annotation
    threshold = p95_d
    high_mask = disagree > threshold
    pct_high  = high_mask.mean() * 100
    axes[1].text(0.98, 0.98,
                 f'Top 5% disagreement\ncovers {pct_high:.1f}% of pixels\n'
                 f'(σ > {threshold:.4f})',
                 transform=axes[1].transAxes, color='#FF9F1C', fontsize=9,
                 ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    fig.suptitle(
        f'Reconstruction Uncertainty Proxy — Patient {pid}, Slice {sidx}\n'
        f'Brighter pixels = higher inter-model disagreement = less confident reconstruction',
        color=C_TEXT, fontsize=11, fontweight='bold', y=1.03
    )
    plt.tight_layout(pad=0.5)
    p = OUT / f'fig_uncertainty_compact_{pid}_s{sidx}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved: {p.name}')
    return p

# ── Multi-slice summary figure ─────────────────────────────────────────────────
def plot_multi_slice_summary(cases):
    """
    Thesis-level summary: N rows × 4 cols
    [Low-dose | Disagreement | Best SNN output | Full-dose]
    """
    n = len(cases)
    fig = plt.figure(figsize=(20, 5.5 * n))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.05, wspace=0.04)

    col_titles  = ['Low-Dose Input', 'Inter-Model Disagreement\n(Uncertainty Proxy)',
                   'Best SNN — SNN-Direct-IF (T=8)', 'Full-Dose Reference']
    col_colors  = [C_INPUT, '#FF9F1C', C_DIF, C_REF]

    for ri, (pid, sidx, low, full, preds, disagree) in enumerate(cases):
        best_snn = preds['SNN-Direct-IF (T=8)']
        imgs     = [low, disagree, best_snn, full]
        cmaps    = ['gray', 'hot', 'gray', 'gray']
        vmins    = [0, 0, 0, 0]
        vmaxs    = [1, disagree.max(), 1, 1]

        for ci in range(4):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(imgs[ci], cmap=cmaps[ci], vmin=vmins[ci], vmax=vmaxs[ci], aspect='equal')
            ax.axis('off')

            if ri == 0:
                ax.set_title(col_titles[ci], color=col_colors[ci],
                             fontsize=11, fontweight='bold', pad=8)
            if ci == 0:
                ax.set_ylabel(f'{pid}\nSlice {sidx}',
                              color=C_TEXT, fontsize=9.5, rotation=0,
                              labelpad=68, va='center')

            # Metrics
            if ci == 2:
                p_val = psnr_fn(full, best_snn, data_range=1.0)
                s_val = ssim_fn(full, best_snn, data_range=1.0)
                ax.text(0.02, 0.02, f'PSNR: {p_val:.2f} dB\nSSIM: {s_val:.3f}',
                        transform=ax.transAxes, color='white', fontsize=8.5, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))
            if ci == 1:
                mean_d = disagree.mean()
                ax.text(0.02, 0.02, f'Mean σ: {mean_d:.4f}',
                        transform=ax.transAxes, color='white', fontsize=8.5, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    fig.suptitle(
        'Inter-Model Disagreement as Reconstruction Uncertainty Proxy\n'
        'Pixel-wise std across CNN-Final, SNN-Direct-IF, SNN-Direct-LIF, SNN-IF Latency',
        color=C_TEXT, fontsize=13, fontweight='bold', y=1.01
    )
    p = OUT / 'fig_uncertainty_summary.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved: {p.name}')
    return p

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('='*60)
    print('Inter-Model Disagreement — Uncertainty Proxy')
    print('='*60)
    print(f'Device: {DEVICE}')

    cnn, dif, dlf, lat = load_models()
    split = load_split(SPLIT)

    # Select representative slices: one from each of 3 test patients, best anatomical coverage
    CASES = [
        ('C004', 50),
        ('C004', 100),
        ('C050', 50),
        ('C050', 100),
        ('C111', 100),
    ]

    print(f'\nProcessing {len(CASES)} cases...')

    summary_cases = []

    with h5py.File(H5, 'r') as hf:
        for pid, sidx in CASES:
            print(f'\n  {pid} slice {sidx}:')
            low  = hf[pid]['low_dose'][sidx].astype(np.float32)
            full = hf[pid]['full_dose'][sidx].astype(np.float32)

            preds    = infer_all(cnn, dif, dlf, lat, low)
            disagree = compute_disagreement(preds)

            print(f'    Disagreement: mean={disagree.mean():.4f}  '
                  f'max={disagree.max():.4f}  '
                  f'p95={np.percentile(disagree,95):.4f}')

            # Per-model PSNR for context
            for name, pred in preds.items():
                p = psnr_fn(full, pred, data_range=1.0)
                print(f'    {name:<30}: {p:.2f} dB')

            # Full 7-panel figure
            plot_full(pid, sidx, low, full, preds, disagree)

            # Compact 3-panel figure
            plot_compact(pid, sidx, low, full, preds, disagree)

            summary_cases.append((pid, sidx, low, full, preds, disagree))

    # Multi-slice summary figure (thesis figure)
    print('\nGenerating summary figure...')
    plot_multi_slice_summary(summary_cases[:3])

    print('\n' + '='*60)
    print('Done.')
    print(f'Figures saved to: {OUT}')
    print()
    print('Thesis figures:')
    print('  fig_uncertainty_summary.png          ← main thesis figure')
    print('  fig_uncertainty_compact_*.png         ← compact per-slice')
    print('  fig_uncertainty_*.png                 ← full 7-panel per-slice')

if __name__ == '__main__':
    main()