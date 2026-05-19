"""
scripts/eval/visualize_all_final.py
------------------------------------
Generates ALL thesis and publication figures for SpikeCT-Denoise.

Figures produced:
  Fig 1:  4-panel per-slice comparison (Low | CNN | SNN-Direct-IF | Full)
           → one per patient × 3 slices = 15 figures
  Fig 2:  2-panel difference maps (CNN error | SNN-Direct-IF error)
           → one per patient × 3 slices = 15 figures
  Fig 3:  4-model comparison strip (all 4 models side by side)
           → 3 representative slices
  Fig 4:  Combined multi-row thesis figure (3 rows × 5 cols)
  Fig 5:  Training curves — all 4 models, annotated
  Fig 6:  Ablation bar chart — PSNR and SSIM
  Fig 7:  Per-patient PSNR grouped bar chart
  Fig 8:  Efficiency comparison (latency, throughput, sparsity)
  Fig 9:  Quality-efficiency scatter (PSNR vs latency)
  Fig 10: Energy estimate bar chart
  Fig 11: Encoding comparison (latency vs direct, same slice)

Run from project root:
    python scripts/eval/visualize_all_final.py
"""

import sys, json, csv, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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

LOG_DIR = PROJECT_ROOT / 'logs'

# ── Colour palette (consistent across all figures) ────────────────────────────
BG      = '#0D1117'
BG2     = '#161B22'
GRID    = '#21262D'
C_CNN   = '#3B82F6'   # blue
C_DIF   = '#10B981'   # green  — SNN-Direct-IF (best SNN)
C_DLF   = '#8B5CF6'   # purple — SNN-Direct-LIF
C_LAT   = '#F59E0B'   # amber  — SNN-IF Latency
C_INPUT = '#EF4444'   # red    — low-dose input
C_REF   = '#D1D5DB'   # grey   — full-dose reference
C_TEXT  = '#E6EDF3'
C_MUTED = '#8B949E'

MODEL_COLORS  = [C_CNN, C_DIF, C_DLF, C_LAT]
MODEL_NAMES   = ['CNN-Final', 'SNN-Direct-IF\n(T=8)', 'SNN-Direct-LIF\n(T=8)', 'SNN-IF Latency\n(T=16)']
MODEL_LABELS  = ['CNN-Final', 'SNN-Direct-IF (T=8)', 'SNN-Direct-LIF (T=8)', 'SNN-IF Latency (T=16)']

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   BG2,
    'figure.facecolor': BG,
    'axes.edgecolor':   GRID,
    'axes.labelcolor':  C_TEXT,
    'xtick.color':      C_MUTED,
    'ytick.color':      C_MUTED,
    'grid.color':       GRID,
    'grid.linewidth':   0.6,
    'text.color':       C_TEXT,
    'legend.facecolor': BG2,
    'legend.edgecolor': GRID,
    'legend.labelcolor':C_TEXT,
})

# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    print('Loading models...')
    ckpts = {
        'cnn':   PROJECT_ROOT / 'checkpoints/cnn_50_best.pth',
        'dif':   PROJECT_ROOT / 'checkpoints/snn_direct_if_best.pth',
        'dlf':   PROJECT_ROOT / 'checkpoints/snn_direct_lif_best.pth',
        'lat':   PROJECT_ROOT / 'checkpoints/snn_if_50_best.pth',
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

    print(f'  All models loaded on {DEVICE}')
    return cnn, dif, dlf, lat

# ── Inference helpers ─────────────────────────────────────────────────────────
def infer_cnn(model, arr):
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            return model(x).squeeze().cpu().float().numpy().clip(0,1)

def infer_direct(model, arr, T):
    img = torch.from_numpy(arr).unsqueeze(0)
    di  = direct_encode(img, T=T).unsqueeze(1).to(DEVICE)
    functional.reset_net(model)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            return model(di).squeeze().cpu().float().numpy().clip(0,1)

def infer_latency(model, arr, T=16):
    img    = torch.from_numpy(arr).unsqueeze(0)
    spikes = latency_encode(img, T=T).unsqueeze(1).to(DEVICE)
    functional.reset_net(model)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            return model(spikes).squeeze().cpu().float().numpy().clip(0,1)

def metrics(pred, ref):
    p = np.clip(pred, 0, 1)
    return psnr_fn(ref, p, data_range=1.0), ssim_fn(ref, p, data_range=1.0)

def add_metrics(ax, psnr, ssim, fontsize=9):
    ax.text(0.02, 0.02, f'PSNR: {psnr:.2f} dB\nSSIM: {ssim:.3f}',
            transform=ax.transAxes, color='white', fontsize=fontsize,
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

# ── Load logs ─────────────────────────────────────────────────────────────────
def load_log(name):
    path = LOG_DIR / f'{name}_training_log.csv'
    if not path.exists(): return [], []
    epochs, psnrs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(float(row['epoch']))
            psnrs.append(float(row['val_psnr']))
    return epochs, psnrs

# ═════════════════════════════════════════════════════════════════════════════
# RUN INFERENCE ON ALL TEST CASES
# ═════════════════════════════════════════════════════════════════════════════
SLICE_INDICES = [50, 100, 150]

def collect_all_results(cnn, dif, dlf, lat):
    split = load_split(SPLIT)
    results = []
    print('Running inference on all test cases...')
    with h5py.File(H5, 'r') as hf:
        for pid in split['test']:
            for sidx in SLICE_INDICES:
                low  = hf[pid]['low_dose'][sidx].astype(np.float32)
                full = hf[pid]['full_dose'][sidx].astype(np.float32)
                cnn_pred = infer_cnn(cnn, low)
                dif_pred = infer_direct(dif, low, T=8)
                dlf_pred = infer_direct(dlf, low, T=8)
                lat_pred = infer_latency(lat, low, T=16)
                r = {
                    'pid': pid, 'sidx': sidx,
                    'low': low, 'full': full,
                    'cnn': cnn_pred, 'dif': dif_pred,
                    'dlf': dlf_pred, 'lat': lat_pred,
                    'low_psnr':  psnr_fn(full, np.clip(low,0,1), data_range=1.0),
                    'cnn_psnr':  psnr_fn(full, cnn_pred, data_range=1.0),
                    'dif_psnr':  psnr_fn(full, dif_pred, data_range=1.0),
                    'dlf_psnr':  psnr_fn(full, dlf_pred, data_range=1.0),
                    'lat_psnr':  psnr_fn(full, lat_pred, data_range=1.0),
                    'cnn_ssim':  ssim_fn(full, cnn_pred, data_range=1.0),
                    'dif_ssim':  ssim_fn(full, dif_pred, data_range=1.0),
                    'dlf_ssim':  ssim_fn(full, dlf_pred, data_range=1.0),
                    'lat_ssim':  ssim_fn(full, lat_pred, data_range=1.0),
                    'cnn_diff':  np.abs(cnn_pred - full),
                    'dif_diff':  np.abs(dif_pred - full),
                    'dlf_diff':  np.abs(dlf_pred - full),
                    'lat_diff':  np.abs(lat_pred - full),
                }
                results.append(r)
                print(f'  {pid} s{sidx}: CNN={r["cnn_psnr"]:.2f} DIF={r["dif_psnr"]:.2f} DLF={r["dlf_psnr"]:.2f} LAT={r["lat_psnr"]:.2f}')
    return results

# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — 4-panel: Low | CNN | SNN-Direct-IF | Full
# ═════════════════════════════════════════════════════════════════════════════
def fig1_comparison(r):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.patch.set_facecolor(BG)
    panels = [
        (r['low'],  'Low-Dose Input',         C_INPUT, r['low_psnr'], None),
        (r['cnn'],  'CNN-Final',               C_CNN,   r['cnn_psnr'], r['cnn_ssim']),
        (r['dif'],  'SNN-Direct-IF (T=8)',     C_DIF,   r['dif_psnr'], r['dif_ssim']),
        (r['full'], 'Full-Dose Reference',     C_REF,   None,          None),
    ]
    for ax, (img, title, col, psnr, ssim_v) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, color=col, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')
        if psnr is not None:
            add_metrics(ax, psnr, ssim_v if ssim_v else 0)
    fig.suptitle(f'Patient {r["pid"]} — Slice {r["sidx"]}  |  Low-Dose CT Denoising',
                 color=C_TEXT, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.4)
    p = OUT / f'fig1_comparison_{r["pid"]}_s{r["sidx"]}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Difference maps with correct colorbar placement
# ═════════════════════════════════════════════════════════════════════════════
def fig2_difference(r):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    vmax = max(r['cnn_diff'].max(), r['dif_diff'].max(), 0.04)
    panels = [
        (r['cnn_diff'], 'CNN-Final Error  |CNN − Reference|',      C_CNN, np.mean(r['cnn_diff']**2)),
        (r['dif_diff'], 'SNN-Direct-IF Error  |SNN − Reference|',  C_DIF, np.mean(r['dif_diff']**2)),
    ]
    im = None
    for ax, (diff, title, col, mse) in zip(axes, panels):
        im = ax.imshow(diff, cmap='hot', vmin=0, vmax=vmax, aspect='equal')
        ax.set_title(title, color=col, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')
        ax.text(0.02, 0.02, f'MSE: {mse:.5f}',
                transform=ax.transAxes, color='white', fontsize=10, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, aspect=25)
    cbar.set_label('Absolute Error Magnitude', color=C_TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED, fontsize=8)
    fig.suptitle(f'Patient {r["pid"]} — Slice {r["sidx"]}  |  Residual Error Maps',
                 color=C_TEXT, fontsize=11, fontweight='bold', y=1.03)
    plt.tight_layout(pad=0.5)
    p = OUT / f'fig2_difference_{r["pid"]}_s{r["sidx"]}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — All 4 models side by side (best 3 slices)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_all_models(results_subset):
    fig, axes = plt.subplots(len(results_subset), 6, figsize=(28, 5.5*len(results_subset)))
    fig.patch.set_facecolor(BG)
    if len(results_subset) == 1:
        axes = [axes]

    col_titles  = ['Low-Dose Input', 'CNN-Final', 'SNN-Direct-IF\n(T=8)', 'SNN-Direct-LIF\n(T=8)', 'SNN-IF Latency\n(T=16)', 'Full-Dose\nReference']
    col_colors  = [C_INPUT, C_CNN, C_DIF, C_DLF, C_LAT, C_REF]

    for ri, r in enumerate(results_subset):
        imgs   = [r['low'], r['cnn'], r['dif'], r['dlf'], r['lat'], r['full']]
        psnrs  = [r['low_psnr'], r['cnn_psnr'], r['dif_psnr'], r['dlf_psnr'], r['lat_psnr'], None]
        ssims  = [None, r['cnn_ssim'], r['dif_ssim'], r['dlf_ssim'], r['lat_ssim'], None]

        for ci in range(6):
            ax = axes[ri][ci]
            ax.imshow(imgs[ci], cmap='gray', vmin=0, vmax=1, aspect='equal')
            ax.axis('off')
            if ri == 0:
                ax.set_title(col_titles[ci], color=col_colors[ci],
                             fontsize=10.5, fontweight='bold', pad=6)
            if ci == 0:
                ax.set_ylabel(f'{r["pid"]}\nSlice {r["sidx"]}',
                              color=C_TEXT, fontsize=9.5, rotation=0,
                              labelpad=72, va='center')
            if psnrs[ci] is not None and ssims[ci] is not None:
                add_metrics(ax, psnrs[ci], ssims[ci], fontsize=8)
            elif psnrs[ci] is not None:
                ax.text(0.02, 0.02, f'PSNR: {psnrs[ci]:.2f} dB',
                        transform=ax.transAxes, color='white', fontsize=8, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    fig.suptitle('All-Model Comparison — CNN-Final vs Three SNN Variants\nLow-Dose CT Denoising on Mayo Test Set',
                 color=C_TEXT, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.08)
    p = OUT / 'fig3_all_models_comparison.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Combined thesis figure: 3 rows × 5 cols (Low|CNN|DIF|LAT|Full)
# ═════════════════════════════════════════════════════════════════════════════
def fig4_combined_thesis(results_subset):
    n = len(results_subset)
    fig = plt.figure(figsize=(22, 5.5*n))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(n, 5, figure=fig, hspace=0.04, wspace=0.03)
    col_titles  = ['Low-Dose Input', 'CNN-Final', 'SNN-Direct-IF (T=8)', 'SNN-IF Latency (T=16)', 'Full-Dose Reference']
    col_colors  = [C_INPUT, C_CNN, C_DIF, C_LAT, C_REF]

    for ri, r in enumerate(results_subset):
        imgs  = [r['low'], r['cnn'], r['dif'], r['lat'], r['full']]
        psnrs = [r['low_psnr'], r['cnn_psnr'], r['dif_psnr'], r['lat_psnr'], None]
        ssims = [None, r['cnn_ssim'], r['dif_ssim'], r['lat_ssim'], None]
        for ci in range(5):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(imgs[ci], cmap='gray', vmin=0, vmax=1, aspect='equal')
            ax.axis('off')
            if ri == 0:
                ax.set_title(col_titles[ci], color=col_colors[ci],
                             fontsize=11, fontweight='bold', pad=8)
            if ci == 0:
                ax.set_ylabel(f'{r["pid"]}  Slice {r["sidx"]}',
                              color=C_TEXT, fontsize=9, rotation=0,
                              labelpad=70, va='center')
            if psnrs[ci] is not None and ssims[ci] is not None:
                add_metrics(ax, psnrs[ci], ssims[ci], fontsize=8.5)
            elif psnrs[ci] is not None:
                ax.text(0.02, 0.02, f'PSNR: {psnrs[ci]:.2f} dB',
                        transform=ax.transAxes, color='white', fontsize=8.5, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.65))
    fig.suptitle('Low-Dose CT Denoising — CNN-Final vs SNN Variants\nVisual Comparison Across Representative Test Slices  (PSNR / SSIM on each panel)',
                 color=C_TEXT, fontsize=13, fontweight='bold', y=1.01)
    p = OUT / 'fig4_combined_thesis.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Training curves, all 4 models
# ═════════════════════════════════════════════════════════════════════════════
def fig5_training_curves():
    cnn_e, cnn_p = load_log('cnn_50')
    dif_e, dif_p = load_log('snn_direct_if')
    dlf_e, dlf_p = load_log('snn_direct_lif')

    # SNN-IF Latency reconstructed from known values
    lat_known = [
        (1,13.73),(2,18.49),(3,19.43),(4,19.28),(5,20.47),(6,19.56),(7,21.17),(8,21.53),
        (9,21.77),(10,22.15),(11,22.57),(12,22.80),(13,23.44),(14,23.98),(15,24.20),
        (16,24.54),(17,24.76),(18,24.87),(19,24.30),(20,24.84),(21,24.84),(22,25.12),
        (23,23.58),(24,21.78),(25,22.68),(26,23.55),(27,23.92),(28,24.33),(29,24.51),
        (30,24.75),(31,24.80),(32,25.02),(33,25.14),(34,25.28),(35,25.33),(36,25.26),
        (37,25.57),(38,25.65),(39,25.76),(40,25.90),(41,25.97),(42,26.01),(43,26.06),
        (44,26.00),(45,25.90),(46,26.17),(47,26.19),(48,26.30),(49,26.35),(50,26.11),
        (51,26.40),(52,26.45),(53,26.29),(54,26.31),(55,26.42),(56,26.51),(57,26.29),
        (58,26.41),(59,26.49),(60,26.66),(61,26.46),(62,26.64),(63,26.28),(64,26.64),
        (65,26.70),(66,26.74),(67,26.74),(68,26.85),(69,26.70),(70,26.82),(71,26.90),
        (72,26.89),(73,26.96),(74,27.00),(75,27.03),(76,26.98),(77,26.98),(78,27.09),
        (79,27.01),(80,27.14),(81,27.04),(82,26.98),(83,27.11),(84,27.13),(85,27.14),
        (86,26.92),(87,27.14),(88,27.19),(89,27.17),(90,27.17),(91,27.18),(92,27.20),
        (93,27.22),(94,27.28),(95,27.30),(96,27.33),(97,27.36),
    ]
    lat_e = [x[0] for x in lat_known]
    lat_p = [x[1] for x in lat_known]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    for ax in [ax1, ax2]:
        ax.set_facecolor(BG2)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Validation PSNR (dB)', fontsize=11)

    # Full history
    if cnn_e: ax1.plot(cnn_e, cnn_p, color=C_CNN, lw=2, label='CNN-Final', zorder=4)
    ax1.plot(lat_e, lat_p, color=C_LAT, lw=1.8, label='SNN-IF Latency (T=16)', alpha=0.9, zorder=3)
    if dif_e: ax1.plot(dif_e, dif_p, color=C_DIF, lw=2, label='SNN-Direct-IF (T=8)', zorder=4)
    if dlf_e: ax1.plot(dlf_e, dlf_p, color=C_DLF, lw=1.8, label='SNN-Direct-LIF (T=8)', alpha=0.9, zorder=3)

    # Spike storm annotation
    ax1.axvspan(22, 25, alpha=0.12, color='red', zorder=1)
    ax1.text(24.5, 20.5, 'spike\nstorm', color='#FF6B6B', fontsize=8, ha='center')

    # LIF delayed start annotation
    ax1.axvspan(1, 21, alpha=0.08, color=C_DLF, zorder=1)
    ax1.text(11, 11.5, 'LIF warm-up\nepochs 1–20', color=C_DLF, fontsize=7.5, ha='center', alpha=0.9)

    ax1.axhline(y=29.1151, color=C_CNN, lw=1, ls='--', alpha=0.4)
    ax1.text(5, 29.35, 'CNN best 29.12 dB', color=C_CNN, fontsize=8, alpha=0.7)
    ax1.set_title('Full Training History', color=C_TEXT, fontsize=11)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_ylim(9, 31)

    # Convergence zone
    if cnn_e: ax2.plot(cnn_e, cnn_p, color=C_CNN, lw=2, label=f'CNN-Final (best {max(cnn_p):.2f} dB)' if cnn_p else 'CNN-Final')
    ax2.plot(lat_e, lat_p, color=C_LAT, lw=1.8, label=f'SNN-IF Latency (best 27.36 dB)', alpha=0.9)
    if dif_e: ax2.plot(dif_e, dif_p, color=C_DIF, lw=2, label=f'SNN-Direct-IF (best {max(dif_p):.2f} dB)' if dif_p else 'SNN-Direct-IF')
    if dlf_e: ax2.plot(dlf_e, dlf_p, color=C_DLF, lw=1.8, label=f'SNN-Direct-LIF (best {max(dlf_p):.2f} dB)' if dlf_p else 'SNN-Direct-LIF', alpha=0.9)

    ax2.axhline(y=29.1151, color=C_CNN, lw=1, ls='--', alpha=0.4)
    ax2.set_xlim(20, 130)
    ax2.set_ylim(22, 31)
    ax2.set_title('Convergence Zone (Epoch 20+)', color=C_TEXT, fontsize=11)
    ax2.legend(fontsize=8.5, loc='lower right')

    fig.suptitle('Training Curves — Validation PSNR vs Epoch (All Models)',
                 color=C_TEXT, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    p = OUT / 'fig5_training_curves.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Ablation bar chart
# ═════════════════════════════════════════════════════════════════════════════
def fig6_ablation():
    names   = ['CNN-Final', 'SNN-Direct-IF\n(T=8)', 'SNN-Direct-LIF\n(T=8)', 'SNN-IF Latency\n(T=16)']
    psnr    = [30.1296, 29.4094, 29.1467, 28.0794]
    ssim_v  = [0.6854,  0.6692,  0.6556,  0.6225]
    psnr_s  = [1.4837,  1.2630,  1.2084,  1.0740]
    ssim_s  = [0.0685,  0.0645,  0.0644,  0.0600]
    colors  = [C_CNN, C_DIF, C_DLF, C_LAT]
    x = np.arange(len(names))
    w = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(BG2)
        ax.grid(True, axis='y', alpha=0.35)
        ax.set_axisbelow(True)

    # PSNR
    bars = ax1.bar(x, psnr, w, color=colors, alpha=0.85,
                   yerr=psnr_s, capsize=5,
                   error_kw=dict(ecolor=C_TEXT, capsize=5, elinewidth=1.5))
    ax1.axhline(y=21.52, color=C_INPUT, lw=1.5, ls=':', alpha=0.7)
    ax1.text(3.4, 21.7, 'Low-dose input\n21.52 dB', color=C_INPUT, fontsize=8, ha='right', alpha=0.8)
    ax1.axhline(y=30.1296, color=C_CNN, lw=1.2, ls='--', alpha=0.45)
    for bar, v, s, g in zip(bars, psnr, psnr_s, [0, -0.72, -0.98, -2.05]):
        ax1.text(bar.get_x()+bar.get_width()/2, v+s+0.15,
                 f'{v:.2f}', ha='center', fontsize=10, color=C_TEXT, fontweight='bold')
        if g != 0:
            ax1.text(bar.get_x()+bar.get_width()/2, 26.5,
                     f'{g:.2f} dB', ha='center', fontsize=9, color=colors[list(psnr).index(v)])
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9.5)
    ax1.set_ylabel('Test PSNR (dB)', fontsize=11)
    ax1.set_title('Test PSNR — mean ± std', color=C_TEXT, fontsize=11)
    ax1.set_ylim(19, 33)

    # SSIM
    bars2 = ax2.bar(x, ssim_v, w, color=colors, alpha=0.85,
                    yerr=ssim_s, capsize=5,
                    error_kw=dict(ecolor=C_TEXT, capsize=5, elinewidth=1.5))
    ax2.axhline(y=0.6854, color=C_CNN, lw=1.2, ls='--', alpha=0.45)
    for bar, v, s in zip(bars2, ssim_v, ssim_s):
        ax2.text(bar.get_x()+bar.get_width()/2, v+s+0.003,
                 f'{v:.3f}', ha='center', fontsize=10, color=C_TEXT, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9.5)
    ax2.set_ylabel('Test SSIM', fontsize=11)
    ax2.set_title('Test SSIM — mean ± std', color=C_TEXT, fontsize=11)
    ax2.set_ylim(0.55, 0.77)

    fig.suptitle('Ablation Results — Test Set Performance (All Models)',
                 color=C_TEXT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUT / 'fig6_ablation_bars.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — Per-patient PSNR
# ═════════════════════════════════════════════════════════════════════════════
def fig7_per_patient(results):
    patients = ['C004', 'C050', 'C111', 'C121', 'C249']
    per_pat  = {pid: {'low':[], 'cnn':[], 'dif':[], 'dlf':[], 'lat':[]} for pid in patients}
    for r in results:
        pid = r['pid']
        per_pat[pid]['low'].append(r['low_psnr'])
        per_pat[pid]['cnn'].append(r['cnn_psnr'])
        per_pat[pid]['dif'].append(r['dif_psnr'])
        per_pat[pid]['dlf'].append(r['dlf_psnr'])
        per_pat[pid]['lat'].append(r['lat_psnr'])

    means = {k: {m: np.mean(v[m]) for m in v} for k, v in per_pat.items()}
    x = np.arange(len(patients))
    w = 0.16

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.grid(True, axis='y', alpha=0.35); ax.set_axisbelow(True)

    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    keys    = ['cnn', 'dif', 'dlf', 'lat']
    cols    = [C_CNN, C_DIF, C_DLF, C_LAT]
    labels  = ['CNN-Final', 'SNN-Direct-IF (T=8)', 'SNN-Direct-LIF (T=8)', 'SNN-IF Latency (T=16)']

    for off, key, col, lbl in zip(offsets, keys, cols, labels):
        vals = [means[pid][key] for pid in patients]
        ax.bar(x+off, vals, w, color=col, alpha=0.85, label=lbl)
        for xi, v in zip(x+off, vals):
            ax.text(xi, v+0.08, f'{v:.1f}', ha='center', fontsize=8, color=C_TEXT)

    ax.set_xticks(x); ax.set_xticklabels(patients, fontsize=11)
    ax.set_ylabel('Mean PSNR (dB)', fontsize=11)
    ax.set_title('Per-Patient Mean PSNR — Test Set (3 slices per patient)',
                 color=C_TEXT, fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(25, 34)

    fig.suptitle('Per-Patient Test PSNR — All Models',
                 color=C_TEXT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUT / 'fig7_per_patient_psnr.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 8 — Efficiency comparison
# ═════════════════════════════════════════════════════════════════════════════
def fig8_efficiency():
    names     = ['CNN-Final', 'SNN-Direct-IF\n(T=8)', 'SNN-Direct-LIF\n(T=8)', 'SNN-IF Latency\n(T=16)']
    latency   = [0.95,  12.90, 13.35, 50.87]
    throughp  = [1047.7, 77.5,  74.9,  19.7]
    sparsity  = [0.0,   82.3,  79.9,  90.0]
    colors    = [C_CNN, C_DIF, C_DLF, C_LAT]
    x = np.arange(len(names))
    w = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG2); ax.grid(True, axis='y', alpha=0.35); ax.set_axisbelow(True)

    # Latency — log scale
    bars = axes[0].bar(x, latency, w, color=colors, alpha=0.85)
    for bar, v in zip(bars, latency):
        axes[0].text(bar.get_x()+bar.get_width()/2, v*1.08,
                     f'{v:.2f}ms', ha='center', fontsize=9, color=C_TEXT, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel('Inference Latency (ms/slice)', fontsize=10)
    axes[0].set_title('GPU Inference Latency\n(log scale)', color=C_TEXT, fontsize=10)
    axes[0].set_yscale('log'); axes[0].set_ylim(0.5, 120)

    # Throughput
    bars2 = axes[1].bar(x, throughp, w, color=colors, alpha=0.85)
    for bar, v in zip(bars2, throughp):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+20,
                     f'{v:.0f}', ha='center', fontsize=9, color=C_TEXT, fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel('Throughput (slices/sec)', fontsize=10)
    axes[1].set_title('Inference Throughput', color=C_TEXT, fontsize=10)

    # Sparsity
    bars3 = axes[2].bar(x, sparsity, w, color=colors, alpha=0.85)
    for bar, v in zip(bars3, sparsity):
        label = f'{v:.1f}%' if v > 0 else 'N/A\n(dense)'
        axes[2].text(bar.get_x()+bar.get_width()/2, v+1.5,
                     label, ha='center', fontsize=9, color=C_TEXT, fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=9)
    axes[2].set_ylabel('Neural Sparsity (%)', fontsize=10)
    axes[2].set_title('Spike Sparsity at Inference\n(SNN only)', color=C_TEXT, fontsize=10)
    axes[2].set_ylim(0, 100)

    fig.suptitle('Inference Efficiency Comparison — GPU Hardware (RTX 5060)',
                 color=C_TEXT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUT / 'fig8_efficiency.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 9 — PSNR vs Latency scatter
# ═════════════════════════════════════════════════════════════════════════════
def fig9_scatter():
    data = [
        ('CNN-Final',             30.1296, 0.95,  C_CNN, 200),
        ('SNN-Direct-IF\n(T=8)',  29.4094, 12.90, C_DIF, 200),
        ('SNN-Direct-LIF\n(T=8)', 29.1467, 13.35, C_DLF, 200),
        ('SNN-IF Latency\n(T=16)',28.0794, 50.87, C_LAT, 200),
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)

    for name, psnr, lat, col, sz in data:
        ax.scatter(lat, psnr, color=col, s=sz, zorder=5,
                   edgecolors='white', linewidth=1.2)
        xoff = 1.5 if 'Latency' in name else 0.8
        yoff = 0.05 if 'CNN' in name else -0.1
        ax.text(lat+xoff, psnr+yoff, name, color=col, fontsize=9.5,
                fontweight='bold', va='center')

    ax.axvline(x=5, color=GRID, lw=1, ls=':', alpha=0.5)
    ax.axhline(y=29.0, color=GRID, lw=1, ls=':', alpha=0.5)
    ax.set_xlabel('Inference Latency (ms/slice) — log scale', fontsize=11)
    ax.set_ylabel('Test PSNR (dB)', fontsize=11)
    ax.set_title('Quality vs Inference Speed Tradeoff\n← faster / higher quality →',
                 color=C_TEXT, fontsize=12, fontweight='bold')
    ax.set_xscale('log'); ax.set_xlim(0.7, 90); ax.set_ylim(27.5, 31)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Performance-Efficiency Tradeoff — Test Set',
                 color=C_TEXT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUT / 'fig9_psnr_vs_latency.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 10 — Energy estimate
# ═════════════════════════════════════════════════════════════════════════════
def fig10_energy():
    models_e = ['CNN-Final\n(GPU)', 'SNN-Direct-IF\n(neuromorphic)', 'SNN-Direct-LIF\n(neuromorphic)', 'SNN-IF Latency\n(neuromorphic)']
    energy   = [5916, 5421, 6071, 6380]
    colors_e = [C_CNN, C_DIF, C_DLF, C_LAT]
    ops_v    = [1.286, 6.023, 6.746, 7.089]
    x = np.arange(len(models_e))
    w = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(BG2); ax.grid(True, axis='y', alpha=0.35); ax.set_axisbelow(True)

    bars = ax1.bar(x, energy, w, color=colors_e, alpha=0.85)
    ax1.axhline(y=5916, color=C_CNN, lw=1.5, ls='--', alpha=0.6)
    ax1.text(3.4, 6000, 'CNN baseline\n5,916 nJ', color=C_CNN, fontsize=8, ha='right')
    for bar, v in zip(bars, energy):
        diff = v - 5916
        col_d = '#69DB7C' if diff < 0 else '#FF6B6B'
        sign  = f'{diff:+.0f} nJ'
        ax1.text(bar.get_x()+bar.get_width()/2, v+50,
                 f'{v:,}\n({sign})', ha='center', fontsize=9, color=col_d, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(models_e, fontsize=9)
    ax1.set_ylabel('Est. Energy per Inference (nJ)', fontsize=10)
    ax1.set_title('Energy per 512×512 Slice\n(Horowitz 2014: E_MAC=4.6 pJ, E_SynOp=0.9 pJ)',
                  color=C_TEXT, fontsize=10)
    ax1.set_ylim(4800, 7200)

    bars2 = ax2.bar(x, ops_v, w, color=colors_e, alpha=0.85)
    ax2.axhline(y=6.571, color='#F0C040', lw=1.5, ls='--', alpha=0.8)
    ax2.text(3.4, 6.75, 'Break-even\n6.57 GSynOps', color='#F0C040', fontsize=8.5, ha='right')
    for bar, v in zip(bars2, ops_v):
        above = v > 6.571
        col_d = '#FF6B6B' if above else '#69DB7C'
        marker = '↑ above' if above else '✓ below'
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.1,
                 f'{v:.3f}\n{marker}', ha='center', fontsize=9, color=col_d, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(models_e, fontsize=9)
    ax2.set_ylabel('Compute (GMACs or GSynOps)', fontsize=10)
    ax2.set_title('Compute Operations vs Break-even\n(SynOps/MACs ≤ 5.11 for energy saving)',
                  color=C_TEXT, fontsize=10)
    ax2.set_ylim(0, 9)

    fig.suptitle('Neuromorphic Energy Estimate — Analytical (Horowitz 2014 Model)',
                 color=C_TEXT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUT / 'fig10_energy.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# FIG 11 — Encoding comparison (latency vs direct, same slice)
# ═════════════════════════════════════════════════════════════════════════════
def fig11_encoding_comparison(r):
    """Show latency vs direct coding on same slice to illustrate encoding effect."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))
    fig.patch.set_facecolor(BG)

    panels = [
        (r['low'],  'Low-Dose Input',          C_INPUT, r['low_psnr'],  None),
        (r['cnn'],  'CNN-Final\n(Reference)',   C_CNN,   r['cnn_psnr'],  r['cnn_ssim']),
        (r['dif'],  'SNN-Direct-IF (T=8)\nDirect Coding', C_DIF, r['dif_psnr'], r['dif_ssim']),
        (r['lat'],  'SNN-IF Latency (T=16)\nLatency Encoding', C_LAT, r['lat_psnr'], r['lat_ssim']),
        (r['full'], 'Full-Dose Reference',      C_REF,   None,           None),
    ]
    for ax, (img, title, col, psnr, ssim_v) in zip(axes, panels):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, color=col, fontsize=10.5, fontweight='bold', pad=6)
        ax.axis('off')
        if psnr is not None and ssim_v is not None:
            add_metrics(ax, psnr, ssim_v, fontsize=8.5)
        elif psnr is not None:
            ax.text(0.02, 0.02, f'PSNR: {psnr:.2f} dB',
                    transform=ax.transAxes, color='white', fontsize=8.5, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    # Encoding gain annotation
    gain = r['dif_psnr'] - r['lat_psnr']
    axes[2].text(0.98, 0.98, f'Direct coding\n+{gain:.2f} dB vs Latency',
                 transform=axes[2].transAxes, color=C_DIF, fontsize=9,
                 ha='right', va='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65))

    fig.suptitle(f'Encoding Strategy Comparison — Patient {r["pid"]} Slice {r["sidx"]}\n'
                 f'Direct coding (+{gain:.2f} dB) vs Latency encoding — same architecture, same neuron model (IF)',
                 color=C_TEXT, fontsize=11, fontweight='bold', y=1.03)
    plt.tight_layout(pad=0.4)
    p = OUT / f'fig11_encoding_comparison_{r["pid"]}_s{r["sidx"]}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return p

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print('='*60)
    print('SpikeCT-Denoise — Complete Visual Evaluation')
    print('='*60)

    cnn, dif, dlf, lat = load_models()
    results = collect_all_results(cnn, dif, dlf, lat)

    print(f'\nGenerating figures → {OUT}')

    saved = []

    # Per-slice figures
    print('\n[Fig 1] 4-panel comparison...')
    for r in results:
        saved.append(fig1_comparison(r))

    print('[Fig 2] Difference maps...')
    for r in results:
        saved.append(fig2_difference(r))

    print('[Fig 3] All 4 models side by side...')
    best_slices = [r for r in results if r['sidx'] == 100][:3]
    saved.append(fig3_all_models(best_slices))

    print('[Fig 4] Combined thesis figure...')
    thesis_cases = [
        next(r for r in results if r['pid']=='C004' and r['sidx']==50),
        next(r for r in results if r['pid']=='C050' and r['sidx']==100),
        next(r for r in results if r['pid']=='C111' and r['sidx']==150),
    ]
    saved.append(fig4_combined_thesis(thesis_cases))

    print('[Fig 5] Training curves...')
    saved.append(fig5_training_curves())

    print('[Fig 6] Ablation bar chart...')
    saved.append(fig6_ablation())

    print('[Fig 7] Per-patient PSNR...')
    saved.append(fig7_per_patient(results))

    print('[Fig 8] Efficiency comparison...')
    saved.append(fig8_efficiency())

    print('[Fig 9] PSNR vs latency scatter...')
    saved.append(fig9_scatter())

    print('[Fig 10] Energy estimate...')
    saved.append(fig10_energy())

    print('[Fig 11] Encoding comparison...')
    best_r = max(results, key=lambda r: r['dif_psnr'] - r['lat_psnr'])
    saved.append(fig11_encoding_comparison(best_r))

    print(f'\n{"="*60}')
    print(f'Done. {len(saved)} figures saved to:')
    print(f'  {OUT}')
    print()
    print('Thesis-ready figures:')
    print('  fig4_combined_thesis.png    ← main visual comparison figure')
    print('  fig5_training_curves.png    ← learning dynamics, all models')
    print('  fig6_ablation_bars.png      ← ablation results')
    print('  fig7_per_patient_psnr.png   ← generalisation across patients')
    print('  fig8_efficiency.png         ← GPU efficiency metrics')
    print('  fig9_psnr_vs_latency.png    ← quality-efficiency tradeoff')
    print('  fig10_energy.png            ← neuromorphic energy estimate')
    print('  fig11_encoding_comparison_*.png ← encoding strategy effect')
    print()
    print('Per-slice figures:')
    print('  fig1_comparison_*.png       ← 4-panel: Low|CNN|DIF|Full')
    print('  fig2_difference_*.png       ← error maps with correct colorbar')
    print('  fig3_all_models_comparison.png ← 6-panel: all 4 models')

if __name__ == '__main__':
    main()