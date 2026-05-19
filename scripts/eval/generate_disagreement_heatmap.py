"""
generate_disagreement_heatmap.py
=================================
SpikeCT-Denoise — Inter-Model Disagreement Heatmap

Inference code copied exactly from scripts/eval/evaluate_all_models.py.
Data loading uses your project dataset classes — no custom preprocessing.

Run from project root:
    cd /mnt/c/Projects/SpikeCT-Denoise
    source venv/bin/activate
    python scripts/eval/generate_disagreement_heatmap.py
    python scripts/eval/generate_disagreement_heatmap.py --patient C050 --slice 100

Output: outputs/figures/disagreement_heatmap_{patient}_s{slice}.png
        outputs/figures/disagreement_stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ── Project root (same logic as evaluate_all_models.py) ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.models.cnn import CNNFinal
from src.models.snn import SNNFinalIF
from src.models.snn_direct import SNNDirectIF, SNNDirectLIF
from spikingjelly.activation_based import functional

set_seed(42)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_FILE    = Path.home() / "SpikeCT_Data/mayo_50patients_fast.h5"

CHECKPOINTS = {
    "CNN-Final":      PROJECT_ROOT / "checkpoints" / "cnn_50_best.pth",
    "SNN-IF Latency": PROJECT_ROOT / "checkpoints" / "snn_if_50_best.pth",
    "SNN-Direct-IF":  PROJECT_ROOT / "checkpoints" / "snn_direct_if_best.pth",
    "SNN-Direct-LIF": PROJECT_ROOT / "checkpoints" / "snn_direct_lif_best.pth",
}

# ── Encoding (copied from src/data/dataset.py pattern) ────────────────────────

def latency_encode_single(image: torch.Tensor, T: int = 16) -> torch.Tensor:
    """
    image  : (1, H, W) float32 in [0,1]  — already preprocessed
    returns: (T, 1, H, W) binary spike tensor
    Matches latency_encode() in src/data/dataset.py
    """
    t_spike = torch.floor((1.0 - image) * (T - 1)).long().clamp(0, T - 1)
    spikes = torch.zeros(T, *image.shape, device=image.device)
    for t in range(T):
        spikes[t] = (t_spike == t).float()
    return spikes  # (T, 1, H, W)


def direct_encode_single(image: torch.Tensor, T: int = 8) -> torch.Tensor:
    """
    image  : (1, H, W) float32 in [0,1]  — already preprocessed
    returns: (T, 1, H, W)
    Matches direct_encode() in src/models/snn_direct.py
    """
    return image.unsqueeze(0).expand(T, -1, -1, -1).contiguous()  # (T, 1, H, W)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(name: str) -> torch.nn.Module:
    ckpt_path = CHECKPOINTS[name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_cls = {
        "CNN-Final":      CNNFinal,
        "SNN-IF Latency": SNNFinalIF,
        "SNN-Direct-IF":  SNNDirectIF,
        "SNN-Direct-LIF": SNNDirectLIF,
    }
    model = model_cls[name]()

    state = torch.load(str(ckpt_path), map_location=DEVICE)
    # Unwrap checkpoint dict if needed
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ── Inference (shape logic copied from evaluate_all_models.py) ────────────────

@torch.no_grad()
def run_inference(name: str, model: torch.nn.Module,
                  image_np: np.ndarray) -> np.ndarray:
    """
    image_np: (H, W) float32 already in [0,1] — straight from HDF5
    returns : (H, W) float32 in [0,1]

    Shape conventions (matching evaluate_all_models.py):
      CNN       : input  (1, 1, H, W)
      SNN Lat   : input  (T, 1, 1, H, W)  — (T, B, C, H, W)
      SNN Direct: input  (T, 1, 1, H, W)  — (T, B, C, H, W)
    """
    # (H,W) → (1, H, W) → add batch → (1, 1, H, W)
    img = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

    if name == "CNN-Final":
        with torch.amp.autocast('cuda'):
            pred = model(img)                          # (1, 1, H, W)

    elif name == "SNN-IF Latency":
        functional.reset_net(model)
        # encode: (1, H, W) → (T, 1, H, W) → add batch dim → (T, 1, 1, H, W)
        spikes = latency_encode_single(img.squeeze(0), T=16)  # (T, 1, H, W)
        spikes = spikes.unsqueeze(1)                           # (T, 1, 1, H, W)
        with torch.amp.autocast('cuda'):
            pred = model(spikes)                       # (1, 1, H, W)
        functional.reset_net(model)

    elif name in ("SNN-Direct-IF", "SNN-Direct-LIF"):
        functional.reset_net(model)
        # encode: (1, H, W) → (T, 1, H, W) → add batch dim → (T, 1, 1, H, W)
        encoded = direct_encode_single(img.squeeze(0), T=8)   # (T, 1, H, W)
        encoded = encoded.unsqueeze(1)                         # (T, 1, 1, H, W)
        with torch.amp.autocast('cuda'):
            pred = model(encoded)                      # (1, 1, H, W)
        functional.reset_net(model)

    else:
        raise ValueError(f"Unknown model: {name}")

    return pred.squeeze().cpu().float().numpy().clip(0.0, 1.0).astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient",  default="C004",
                        choices=["C004","C050","C111","C121","C249"])
    parser.add_argument("--slice",    type=int, default=50)
    parser.add_argument("--save_dir", default=str(PROJECT_ROOT / "outputs" / "figures"))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("SpikeCT — Inter-Model Disagreement Heatmap")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Device       : {DEVICE}")
    print(f"Patient      : {args.patient}  |  Slice: {args.slice}")
    print("=" * 60)

    # ── Load slice — data already in [0,1], no preprocessing needed ───────────
    print("\nLoading CT slice (data already preprocessed in HDF5)...")
    with h5py.File(str(H5_FILE), "r") as f:
        low_np  = f[args.patient]["low_dose"][args.slice].astype(np.float32)
        full_np = f[args.patient]["full_dose"][args.slice].astype(np.float32)

    print(f"  low  — min:{low_np.min():.4f} max:{low_np.max():.4f} mean:{low_np.mean():.4f}")
    print(f"  full — min:{full_np.min():.4f} max:{full_np.max():.4f} mean:{full_np.mean():.4f}")

    input_psnr = float(psnr_fn(full_np, low_np, data_range=1.0))
    input_ssim = float(ssim_fn(full_np, low_np, data_range=1.0))
    print(f"  Input PSNR={input_psnr:.4f} dB  SSIM={input_ssim:.4f}\n")

    # ── Run all four models ────────────────────────────────────────────────────
    model_names = ["CNN-Final", "SNN-IF Latency", "SNN-Direct-IF", "SNN-Direct-LIF"]
    outputs, metrics = {}, {}

    for name in model_names:
        print(f"  [{name}] loading checkpoint & running inference...")
        model = load_model(name)
        pred  = run_inference(name, model, low_np)
        outputs[name] = pred

        p = float(psnr_fn(full_np, pred, data_range=1.0))
        s = float(ssim_fn(full_np, pred, data_range=1.0))
        metrics[name] = {"psnr": p, "ssim": s}
        print(f"  [{name}] PSNR={p:.4f} dB  SSIM={s:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Disagreement map ───────────────────────────────────────────────────────
    stack        = np.stack(list(outputs.values()), axis=0)  # (4, H, W)
    disagreement = np.std(stack, axis=0)                      # (H, W)

    stats = {k: float(v) for k, v in zip(
        ["mean","std","min","max","p50","p95","p99"],
        [disagreement.mean(), disagreement.std(), disagreement.min(),
         disagreement.max(),
         np.percentile(disagreement, 50),
         np.percentile(disagreement, 95),
         np.percentile(disagreement, 99)]
    )}

    # ── Figure: 3 rows × 3 cols ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 19))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes.flat:
        ax.set_facecolor("#0f0f1a")
    tc = "white"

    fig.suptitle(
        f"Inter-Model Disagreement  |  Patient {args.patient}  Slice {args.slice}\n"
        "Pixel σ across: CNN-Final · SNN-IF Latency · SNN-Direct-IF · SNN-Direct-LIF",
        fontsize=12, fontweight="bold", color=tc, y=1.005
    )

    def show_ct(ax, img, title):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color=tc, pad=5)
        ax.axis("off")

    def show_diff(ax, pred, ref, title):
        diff = np.abs(pred - ref)
        im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=0.15)
        ax.set_title(title, fontsize=9, color=tc, pad=5)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=tc)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=tc)

    # Row 0: input | reference | disagreement heatmap
    show_ct(axes[0,0], low_np,
            f"Low-Dose Input\nPSNR {input_psnr:.2f} dB  SSIM {input_ssim:.4f}")
    show_ct(axes[0,1], full_np, "Full-Dose Reference (Ground Truth)")

    vmax_d = max(stats["p99"], 1e-8)
    im_d = axes[0,2].imshow(disagreement, cmap="hot", vmin=0, vmax=vmax_d)
    axes[0,2].set_title(
        f"Disagreement Map (pixel-wise σ)\n"
        f"mean={stats['mean']:.4f}  p95={stats['p95']:.4f}  max={stats['max']:.4f}",
        fontsize=9, color=tc, pad=5)
    axes[0,2].axis("off")
    cb = plt.colorbar(im_d, ax=axes[0,2], fraction=0.046, pad=0.04)
    cb.set_label("σ (normalised HU)", color=tc, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=tc)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=tc)

    # Row 1: CNN | Direct-IF | Direct-LIF
    for i, name in enumerate(["CNN-Final", "SNN-Direct-IF", "SNN-Direct-LIF"]):
        p, s = metrics[name]["psnr"], metrics[name]["ssim"]
        show_ct(axes[1,i], outputs[name],
                f"{name}\nPSNR {p:.2f} dB  SSIM {s:.4f}")

    # Row 2: Latency | error CNN | error Direct-IF
    p, s = metrics["SNN-IF Latency"]["psnr"], metrics["SNN-IF Latency"]["ssim"]
    show_ct(axes[2,0], outputs["SNN-IF Latency"],
            f"SNN-IF Latency\nPSNR {p:.2f} dB  SSIM {s:.4f}")
    show_diff(axes[2,1], outputs["CNN-Final"], full_np,
              "Error Map: |CNN-Final − Reference|")
    show_diff(axes[2,2], outputs["SNN-Direct-IF"], full_np,
              "Error Map: |SNN-Direct-IF − Reference|")

    plt.tight_layout()
    out_img = os.path.join(args.save_dir,
        f"disagreement_heatmap_{args.patient}_s{args.slice}.png")
    plt.savefig(out_img, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nFigure → {out_img}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    result = {
        "patient": args.patient, "slice": args.slice,
        "seed": 42, "device": str(DEVICE),
        "input_psnr": input_psnr, "input_ssim": input_ssim,
        "model_metrics": metrics,
        "disagreement_stats": stats,
        "note": (
            "Disagreement = pixel-wise std across 4 deterministic model outputs. "
            "Inter-model disagreement proxy, NOT Bayesian uncertainty. "
            "All models deterministic (seed=42, no dropout). "
            "Data loaded directly from HDF5 — already in [0,1], no preprocessing applied."
        )
    }
    out_json = os.path.join(args.save_dir, "disagreement_stats.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Stats  → {out_json}")

    # ── Console summary ────────────────────────────────────────────────────────
    print()
    print(f"{'Model':<22} {'PSNR (dB)':>10} {'SSIM':>7}")
    print("-" * 42)
    print(f"{'Low-Dose Input':<22} {input_psnr:>10.4f} {input_ssim:>7.4f}")
    for n in model_names:
        print(f"{n:<22} {metrics[n]['psnr']:>10.4f} {metrics[n]['ssim']:>7.4f}")
    print()
    print("Disagreement statistics:")
    for k, v in stats.items():
        print(f"  {k:>5}: {v:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()