"""
app/demo.py
-----------
SpikeCT-Denoise — Interactive Demo

Run:
    python app/demo.py

Opens:
    http://localhost:7860

Input:
    .npy file, float32 array
    shape (512,512) or (N,512,512)
    normalized to [0,1]

Models:
    - CNN-Final
    - SNN-Direct-IF (T=8)
    - SNN-Direct-LIF (T=8)
    - SNN-IF Latency (T=16)
"""

import time
import sys
from pathlib import Path

import numpy as np
import torch
import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ─────────────────────────────────────────────────────────────
# Project path
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────

from src.models.cnn import CNNFinal
from src.models.snn import SNNFinalIF, SNNFinalLIF
from src.data.dataset import latency_encode
from spikingjelly.activation_based import functional

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ─────────────────────────────────────────────────────────────
# Direct encoding
# ─────────────────────────────────────────────────────────────

def direct_encode(x: torch.Tensor, T: int = 8):
    """
    Direct/rate encoding.

    Input:
        (B,H,W)

    Output:
        (T,B,H,W)
    """
    return x.unsqueeze(0).repeat(T, 1, 1, 1)

# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_model(model_cls, ckpt_path, **kwargs):

    model = model_cls(**kwargs).to(DEVICE)

    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model

print("Loading models...")

MODELS = {}

ckpt_dir = PROJECT_ROOT / "checkpoints"

# CNN
try:

    cnn_ckpt = (
        ckpt_dir / "cnn_final_best.pth"
        if (ckpt_dir / "cnn_final_best.pth").exists()
        else ckpt_dir / "cnn_50_best.pth"
    )

    MODELS["CNN-Final"] = load_model(
        CNNFinal,
        cnn_ckpt
    )

    print("  ✓ CNN-Final loaded")

except Exception as e:
    print(f"  ✗ CNN-Final: {e}")

# SNN Direct IF
try:

    MODELS["SNN-Direct-IF (T=8)"] = load_model(
        SNNFinalIF,
        ckpt_dir / "snn_direct_if_best.pth",
        T=8
    )

    print("  ✓ SNN-Direct-IF loaded")

except Exception as e:
    print(f"  ✗ SNN-Direct-IF: {e}")

# SNN Direct LIF
try:

    MODELS["SNN-Direct-LIF (T=8)"] = load_model(
        SNNFinalLIF,
        ckpt_dir / "snn_direct_lif_best.pth",
        T=8
    )

    print("  ✓ SNN-Direct-LIF loaded")

except Exception as e:
    print(f"  ✗ SNN-Direct-LIF: {e}")

# SNN Latency IF
try:

    MODELS["SNN-IF Latency (T=16)"] = load_model(
        SNNFinalIF,
        ckpt_dir / "snn_if_50_best.pth",
        T=16
    )

    print("  ✓ SNN-IF Latency loaded")

except Exception as e:
    print(f"  ✗ SNN-IF Latency: {e}")

print(f"\nDevice: {DEVICE}")
print(f"Models ready: {list(MODELS.keys())}\n")

# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(model, model_name, img_np):

    t0 = time.time()

    # CNN
    if model_name == "CNN-Final":

        x = (
            torch.from_numpy(img_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        with torch.no_grad():

            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(x)
            else:
                pred = model(x)

    # Direct coding SNNs
    elif "Direct" in model_name:

        img_t = (
            torch.from_numpy(img_np)
            .unsqueeze(0)
            .float()
        )  # (B,H,W)

        T = 8

        spikes = (
            direct_encode(img_t, T=T)
            .unsqueeze(2)
            .to(DEVICE)
        )  # (T,B,1,H,W)

        functional.reset_net(model)

        with torch.no_grad():

            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(spikes)
            else:
                pred = model(spikes)

    # Latency coding
    elif "Latency" in model_name:

        img_t = (
            torch.from_numpy(img_np)
            .unsqueeze(0)
            .float()
        )

        T = 16

        spikes = (
            latency_encode(img_t, T=T)
            .unsqueeze(2)
            .to(DEVICE)
        )

        functional.reset_net(model)

        with torch.no_grad():

            if DEVICE.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(spikes)
            else:
                pred = model(spikes)

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (
        time.time() - t0
    ) * 1000

    pred_np = (
        pred.squeeze()
        .detach()
        .cpu()
        .float()
        .numpy()
        .clip(0, 1)
    )

    return pred_np, elapsed_ms

# ─────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────

def make_figure(
    input_img,
    predictions,
    reference=None
):

    has_ref = reference is not None

    n_cols = (
        1 +
        len(predictions) +
        (1 if has_ref else 0)
    )

    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(5 * n_cols, 5)
    )

    if n_cols == 1:
        axes = [axes]

    col = 0

    # Input
    axes[col].imshow(
        input_img,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[col].set_title(
        "Low-Dose Input"
    )

    axes[col].axis("off")

    col += 1

    # Predictions
    for name, pred, elapsed in predictions:

        ax = axes[col]

        ax.imshow(
            pred,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(name)

        ax.axis("off")

        if has_ref:

            p = psnr_fn(
                reference,
                pred,
                data_range=1.0
            )

            s = ssim_fn(
                reference,
                pred,
                data_range=1.0
            )

            txt = (
                f"PSNR: {p:.2f} dB\n"
                f"SSIM: {s:.3f}\n"
                f"{elapsed:.0f} ms"
            )

        else:

            txt = f"{elapsed:.0f} ms"

        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round",
                facecolor="black",
                alpha=0.7
            )
        )

        col += 1

    # Reference
    if has_ref:

        axes[col].imshow(
            reference,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[col].set_title(
            "Full-Dose Reference"
        )

        axes[col].axis("off")

    plt.tight_layout()

    return fig

# ─────────────────────────────────────────────────────────────
# Metrics text
# ─────────────────────────────────────────────────────────────

def make_metrics_text(
    predictions,
    reference,
    input_img
):

    if reference is None:

        lines = [
            "No reference image provided.\n"
        ]

        for name, _, elapsed in predictions:
            lines.append(
                f"- {name}: {elapsed:.1f} ms"
            )

        return "\n".join(lines)

    p_in = psnr_fn(
        reference,
        input_img,
        data_range=1.0
    )

    lines = [
        f"Input PSNR: {p_in:.2f} dB",
        ""
    ]

    for name, pred, elapsed in predictions:

        p = psnr_fn(
            reference,
            pred,
            data_range=1.0
        )

        s = ssim_fn(
            reference,
            pred,
            data_range=1.0
        )

        gain = p - p_in

        lines.extend([
            f"{name}",
            f"  PSNR: {p:.4f} dB",
            f"  Gain: {gain:+.2f} dB",
            f"  SSIM: {s:.4f}",
            f"  Time: {elapsed:.1f} ms",
            ""
        ])

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────

def denoise(
    input_file,
    reference_file,
    selected_models
):

    if input_file is None:
        return None, "Upload an input file."

    if not selected_models:
        return None, "Select at least one model."

    try:

        arr = np.load(
            input_file.name
        ).astype(np.float32)

        if arr.ndim == 3:
            arr = arr[
                arr.shape[0] // 2
            ]

        if arr.shape != (512, 512):
            return None, (
                f"Expected (512,512), got {arr.shape}"
            )

    except Exception as e:

        return None, f"Input error: {e}"

    reference = None

    if reference_file is not None:

        try:

            reference = np.load(
                reference_file.name
            ).astype(np.float32)

            if reference.ndim == 3:
                reference = reference[
                    reference.shape[0] // 2
                ]

        except:
            reference = None

    predictions = []

    for name in selected_models:

        try:

            pred, elapsed = run_inference(
                MODELS[name],
                name,
                arr
            )

            predictions.append(
                (name, pred, elapsed)
            )

        except Exception as e:

            print(f"{name}: {e}")

    if len(predictions) == 0:
        return None, "No models ran."

    fig = make_figure(
        arr,
        predictions,
        reference
    )

    metrics = make_metrics_text(
        predictions,
        reference,
        arr
    )

    return fig, metrics

# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

MODEL_CHOICES = list(MODELS.keys())

with gr.Blocks(
    title="SpikeCT-Denoise"
) as demo:

    gr.Markdown(
        "# SpikeCT-Denoise\n"
        "Low-Dose CT Restoration — "
        "CNN vs Spiking Neural Networks"
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_file = gr.File(
                label="Low-dose input (.npy)",
                file_types=[".npy"]
            )

            reference_file = gr.File(
                label="Full-dose reference (.npy)",
                file_types=[".npy"]
            )

            model_selector = gr.CheckboxGroup(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES,
                label="Models"
            )

            run_btn = gr.Button(
                "Run Denoising",
                variant="primary"
            )

        with gr.Column(scale=3):

            output_plot = gr.Plot(
                label="Denoising Comparison"
            )

            output_metrics = gr.Markdown()

    run_btn.click(
        fn=denoise,
        inputs=[
            input_file,
            reference_file,
            model_selector
        ],
        outputs=[
            output_plot,
            output_metrics
        ]
    )

# ─────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )