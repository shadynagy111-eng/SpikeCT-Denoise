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
# Gallery generation
# ─────────────────────────────────────────────────────────────

def make_gallery_items(
    input_img,
    predictions,
    reference=None
):
    """
    Returns a list of (image, label) tuples for gr.Gallery.
    """
    items = []

    # Helper to convert grayscale float32 [0,1] to uint8 [0,255]
    def to_uint8(img):
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Input
    items.append((to_uint8(input_img), "Low-Dose Input"))

    # Predictions
    for name, pred, elapsed in predictions:
        items.append((to_uint8(pred), f"{name} ({elapsed:.0f}ms)"))

    # Reference
    if reference is not None:
        items.append((to_uint8(reference), "Full-Dose Reference"))

    return items

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
            "### Evaluation Summary (No Reference Provided)\n",
            "> **Note:** Without a reference image, PSNR and SSIM gains cannot be calculated. ",
            "The model output is purely based on the low-dose input.\n"
        ]

        for name, _, elapsed in predictions:
            lines.append(
                f"- **{name}**: {elapsed:.1f} ms"
            )

        return "\n".join(lines)

    p_in = psnr_fn(
        reference,
        input_img,
        data_range=1.0
    )

    lines = [
        f"### Evaluation Summary",
        f"**Input PSNR:** {p_in:.2f} dB",
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
            f"#### {name}",
            f"- **PSNR:** {p:.4f} dB ({gain:+.2f} dB gain)",
            f"- **SSIM:** {s:.4f}",
            f"- **Time:** {elapsed:.1f} ms",
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

    gallery = make_gallery_items(
        arr,
        predictions,
        reference
    )

    metrics = make_metrics_text(
        predictions,
        reference,
        arr
    )

    return gallery, metrics

# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────

MODEL_CHOICES = list(MODELS.keys())

with gr.Blocks(
    title="SpikeCT-Denoise",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        "# SpikeCT-Denoise\n"
        "Low-Dose CT Restoration — CNN vs Spiking Neural Networks"
    )

    with gr.Row():

        with gr.Column(scale=1):

            with gr.Group():
                gr.Markdown("### 1. Upload Data")
                input_file = gr.File(
                    label="Low-dose input (.npy)",
                    file_types=[".npy"]
                )

                reference_file = gr.File(
                    label="Full-dose reference (.npy)",
                    file_types=[".npy"],
                )
                gr.Markdown(
                    "*The reference is **optional**. If provided, it is only used to calculate "
                    "quality metrics (PSNR/SSIM) and compare results. It does not affect the model output.*"
                )

            with gr.Group():
                gr.Markdown("### 2. Select Models")
                model_selector = gr.CheckboxGroup(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES,
                    label="Active Models"
                )

            run_btn = gr.Button(
                "🚀 Run Denoising",
                variant="primary"
            )

        with gr.Column(scale=2):
            
            output_gallery = gr.Gallery(
                label="Denoising Results",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )

            output_metrics = gr.Markdown()

    with gr.Accordion("Technical Details & Metric Explanations", open=False):
        gr.Markdown(
            "### How it works\n"
            "- **Training:** These models were trained using pairs of Low-Dose and Full-Dose images. "
            "The Full-Dose image acted as the 'Ground Truth'.\n"
            "- **Inference (Demo):** The model only sees the Low-Dose input. It uses its learned weights "
            "to predict the clean image.\n"
            "- **PSNR (Peak Signal-to-Noise Ratio):** Measures the ratio between the maximum possible power "
            "of a signal and the power of corrupting noise. Higher is better.\n"
            "- **SSIM (Structural Similarity Index):** Measures the similarity between two images based on "
            "luminance, contrast, and structure. Ranges from 0 to 1; higher is better."
        )

    run_btn.click(
        fn=denoise,
        inputs=[
            input_file,
            reference_file,
            model_selector
        ],
        outputs=[
            output_gallery,
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