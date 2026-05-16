"""
src/models/snn.py
-----------------
SNN-Final: Spiking version of the 4+4 symmetric CDAE.

Identical convolutional backbone to CNNFinal:
  1 -> 32 -> 64 -> 96 -> 96 -> 96 -> 64 -> 32 -> 1

ReLU activations replaced with spiking neurons (IF or LIF).
Final layer uses an integrate-without-fire membrane accumulator
(readout at last timestep), as in SPIDEN (Castagnetti 2023).

Two variants:
  SNNFinalIF:  Integrate-and-Fire neurons (no leak)
               Uniform quantization. Compatible with late-weighted
               latency encoding. Recommended primary model.
               Citation: SPIDEN 2023 (ATIF/IF outperforms LIF on denoising)

  SNNFinalLIF: Leaky Integrate-and-Fire neurons
               Non-uniform quantization due to leak factor.
               Included for ablation comparison.
               Citation: SPIDEN 2023 (DnSNN-LIF ~1 dB below ATIF)

Training:
  Input:  (T, B, 1, 512, 512) spike tensor from latency encoding
  Target: (B, 1, 512, 512)    normalized full-dose CT
  Output: (B, 1, 512, 512)    continuous membrane readout
  Loss:   MSE on output vs target
  Method: BPTT with piecewise linear surrogate gradient (Wu 2018, Neftci 2019)

Parameters: 314,401 (identical to CNNFinal — neurons have no learned params)
T:          30 timesteps (frozen from encoding validation)
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate

# ATan surrogate gradient — smooth, numerically stable, available in
# SpikingJelly 0.0.0.0.14. Used for both IF and LIF variants.
# Citation: Neftci 2019 (surrogate gradient methods for SNNs)
_SURROGATE = surrogate.ATan()


# ── IF variant ────────────────────────────────────────────────────────────────

class SNNFinalIF(nn.Module):
    """
    SNN with Integrate-and-Fire neurons (no leak).
    Membrane potential accumulates without decay.
    Implements uniform quantization — most compatible with
    late-weighted latency-encoded CT data.
    Primary model. Citation: SPIDEN 2023.
    """

    def __init__(self, T: int = 30, v_threshold: float = 1.0):
        super().__init__()
        self.T = T

        # Encoder
        self.enc1_conv = nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1)
        self.enc1_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)
        self.enc2_conv = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc2_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)
        self.enc3_conv = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.enc3_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)
        self.enc4_conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.enc4_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)

        # Decoder
        self.dec1_conv = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.dec1_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)
        self.dec2_conv = nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.dec2_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)
        self.dec3_conv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.dec3_if   = neuron.IFNode(v_threshold=v_threshold,
                                       surrogate_function=_SURROGATE,
                                       detach_reset=True)

        # Final conv — no spiking neuron
        self.dec4_conv = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)

        # Integrate-without-fire readout: v_threshold=inf → never fires,
        # acts as pure membrane accumulator. Read .v at last timestep.
        self.readout = neuron.IFNode(v_threshold=float('inf'),
                                     surrogate_function=_SURROGATE)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_input: (T, B, 1, 512, 512)
        Returns:
            (B, 1, 512, 512) — membrane readout normalized by T
        """
        functional.reset_net(self)
        for t in range(spike_input.shape[0]):
            x = spike_input[t]
            x = self.enc1_if(self.enc1_conv(x))
            x = self.enc2_if(self.enc2_conv(x))
            x = self.enc3_if(self.enc3_conv(x))
            x = self.enc4_if(self.enc4_conv(x))
            x = self.dec1_if(self.dec1_conv(x))
            x = self.dec2_if(self.dec2_conv(x))
            x = self.dec3_if(self.dec3_conv(x))
            x = self.dec4_conv(x)
            self.readout(x)
        return self.readout.v / self.T


# ── LIF variant ───────────────────────────────────────────────────────────────

class SNNFinalLIF(nn.Module):
    """
    SNN with Leaky Integrate-and-Fire neurons.
    Membrane potential decays between timesteps.
    tau=2.0 → decay factor beta ≈ 0.607 (moderately slow decay).
    Ablation comparison against IF variant.
    Citation: SPIDEN 2023 (LIF ~1 dB below IF on denoising tasks).
    """

    def __init__(self, T: int = 30, v_threshold: float = 1.0, tau: float = 2.0):
        super().__init__()
        self.T = T

        # Encoder
        self.enc1_conv = nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1)
        self.enc1_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)
        self.enc2_conv = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc2_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)
        self.enc3_conv = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.enc3_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)
        self.enc4_conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.enc4_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)

        # Decoder
        self.dec1_conv = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.dec1_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)
        self.dec2_conv = nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.dec2_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)
        self.dec3_conv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.dec3_lif  = neuron.LIFNode(tau=tau, v_threshold=v_threshold,
                                        surrogate_function=_SURROGATE,
                                        detach_reset=True)

        self.dec4_conv = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)
        self.readout   = neuron.IFNode(v_threshold=float('inf'),
                                       surrogate_function=_SURROGATE)

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_input: (T, B, 1, 512, 512)
        Returns:
            (B, 1, 512, 512)
        """
        functional.reset_net(self)
        for t in range(spike_input.shape[0]):
            x = spike_input[t]
            x = self.enc1_lif(self.enc1_conv(x))
            x = self.enc2_lif(self.enc2_conv(x))
            x = self.enc3_lif(self.enc3_conv(x))
            x = self.enc4_lif(self.enc4_conv(x))
            x = self.dec1_lif(self.dec1_conv(x))
            x = self.dec2_lif(self.dec2_conv(x))
            x = self.dec3_lif(self.dec3_conv(x))
            x = self.dec4_conv(x)
            self.readout(x)
        return self.readout.v / self.T


# ── Shared utilities ──────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count learnable parameters. Spiking neurons contribute none."""
    return sum(p.numel() for p in model.parameters())


def verify_snn(model: nn.Module, T: int = 30,
               device: torch.device = torch.device("cpu")) -> None:
    """Verify SNN output shape and parameter count with a dummy forward pass."""
    model = model.to(device)
    n = count_parameters(model)
    assert 280_000 <= n <= 330_000, \
        f"Parameter count {n} outside expected band [280k, 330k]"
    x = torch.zeros(T, 1, 1, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 512, 512), \
        f"Output shape {y.shape}, expected (1, 1, 512, 512)"
    print(f"{model.__class__.__name__} verified: {n:,} parameters, "
          f"input (T={T}, B=1, 1, 512, 512) -> output {tuple(y.shape)}")