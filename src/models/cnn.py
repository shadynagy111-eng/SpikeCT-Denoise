"""
src/models/cnn.py
-----------------
CNN-Final: 4+4 symmetric Convolutional Denoising Autoencoder.

Architecture (FROZEN):
  Encoder: Conv2d  1 -> 32 -> 64 -> 96 -> 96  (stride=2, kernel=3)
  Decoder: ConvTranspose2d 96 -> 96 -> 64 -> 32 -> 1 (stride=2, kernel=3)

Properties:
  Parameters:       314,401
  Input shape:      (B, 1, 512, 512)
  Output shape:     (B, 1, 512, 512)
  No batch norm, no skip connections, no pooling.
  Linear output — do NOT apply sigmoid during training.
  Clamp output to [0, 1] at evaluation/inference only.
"""

import torch
import torch.nn as nn


class CNNFinal(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 512, 512) normalized low-dose CT, values in [0, 1]
        Returns:
            (B, 1, 512, 512) reconstructed full-dose CT, linear output
        """
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def verify_architecture(device: torch.device = torch.device("cpu")) -> None:
    """
    Instantiate the model, verify parameter count and output shape.
    Raises AssertionError if anything is wrong.
    Call this once at the start of training to confirm integrity.
    """
    model = CNNFinal().to(device)
    n = count_parameters(model)
    assert 280_000 <= n <= 330_000, \
        f"Parameter count {n} outside expected band [280k, 330k]"

    x = torch.zeros(1, 1, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape, \
        f"Output shape {y.shape} does not match input shape {x.shape}"

    print(f"CNNFinal verified: {n:,} parameters, "
          f"input {tuple(x.shape)} -> output {tuple(y.shape)}")