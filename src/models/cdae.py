"""
Convolutional Denoising Autoencoder (CDAE)
Baseline CNN for CT image denoising
Input: [B, 1, 512, 512]
"""

import torch
import torch.nn as nn


class CDAE(nn.Module):
    """
    Symmetric Convolutional Denoising Autoencoder
    Encoder: 3x Conv2d with stride 2 (downsampling)
    Decoder: 3x ConvTranspose2d with stride 2 (upsampling)
    No skip connections, no batchnorm — minimal baseline
    """

    def __init__(self, in_channels: int = 1):
        super(CDAE, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # DECODER
        self.decoder = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            # Layer 2: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            # Layer 3: 32 -> 1
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CDAE()
    print(f"Parameters: {model.get_num_params():,}")
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert x.shape == y.shape
    print("✓ Model test passed")