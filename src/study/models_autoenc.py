from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FragmentAutoencoder(nn.Module):
    """A lightweight CNN autoencoder for 16x16 RGB fragments.

    Encoder outputs an embedding vector; decoder reconstructs the patch.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(128 * 4 * 4, embedding_dim)
        # Decoder
        self.dec_fc = nn.Linear(embedding_dim, 128 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)
        h = h.view(h.size(0), 128, 4, 4)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
