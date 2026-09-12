"""
Convolutional Autoencoder architecture for anomaly detection.

Design: symmetric encoder/decoder using strided convolutions (encoder)
and transposed convolutions (decoder), with a linear bottleneck layer
that forces the network to compress each image into a fixed-size vector.

This bottleneck size is a deliberate hyperparameter, not an afterthought:
too large and the network can memorize enough detail to reconstruct
defects too; too small and it can't reconstruct normal images well
either. See training/evaluation notes for how we validate this choice.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Compresses a (3, H, W) image into a (latent_dim,) vector."""

    def __init__(self, image_size: int = 256, latent_dim: int = 256) -> None:
        super().__init__()

        # 5 stride-2 conv layers: each halves spatial size, doubles channels
        # (roughly). 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 256 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 32 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),# 16 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # After 5 halvings: image_size / 32. For image_size=256 -> 8.
        self.feature_map_size = image_size // 32
        flattened_dim = 512 * self.feature_map_size * self.feature_map_size

        self.fc = nn.Linear(flattened_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # keep batch dim, flatten rest
        latent = self.fc(x)
        return latent


class Decoder(nn.Module):
    """Reconstructs a (3, H, W) image from a (latent_dim,) vector."""

    def __init__(self, image_size: int = 256, latent_dim: int = 256) -> None:
        super().__init__()

        self.feature_map_size = image_size // 32
        flattened_dim = 512 * self.feature_map_size * self.feature_map_size

        self.fc = nn.Linear(latent_dim, flattened_dim)

        # Mirror of the encoder: 5 stride-2 transposed convs.
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 128 -> 256
            nn.Tanh(),  # output roughly in [-1, 1]; see note below
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.view(-1, 512, self.feature_map_size, self.feature_map_size)
        reconstruction = self.deconv_layers(x)
        return reconstruction


class ConvAutoencoder(nn.Module):
    """Full autoencoder: Encoder -> bottleneck -> Decoder."""

    def __init__(self, image_size: int = 256, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(image_size=image_size, latent_dim=latent_dim)
        self.decoder = Decoder(image_size=image_size, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction