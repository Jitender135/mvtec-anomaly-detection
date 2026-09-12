"""
Throwaway test script (not part of the permanent architecture).

Purpose: verify that PyTorch can build a small CNN, move it to the
correct device, and run a forward pass on a realistic-shaped input
BEFORE we write any real model code.
"""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """A minimal CNN, used only to sanity-check the PyTorch pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Simulate a batch of 1 RGB image, 256x256 pixels.
    # Shape convention in PyTorch: (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    print(f"Input shape : {tuple(dummy_input.shape)}")

    model = TinyCNN().to(device)

    output = model(dummy_input)
    print(f"Output shape: {tuple(output.shape)}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {num_params}")

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Output is on device: {output.device}")


if __name__ == "__main__":
    main()