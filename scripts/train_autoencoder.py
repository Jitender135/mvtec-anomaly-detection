"""
Training script for the baseline convolutional autoencoder.

Trains exclusively on train/good images for one MVTec AD category.
Holds out a validation slice of train/good (never used for gradient
updates) purely to monitor for overfitting during training. This is
DIFFERENT from threshold calibration, which happens later using the
test/good split (see Phase 6 / threshold calibration).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import load_merged_config, resolve_device
from src.utils.seed import set_seed
from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import get_autoencoder_image_transform, get_mask_transform
from src.models.autoencoder.network import ConvAutoencoder


def main() -> None:
    config = load_merged_config("configs/base.yaml", "configs/autoencoder.yaml")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    print(f"Using device: {device}")

    category = config["category"]
    image_size = config["data"]["image_size"]

    image_transform = get_autoencoder_image_transform(image_size)
    mask_transform = get_mask_transform(image_size)

    full_train_ds = MVTecDataset(
        root=config["data"]["root"],
        category=category,
        split="train",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    val_fraction = config["training"]["val_split"]
    val_size = int(len(full_train_ds) * val_fraction)
    train_size = len(full_train_ds) - val_size

    # Seeded generator ensures the split is reproducible across runs.
    generator = torch.Generator().manual_seed(config["seed"])
    train_subset, val_subset = random_split(
        full_train_ds, [train_size, val_size], generator=generator
    )

    print(f"Train samples: {train_size} | Validation samples: {val_size}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = ConvAutoencoder(
        image_size=image_size,
        latent_dim=config["model"]["latent_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    num_epochs = config["training"]["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)

        avg_train_loss = train_loss_total / train_size

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                reconstructions = model(images)
                loss = criterion(reconstructions, images)
                val_loss_total += loss.item() * images.size(0)

        avg_val_loss = val_loss_total / val_size

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train loss: {avg_train_loss:.6f} | "
              f"Val loss: {avg_val_loss:.6f}")

    checkpoint_path = config["checkpoint"]["save_path"].format(category=category)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()