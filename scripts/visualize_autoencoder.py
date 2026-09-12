"""
Visualize autoencoder reconstruction quality on a defective test image.

Shows: original image, reconstruction, ground-truth defect mask,
reconstruction-error heatmap, and the error heatmap overlaid on the
original image. This is a QUALITATIVE check -- it lets us see whether
reconstruction error actually rises in the true defect region, before
we compute any formal evaluation metrics (Phase 12) or calibrate a
real detection threshold (Phase 6).
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_merged_config, resolve_device
from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import get_autoencoder_image_transform, get_mask_transform
from src.models.autoencoder.network import ConvAutoencoder
from src.visualization.heatmaps import denormalize_autoencoder_image, overlay_mask_on_image


def compute_error_map(original: torch.Tensor, reconstruction: torch.Tensor) -> np.ndarray:
    """
    Compute a per-pixel reconstruction error map.

    Args:
        original: (3, H, W) tensor, autoencoder-normalized ([-1,1] range).
        reconstruction: (3, H, W) tensor, same range.

    Returns:
        (H, W) numpy array of mean absolute error across channels.
    """
    diff = torch.abs(original - reconstruction)      # (3, H, W)
    error_map = diff.mean(dim=0)                      # (H, W), averaged over channels
    return error_map.detach().cpu().numpy()


def normalize_for_display(error_map: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an error map to [0,1] PURELY for visualization.

    NOTE: this is not the calibrated anomaly threshold from Phase 6 --
    it just rescales this single image's error map so it displays with
    good contrast. Do not use this normalization for actual detection.
    """
    min_val, max_val = error_map.min(), error_map.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(error_map)
    return (error_map - min_val) / (max_val - min_val)


def main(category: str, show_normal: bool = False, image_size: int = 256) -> None:
    config = load_merged_config("configs/base.yaml", "configs/autoencoder.yaml")
    device = resolve_device(config["device"])

    checkpoint_path = config["checkpoint"]["save_path"].format(category=category)

    model = ConvAutoencoder(
        image_size=image_size,
        latent_dim=config["model"]["latent_dim"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    image_transform = get_autoencoder_image_transform(image_size)
    mask_transform = get_mask_transform(image_size)

    test_ds = MVTecDataset(
        root=config["data"]["root"],
        category=category,
        split="test",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    target_label = 0 if show_normal else 1
    selected_sample = next(
        test_ds[i] for i in range(len(test_ds)) if test_ds[i]["label"] == target_label
    )

    original = selected_sample["image"].to(device)          # (3, H, W)
    ground_truth_mask = selected_sample["mask"].squeeze(0).numpy()  # (H, W)

    with torch.no_grad():
        reconstruction = model(original.unsqueeze(0)).squeeze(0)  # (3, H, W)

    error_map = compute_error_map(original, reconstruction)
    error_map_display = normalize_for_display(error_map)

    original_display = denormalize_autoencoder_image(original)
    reconstruction_display = denormalize_autoencoder_image(reconstruction)

    overlay = overlay_mask_on_image(original_display, error_map_display)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(original_display)
    axes[0].set_title(f"Original ({selected_sample['defect_type']})")

    axes[1].imshow(reconstruction_display)
    axes[1].set_title("Reconstruction")

    axes[2].imshow(ground_truth_mask, cmap="gray")
    axes[2].set_title("Ground-truth mask")

    axes[3].imshow(error_map_display, cmap="jet")
    axes[3].set_title("Reconstruction error map")

    axes[4].imshow(overlay)
    axes[4].set_title("Error overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Autoencoder reconstruction quality -- category: {category}")
    plt.tight_layout()

    suffix = "normal" if show_normal else "defective"
    output_path = f"artifacts/autoencoder_reconstruction_{category}_{suffix}.png"
    plt.savefig(output_path, dpi=120)
    print(f"Saved visualization to: {output_path}")

    plt.show()


if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    show_normal = "--normal" in sys.argv
    main(category=category, show_normal=show_normal)