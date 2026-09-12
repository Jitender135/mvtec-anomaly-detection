"""
Visualize MVTec AD samples: a normal training image, a defective test
image, its ground-truth mask, and the mask overlaid on the image.

Usage:
    python -m scripts.visualize_dataset bottle
"""

import sys

import matplotlib.pyplot as plt

from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import get_image_transform, get_mask_transform
from src.visualization.heatmaps import (
    denormalize_image,
    mask_to_numpy,
    overlay_mask_on_image,
)


def main(category: str, image_size: int = 256) -> None:
    image_transform = get_image_transform(image_size)
    mask_transform = get_mask_transform(image_size)

    train_ds = MVTecDataset(
        root="data/raw/mvtec_ad",
        category=category,
        split="train",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    test_ds = MVTecDataset(
        root="data/raw/mvtec_ad",
        category=category,
        split="test",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    # A normal training sample.
    normal_sample = train_ds[0]

    # First defective test sample found.
    defective_sample = next(
        test_ds[i] for i in range(len(test_ds)) if test_ds[i]["label"] == 1
    )

    normal_img = denormalize_image(normal_sample["image"])
    defective_img = denormalize_image(defective_sample["image"])
    defect_mask = mask_to_numpy(defective_sample["mask"])
    overlay = overlay_mask_on_image(defective_img, defect_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(normal_img)
    axes[0].set_title("Normal (train)")

    axes[1].imshow(defective_img)
    axes[1].set_title(f"Defective (test): {defective_sample['defect_type']}")

    axes[2].imshow(defect_mask, cmap="gray")
    axes[2].set_title("Ground-truth mask")

    axes[3].imshow(overlay)
    axes[3].set_title("Mask overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"MVTec AD -- category: {category}")
    plt.tight_layout()

    output_path = f"artifacts/dataset_preview_{category}.png"
    plt.savefig(output_path, dpi=120)
    print(f"Saved visualization to: {output_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.visualize_dataset <category>")
        sys.exit(1)

    main(category=sys.argv[1])