"""
Run the trained autoencoder across the FULL test set for one category,
computing an image-level anomaly score per image via top-K reconstruction
error aggregation. Also scores the held-out VALIDATION split (from
training) separately -- these are genuinely normal, never-trained-on
images, and are the methodologically correct basis for percentile-based
threshold calibration (see src/evaluation/threshold.py).

Saves two CSVs:
  - artifacts/autoencoder_scores_{category}.csv       (test set)
  - artifacts/autoencoder_scores_{category}_val.csv    (validation split)
"""

import sys

import pandas as pd
import torch
from torch.utils.data import random_split

from src.config import load_merged_config, resolve_device
from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import get_autoencoder_image_transform, get_mask_transform
from src.models.autoencoder.network import ConvAutoencoder
from src.evaluation.metrics import compute_image_score_topk
from scripts.visualize_autoencoder import compute_error_map


def load_model(category: str, config: dict, image_size: int, device: str) -> ConvAutoencoder:
    checkpoint_path = config["checkpoint"]["save_path"].format(category=category)
    model = ConvAutoencoder(
        image_size=image_size,
        latent_dim=config["model"]["latent_dim"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def score_dataset(model, dataset, device: str, k_fraction: float, label_override: int = None) -> list[dict]:
    """
    Run the model over every sample in `dataset` and compute an
    image-level anomaly score for each.

    Args:
        label_override: if provided, force this label for every sample
            (used for the validation split, where every image is known
            to be normal by construction -- label=0 -- regardless of
            what MVTecDataset's own label field would report, since
            MVTecDataset doesn't distinguish train/val internally).
    """
    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            original = sample["image"].to(device)

            reconstruction = model(original.unsqueeze(0)).squeeze(0)
            error_map = compute_error_map(original, reconstruction)
            score = compute_image_score_topk(error_map, k_fraction=k_fraction)

            label = label_override if label_override is not None else sample["label"]

            results.append({
                "image_path": sample["image_path"],
                "defect_type": sample["defect_type"],
                "label": label,
                "anomaly_score": score,
            })
    return results


def main(category: str, image_size: int = 256, k_fraction: float = 0.01) -> None:
    config = load_merged_config("configs/base.yaml", "configs/autoencoder.yaml")
    device = resolve_device(config["device"])

    model = load_model(category, config, image_size, device)

    image_transform = get_autoencoder_image_transform(image_size)
    mask_transform = get_mask_transform(image_size)

    # ---- TEST SET ----
    test_ds = MVTecDataset(
        root=config["data"]["root"],
        category=category,
        split="test",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    print(f"Total test images: {len(test_ds)}")

    test_results = score_dataset(model, test_ds, device, k_fraction)
    test_df = pd.DataFrame(test_results)
    test_output_path = f"artifacts/autoencoder_scores_{category}.csv"
    test_df.to_csv(test_output_path, index=False)
    print(f"Saved test scores to: {test_output_path}")

    print("\nTest set summary:")
    print(test_df.groupby("label")["anomaly_score"].describe())

    # ---- VALIDATION SPLIT (reproducing the exact split from training) ----
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

    generator = torch.Generator().manual_seed(config["seed"])
    _, val_subset = random_split(full_train_ds, [train_size, val_size], generator=generator)

    print(f"\nRecovered validation split: {len(val_subset)} images "
          f"(must match training run's 'Validation samples' count)")

    val_results = score_dataset(model, val_subset, device, k_fraction, label_override=0)
    val_df = pd.DataFrame(val_results)
    val_output_path = f"artifacts/autoencoder_scores_{category}_val.csv"
    val_df.to_csv(val_output_path, index=False)
    print(f"Saved validation scores to: {val_output_path}")
    print("\nValidation set summary:")
    print(val_df["anomaly_score"].describe())


if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    main(category=category)