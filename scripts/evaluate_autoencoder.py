"""
Run the trained autoencoder across the FULL test set for one category,
computing an image-level anomaly score per image via top-K reconstruction
error aggregation. Saves results to a CSV for use by threshold calibration
(Phase 6) and formal evaluation metrics (Phase 12).
"""

import sys

import pandas as pd
import torch

from src.config import load_merged_config, resolve_device
from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import get_autoencoder_image_transform, get_mask_transform
from src.models.autoencoder.network import ConvAutoencoder
from src.evaluation.metrics import compute_image_score_topk
from scripts.visualize_autoencoder import compute_error_map


def main(category: str, image_size: int = 256, k_fraction: float = 0.01) -> None:
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
    print(f"Total test images: {len(test_ds)}")

    results = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            sample = test_ds[i]
            original = sample["image"].to(device)

            reconstruction = model(original.unsqueeze(0)).squeeze(0)
            error_map = compute_error_map(original, reconstruction)
            score = compute_image_score_topk(error_map, k_fraction=k_fraction)

            results.append({
                "image_path": sample["image_path"],
                "defect_type": sample["defect_type"],
                "label": sample["label"],
                "anomaly_score": score,
            })

    df = pd.DataFrame(results)

    output_path = f"artifacts/autoencoder_scores_{category}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved scores to: {output_path}")

    print("\nSummary:")
    print(df.groupby("label")["anomaly_score"].describe())


if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    main(category=category)