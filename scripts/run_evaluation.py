"""
Full evaluation of the baseline autoencoder on MVTec AD test data.

Reports:
  - ROC-AUC, PR-AUC (threshold-independent, primary metrics)
  - Classification metrics (precision/recall/F1/accuracy) at three
    different threshold strategies, with explicit notes on which are
    methodologically validated vs. illustrative (see threshold.py docstring)
"""

import sys

import pandas as pd
import torch
from torch.utils.data import random_split

from src.config import load_merged_config
from src.data.mvtec_dataset import MVTecDataset
from src.data.transforms import get_autoencoder_image_transform, get_mask_transform
from src.evaluation.metrics import compute_roc_auc, compute_pr_auc, compute_classification_metrics
from src.evaluation.threshold import (
    select_threshold_percentile,
    select_threshold_youden,
    select_threshold_f1_optimal,
)


def get_validation_normal_scores(category: str, config: dict) -> list[float]:
    """
    Recompute the SAME train/val split used during training (same seed),
    and return anomaly scores for just the validation slice. These are
    genuinely-normal, held-out-from-training images -- the correct basis
    for percentile-based threshold calibration.
    """
    # NOTE: this recomputation approach is a placeholder -- see caveat below.
    raise NotImplementedError


def main(category: str) -> None:
    config = load_merged_config("configs/base.yaml", "configs/autoencoder.yaml")

    scores_path = f"artifacts/autoencoder_scores_{category}.csv"
    df = pd.read_csv(scores_path)
    print(f"Loaded {len(df)} scored test images from: {scores_path}")

    labels = df["label"].tolist()
    scores = df["anomaly_score"].tolist()

    # ---- Threshold-independent metrics (primary) ----
    roc_auc = compute_roc_auc(labels, scores)
    pr_auc = compute_pr_auc(labels, scores)

    print("\n" + "=" * 60)
    print("THRESHOLD-INDEPENDENT METRICS (primary)")
    print("=" * 60)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    # ---- Threshold-dependent metrics (illustrative, see notes) ----
    val_scores_path = f"artifacts/autoencoder_scores_{category}_val.csv"
    val_df = pd.read_csv(val_scores_path)
    validation_normal_scores = val_df["anomaly_score"].tolist()
    print(f"Loaded {len(validation_normal_scores)} validation scores from: {val_scores_path}")

    percentile_threshold = select_threshold_percentile(validation_normal_scores, percentile=95.0)
    youden_result = select_threshold_youden(labels, scores)
    f1_result = select_threshold_f1_optimal(labels, scores)

    print("\n" + "=" * 60)
    print("THRESHOLD STRATEGIES (illustrative -- see README methodology notes)")
    print("=" * 60)

    for name, threshold in [
        ("Percentile (95th, VALIDATION normals -- methodologically clean)", percentile_threshold),
        ("Youden's J (ROC-based)", youden_result["threshold"]),
        ("F1-optimal", f1_result["threshold"]),
    ]:
        metrics = compute_classification_metrics(labels, scores, threshold)
        print(f"\n{name}:")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    main(category=category)