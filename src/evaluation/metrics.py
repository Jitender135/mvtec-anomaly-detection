"""
Evaluation metrics for anomaly detection.

Includes score aggregation (per-pixel error map -> single image-level
score) as well as, later, ROC-AUC/PR-AUC/precision/recall/F1 computation
across a full test set.
"""

import numpy as np


def compute_image_score_topk(error_map: np.ndarray, k_fraction: float = 0.01) -> float:
    """
    Aggregate a per-pixel error map into a single image-level anomaly score
    by averaging the top-K% highest-error pixels.

    Rationale: real defects are typically small, localized regions. A plain
    whole-image mean would dilute a small defect against a large background
    of low-error normal pixels. Taking the single max pixel is sensitive to
    isolated noise (e.g. checkerboard artifacts). Averaging the top-K% of
    pixels balances sensitivity to localized anomalies against robustness
    to single-pixel noise.

    Args:
        error_map: (H, W) array of per-pixel reconstruction error.
        k_fraction: fraction of pixels (0 < k_fraction <= 1) to average.
                    e.g. 0.01 = top 1% of pixels by error value.

    Returns:
        A single float anomaly score for the image.
    """
    if not (0 < k_fraction <= 1):
        raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}")

    flattened = error_map.flatten()
    num_pixels = flattened.shape[0]
    k = max(1, int(num_pixels * k_fraction))  # always at least 1 pixel

    # np.partition is O(n) and avoids a full sort -- we only need the
    # top-k values, not a fully ordered ranking of every pixel.
    top_k_values = np.partition(flattened, -k)[-k:]
    return float(top_k_values.mean())

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)


def compute_roc_auc(labels: list[int], scores: list[float]) -> float:
    """
    ROC-AUC: probability that a randomly chosen anomalous image scores
    higher than a randomly chosen normal image. Threshold-independent --
    evaluates ranking quality across ALL possible thresholds at once.
    1.0 = perfect separation, 0.5 = no better than random guessing.
    """
    return roc_auc_score(labels, scores)


def compute_pr_auc(labels: list[int], scores: list[float]) -> float:
    """
    PR-AUC (average precision): area under the precision-recall curve.
    More informative than ROC-AUC when classes are imbalanced (which
    ours are: 20 normal vs 63 anomalous), since it focuses on performance
    with respect to the positive (anomalous) class specifically.
    """
    return average_precision_score(labels, scores)


def compute_classification_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> dict:
    """
    Compute precision, recall, F1, and accuracy at a SPECIFIC threshold.

    Unlike ROC-AUC/PR-AUC, these metrics depend entirely on the chosen
    threshold -- report the threshold alongside these numbers always.
    """
    predictions = [1 if s >= threshold else 0 for s in scores]

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }