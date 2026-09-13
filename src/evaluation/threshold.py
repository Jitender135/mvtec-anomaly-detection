"""
Threshold calibration strategies.

IMPORTANT METHODOLOGICAL NOTE:
- Percentile-based calibration uses ONLY normal (validation) scores --
  it never looks at labeled anomalies, so it is methodologically clean
  to then evaluate on the held-out test set.
- ROC-based and F1-optimal calibration REQUIRE labeled anomalies to
  compute, which MVTec AD only provides in the test set. Using these
  strategies means the threshold is tuned on the same data used for
  final evaluation -- we report this explicitly as "illustrative"
  rather than a validated deployment threshold. See README for a full
  discussion of this tradeoff.
"""

import numpy as np
from sklearn.metrics import roc_curve, f1_score


def select_threshold_percentile(normal_scores: list[float], percentile: float = 95.0) -> float:
    """
    Set the threshold at a given percentile of NORMAL (non-anomalous)
    scores only. E.g. percentile=95 means we accept ~5% of normal
    validation images scoring above the threshold (false positives),
    by design choice.

    This is the only strategy here that requires NO labeled anomalies.
    """
    return float(np.percentile(normal_scores, percentile))


def select_threshold_youden(labels: list[int], scores: list[float]) -> dict:
    """
    Select the threshold that maximizes Youden's J statistic (TPR - FPR)
    from the ROC curve -- the point on the curve furthest from random
    guessing. Requires labeled anomalies (test set).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_index = int(np.argmax(j_scores))

    return {
        "threshold": float(thresholds[best_index]),
        "tpr": float(tpr[best_index]),
        "fpr": float(fpr[best_index]),
        "youden_j": float(j_scores[best_index]),
    }


def select_threshold_f1_optimal(labels: list[int], scores: list[float]) -> dict:
    """
    Sweep over candidate thresholds (each unique score value) and return
    the one maximizing F1. Requires labeled anomalies (test set).
    """
    candidate_thresholds = sorted(set(scores))
    best_f1 = -1.0
    best_threshold = candidate_thresholds[0]

    for t in candidate_thresholds:
        predictions = [1 if s >= t else 0 for s in scores]
        f1 = f1_score(labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return {"threshold": float(best_threshold), "f1": float(best_f1)}