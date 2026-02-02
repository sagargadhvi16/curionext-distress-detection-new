"""
Evaluation metrics for distress detection.
Focus on minimizing false negatives (missed distress cases).
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_false_negative_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute False Negative Rate (FNR).

    FNR = FN / (FN + TP)

    False negatives are cases where distress was present but not detected.
    This is the most critical error type for distress detection.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # [[TN, FP],
    #  [FN, TP]]
    if cm.shape == (2, 2):
        _, _, FN, TP = cm.ravel()
    else:
        # Edge cases (single class present)
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TP = np.sum((y_true == 1) & (y_pred == 1))

    if (FN + TP) == 0:
        return 0.0

    return float(FN / (FN + TP))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None
) -> Dict[str, float]:
    """
    Compute binary classification metrics for distress detection.
    Used for evaluation and ablation studies.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fnr": compute_false_negative_rate(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics
