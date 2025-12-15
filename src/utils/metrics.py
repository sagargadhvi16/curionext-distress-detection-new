"""Evaluation metrics for distress detection."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics

    TODO: Implement metric computation
    IMPORTANT: Focus on minimizing False Negatives (missed distress cases)
    """
    pass  # To be implemented


def compute_false_negative_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute false negative rate (critical metric).

    FNR = FN / (FN + TP)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        False negative rate

    TODO: Implement FNR computation
    """
    pass  # To be implemented
