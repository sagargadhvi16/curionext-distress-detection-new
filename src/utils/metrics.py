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
        y_prob: Predicted probabilities (optional, for AUC calculation)

    Returns:
        Dictionary of metrics including accuracy, precision, recall, F1, and optionally AUC

    IMPORTANT: Focus on minimizing False Negatives (missed distress cases)
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Handle case where only one class present
            metrics['auc'] = 0.0
    
    return metrics


def compute_false_negative_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute false negative rate (critical metric).

    FNR = FN / (FN + TP)
    
    False negatives are cases where distress was present but not detected.
    This is the most critical error type for distress detection.

    Args:
        y_true: Ground truth labels (1 = distress, 0 = no distress)
        y_pred: Predicted labels (1 = distress, 0 = no distress)

    Returns:
        False negative rate (0-1), or 0.0 if no positive cases in ground truth
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Extract values
    # [[TN, FP],
    #  [FN, TP]]
    if cm.shape == (2, 2):
        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]
    else:
        # Handle edge cases (single class present)
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                # Only negative class
                FN, TP = 0, 0
            else:
                # Only positive class
                FN = np.sum((y_true == 1) & (y_pred == 0))
                TP = np.sum((y_true == 1) & (y_pred == 1))
        else:
            # Fallback
            FN = np.sum((y_true == 1) & (y_pred == 0))
            TP = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate FNR
    if (FN + TP) == 0:
        # No positive cases in ground truth
        return 0.0
    
    fnr = FN / (FN + TP)
    return float(fnr)
