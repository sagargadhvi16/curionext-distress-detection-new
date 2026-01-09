"""Comprehensive metrics calculator for model evaluation."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from typing import Dict, Tuple, Optional, List
import warnings

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.
    
    Computes all standard evaluation metrics including:
    - Accuracy, Precision, Recall, F1-Score
    - Area Under ROC Curve (AUC-ROC)
    - Confusion Matrix
    - False Negative Rate (critical for distress detection)
    - Matthew's Correlation Coefficient
    - Cohen's Kappa
    """
    
    def __init__(self, num_classes: int = 2, average: str = "binary"):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes (2 for binary, > 2 for multi-class)
            average: Averaging method for multi-class ('binary', 'micro', 'macro', 'weighted')
        """
        self.num_classes = num_classes
        self.average = average
        
        # For multi-class, use weighted average unless specified
        if num_classes > 2 and average == "binary":
            self.average = "weighted"
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (batch_size, num_classes)
            sample_weight: Optional sample weights
        
        Returns:
            Dictionary with all computed metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_true) == 0:
            logger.warning("Empty arrays provided to compute_all_metrics")
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._compute_basic_metrics(y_true, y_pred, sample_weight))
        
        # Confusion matrix and related metrics
        metrics.update(self._compute_confusion_matrix_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._compute_probability_metrics(y_true, y_proba))
        
        return metrics
    
    def _compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute basic classification metrics."""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        except Exception as e:
            logger.warning(f"Error computing accuracy: {e}")
            metrics['accuracy'] = 0.0
        
        try:
            metrics['precision'] = precision_score(
                y_true, y_pred,
                average=self.average,
                sample_weight=sample_weight,
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error computing precision: {e}")
            metrics['precision'] = 0.0
        
        try:
            metrics['recall'] = recall_score(
                y_true, y_pred,
                average=self.average,
                sample_weight=sample_weight,
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error computing recall: {e}")
            metrics['recall'] = 0.0
        
        try:
            metrics['f1'] = f1_score(
                y_true, y_pred,
                average=self.average,
                sample_weight=sample_weight,
                zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error computing F1 score: {e}")
            metrics['f1'] = 0.0
        
        # Additional metrics
        try:
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Error computing Matthews correlation coefficient: {e}")
            metrics['mcc'] = 0.0
        
        try:
            metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Error computing Cohen's kappa: {e}")
            metrics['kappa'] = 0.0
        
        return metrics
    
    def _compute_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute confusion matrix and related metrics."""
        metrics = {}
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm
            
            # For binary classification, extract TPR, FPR, TNR, FNR
            if self.num_classes == 2 and cm.shape == (2, 2):
                tn, fp = cm[0, 0], cm[0, 1]
                fn, tp = cm[1, 0], cm[1, 1]
                
                # True positive rate (sensitivity/recall)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['tpr'] = tpr
                
                # False positive rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                metrics['fpr'] = fpr
                
                # True negative rate (specificity)
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['tnr'] = tnr
                
                # False negative rate (critical for distress detection)
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                metrics['fnr'] = fnr
                
                # Specificity
                metrics['specificity'] = tnr
        
        except Exception as e:
            logger.warning(f"Error computing confusion matrix metrics: {e}")
            metrics['confusion_matrix'] = None
        
        return metrics
    
    def _compute_probability_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics based on predicted probabilities."""
        metrics = {}
        
        y_proba = np.asarray(y_proba)
        
        # Handle different probability shapes
        if y_proba.ndim == 1:
            # Binary case with single probability
            y_prob_positive = y_proba
        elif y_proba.ndim == 2:
            if y_proba.shape[1] == 2:
                # Binary case with probabilities for both classes
                y_prob_positive = y_proba[:, 1]
            else:
                # Multi-class case - can't compute standard ROC-AUC
                y_prob_positive = None
        else:
            y_prob_positive = None
        
        if y_prob_positive is not None:
            try:
                # ROC-AUC
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob_positive)
            except Exception as e:
                logger.warning(f"Error computing AUC-ROC: {e}")
                metrics['auc_roc'] = 0.0
            
            try:
                # Average precision
                metrics['average_precision'] = average_precision_score(y_true, y_prob_positive)
            except Exception as e:
                logger.warning(f"Error computing average precision: {e}")
                metrics['average_precision'] = 0.0
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return dictionary with zero values for all metrics."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_roc': 0.0,
            'average_precision': 0.0,
            'mcc': 0.0,
            'kappa': 0.0,
            'tpr': 0.0,
            'fpr': 0.0,
            'tnr': 0.0,
            'fnr': 0.0,
            'specificity': 0.0,
            'confusion_matrix': None
        }
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Optional class names for display
        
        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities (for positive class)
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if y_proba.ndim == 2:
            # Get probabilities for positive class
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return fpr, tpr, thresholds
    
    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get precision-recall curve.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities (for positive class)
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if y_proba.ndim == 2:
            # Get probabilities for positive class
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        return precision, recall, thresholds
    
    def format_metrics_string(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics as a readable string.
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            Formatted string
        """
        lines = []
        
        # Group metrics
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1']
        prob_metrics = ['auc_roc', 'average_precision']
        corr_metrics = ['mcc', 'kappa']
        binary_metrics = ['tpr', 'fpr', 'tnr', 'fnr', 'specificity']
        
        if any(m in metrics for m in basic_metrics):
            lines.append("Basic Metrics:")
            for m in basic_metrics:
                if m in metrics:
                    lines.append(f"  {m}: {metrics[m]:.4f}")
        
        if any(m in metrics for m in prob_metrics):
            lines.append("Probability Metrics:")
            for m in prob_metrics:
                if m in metrics:
                    lines.append(f"  {m}: {metrics[m]:.4f}")
        
        if any(m in metrics for m in corr_metrics):
            lines.append("Correlation Metrics:")
            for m in corr_metrics:
                if m in metrics:
                    lines.append(f"  {m}: {metrics[m]:.4f}")
        
        if any(m in metrics for m in binary_metrics):
            lines.append("Binary Classification Metrics:")
            for m in binary_metrics:
                if m in metrics:
                    lines.append(f"  {m}: {metrics[m]:.4f}")
        
        return "\n".join(lines)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names
    
    Returns:
        Dictionary mapping class to metrics dictionary
    """
    unique_classes = np.unique(y_true)
    
    if class_names is None:
        class_names = [str(c) for c in unique_classes]
    
    per_class_metrics = {}
    
    for class_idx, class_label in enumerate(unique_classes):
        # Binary classification: class vs. rest
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)
        
        calculator = MetricsCalculator(num_classes=2)
        metrics = calculator.compute_all_metrics(y_true_binary, y_pred_binary)
        
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_label}"
        per_class_metrics[class_name] = metrics
    
    return per_class_metrics
