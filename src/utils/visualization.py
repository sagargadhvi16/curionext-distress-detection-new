"""Training visualization utilities."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Use non-interactive backend for server environments
matplotlib.use('Agg')


class TrainingVisualizer:
    """Create and save training visualization plots."""
    
    def __init__(self, output_dir: str = "logs/plots", dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curves(
        self,
        train_loss: List[float],
        val_loss: List[float],
        train_accuracy: Optional[List[float]] = None,
        val_accuracy: Optional[List[float]] = None,
        train_f1: Optional[List[float]] = None,
        val_f1: Optional[List[float]] = None,
        epochs: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        title: str = "Training Curves",
        show: bool = False
    ) -> Path:
        """
        Plot training and validation curves.
        
        Args:
            train_loss: List of training losses
            val_loss: List of validation losses
            train_accuracy: List of training accuracies (optional)
            val_accuracy: List of validation accuracies (optional)
            train_f1: List of training F1 scores (optional)
            val_f1: List of validation F1 scores (optional)
            epochs: List of epoch numbers (default: 0 to len-1)
            save_path: Path to save figure
            title: Plot title
            show: Whether to display the plot
        
        Returns:
            Path where figure was saved
        """
        if epochs is None:
            epochs = list(range(len(train_loss)))
        
        # Create figure with subplots
        num_plots = 1
        if train_accuracy is not None:
            num_plots += 1
        if train_f1 is not None:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Loss plot
        axes[plot_idx].plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        axes[plot_idx].plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # Accuracy plot
        if train_accuracy is not None and val_accuracy is not None:
            axes[plot_idx].plot(epochs, train_accuracy, 'b-o', label='Train Acc', linewidth=2, markersize=4)
            axes[plot_idx].plot(epochs, val_accuracy, 'r-s', label='Val Acc', linewidth=2, markersize=4)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].set_title('Accuracy')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # F1 plot
        if train_f1 is not None and val_f1 is not None:
            axes[plot_idx].plot(epochs, train_f1, 'b-o', label='Train F1', linewidth=2, markersize=4)
            axes[plot_idx].plot(epochs, val_f1, 'r-s', label='Val F1', linewidth=2, markersize=4)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('F1 Score')
            axes[plot_idx].set_title('F1 Score')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        else:
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def plot_learning_rate(
        self,
        learning_rates: List[float],
        epochs: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        title: str = "Learning Rate Schedule",
        show: bool = False
    ) -> Path:
        """
        Plot learning rate schedule.
        
        Args:
            learning_rates: List of learning rates per epoch
            epochs: List of epoch numbers (default: 0 to len-1)
            save_path: Path to save figure
            title: Plot title
            show: Whether to display the plot
        
        Returns:
            Path where figure was saved
        """
        if epochs is None:
            epochs = list(range(len(learning_rates)))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(epochs, learning_rates, 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if learning rates vary significantly
        if max(learning_rates) / min(learning_rates) > 10:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "learning_rate.png"
        else:
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved learning rate plot to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix",
        show: bool = False,
        normalize: bool = False
    ) -> Path:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix (numpy array)
            class_names: Names of classes
            save_path: Path to save figure
            title: Plot title
            show: Whether to display the plot
            normalize: Whether to normalize the confusion matrix
        
        Returns:
            Path where figure was saved
        """
        cm = np.asarray(cm)
        
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count' if not normalize else 'Proportion', rotation=270, labelpad=20)
        
        # Set labels and ticks
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j, i, format(cm[i, j], fmt),
                             ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.png"
        else:
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        save_path: Optional[str] = None,
        title: str = "ROC Curve",
        show: bool = False
    ) -> Path:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under curve
            save_path: Path to save figure
            title: Plot title
            show: Whether to display the plot
        
        Returns:
            Path where figure was saved
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "roc_curve.png"
        else:
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def plot_precision_recall_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        ap: float,
        save_path: Optional[str] = None,
        title: str = "Precision-Recall Curve",
        show: bool = False
    ) -> Path:
        """
        Plot precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            ap: Average precision
            save_path: Path to save figure
            title: Plot title
            show: Whether to display the plot
        
        Returns:
            Path where figure was saved
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'Precision-Recall (AP = {ap:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "precision_recall_curve.png"
        else:
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, float],
        save_path: Optional[str] = None,
        title: str = "Metrics Comparison",
        show: bool = False
    ) -> Path:
        """
        Plot bar chart of metrics.
        
        Args:
            metrics_dict: Dictionary of metric names to values
            save_path: Path to save figure
            title: Plot title
            show: Whether to display the plot
        
        Returns:
            Path where figure was saved
        """
        # Filter out confusion matrix and other non-scalar metrics
        scalar_metrics = {k: v for k, v in metrics_dict.items() 
                         if isinstance(v, (int, float)) and not np.isnan(v)}
        
        if not scalar_metrics:
            logger.warning("No scalar metrics to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(scalar_metrics.keys())
        values = list(scalar_metrics.values())
        
        # Create bar chart
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
        ax.bar(metrics, values, color=colors, alpha=0.7)
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "metrics_comparison.png"
        else:
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved metrics comparison to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return save_path


# Convenience functions
def plot_training_curves(
    train_loss: List[float],
    val_loss: List[float],
    train_accuracy: Optional[List[float]] = None,
    val_accuracy: Optional[List[float]] = None,
    output_dir: str = "logs/plots",
    save_path: Optional[str] = None,
    show: bool = False
) -> Path:
    """
    Plot training curves (convenience function).
    
    Args:
        train_loss: Training losses per epoch
        val_loss: Validation losses per epoch
        train_accuracy: Training accuracies (optional)
        val_accuracy: Validation accuracies (optional)
        output_dir: Output directory for plots
        save_path: Path to save figure
        show: Whether to display the plot
    
    Returns:
        Path where figure was saved
    """
    visualizer = TrainingVisualizer(output_dir)
    return visualizer.plot_training_curves(
        train_loss,
        val_loss,
        train_accuracy,
        val_accuracy,
        save_path=save_path,
        show=show
    )
