"""Early stopping implementation for training."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training if no improvement is observed
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        metric: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
        restore_best_weights: bool = True,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize early stopping.
        
        Args:
            metric: Name of metric to monitor (e.g., 'val_loss', 'val_f1', 'val_recall')
            patience: Number of epochs with no improvement after which training is stopped
            min_delta: Minimum change in the monitored metric to qualify as an improvement
            mode: 'min' if lower metric value is better, 'max' if higher is better
            verbose: Whether to print messages
            restore_best_weights: Whether to restore model weights from epoch with best metric
            checkpoint_dir: Directory to save best model checkpoint
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize tracking variables
        self.counter = 0  # Counter for epochs without improvement
        self.best_epoch = 0
        self.best_value = None
        self.best_weights = None
        self.wait_count = 0  # Number of epochs to wait
        
        if mode == "min":
            self.best_value = np.inf
            self.compare = lambda x: x < self.best_value - self.min_delta
        else:
            self.best_value = -np.inf
            self.compare = lambda x: x > self.best_value + self.min_delta
    
    def __call__(
        self,
        current_value: float,
        epoch: int,
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current value of the metric being monitored
            epoch: Current epoch number
            model: Model to checkpoint if this is the best epoch
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            # First call
            self.best_value = current_value
            self.best_epoch = epoch
            self._save_best_weights(model, epoch)
            if self.verbose:
                logger.info(f"Initial {self.metric}: {current_value:.6f}")
            return False
        
        if self.compare(current_value):
            # Improvement detected
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            self._save_best_weights(model, epoch)
            
            if self.verbose:
                delta = current_value - self.best_value if self.mode == "max" else self.best_value - current_value
                logger.info(
                    f"Epoch {epoch}: {self.metric} improved to {current_value:.6f}"
                )
            return False
        
        else:
            # No improvement
            self.counter += 1
            
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.metric} = {current_value:.6f} "
                    f"(no improvement for {self.counter}/{self.patience} epochs)"
                )
            
            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered! Best {self.metric}: {self.best_value:.6f} "
                        f"at epoch {self.best_epoch}"
                    )
                return True
        
        return False
    
    def _save_best_weights(self, model: Optional[nn.Module], epoch: int) -> None:
        """Save the best model weights."""
        if model is None:
            return
        
        if self.restore_best_weights:
            # Store weights in memory
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Optionally save to disk
        if self.checkpoint_dir is not None:
            checkpoint_path = Path(self.checkpoint_dir) / f"best_model_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.debug(f"Saved best model checkpoint to {checkpoint_path}")
    
    def restore_best_model(self, model: nn.Module, device: torch.device) -> None:
        """
        Restore model weights from best epoch.
        
        Args:
            model: Model to restore weights to
            device: Device to move weights to
        """
        if self.best_weights is None:
            logger.warning("No best weights saved. Skipping restoration.")
            return
        
        # Move weights to correct device
        best_weights_on_device = {k: v.to(device) for k, v in self.best_weights.items()}
        model.load_state_dict(best_weights_on_device)
        
        if self.verbose:
            logger.info(f"Restored model weights from epoch {self.best_epoch}")
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None
        
        if self.mode == "min":
            self.best_value = np.inf
        else:
            self.best_value = -np.inf
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get early stopping state for checkpointing.
        
        Returns:
            Dictionary with early stopping state
        """
        return {
            "metric": self.metric,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "counter": self.counter,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load early stopping state from checkpoint.
        
        Args:
            state: Dictionary with early stopping state
        """
        self.counter = state.get("counter", 0)
        self.best_epoch = state.get("best_epoch", 0)
        self.best_value = state.get("best_value", None)
