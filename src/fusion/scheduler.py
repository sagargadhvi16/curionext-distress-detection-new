"""Learning rate schedulers for training."""
import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Dict


class AdaptiveLRScheduler:
    """
    Wrapper for adaptive learning rate scheduling.
    Supports:
    - ReduceLROnPlateau: Reduce LR when metric plateaus
    - CosineAnnealingLR: Cosine annealing schedule
    - CosineAnnealingWarmRestarts: Cosine annealing with warm restarts
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "reduce_on_plateau",
        **kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('reduce_on_plateau', 'cosine_annealing', 'cosine_warm_restarts')
            **kwargs: Additional arguments for scheduler:
                     For reduce_on_plateau:
                       - factor: Factor to reduce LR (default: 0.5)
                       - patience: Patience epochs (default: 5)
                       - min_lr: Minimum learning rate (default: 1e-6)
                       - mode: 'min' or 'max' (default: 'min')
                     For cosine_annealing:
                       - T_max: Maximum number of epochs (required)
                       - eta_min: Minimum learning rate (default: 0)
                     For cosine_warm_restarts:
                       - T_0: Initial restart period (default: 10)
                       - T_mult: Multiplier for restart period (default: 2)
                       - eta_min: Minimum learning rate (default: 0)
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type.lower()
        self.scheduler = None
        
        if self.scheduler_type == "reduce_on_plateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
            self.is_metric_based = True
        elif self.scheduler_type == "cosine_annealing":
            T_max = kwargs.get('T_max')
            if T_max is None:
                raise ValueError("T_max is required for cosine_annealing scheduler")
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=kwargs.get('eta_min', 0)
            )
            self.is_metric_based = False
        elif self.scheduler_type == "cosine_warm_restarts":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 0)
            )
            self.is_metric_based = False
        else:
            raise ValueError(
                f"Unknown scheduler type: {scheduler_type}. "
                f"Supported types: 'reduce_on_plateau', 'cosine_annealing', 'cosine_warm_restarts'"
            )
    
    def step(self, metrics: Optional[float] = None):
        """
        Step the scheduler.
        
        Args:
            metrics: Metric value for ReduceLROnPlateau (loss or validation metric)
                    Ignored for other schedulers
        """
        if self.is_metric_based:
            if metrics is None:
                raise ValueError("Metrics required for ReduceLROnPlateau scheduler")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> list:
        """Get current learning rate(s)."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            # For ReduceLROnPlateau, get from optimizer
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get scheduler state dict."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Optional[Dict] = None
) -> AdaptiveLRScheduler:
    """
    Factory function to create scheduler from config dictionary.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary with keys:
                - 'type': Scheduler type ('reduce_on_plateau', 'cosine_annealing', 'cosine_warm_restarts')
                - Additional keys depend on scheduler type (see AdaptiveLRScheduler.__init__)
    
    Returns:
        AdaptiveLRScheduler instance
    """
    if config is None:
        # Default: reduce on plateau
        return AdaptiveLRScheduler(optimizer, scheduler_type="reduce_on_plateau")
    
    scheduler_type = config.get('type', 'reduce_on_plateau')
    kwargs = {k: v for k, v in config.items() if k != 'type'}
    
    return AdaptiveLRScheduler(optimizer, scheduler_type=scheduler_type, **kwargs)

