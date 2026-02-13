"""Model checkpoint saving and loading utilities."""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    metrics: Optional[Dict[str, float]] = None,
    training_state: Optional[Dict[str, Any]] = None,
    save_optimizer: bool = True,
    save_scheduler: bool = True,
    save_training_state: bool = True,
    is_best: bool = False,
    keep_best: int = 5,
    best_metric_name: Optional[str] = None
) -> Path:
    """
    Save model checkpoint to disk.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: LR scheduler state (optional)
        metrics: Dictionary of current metrics (optional)
        training_state: Additional training state to save (optional)
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        save_training_state: Whether to save training state
        is_best: Whether this is the best checkpoint so far
        keep_best: Number of best checkpoints to keep
        best_metric_name: Name of metric used to determine best checkpoint
    
    Returns:
        Path to saved checkpoint
    
    Raises:
        IOError: If checkpoint directory cannot be created
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer).__name__
    
    if save_scheduler and scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        checkpoint['scheduler_type'] = type(scheduler).__name__
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if save_training_state and training_state is not None:
        checkpoint['training_state'] = training_state
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Handle best checkpoint
    if is_best:
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
        torch.save(checkpoint, best_checkpoint_path)
        logger.info(f"Saved best checkpoint to {best_checkpoint_path}")
        
        # Update best checkpoint info
        _save_best_checkpoint_info(
            checkpoint_dir,
            epoch,
            metrics,
            best_metric_name
        )
    
    # Clean up old checkpoints
    _cleanup_old_checkpoints(
        checkpoint_dir,
        keep_best=keep_best,
        is_best=is_best
    )
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: str = "cpu",
    load_optimizer: bool = True,
    load_scheduler: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)
        device: Device to load checkpoint to
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
    
    Returns:
        Dictionary with checkpoint information including epoch, metrics, etc.
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If checkpoint is corrupted or missing expected keys
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model state from checkpoint")
    
    # Load optimizer state
    if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    # Load scheduler state
    if load_scheduler and scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"Loaded scheduler state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'training_state': checkpoint.get('training_state', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown')
    }
    
    logger.info(f"Checkpoint epoch: {metadata['epoch']}")
    if metadata['metrics']:
        logger.info(f"Checkpoint metrics: {metadata['metrics']}")
    
    return metadata


def load_best_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load the best checkpoint from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)
        device: Device to load checkpoint to
    
    Returns:
        Dictionary with checkpoint information
    
    Raises:
        FileNotFoundError: If best checkpoint not found
    """
    checkpoint_path = Path(checkpoint_dir) / "best_checkpoint.pt"
    
    if not checkpoint_path.exists():
        # Try to find best checkpoint in file
        info_path = Path(checkpoint_dir) / "best_checkpoint_info.txt"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = f.readlines()
                if info:
                    checkpoint_path = Path(checkpoint_dir) / info[0].split()[-1]
    
    return load_checkpoint(
        str(checkpoint_path),
        model,
        optimizer,
        scheduler,
        device,
        load_optimizer=True,
        load_scheduler=True
    )


def _cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_best: int = 5,
    is_best: bool = False
) -> None:
    """
    Remove old checkpoints, keeping only the best ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Number of best checkpoints to keep
        is_best: Whether current checkpoint is the best
    """
    checkpoint_files = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    
    # Keep the best checkpoint file separate
    if (checkpoint_dir / "best_checkpoint.pt").exists():
        checkpoint_files = [f for f in checkpoint_files if f.name != "best_checkpoint.pt"]
    
    # Remove old checkpoints
    if len(checkpoint_files) > keep_best:
        for old_checkpoint in checkpoint_files[:-keep_best]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")


def _save_best_checkpoint_info(
    checkpoint_dir: Path,
    epoch: int,
    metrics: Optional[Dict[str, float]],
    best_metric_name: Optional[str]
) -> None:
    """
    Save information about the best checkpoint to a text file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        epoch: Epoch number of best checkpoint
        metrics: Dictionary of metrics
        best_metric_name: Name of metric used to determine best
    """
    info_path = checkpoint_dir / "best_checkpoint_info.txt"
    
    with open(info_path, 'w') as f:
        f.write(f"Best checkpoint: epoch {epoch}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Best metric: {best_metric_name}\n")
        if metrics:
            f.write("Metrics:\n")
            for name, value in metrics.items():
                f.write(f"  {name}: {value:.6f}\n")
        f.write(f"File: checkpoint_epoch_{epoch:03d}.pt\n")
    
    logger.debug(f"Saved best checkpoint info to {info_path}")


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and return the latest
    latest = sorted(
        checkpoint_files,
        key=lambda p: int(p.stem.split("_")[-1])
    )[-1]
    
    return latest


def resume_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    device: str = "cpu"
) -> Tuple[int, Dict[str, Any]]:
    """
    Resume training from the latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: LR scheduler to load state into (optional)
        device: Device to load checkpoint to
    
    Returns:
        Tuple of (starting_epoch, checkpoint_info)
    
    Raises:
        FileNotFoundError: If no checkpoint found in directory
    """
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
    
    info = load_checkpoint(
        str(latest_checkpoint),
        model,
        optimizer,
        scheduler,
        device
    )
    
    # Start from next epoch
    start_epoch = info['epoch'] + 1
    
    return start_epoch, info
