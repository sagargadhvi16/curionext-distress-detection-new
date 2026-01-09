"""Complete training script for distress detection model."""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from datetime import datetime
import warnings
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logger
from src.fusion.model import DistressDetectionModel
from src.fusion.loss import MultiTaskLoss
from src.fusion.training import train_epoch, validate_epoch
from src.fusion.early_stopping import EarlyStopping
from src.fusion.checkpoint import save_checkpoint, load_checkpoint, resume_from_checkpoint
from src.fusion.metrics_calculator import MetricsCalculator
from src.utils.visualization import TrainingVisualizer
from src.audio.encoder import AudioEncoder
from src.biometric.encoder import BiometricEncoder
from src.fusion.dataset import MultimodalDataset
from src.fusion.pairing import load_paired_data, split_data

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train distress detection model")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Path to training config file")
    
    # Override training parameters
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    
    # Device and model
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, cuda:0, etc.)")
    
    # Training control
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model checkpoint")
    
    # Logging and saving
    parser.add_argument("--save-dir", type=str, default="models/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--plot-dir", type=str, default="logs/plots",
                        help="Directory to save plots")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory with paired data files")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plot generation")
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    import logging
    setup_logger(
        name='train',
        log_file=Path(args.log_dir) / "training.log",
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"Random seed set to {args.seed}")


def get_device(device_str: str) -> torch.device:
    """Get device from string specification."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    return device


def load_config_and_override(config_path: str, args) -> dict:
    """Load config and apply command line overrides."""
    config = load_config(config_path)
    
    # Apply overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Training config: epochs={config['training']['epochs']}, "
                f"batch_size={config['training']['batch_size']}, "
                f"lr={config['training']['learning_rate']}")
    
    return config


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create the model."""
    logger.info("Creating model...")
    
    model_config = config.get('model', {})
    
    # Create encoders
    audio_encoder = AudioEncoder()
    biometric_encoder = BiometricEncoder()
    
    # Create model
    model = DistressDetectionModel(
        audio_encoder=audio_encoder,
        biometric_encoder=biometric_encoder,
        use_attention_fusion=model_config.get('use_attention_fusion', True),
        audio_dim=model_config.get('audio_dim', 256),
        bio_dim=model_config.get('bio_dim', 256),
        context_dim=model_config.get('context_dim', 64),
        fusion_hidden_dims=model_config.get('fusion_hidden_dims', [512, 256]),
        classifier_hidden_dim=model_config.get('classifier_hidden_dim', 128),
        dropout=model_config.get('dropout', 0.4),
        num_distress_types=model_config.get('num_distress_types', 5)
    )
    
    model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def create_optimizer_and_scheduler(model: nn.Module, config: dict):
    """Create optimizer and learning rate scheduler."""
    train_config = config['training']
    lr = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 0.0)
    optimizer_type = train_config.get('optimizer', 'adam').lower()
    
    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **train_config.get('optimizer_params', {}).get('adam', {})
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **train_config.get('optimizer_params', {}).get('sgd', {})
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type.upper()} optimizer with lr={lr}")
    
    # Create scheduler
    scheduler_config = train_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6),
            cooldown=scheduler_config.get('cooldown', 0)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('t_max', 100)
        )
    else:
        scheduler = None
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler")
    
    return optimizer, scheduler


def create_loss(config: dict) -> MultiTaskLoss:
    """Create loss function."""
    loss_config = config['loss']
    
    distress_weights = loss_config.get('distress_class_weights', [1.0, 2.0])
    task_weights = loss_config.get('task_weights', {'distress': 1.0, 'severity': 0.5, 'type': 0.5})
    
    criterion = MultiTaskLoss(
        distress_weight=task_weights.get('distress', 1.0),
        severity_weight=task_weights.get('severity', 0.5),
        type_weight=task_weights.get('type', 0.5),
        distress_class_weights=distress_weights
    )
    
    logger.info(f"Created loss function with task weights: {task_weights}")
    
    return criterion


def create_dataloaders(config: dict, data_dir: str = "data/processed"):
    """Create train, validation, and test dataloaders."""
    logger.info("Creating dataloaders...")
    
    # Load paired data
    try:
        paired_samples = load_paired_data(data_dir)
        logger.info(f"Loaded {len(paired_samples)} paired samples")
    except Exception as e:
        logger.error(f"Failed to load paired data: {e}")
        logger.warning("Using dummy data for testing")
        # Create dummy paired samples for testing
        paired_samples = []
    
    # Split data
    data_config = config.get('data', {})
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    random_seed = data_config.get('random_seed', 42)
    
    train_samples, val_samples, test_samples = split_data(
        paired_samples,
        train_split=train_split,
        val_split=val_split,
        random_seed=random_seed
    )
    
    logger.info(f"Data split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    # Create datasets
    train_dataset = MultimodalDataset(train_samples)
    val_dataset = MultimodalDataset(val_samples)
    test_dataset = MultimodalDataset(test_samples)
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    val_batch_size = config.get('validation', {}).get('val_batch_size', batch_size * 2)
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def get_current_learning_rate(optimizer: optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    config: dict,
    device: torch.device,
    save_dir: str,
    plot_dir: str,
    start_epoch: int = 0
):
    """Main training loop."""
    
    # Setup early stopping
    early_stop_config = config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        metric=early_stop_config.get('metric', 'val_f1'),
        patience=early_stop_config.get('patience', 15),
        min_delta=early_stop_config.get('min_delta', 1e-4),
        mode=early_stop_config.get('mode', 'max'),
        verbose=early_stop_config.get('verbose', True),
        restore_best_weights=early_stop_config.get('restore_best_weights', True),
        checkpoint_dir=save_dir
    )
    
    # Setup visualization
    visualizer = TrainingVisualizer(plot_dir)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rates': [],
        'best_metrics': {}
    }
    
    # Training config
    num_epochs = config['training']['epochs']
    max_grad_norm = config['training'].get('gradient_clip_norm', 1.0)
    log_interval = config['logging'].get('log_every', 10)
    eval_interval = config.get('validation', {}).get('eval_interval', 1)
    checkpoint_config = config.get('checkpointing', {})
    save_every = checkpoint_config.get('save_every', 1)
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Get current learning rate
        current_lr = get_current_learning_rate(optimizer)
        history['learning_rates'].append(current_lr)
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Training epoch
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=max_grad_norm,
            log_interval=log_interval
        )
        
        history['train_loss'].append(train_metrics.get('loss', 0.0))
        history['train_accuracy'].append(train_metrics.get('distress_accuracy', 0.0))
        history['train_f1'].append(train_metrics.get('distress_f1', 0.0))
        
        logger.info(f"Train - Loss: {train_metrics.get('loss', 0.0):.4f}, "
                   f"Accuracy: {train_metrics.get('distress_accuracy', 0.0):.4f}, "
                   f"F1: {train_metrics.get('distress_f1', 0.0):.4f}")
        
        # Validation epoch
        if (epoch + 1) % eval_interval == 0:
            val_metrics = validate_epoch(
                model,
                val_loader,
                criterion,
                device
            )
            
            history['val_loss'].append(val_metrics.get('loss', 0.0))
            history['val_accuracy'].append(val_metrics.get('distress_accuracy', 0.0))
            history['val_f1'].append(val_metrics.get('distress_f1', 0.0))
            
            logger.info(f"Val - Loss: {val_metrics.get('loss', 0.0):.4f}, "
                       f"Accuracy: {val_metrics.get('distress_accuracy', 0.0):.4f}, "
                       f"F1: {val_metrics.get('distress_f1', 0.0):.4f}, "
                       f"Recall: {val_metrics.get('distress_recall', 0.0):.4f}, "
                       f"FNR: {val_metrics.get('false_negative_rate', 0.0):.4f}")
            
            # Step scheduler (if using reduce_on_plateau)
            if scheduler is not None and hasattr(scheduler, 'step') and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_metric = val_metrics.get(early_stop_config.get('metric', 'loss'), 0.0)
                scheduler.step(scheduler_metric)
            
            # Check early stopping
            early_stop_metric = val_metrics.get(early_stop_config.get('metric', 'val_f1'), 0.0)
            should_stop = early_stopping(early_stop_metric, epoch, model)
            
            # Save checkpoint if best
            is_best = (early_stopping.counter == 0)
            
            history['best_metrics'] = val_metrics
            
            if should_stop:
                logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
                early_stopping.restore_best_model(model, device)
                break
        
        else:
            # Step other schedulers
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                save_dir,
                epoch,
                model,
                optimizer,
                scheduler,
                metrics=train_metrics,
                training_state={
                    'epoch': epoch,
                    'iterations_done': (epoch + 1) * len(train_loader)
                },
                save_optimizer=checkpoint_config.get('save_optimizer', True),
                save_scheduler=checkpoint_config.get('save_scheduler', True),
                save_training_state=checkpoint_config.get('save_training_state', True),
                is_best=(epoch + 1) % eval_interval == 0 and early_stopping.counter == 0,
                keep_best=checkpoint_config.get('keep_best', 5),
                best_metric_name=early_stop_config.get('metric')
            )
    
    # Plot training curves
    if not config.get('visualization', {}).get('plot_training_curves', True):
        pass
    else:
        visualizer.plot_training_curves(
            history['train_loss'],
            history['val_loss'],
            history['train_accuracy'],
            history['val_accuracy'],
            history['train_f1'],
            history['val_f1'],
            save_path="training_curves_final.png"
        )
        
        visualizer.plot_learning_rate(
            history['learning_rates'],
            save_path="learning_rate_schedule.png"
        )
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    return model, history


def main():
    """Main training function."""
    args = parse_args()
    
    # Print header
    print("=" * 80)
    print("CurioNext Distress Detection - Training Pipeline")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Setup
    setup_environment(args)
    device = get_device(args.device)
    config = load_config_and_override(args.config, args)
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create loss
    criterion = create_loss(config)
    
    # Load pretrained if specified
    start_epoch = 0
    if args.pretrained:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        load_checkpoint(args.pretrained, model, optimizer, scheduler, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info("Resuming from latest checkpoint...")
        try:
            start_epoch, info = resume_from_checkpoint(args.save_dir, model, optimizer, scheduler, device)
            logger.info(f"Resumed from epoch {start_epoch}")
        except FileNotFoundError:
            logger.warning("No checkpoint found, starting from epoch 0")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, args.data_dir)
    
    # Training loop
    model, history = training_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device,
        args.save_dir,
        args.plot_dir,
        start_epoch
    )
    
    # Save final model
    final_path = Path(args.save_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Save training history
    history_path = Path(args.log_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(
            {k: v for k, v in history.items() if k != 'best_metrics'},
            f,
            indent=2
        )
    logger.info(f"Saved training history to {history_path}")
    
    print("\n" + "=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("Training complete! Checkpoints saved to:", args.save_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
