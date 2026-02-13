"""Integration script for WER-SSL fusion approach in CurioNext."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionTrainer:
    """Trainer for the distress detection model with WER-SSL fusion approach."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        use_augmentation: bool = True,
        use_tcn_refiner: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Distress detection model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            use_augmentation: Whether to use signal augmentation
            use_tcn_refiner: Whether TCN refiner is enabled in model
        """
        self.model = model.to(device)
        self.device = device
        self.use_augmentation = use_augmentation
        self.use_tcn_refiner = use_tcn_refiner
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions for multi-task learning
        self.loss_distress = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_severity = nn.MSELoss()
        self.loss_type = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Loss weights (can be tuned)
        self.loss_weights = {
            'distress': 1.0,
            'severity': 0.5,
            'type': 0.8
        }
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs
            targets: Target labels (dict or tensor)
            
        Returns:
            Total weighted loss
        """
        loss_total = 0.0
        
        # Handle both dict and tensor targets
        if not isinstance(targets, dict):
            # If targets is a tensor, assume it's just distress labels
            distress_labels = targets
            targets = {'distress': distress_labels}
        
        # Distress detection loss
        if 'distress_logits' in outputs and 'distress' in targets:
            loss_dist = self.loss_distress(
                outputs['distress_logits'],
                targets['distress'].long()
            )
            loss_total += self.loss_weights['distress'] * loss_dist
        
        # Severity regression loss
        if 'severity' in outputs and 'severity' in targets:
            loss_sev = self.loss_severity(
                outputs['severity'],
                targets['severity'].float()
            )
            loss_total += self.loss_weights['severity'] * loss_sev
        
        # Distress type classification loss
        if 'type_logits' in outputs and 'type' in targets:
            loss_type = self.loss_type(
                outputs['type_logits'],
                targets['type'].long()
            )
            loss_total += self.loss_weights['type'] * loss_type
        
        return loss_total
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        augmentor: Optional[object] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            augmentor: Optional signal augmentor (from src.utils.signal_augmentation)
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Extract batch data
            if isinstance(batch_data, (list, tuple)):
                audio, biometric, targets = batch_data[0], batch_data[1], batch_data[2]
            else:
                # Handle dict format
                audio = batch_data.get('audio')
                biometric = batch_data.get('biometric')
                targets = batch_data.get('targets')
            
            # Move to device
            audio = audio.to(self.device) if audio is not None else None
            biometric = biometric.to(self.device) if biometric is not None else None
            
            # Apply augmentation if enabled
            if self.use_augmentation and augmentor is not None and audio is not None:
                # Augmentation would be applied here
                # Example: audio = apply_augmentation(audio, augmentor)
                pass
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(audio, biometric)
            
            # Move targets to device
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in targets.items()}
            
            # Compute loss
            loss = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        
        return {"train_loss": avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        distress_correct = 0
        type_correct = 0
        total_samples = 0
        
        for batch_data in val_loader:
            # Extract batch data
            if isinstance(batch_data, (list, tuple)):
                audio, biometric, targets = batch_data[0], batch_data[1], batch_data[2]
            else:
                audio = batch_data.get('audio')
                biometric = batch_data.get('biometric')
                targets = batch_data.get('targets')
            
            # Move to device
            audio = audio.to(self.device) if audio is not None else None
            biometric = biometric.to(self.device) if biometric is not None else None
            
            # Move targets to device and convert to dict if needed
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in targets.items()}
            else:
                # Convert tensor targets to dict
                targets = {'distress': targets.to(self.device) if torch.is_tensor(targets) else targets}
            
            # Forward pass
            outputs = self.model(audio, biometric)
            
            # Compute loss
            loss = self.compute_loss(outputs, targets)
            val_loss += loss.item()
            
            # Compute accuracy
            if 'distress_logits' in outputs and 'distress' in targets:
                preds = torch.argmax(outputs['distress_logits'], dim=1)
                distress_correct += (preds == targets['distress']).sum().item()
            
            if 'type_logits' in outputs and 'type' in targets:
                preds = torch.argmax(outputs['type_logits'], dim=1)
                type_correct += (preds == targets['type']).sum().item()
            
            total_samples += audio.shape[0] if audio is not None else biometric.shape[0]
            num_batches += 1
        
        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
        distress_acc = distress_correct / total_samples if total_samples > 0 else 0.0
        type_acc = type_correct / total_samples if total_samples > 0 else 0.0
        
        self.val_losses.append(avg_loss)
        
        return {
            "val_loss": avg_loss,
            "distress_accuracy": distress_acc,
            "type_accuracy": type_acc
        }


def create_dummy_dataloaders(
    num_samples: int = 100,
    batch_size: int = 32,
    audio_dim: int = 256,
    bio_dim: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy data loaders for testing.
    
    Args:
        num_samples: Number of samples
        batch_size: Batch size
        audio_dim: Audio feature dimension
        bio_dim: Biometric feature dimension
        device: Device
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dummy data
    audio_data = torch.randn(num_samples, audio_dim)
    bio_data = torch.randn(num_samples, bio_dim)
    
    distress_labels = torch.randint(0, 2, (num_samples,))
    severity_labels = torch.rand(num_samples, 1) * 10
    type_labels = torch.randint(0, 5, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(audio_data, bio_data, distress_labels, severity_labels, type_labels)
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def train_with_fusion(
    fusion_type: str = "transformer",
    use_tcn_refiner: bool = False,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = None
):
    """
    Complete training pipeline with WER-SSL fusion approach.
    
    Args:
        fusion_type: Type of fusion ("transformer", "attention", or "late")
        use_tcn_refiner: Whether to use TCN refiner for audio
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Trained model and metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Fusion type: {fusion_type}")
    logger.info(f"TCN refiner enabled: {use_tcn_refiner}")
    
    # Import model
    from src.fusion.model import DistressDetectionModel
    
    # Create model
    model = DistressDetectionModel(
        fusion_type=fusion_type,
        use_tcn_refiner=use_tcn_refiner,
        audio_dim=256,
        bio_dim=256,
        context_dim=64,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dropout=0.2
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dummy_dataloaders(
        num_samples=200,
        batch_size=batch_size,
        device=device
    )
    
    # Create trainer
    trainer = FusionTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        use_augmentation=True,
        use_tcn_refiner=use_tcn_refiner
    )
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"Distress Accuracy: {val_metrics['distress_accuracy']:.4f}")
        logger.info(f"Type Accuracy: {val_metrics['type_accuracy']:.4f}")
    
    return model, {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses
    }


if __name__ == "__main__":
    # Example usage
    logger.info("=== WER-SSL Fusion Integration Test ===\n")
    
    # Train with Transformer fusion (recommended)
    logger.info("Training with Transformer fusion (WER-SSL approach)...")
    model_transformer, metrics_transformer = train_with_fusion(
        fusion_type="transformer",
        use_tcn_refiner=True,
        num_epochs=5,
        batch_size=32
    )
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Final train loss: {metrics_transformer['train_losses'][-1]:.4f}")
    logger.info(f"Final val loss: {metrics_transformer['val_losses'][-1]:.4f}")
