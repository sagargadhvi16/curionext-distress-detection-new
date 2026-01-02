"""Training and validation functions for multi-task distress detection."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm

from src.fusion.loss import MultiTaskLoss
from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics, compute_false_negative_rate

logger = get_logger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    Train model for one epoch with gradient clipping and logging.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function (MultiTaskLoss)
        optimizer: Optimizer
        device: Device to run on
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: Log every N batches
    
    Returns:
        Dictionary with average losses and metrics for the epoch
    """
    model.train()
    
    # Accumulators
    total_loss = 0.0
    distress_loss_sum = 0.0
    severity_loss_sum = 0.0
    type_loss_sum = 0.0
    
    # Predictions and targets for metrics
    all_distress_preds = []
    all_distress_labels = []
    all_severity_preds = []
    all_severity_targets = []
    all_type_preds = []
    all_type_labels = []
    
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        audio = batch['audio'].to(device)
        biometric = batch['biometric'].to(device)
        context = batch.get('context')
        if context is not None:
            context = context.to(device)
        
        labels = batch['label'].to(device)
        severity_targets = batch['severity'].to(device)
        type_labels = batch['distress_type'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get model predictions
        outputs = model(
            audio_features=audio,
            biometric_features=biometric,
            context_features=context
        )
        
        # Prepare targets
        targets = {
            'label': labels,
            'severity': severity_targets,
            'distress_type': type_labels
        }
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        distress_loss_sum += loss_dict['distress_loss'].item()
        severity_loss_sum += loss_dict['severity_loss'].item()
        type_loss_sum += loss_dict['type_loss'].item()
        
        # Collect predictions for metrics
        with torch.no_grad():
            # Distress predictions
            distress_probs = torch.softmax(outputs['distress_logits'], dim=-1)
            distress_preds = torch.argmax(outputs['distress_logits'], dim=-1)
            all_distress_preds.extend(distress_preds.cpu().numpy())
            all_distress_labels.extend(labels.cpu().numpy())
            
            # Severity predictions
            all_severity_preds.extend(outputs['severity'].cpu().numpy().flatten())
            all_severity_targets.extend(severity_targets.cpu().numpy().flatten())
            
            # Type predictions
            type_preds = torch.argmax(outputs['type_logits'], dim=-1)
            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())
        
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            current_loss = loss.item()
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'd_loss': f'{loss_dict["distress_loss"].item():.4f}',
                's_loss': f'{loss_dict["severity_loss"].item():.4f}',
                't_loss': f'{loss_dict["type_loss"].item():.4f}'
            })
            logger.debug(
                f"Batch {batch_idx + 1}/{len(dataloader)}: "
                f"loss={current_loss:.4f}, "
                f"distress_loss={loss_dict['distress_loss'].item():.4f}, "
                f"severity_loss={loss_dict['severity_loss'].item():.4f}, "
                f"type_loss={loss_dict['type_loss'].item():.4f}"
            )
    
    # Compute average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_distress_loss = distress_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_severity_loss = severity_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_type_loss = type_loss_sum / num_batches if num_batches > 0 else 0.0
    
    # Compute metrics
    all_distress_preds = np.array(all_distress_preds)
    all_distress_labels = np.array(all_distress_labels)
    all_severity_preds = np.array(all_severity_preds)
    all_severity_targets = np.array(all_severity_targets)
    all_type_preds = np.array(all_type_preds)
    all_type_labels = np.array(all_type_labels)
    
    # Distress metrics
    distress_metrics = compute_metrics(
        all_distress_labels,
        all_distress_preds,
        y_prob=None
    )
    
    # Severity metrics (MAE and RMSE)
    severity_mae = np.mean(np.abs(all_severity_preds - all_severity_targets))
    severity_rmse = np.sqrt(np.mean((all_severity_preds - all_severity_targets) ** 2))
    
    # Type metrics
    type_accuracy = np.mean(all_type_preds == all_type_labels)
    
    # False negative rate (critical metric)
    fnr = compute_false_negative_rate(all_distress_labels, all_distress_preds)
    
    metrics = {
        'loss': avg_loss,
        'distress_loss': avg_distress_loss,
        'severity_loss': avg_severity_loss,
        'type_loss': avg_type_loss,
        'distress_accuracy': distress_metrics.get('accuracy', 0.0),
        'distress_precision': distress_metrics.get('precision', 0.0),
        'distress_recall': distress_metrics.get('recall', 0.0),
        'distress_f1': distress_metrics.get('f1', 0.0),
        'false_negative_rate': fnr,
        'severity_mae': severity_mae,
        'severity_rmse': severity_rmse,
        'type_accuracy': type_accuracy
    }
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model for one epoch (no gradient updates).
    
    Computes all metrics including:
    - Loss components
    - Distress detection metrics (accuracy, precision, recall, F1, FNR)
    - Severity regression metrics (MAE, RMSE)
    - Type classification accuracy
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function (MultiTaskLoss)
        device: Device to run on
    
    Returns:
        Dictionary with average losses and metrics for the epoch
    """
    model.eval()
    
    # Accumulators
    total_loss = 0.0
    distress_loss_sum = 0.0
    severity_loss_sum = 0.0
    type_loss_sum = 0.0
    
    # Predictions and targets for metrics
    all_distress_preds = []
    all_distress_labels = []
    all_distress_probs = []
    all_severity_preds = []
    all_severity_targets = []
    all_type_preds = []
    all_type_labels = []
    
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch in pbar:
            # Move batch to device
            audio = batch['audio'].to(device)
            biometric = batch['biometric'].to(device)
            context = batch.get('context')
            if context is not None:
                context = context.to(device)
            
            labels = batch['label'].to(device)
            severity_targets = batch['severity'].to(device)
            type_labels = batch['distress_type'].to(device)
            
            # Forward pass
            outputs = model(
                audio_features=audio,
                biometric_features=biometric,
                context_features=context
            )
            
            # Prepare targets
            targets = {
                'label': labels,
                'severity': severity_targets,
                'distress_type': type_labels
            }
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Accumulate losses
            total_loss += loss.item()
            distress_loss_sum += loss_dict['distress_loss'].item()
            severity_loss_sum += loss_dict['severity_loss'].item()
            type_loss_sum += loss_dict['type_loss'].item()
            
            # Collect predictions for metrics
            # Distress predictions
            distress_probs = torch.softmax(outputs['distress_logits'], dim=-1)
            distress_preds = torch.argmax(outputs['distress_logits'], dim=-1)
            all_distress_preds.extend(distress_preds.cpu().numpy())
            all_distress_labels.extend(labels.cpu().numpy())
            all_distress_probs.extend(distress_probs[:, 1].cpu().numpy())  # Probability of distress class
            
            # Severity predictions
            all_severity_preds.extend(outputs['severity'].cpu().numpy().flatten())
            all_severity_targets.extend(severity_targets.cpu().numpy().flatten())
            
            # Type predictions
            type_preds = torch.argmax(outputs['type_logits'], dim=-1)
            all_type_preds.extend(type_preds.cpu().numpy())
            all_type_labels.extend(type_labels.cpu().numpy())
            
            num_batches += 1
    
    # Compute average losses
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_distress_loss = distress_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_severity_loss = severity_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_type_loss = type_loss_sum / num_batches if num_batches > 0 else 0.0
    
    # Convert to numpy arrays
    all_distress_preds = np.array(all_distress_preds)
    all_distress_labels = np.array(all_distress_labels)
    all_distress_probs = np.array(all_distress_probs)
    all_severity_preds = np.array(all_severity_preds)
    all_severity_targets = np.array(all_severity_targets)
    all_type_preds = np.array(all_type_preds)
    all_type_labels = np.array(all_type_labels)
    
    # Distress metrics
    distress_metrics = compute_metrics(
        all_distress_labels,
        all_distress_preds,
        y_prob=all_distress_probs
    )
    
    # Severity metrics (MAE and RMSE)
    severity_mae = np.mean(np.abs(all_severity_preds - all_severity_targets))
    severity_rmse = np.sqrt(np.mean((all_severity_preds - all_severity_targets) ** 2))
    
    # Type metrics
    type_accuracy = np.mean(all_type_preds == all_type_labels)
    
    # False negative rate (critical metric)
    fnr = compute_false_negative_rate(all_distress_labels, all_distress_preds)
    
    metrics = {
        'loss': avg_loss,
        'distress_loss': avg_distress_loss,
        'severity_loss': avg_severity_loss,
        'type_loss': avg_type_loss,
        'distress_accuracy': distress_metrics.get('accuracy', 0.0),
        'distress_precision': distress_metrics.get('precision', 0.0),
        'distress_recall': distress_metrics.get('recall', 0.0),
        'distress_f1': distress_metrics.get('f1', 0.0),
        'distress_auc': distress_metrics.get('auc', 0.0),
        'false_negative_rate': fnr,
        'severity_mae': severity_mae,
        'severity_rmse': severity_rmse,
        'type_accuracy': type_accuracy
    }
    
    return metrics

