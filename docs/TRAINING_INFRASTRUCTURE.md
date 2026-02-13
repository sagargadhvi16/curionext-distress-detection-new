# Training Infrastructure Documentation

This document describes the training infrastructure implemented for the fusion layer module.

## Overview

The training infrastructure consists of four main components:

1. **MultiTaskLoss** - Weighted multi-task loss function
2. **train_epoch()** - Training loop with gradient clipping and logging
3. **validate_epoch()** - Validation loop with comprehensive metrics
4. **AdaptiveLRScheduler** - Learning rate scheduling with multiple strategies

## 1. MultiTaskLoss Class (`src/fusion/loss.py`)

### Purpose

Combines three loss functions for multi-task learning:
- **Binary Cross-Entropy (BCE)** for distress detection
- **Mean Squared Error (MSE)** for severity regression
- **Cross-Entropy (CE)** for distress type classification

### Features

- Weighted combination of losses to balance task importance
- Class weights for BCE loss to reduce false negatives (emphasize distress class)
- Configurable reduction method ('mean', 'sum', or 'none')

### Usage

```python
from src.fusion.loss import MultiTaskLoss, create_multitask_loss

# Method 1: Direct instantiation
criterion = MultiTaskLoss(
    distress_weight=1.0,
    severity_weight=0.5,
    type_weight=0.5,
    class_weights=torch.tensor([1.0, 2.0])  # Emphasize distress class
)

# Method 2: From config
config = {
    'distress_weight': 1.0,
    'severity_weight': 0.5,
    'type_weight': 0.5,
    'class_weights': [1.0, 2.0]
}
criterion = create_multitask_loss(config)

# Compute loss
predictions = {
    'distress_logits': model_output['distress_logits'],
    'severity': model_output['severity'],
    'type_logits': model_output['type_logits']
}

targets = {
    'label': labels,
    'severity': severity_targets,
    'distress_type': type_labels
}

loss_dict = criterion(predictions, targets)
total_loss = loss_dict['total_loss']
```

### Loss Dictionary

Returns a dictionary with:
- `total_loss`: Combined weighted loss
- `distress_loss`: Binary classification loss
- `severity_loss`: Regression loss (MSE)
- `type_loss`: Classification loss (CE)

## 2. train_epoch() Function (`src/fusion/training.py`)

### Purpose

Complete training loop for one epoch with:
- Forward pass through model
- Loss calculation
- Backpropagation
- Gradient clipping
- Progress logging
- Metric computation

### Features

- Gradient clipping for stable training
- Per-batch logging with configurable interval
- Accumulates metrics across all batches
- Progress bar with tqdm
- Returns comprehensive metrics dictionary

### Usage

```python
from src.fusion.training import train_epoch

model.train()  # Set model to training mode

metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    max_grad_norm=1.0,  # Gradient clipping threshold
    log_interval=10     # Log every N batches
)

print(f"Training Loss: {metrics['loss']:.4f}")
print(f"Distress F1: {metrics['distress_f1']:.4f}")
print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
```

### Returned Metrics

- `loss`: Total average loss
- `distress_loss`: Average distress classification loss
- `severity_loss`: Average severity regression loss
- `type_loss`: Average type classification loss
- `distress_accuracy`: Binary classification accuracy
- `distress_precision`: Precision score
- `distress_recall`: Recall score
- `distress_f1`: F1 score
- `false_negative_rate`: Critical metric (FN / (FN + TP))
- `severity_mae`: Mean Absolute Error for severity
- `severity_rmse`: Root Mean Squared Error for severity
- `type_accuracy`: Distress type classification accuracy

## 3. validate_epoch() Function (`src/fusion/training.py`)

### Purpose

Validation loop that computes all metrics without gradient updates.

### Features

- No gradient computation (efficiency)
- Comprehensive metric calculation
- Includes AUC for binary classification
- Returns same metrics as training plus additional validation metrics

### Usage

```python
from src.fusion.training import validate_epoch

model.eval()  # Set model to evaluation mode

val_metrics = validate_epoch(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device
)

print(f"Validation Loss: {val_metrics['loss']:.4f}")
print(f"Validation F1: {val_metrics['distress_f1']:.4f}")
print(f"Validation AUC: {val_metrics['distress_auc']:.4f}")
```

### Returned Metrics

Same as `train_epoch()` plus:
- `distress_auc`: ROC AUC score (requires probabilities)

## 4. AdaptiveLRScheduler (`src/fusion/scheduler.py`)

### Purpose

Wrapper for adaptive learning rate scheduling with multiple strategies.

### Supported Schedulers

1. **ReduceLROnPlateau**
   - Reduces LR when metric plateaus
   - Configurable factor, patience, and minimum LR
   - Requires metric value in `step()` call

2. **CosineAnnealingLR**
   - Cosine annealing schedule
   - Requires T_max (maximum epochs)
   - Optional minimum LR (eta_min)

3. **CosineAnnealingWarmRestarts**
   - Cosine annealing with warm restarts
   - Configurable restart period (T_0)
   - Periodic LR increases

### Usage

```python
from src.fusion.scheduler import AdaptiveLRScheduler, create_scheduler

# Method 1: Direct instantiation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ReduceLROnPlateau
scheduler = AdaptiveLRScheduler(
    optimizer,
    scheduler_type="reduce_on_plateau",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    mode='min'  # 'min' for loss, 'max' for accuracy/F1
)

# Step with metric (for ReduceLROnPlateau)
scheduler.step(metrics=val_loss)

# CosineAnnealingLR
scheduler = AdaptiveLRScheduler(
    optimizer,
    scheduler_type="cosine_annealing",
    T_max=50,  # Number of epochs
    eta_min=0.0001
)

# Step without metric (for cosine schedulers)
scheduler.step()

# Method 2: From config
config = {
    'type': 'reduce_on_plateau',
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-6
}
scheduler = create_scheduler(optimizer, config)

# Get current learning rate
current_lr = scheduler.get_last_lr()[0]
```

### Integration in Training Loop

```python
for epoch in range(num_epochs):
    # Training
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validation
    val_metrics = validate_epoch(model, val_loader, criterion, device)
    
    # Update learning rate
    if scheduler.is_metric_based:
        # For ReduceLROnPlateau, use validation metric
        scheduler.step(metrics=val_metrics['loss'])
    else:
        # For cosine schedulers
        scheduler.step()
    
    print(f"Epoch {epoch+1}: LR = {scheduler.get_last_lr()[0]:.6f}")
```

## Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from src.fusion.model import DistressDetectionModel
from src.fusion.loss import create_multitask_loss
from src.fusion.training import train_epoch, validate_epoch
from src.fusion.scheduler import create_scheduler

# Initialize model
model = DistressDetectionModel(use_attention_fusion=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create loss function
loss_config = {
    'distress_weight': 1.0,
    'severity_weight': 0.5,
    'type_weight': 0.5,
    'class_weights': [1.0, 2.0]
}
criterion = create_multitask_loss(loss_config)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Create scheduler
scheduler_config = {
    'type': 'reduce_on_plateau',
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-6
}
scheduler = create_scheduler(optimizer, scheduler_config)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train
    train_metrics = train_epoch(
        model, train_loader, criterion, optimizer, device,
        max_grad_norm=1.0, log_interval=10
    )
    
    # Validate
    val_metrics = validate_epoch(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(metrics=val_metrics['loss'])
    
    # Log metrics
    print(f"Train Loss: {train_metrics['loss']:.4f}, "
          f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Train F1: {train_metrics['distress_f1']:.4f}, "
          f"Val F1: {val_metrics['distress_f1']:.4f}")
    print(f"FNR: {val_metrics['false_negative_rate']:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
```

## Configuration File Integration

The training infrastructure integrates with YAML configuration files:

```yaml
# configs/training_config.yaml

loss:
  distress_weight: 1.0
  severity_weight: 0.5
  type_weight: 0.5
  class_weights: [1.0, 2.0]  # Emphasize distress class

training:
  learning_rate: 0.001
  max_grad_norm: 1.0
  log_interval: 10

scheduler:
  type: "reduce_on_plateau"
  factor: 0.5
  patience: 5
  min_lr: 0.00001
```

## File Structure

```
src/fusion/
├── loss.py          # MultiTaskLoss class
├── training.py      # train_epoch, validate_epoch functions
├── scheduler.py     # AdaptiveLRScheduler class
└── ...

src/utils/
└── metrics.py       # compute_metrics, compute_false_negative_rate

scripts/
└── train.py         # Main training script (uses above functions)
```

## Key Features

1. **Multi-task Learning**: Handles three tasks simultaneously with weighted losses
2. **False Negative Reduction**: Class weights emphasize distress detection
3. **Gradient Clipping**: Prevents exploding gradients
4. **Comprehensive Metrics**: Tracks all task-specific metrics
5. **Flexible Scheduling**: Multiple LR scheduling strategies
6. **Progress Tracking**: Built-in logging and progress bars

## Dependencies

- PyTorch (torch, torch.nn, torch.optim)
- NumPy
- scikit-learn (for metrics)
- tqdm (for progress bars)

## Notes

- All code follows existing codebase conventions
- Only fusion-related code modified (as per Intern 3's scope)
- Compatible with existing model and dataset classes
- Windows-compatible (no Unicode issues)

