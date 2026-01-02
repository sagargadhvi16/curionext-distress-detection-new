"""
Test script for training infrastructure:
- MultiTaskLoss
- train_epoch
- validate_epoch
- AdaptiveLRScheduler
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.fusion.loss import MultiTaskLoss, create_multitask_loss
from src.fusion.training import train_epoch, validate_epoch
from src.fusion.scheduler import AdaptiveLRScheduler, create_scheduler
from src.fusion.model import DistressDetectionModel


def create_dummy_dataset(batch_size=8, num_samples=32):
    """Create dummy dataset for testing."""
    # Create dummy data
    audio_features = torch.randn(num_samples, 256)  # Already encoded
    biometric_features = torch.randn(num_samples, 10, 10)  # Time series
    context_features = torch.randn(num_samples, 64)
    
    # Create dummy labels
    labels = torch.randint(0, 2, (num_samples,))
    severity = torch.rand(num_samples) * 10.0  # 0-10 scale
    distress_type = torch.randint(0, 5, (num_samples,))
    
    # Create dataset
    dataset = TensorDataset(
        audio_features,
        biometric_features,
        context_features,
        labels,
        severity,
        distress_type
    )
    
    # Custom collate function to create dict
    def collate_fn(batch):
        audio, bio, context, label, sev, dtype = zip(*batch)
        return {
            'audio': torch.stack(audio),
            'biometric': torch.stack(bio),
            'context': torch.stack(context),
            'label': torch.stack(label),
            'severity': torch.stack(sev),
            'distress_type': torch.stack(dtype)
        }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


def test_multitask_loss():
    """Test MultiTaskLoss class."""
    print("=" * 60)
    print("Testing MultiTaskLoss")
    print("=" * 60)
    
    # Create loss function
    criterion = MultiTaskLoss(
        distress_weight=1.0,
        severity_weight=0.5,
        type_weight=0.5,
        class_weights=torch.tensor([1.0, 2.0])  # Emphasize distress class
    )
    
    # Create dummy predictions
    batch_size = 4
    predictions = {
        'distress_logits': torch.randn(batch_size, 2),
        'severity': torch.rand(batch_size, 1) * 10.0,
        'type_logits': torch.randn(batch_size, 5)
    }
    
    targets = {
        'label': torch.randint(0, 2, (batch_size,)),
        'severity': torch.rand(batch_size, 1) * 10.0,
        'distress_type': torch.randint(0, 5, (batch_size,))
    }
    
    # Compute loss
    loss_dict = criterion(predictions, targets)
    
    print("\nLoss Components:")
    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Distress Loss: {loss_dict['distress_loss'].item():.4f}")
    print(f"  Severity Loss: {loss_dict['severity_loss'].item():.4f}")
    print(f"  Type Loss: {loss_dict['type_loss'].item():.4f}")
    
    assert loss_dict['total_loss'].item() > 0, "Loss should be positive"
    assert 'distress_loss' in loss_dict
    assert 'severity_loss' in loss_dict
    assert 'type_loss' in loss_dict
    
    print("[OK] MultiTaskLoss test passed!")
    
    # Test factory function
    config = {
        'distress_weight': 1.0,
        'severity_weight': 0.5,
        'type_weight': 0.5,
        'class_weights': [1.0, 2.0]
    }
    criterion2 = create_multitask_loss(config)
    assert isinstance(criterion2, MultiTaskLoss)
    print("[OK] create_multitask_loss factory function works!")
    
    return criterion


def test_train_epoch(criterion):
    """Test train_epoch function."""
    print("\n" + "=" * 60)
    print("Testing train_epoch")
    print("=" * 60)
    
    # Create dummy model
    model = DistressDetectionModel(
        audio_encoder=None,  # Use placeholder
        biometric_encoder=None,  # Use placeholder
        use_attention_fusion=False
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy dataloader
    train_loader = create_dummy_dataset(batch_size=4, num_samples=16)
    
    device = torch.device('cpu')
    
    # Train for one epoch
    print("\nTraining for one epoch...")
    metrics = train_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=1.0,
        log_interval=2
    )
    
    print("\nTraining Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Distress Loss: {metrics['distress_loss']:.4f}")
    print(f"  Severity Loss: {metrics['severity_loss']:.4f}")
    print(f"  Type Loss: {metrics['type_loss']:.4f}")
    print(f"  Distress Accuracy: {metrics['distress_accuracy']:.4f}")
    print(f"  Distress F1: {metrics['distress_f1']:.4f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"  Severity MAE: {metrics['severity_mae']:.4f}")
    print(f"  Type Accuracy: {metrics['type_accuracy']:.4f}")
    
    assert 'loss' in metrics
    assert 'distress_accuracy' in metrics
    assert 'false_negative_rate' in metrics
    
    print("[OK] train_epoch test passed!")
    
    return model


def test_validate_epoch(model, criterion):
    """Test validate_epoch function."""
    print("\n" + "=" * 60)
    print("Testing validate_epoch")
    print("=" * 60)
    
    # Create dummy dataloader
    val_loader = create_dummy_dataset(batch_size=4, num_samples=16)
    
    device = torch.device('cpu')
    
    # Validate
    print("\nValidating...")
    metrics = validate_epoch(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device
    )
    
    print("\nValidation Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Distress Loss: {metrics['distress_loss']:.4f}")
    print(f"  Severity Loss: {metrics['severity_loss']:.4f}")
    print(f"  Type Loss: {metrics['type_loss']:.4f}")
    print(f"  Distress Accuracy: {metrics['distress_accuracy']:.4f}")
    print(f"  Distress Precision: {metrics['distress_precision']:.4f}")
    print(f"  Distress Recall: {metrics['distress_recall']:.4f}")
    print(f"  Distress F1: {metrics['distress_f1']:.4f}")
    print(f"  Distress AUC: {metrics.get('distress_auc', 0.0):.4f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"  Severity MAE: {metrics['severity_mae']:.4f}")
    print(f"  Severity RMSE: {metrics['severity_rmse']:.4f}")
    print(f"  Type Accuracy: {metrics['type_accuracy']:.4f}")
    
    assert 'loss' in metrics
    assert 'distress_auc' in metrics
    assert 'severity_rmse' in metrics
    
    print("[OK] validate_epoch test passed!")
    
    return metrics


def test_scheduler():
    """Test AdaptiveLRScheduler."""
    print("\n" + "=" * 60)
    print("Testing AdaptiveLRScheduler")
    print("=" * 60)
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test ReduceLROnPlateau
    print("\n1. Testing ReduceLROnPlateau scheduler...")
    scheduler1 = AdaptiveLRScheduler(
        optimizer,
        scheduler_type="reduce_on_plateau",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    initial_lr = scheduler1.get_last_lr()[0]
    print(f"   Initial LR: {initial_lr:.6f}")
    
    # Step with decreasing loss (should not reduce LR)
    scheduler1.step(metrics=0.5)
    lr_after_1 = scheduler1.get_last_lr()[0]
    print(f"   LR after step 1 (loss=0.5): {lr_after_1:.6f}")
    assert lr_after_1 == initial_lr, "LR should not change yet"
    
    # Step with same loss multiple times (should reduce after patience)
    for i in range(3):
        scheduler1.step(metrics=0.6)  # Worse metric
    lr_after_patience = scheduler1.get_last_lr()[0]
    print(f"   LR after patience: {lr_after_patience:.6f}")
    
    print("[OK] ReduceLROnPlateau scheduler test passed!")
    
    # Test CosineAnnealingLR
    print("\n2. Testing CosineAnnealingLR scheduler...")
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler2 = AdaptiveLRScheduler(
        optimizer2,
        scheduler_type="cosine_annealing",
        T_max=10,
        eta_min=0.0001
    )
    
    initial_lr2 = scheduler2.get_last_lr()[0]
    print(f"   Initial LR: {initial_lr2:.6f}")
    
    # Step scheduler
    scheduler2.step()
    lr_after_step = scheduler2.get_last_lr()[0]
    print(f"   LR after step: {lr_after_step:.6f}")
    assert lr_after_step != initial_lr2, "LR should change with cosine annealing"
    
    print("[OK] CosineAnnealingLR scheduler test passed!")
    
    # Test factory function
    print("\n3. Testing create_scheduler factory function...")
    optimizer3 = torch.optim.Adam(model.parameters(), lr=0.001)
    config = {
        'type': 'reduce_on_plateau',
        'factor': 0.5,
        'patience': 5
    }
    scheduler3 = create_scheduler(optimizer3, config)
    assert isinstance(scheduler3, AdaptiveLRScheduler)
    print("[OK] create_scheduler factory function works!")
    
    print("\n[OK] All scheduler tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Training Infrastructure Test Suite")
    print("=" * 60)
    
    try:
        # Test loss
        criterion = test_multitask_loss()
        
        # Test training
        model = test_train_epoch(criterion)
        
        # Test validation
        test_validate_epoch(model, criterion)
        
        # Test scheduler
        test_scheduler()
        
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("  - MultiTaskLoss: Working")
        print("  - train_epoch: Working")
        print("  - validate_epoch: Working")
        print("  - AdaptiveLRScheduler: Working")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

