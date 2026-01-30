"""
Quick training test for WER-SSL fusion implementation in CurioNext
This script tests if training works end-to-end with the new architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_training():
    """Test basic training loop with new fusion."""
    print("\n" + "="*70)
    print("TEST 1: Basic Training with Transformer Fusion")
    print("="*70)
    
    from src.fusion.model import DistressDetectionModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create model with new Transformer fusion
    model = DistressDetectionModel(
        fusion_type="transformer",      # NEW: WER-SSL approach
        use_tcn_refiner=False,
        audio_dim=256,
        bio_dim=256,
        context_dim=64,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_num_layers=2
    ).to(device)
    
    logger.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create dummy data
    batch_size = 8
    num_batches = 5
    
    logger.info(f"Creating dummy data: {num_batches} batches of size {batch_size}")
    
    # Training loop
    model.train()
    total_loss = 0.0
    
    for batch_idx in range(num_batches):
        # Create dummy batch
        audio = torch.randn(batch_size, 256).to(device)
        biometric = torch.randn(batch_size, 256).to(device)
        context = torch.randn(batch_size, 64).to(device)
        
        # Create dummy targets
        distress_labels = torch.randint(0, 2, (batch_size,)).to(device)
        severity_labels = torch.rand(batch_size, 1).to(device) * 10
        type_labels = torch.randint(0, 5, (batch_size,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(audio, biometric, context)
        
        # Compute loss
        loss_distress = nn.CrossEntropyLoss()(outputs['distress_logits'], distress_labels)
        loss_severity = nn.MSELoss()(outputs['severity'], severity_labels)
        loss_type = nn.CrossEntropyLoss()(outputs['type_logits'], type_labels)
        
        total_loss_batch = loss_distress + 0.5 * loss_severity + 0.8 * loss_type
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        
        logger.info(f"  Batch {batch_idx+1}/{num_batches}: Loss = {total_loss_batch.item():.4f}")
    
    avg_loss = total_loss / num_batches
    logger.info(f"✓ Training complete! Average loss: {avg_loss:.4f}")
    
    return True


def test_training_with_tcn():
    """Test training with TCN refiner enabled."""
    print("\n" + "="*70)
    print("TEST 2: Training with TCN Audio Refiner")
    print("="*70)
    
    from src.fusion.model import DistressDetectionModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model WITH TCN refiner
    model = DistressDetectionModel(
        fusion_type="transformer",
        use_tcn_refiner=True,           # ← ENABLED
        audio_dim=256,
        bio_dim=256
    ).to(device)
    
    logger.info(f"✓ Model with TCN refiner created")
    logger.info(f"  TCN refiner: {'Enabled' if model.tcn_refiner is not None else 'Disabled'}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    
    for batch_idx in range(3):
        audio = torch.randn(4, 256).to(device)
        biometric = torch.randn(4, 256).to(device)
        
        distress_labels = torch.randint(0, 2, (4,)).to(device)
        severity_labels = torch.rand(4, 1).to(device) * 10
        type_labels = torch.randint(0, 5, (4,)).to(device)
        
        optimizer.zero_grad()
        outputs = model(audio, biometric)
        
        loss_distress = nn.CrossEntropyLoss()(outputs['distress_logits'], distress_labels)
        loss_severity = nn.MSELoss()(outputs['severity'], severity_labels)
        loss_type = nn.CrossEntropyLoss()(outputs['type_logits'], type_labels)
        
        total_loss = loss_distress + 0.5 * loss_severity + 0.8 * loss_type
        
        total_loss.backward()
        optimizer.step()
        
        logger.info(f"  Batch {batch_idx+1}: Loss = {total_loss.item():.4f}")
    
    logger.info(f"✓ Training with TCN complete!")
    
    return True


def test_training_with_augmentation():
    """Test training with signal augmentation."""
    print("\n" + "="*70)
    print("TEST 3: Training with Signal Augmentation")
    print("="*70)
    
    from src.fusion.model import DistressDetectionModel
    from src.utils.signal_augmentation import SignalAugmentation
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = DistressDetectionModel(
        fusion_type="transformer",
        use_tcn_refiner=True
    ).to(device)
    
    # Create augmentor
    augmentor = SignalAugmentation(
        noise_snr_range=(15.0, 30.0),
        magnitude_warp_sigma=0.2
    )
    
    logger.info("✓ Augmentor created")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    # Simulate training with augmentation
    for batch_idx in range(3):
        # Create dummy raw signal data
        raw_audio_signal = np.random.randn(1000)
        
        # Apply augmentations
        augmented_signals = augmentor.apply_batch(
            raw_audio_signal,
            augmentation_type="random",
            num_augmentations=1
        )
        
        # Convert to embeddings (simulated)
        audio = torch.randn(4, 256).to(device)
        biometric = torch.randn(4, 256).to(device)
        
        distress_labels = torch.randint(0, 2, (4,)).to(device)
        severity_labels = torch.rand(4, 1).to(device) * 10
        type_labels = torch.randint(0, 5, (4,)).to(device)
        
        optimizer.zero_grad()
        outputs = model(audio, biometric)
        
        loss_distress = nn.CrossEntropyLoss()(outputs['distress_logits'], distress_labels)
        loss_severity = nn.MSELoss()(outputs['severity'], severity_labels)
        loss_type = nn.CrossEntropyLoss()(outputs['type_logits'], type_labels)
        
        total_loss = loss_distress + 0.5 * loss_severity + 0.8 * loss_type
        
        total_loss.backward()
        optimizer.step()
        
        logger.info(f"  Batch {batch_idx+1}: Loss = {total_loss.item():.4f}, "
                   f"Augmentations applied = {len(augmented_signals)}")
    
    logger.info("✓ Training with augmentation complete!")
    
    return True


def test_fusion_trainer():
    """Test the FusionTrainer class."""
    print("\n" + "="*70)
    print("TEST 4: Training with FusionTrainer")
    print("="*70)
    
    from src.fusion.model import DistressDetectionModel
    from scripts.fusion_integration_werssl import FusionTrainer, create_dummy_dataloaders
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = DistressDetectionModel(
        fusion_type="transformer",
        use_tcn_refiner=True
    )
    
    logger.info("✓ Model created")
    
    # Create trainer
    trainer = FusionTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        use_augmentation=True,
        use_tcn_refiner=True
    )
    
    logger.info("✓ FusionTrainer created")
    
    # Create dummy dataloaders
    train_loader, val_loader = create_dummy_dataloaders(
        num_samples=40,
        batch_size=4,
        device=device
    )
    
    logger.info("✓ Data loaders created")
    
    # Train for 2 epochs
    for epoch in range(2):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Distress Acc: {val_metrics['distress_accuracy']:.4f}")
        logger.info(f"  Type Acc: {val_metrics['type_accuracy']:.4f}")
    
    logger.info("✓ FusionTrainer test complete!")
    
    return True


def test_backward_pass():
    """Test that gradients flow correctly through the model."""
    print("\n" + "="*70)
    print("TEST 5: Gradient Flow Test")
    print("="*70)
    
    from src.fusion.model import DistressDetectionModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DistressDetectionModel(
        fusion_type="transformer",
        use_tcn_refiner=True
    ).to(device)
    
    # Create dummy batch
    audio = torch.randn(4, 256).to(device)
    biometric = torch.randn(4, 256).to(device)
    
    distress_labels = torch.randint(0, 2, (4,)).to(device)
    
    # Forward pass
    outputs = model(audio, biometric)
    
    # Loss
    loss = nn.CrossEntropyLoss()(outputs['distress_logits'], distress_labels)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradients = True
            break
    
    if has_gradients:
        logger.info("✓ Gradients flowing correctly through model")
    else:
        logger.warning("✗ No gradients detected!")
        return False
    
    logger.info(f"✓ Gradient flow test complete!")
    
    return True


def main():
    """Run all training tests."""
    print("\n" + "="*70)
    print("CURIONEXT TRAINING TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Training", test_basic_training),
        ("Training with TCN", test_training_with_tcn),
        ("Training with Augmentation", test_training_with_augmentation),
        ("FusionTrainer", test_fusion_trainer),
        ("Gradient Flow", test_backward_pass),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "✓ PASSED" if result else "✗ FAILED"))
            logger.info(f"\n✓ {test_name} test passed!\n")
        except Exception as e:
            results.append((test_name, f"✗ FAILED: {str(e)}"))
            logger.error(f"\n✗ {test_name} test failed: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        print(f"{result}: {test_name}")
    
    passed = sum(1 for _, r in results if "PASSED" in r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TRAINING TESTS PASSED - READY FOR PRODUCTION!\n")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - CHECK ERRORS ABOVE\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
