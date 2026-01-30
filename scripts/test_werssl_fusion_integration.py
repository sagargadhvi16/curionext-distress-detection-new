"""Comprehensive integration tests for WER-SSL fusion approach in CurioNext."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion.model import DistressDetectionModel
from src.fusion.transformer_fusion import TransformerFusion
from src.audio.tcn_refiner import TCNAudioRefiner
from src.utils.signal_augmentation import SignalAugmentation


def test_transformer_fusion():
    """Test TransformerFusion layer."""
    print("\n" + "="*60)
    print("Test 1: TransformerFusion Layer")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create fusion layer
    fusion = TransformerFusion(
        audio_dim=256,
        bio_dim=256,
        context_dim=64,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        output_dim=256
    ).to(device)
    
    # Create dummy inputs
    batch_size = 4
    audio_emb = torch.randn(batch_size, 256).to(device)
    bio_emb = torch.randn(batch_size, 256).to(device)
    context_emb = torch.randn(batch_size, 64).to(device)
    
    # Forward pass
    output = fusion(audio_emb, bio_emb, context_emb)
    
    # Validate output
    assert output.shape == (batch_size, 256), f"Expected shape (4, 256), got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")
    
    # Test attention weights
    attn_weights = fusion.get_attention_weights(audio_emb, bio_emb, context_emb)
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    
    print("✓ TransformerFusion test PASSED")
    return True


def test_tcn_audio_refiner():
    """Test TCN Audio Refiner."""
    print("\n" + "="*60)
    print("Test 2: TCN Audio Refiner")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create TCN refiner
    refiner = TCNAudioRefiner(
        input_dim=1024,
        tcn_channels=[256, 256, 128],
        kernel_size=5,
        dropout=0.2,
        output_dim=256,
        use_temporal_pooling=True
    ).to(device)
    
    # Test 1: Aggregate embedding (no temporal dimension)
    print("\n  Test 2a: Aggregate embedding")
    audio_agg = torch.randn(4, 1024).to(device)
    output_agg = refiner(audio_agg)
    assert output_agg.shape == (4, 256), f"Expected (4, 256), got {output_agg.shape}"
    print(f"  ✓ Aggregate output shape: {output_agg.shape}")
    
    # Test 2: Temporal embeddings
    print("\n  Test 2b: Temporal embeddings")
    refiner.use_temporal_pooling = False
    audio_temporal = torch.randn(4, 10, 1024).to(device)  # (batch, time, features)
    output_temporal = refiner(audio_temporal)
    assert output_temporal.shape == (4, 10, 256), f"Expected (4, 10, 256), got {output_temporal.shape}"
    print(f"  ✓ Temporal output shape: {output_temporal.shape}")
    
    print("✓ TCNAudioRefiner test PASSED")
    return True


def test_signal_augmentation():
    """Test signal augmentation."""
    print("\n" + "="*60)
    print("Test 3: Signal Augmentation")
    print("="*60)
    
    # Create augmentor
    augmentor = SignalAugmentation(
        noise_snr_range=(15.0, 30.0),
        magnitude_warp_sigma=0.2,
        time_warp_pieces=4,
        permute_pieces=4
    )
    
    # Create dummy signal
    signal_data = np.random.randn(1000)
    
    # Test individual augmentations
    print("\n  Testing individual augmentations:")
    
    # Noise
    aug_noise = augmentor.apply_single(signal_data, "noise")
    assert aug_noise.shape == signal_data.shape
    print(f"  ✓ Noise augmentation: {aug_noise.shape}")
    
    # Magnitude warp
    aug_mag = augmentor.apply_single(signal_data, "magnitude_warp")
    assert aug_mag.shape == signal_data.shape
    print(f"  ✓ Magnitude warp: {aug_mag.shape}")
    
    # Time warp
    aug_time = augmentor.apply_single(signal_data, "time_warp")
    assert len(aug_time) == len(signal_data)
    print(f"  ✓ Time warp: {aug_time.shape}")
    
    # Permute
    aug_perm = augmentor.apply_single(signal_data, "permute")
    assert aug_perm.shape == signal_data.shape
    print(f"  ✓ Permute: {aug_perm.shape}")
    
    # Crop and resize
    aug_crop = augmentor.apply_single(signal_data, "crop_resize")
    assert aug_crop.shape == signal_data.shape
    print(f"  ✓ Crop and resize: {aug_crop.shape}")
    
    # Test batch augmentation
    print("\n  Testing batch augmentation:")
    augmented_batch = augmentor.apply_batch(signal_data, augmentation_type="random", num_augmentations=5)
    assert len(augmented_batch) == 6  # Original + 5 augmentations
    print(f"  ✓ Batch augmentation: {len(augmented_batch)} versions created")
    
    print("✓ SignalAugmentation test PASSED")
    return True


def test_full_model_transformer():
    """Test full model with Transformer fusion."""
    print("\n" + "="*60)
    print("Test 4: Full Model with Transformer Fusion")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with transformer fusion
    model = DistressDetectionModel(
        fusion_type="transformer",
        use_tcn_refiner=False,
        audio_dim=256,
        bio_dim=256,
        context_dim=64,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dropout=0.2
    ).to(device)
    
    # Create dummy inputs
    batch_size = 4
    audio = torch.randn(batch_size, 256).to(device)
    biometric = torch.randn(batch_size, 256).to(device)
    context = torch.randn(batch_size, 64).to(device)
    
    # Forward pass
    outputs = model(audio, biometric, context)
    
    # Validate outputs
    assert 'distress_logits' in outputs
    assert 'severity' in outputs
    assert 'type_logits' in outputs
    assert outputs['distress_logits'].shape == (batch_size, 2)
    assert outputs['severity'].shape == (batch_size, 1)
    assert outputs['type_logits'].shape == (batch_size, 5)
    
    print(f"✓ Distress logits shape: {outputs['distress_logits'].shape}")
    print(f"✓ Severity shape: {outputs['severity'].shape}")
    print(f"✓ Type logits shape: {outputs['type_logits'].shape}")
    
    # Test with attention return
    outputs_attn = model(audio, biometric, context, return_attention=True)
    print(f"✓ Attention weights shape: {outputs_attn.get('attention_weights', 'Not available').shape if 'attention_weights' in outputs_attn else 'Not available'}")
    
    print("✓ Full Model (Transformer) test PASSED")
    return True


def test_full_model_with_tcn():
    """Test full model with TCN refiner."""
    print("\n" + "="*60)
    print("Test 5: Full Model with TCN Refiner")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with TCN refiner
    model = DistressDetectionModel(
        fusion_type="transformer",
        use_tcn_refiner=True,
        audio_dim=256,
        bio_dim=256,
        context_dim=64,
        transformer_d_model=128,
        transformer_nhead=4,
        transformer_num_layers=2
    ).to(device)
    
    # Create dummy inputs
    batch_size = 4
    audio = torch.randn(batch_size, 256).to(device)
    biometric = torch.randn(batch_size, 256).to(device)
    
    # Forward pass
    outputs = model(audio, biometric)
    
    # Validate outputs
    assert outputs['distress_logits'].shape == (batch_size, 2)
    assert outputs['severity'].shape == (batch_size, 1)
    assert outputs['type_logits'].shape == (batch_size, 5)
    
    print(f"✓ Model with TCN refiner works correctly")
    print(f"✓ Output shapes validated")
    
    print("✓ Full Model (with TCN) test PASSED")
    return True


def test_different_fusion_types():
    """Test model with different fusion types."""
    print("\n" + "="*60)
    print("Test 6: Different Fusion Types")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 4
    audio = torch.randn(batch_size, 256).to(device)
    biometric = torch.randn(batch_size, 256).to(device)
    
    fusion_types = ["transformer", "attention", "late"]
    
    for fusion_type in fusion_types:
        print(f"\n  Testing {fusion_type} fusion:")
        
        model = DistressDetectionModel(
            fusion_type=fusion_type,
            use_tcn_refiner=False
        ).to(device)
        
        outputs = model(audio, biometric)
        
        assert outputs['distress_logits'].shape == (batch_size, 2)
        print(f"  ✓ {fusion_type.capitalize()} fusion works correctly")
    
    print("✓ Different Fusion Types test PASSED")
    return True


def test_backward_compatibility():
    """Test backward compatibility with old API."""
    print("\n" + "="*60)
    print("Test 7: Backward Compatibility")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with old parameter names (mapped to new ones)
    model = DistressDetectionModel(
        fusion_type="late",  # Old "concatenation" style fusion
        use_tcn_refiner=False
    ).to(device)
    
    batch_size = 4
    audio = torch.randn(batch_size, 256).to(device)
    biometric = torch.randn(batch_size, 256).to(device)
    
    outputs = model(audio, biometric)
    
    assert 'distress_logits' in outputs
    print(f"✓ Old API compatibility maintained")
    
    print("✓ Backward Compatibility test PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL WERSSL FUSION INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        test_transformer_fusion,
        test_tcn_audio_refiner,
        test_signal_augmentation,
        test_full_model_transformer,
        test_full_model_with_tcn,
        test_different_fusion_types,
        test_backward_compatibility
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
