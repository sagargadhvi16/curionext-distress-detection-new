"""Quick test to verify biometric encoder fix."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.fusion.model import DistressDetectionModel

print("Testing biometric encoder fix...")

# Create model with placeholder encoders
model = DistressDetectionModel(
    audio_encoder=None,
    biometric_encoder=None,
    use_attention_fusion=False
)

# Create test data matching the test_training_infrastructure.py format
batch_size = 4
audio_features = torch.randn(batch_size, 256)  # Pre-encoded audio
biometric_features = torch.randn(batch_size, 10, 10)  # Time-series: (batch, time, features)
context_features = torch.randn(batch_size, 64)

print(f"Audio shape: {audio_features.shape}")
print(f"Biometric shape: {biometric_features.shape}")
print(f"Context shape: {context_features.shape}")

# Try forward pass
try:
    outputs = model(audio_features, biometric_features, context_features)
    print(f"\n✅ Forward pass successful!")
    print(f"Outputs:")
    print(f"  - distress_logits: {outputs['distress_logits'].shape}")
    print(f"  - severity: {outputs['severity'].shape}")
    print(f"  - type_logits: {outputs['type_logits'].shape}")
    print("\n✅ Test PASSED! Biometric encoder handles (batch, 10, 10) input correctly.")
except Exception as e:
    print(f"\n❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
