"""
Demo script for testing fusion layer features:
- Test fusion layers and classifier with dummy inputs
- Visualize attention weights
- Test rule-based system
- Calculate confidence scores
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.fusion.late_fusion import LateFusionLayer
from src.fusion.attention_fusion import AttentionFusion
from src.fusion.classifier import MultiTaskClassifier
from src.fusion.rule_engine import RuleEngine, AlertLevel
from src.fusion.explainer import visualize_attention_weights, calculate_confidence
from src.fusion.model import DistressDetectionModel


def test_fusion_layers():
    """Test fusion layers with dummy inputs."""
    print("=" * 60)
    print("Testing Fusion Layers with Dummy Inputs")
    print("=" * 60)
    
    batch_size = 4
    audio_dim = 256
    bio_dim = 256
    context_dim = 64
    
    # Create dummy inputs
    audio_emb = torch.randn(batch_size, audio_dim)
    bio_emb = torch.randn(batch_size, bio_dim)
    context_emb = torch.randn(batch_size, context_dim)
    
    # Test Late Fusion\n    # Integrate fusion into the testing process\n    audio_data = torch.randn(batch_size, audio_dim)  # Example audio data\n    fused_audio, random_curves = integrate_fusion(audio_data)\n    # Use fused_audio in the testing process\n    print(f"Fused Audio Shape: {fused_audio.shape}")
    print("\n1. Testing Late Fusion Layer...")
    late_fusion = LateFusionLayer(
        audio_dim=audio_dim,
        bio_dim=bio_dim,
        context_dim=context_dim,
        fusion_hidden_dims=[512, 256]
    )
    late_output = late_fusion(audio_emb, bio_emb, context_emb)
    print(f"   Late Fusion Output Shape: {late_output.shape}")
    print(f"   Expected: ({batch_size}, 256)")
    assert late_output.shape == (batch_size, 256), "Late fusion output shape mismatch!"
    
    # Test Attention Fusion
    print("\n2. Testing Attention Fusion Layer...")
    attn_fusion = AttentionFusion(
        audio_dim=audio_dim,
        bio_dim=bio_dim,
        context_dim=context_dim,
        hidden_dim=128
    )
    attn_output, attention_weights = attn_fusion(audio_emb, bio_emb, context_emb)
    print(f"   Attention Fusion Output Shape: {attn_output.shape}")
    print(f"   Attention Weights Shape: {attention_weights.shape}")
    print(f"   Attention Weights (sample 0): {attention_weights[0].detach().cpu().numpy()}")
    print(f"   Expected Output: ({batch_size}, 128)")
    print(f"   Expected Weights: ({batch_size}, 3)")
    assert attn_output.shape == (batch_size, 128), "Attention fusion output shape mismatch!"
    assert attention_weights.shape == (batch_size, 3), "Attention weights shape mismatch!"
    
    # Test Classifier
    print("\n3. Testing Multi-Task Classifier...")
    classifier = MultiTaskClassifier(
        input_dim=256,
        hidden_dim=128,
        num_distress_types=5
    )
    classifier_output = classifier(late_output)
    print(f"   Distress Logits Shape: {classifier_output['distress_logits'].shape}")
    print(f"   Severity Shape: {classifier_output['severity'].shape}")
    print(f"   Type Logits Shape: {classifier_output['type_logits'].shape}")
    assert classifier_output['distress_logits'].shape == (batch_size, 2)
    assert classifier_output['severity'].shape == (batch_size, 1)
    assert classifier_output['type_logits'].shape == (batch_size, 5)
    
    print("\n[OK] All fusion layer tests passed!")
    return attention_weights, classifier_output


def test_attention_visualization(attention_weights):
    """Test attention weight visualization."""
    print("\n" + "=" * 60)
    print("Testing Attention Weight Visualization")
    print("=" * 60)
    
    try:
        # Visualize attention weights
        print("\nCreating attention weight heatmap...")
        fig = visualize_attention_weights(
            attention_weights[:3],  # Use first 3 samples
            modalities=['Audio', 'Biometric', 'Context'],
            sample_indices=['Sample 1', 'Sample 2', 'Sample 3']
        )
        print("[OK] Attention visualization created successfully!")
        print("   (Figure saved in memory, can be displayed or saved)")
        
        # Save if path provided (optional)
        save_path = "logs/attention_weights_heatmap.png"
        os.makedirs("logs", exist_ok=True)
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        except Exception as e:
            print(f"   Could not save figure: {e}")
        
        # Close figure
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except ImportError as e:
        print(f"\n[WARN] Visualization skipped: {e}")
        print("   Install matplotlib and seaborn for visualization: pip install matplotlib seaborn")


def test_rule_engine():
    """Test rule-based system."""
    print("\n" + "=" * 60)
    print("Testing Rule-Based System")
    print("=" * 60)
    
    engine = RuleEngine()
    
    # Test normal values
    print("\n1. Testing normal biometric values...")
    alerts = engine.check_all({
        'heart_rate': 80.0,
        'respiratory_rate': 20.0,
        'temperature': 36.5
    })
    print(f"   Alerts: {len(alerts)} (expected: 0)")
    assert len(alerts) == 0, "Should not have alerts for normal values!"
    
    # Test critical heart rate
    print("\n2. Testing critical heart rate (HR > 180)...")
    alerts = engine.check_heart_rate(heart_rate=185.0)
    if alerts:
        print(f"   Alert Level: {alerts.level.value}")
        print(f"   Message: {alerts.message}")
        print(f"   Value: {alerts.value}, Threshold: {alerts.threshold}")
        assert alerts.level == AlertLevel.EMERGENCY
    else:
        print("   [WARN] No alert generated (unexpected)")
    
    # Test fall detection
    print("\n3. Testing fall detection...")
    acceleration = np.array([20.0, 18.0, 22.0])  # High acceleration
    alert = engine.check_fall_detection(acceleration)
    if alert:
        print(f"   Alert Level: {alert.level.value}")
        print(f"   Message: {alert.message}")
        print(f"   Acceleration Magnitude: {alert.value:.2f} m/s²")
        assert alert.level == AlertLevel.CRITICAL
        assert alert.rule_name == "fall_detected"
    else:
        print("   [WARN] No fall detected (unexpected)")
    
    # Test multiple alerts
    print("\n4. Testing multiple extreme values...")
    alerts = engine.check_all(
        biometrics={
            'heart_rate': 185.0,
            'respiratory_rate': 45.0,
            'temperature': 39.0
        },
        acceleration=np.array([20.0, 18.0, 22.0])
    )
    print(f"   Total Alerts: {len(alerts)}")
    for i, alert in enumerate(alerts, 1):
        print(f"   Alert {i}: {alert.rule_name} - {alert.level.value}")
    
    alert_level = engine.get_alert_level(alerts)
    print(f"   Overall Alert Level: {alert_level.value}")
    assert alert_level == AlertLevel.EMERGENCY
    
    print("\n[OK] All rule engine tests passed!")
    return alerts


def test_confidence_calculation(classifier_output, attention_weights, rule_alerts):
    """Test confidence score calculation."""
    print("\n" + "=" * 60)
    print("Testing Confidence Score Calculation")
    print("=" * 60)
    
    # Calculate confidence
    print("\nCalculating confidence scores...")
    confidence = calculate_confidence(
        model_outputs=classifier_output,
        attention_weights=attention_weights[0:1],  # Use first sample
        rule_alerts=rule_alerts
    )
    
    print("\nConfidence Scores:")
    print(f"   Distress Confidence: {confidence['distress_confidence']:.3f}")
    print(f"   Severity Confidence: {confidence['severity_confidence']:.3f}")
    print(f"   Type Confidence: {confidence['type_confidence']:.3f}")
    print(f"   Modality Agreement: {confidence['modality_agreement']:.3f}")
    print(f"   Rule Agreement: {confidence['rule_agreement']:.3f}")
    print(f"   Overall Confidence: {confidence['overall_confidence']:.3f}")
    
    # Validate confidence scores are in [0, 1]
    for key, value in confidence.items():
        assert 0.0 <= value <= 1.0, f"Confidence {key} out of range: {value}"
    
    print("\n[OK] Confidence calculation test passed!")
    return confidence


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Fusion Layer Features Test Suite")
    print("=" * 60)
    
    try:
        # Test fusion layers
        attention_weights, classifier_output = test_fusion_layers()
        
        # Test attention visualization
        test_attention_visualization(attention_weights)
        
        # Test rule engine
        rule_alerts = test_rule_engine()
        
        # Test confidence calculation
        confidence = test_confidence_calculation(classifier_output, attention_weights, rule_alerts)
        
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Fusion layers: Working")
        print(f"  - Attention visualization: Available")
        print(f"  - Rule engine: {len(rule_alerts)} alerts detected")
        print(f"  - Confidence scores: Calculated (overall: {confidence['overall_confidence']:.3f})")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

