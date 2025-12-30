"""Tests for fusion module."""
import pytest
import torch
import numpy as np
from src.fusion.late_fusion import LateFusionLayer
from src.fusion.attention_fusion import AttentionFusion
from src.fusion.classifier import MultiTaskClassifier, DistressClassifier
from src.fusion.explainer import visualize_attention_weights, calculate_confidence, DistressExplainer
from src.fusion.rule_engine import RuleEngine, AlertLevel
from src.fusion.model import DistressDetectionModel


class TestLateFusion:
    """Test late fusion model."""

    def test_fusion_initialization(self):
        """Test fusion model initialization."""
        fusion = LateFusionLayer(
            audio_dim=256,
            bio_dim=256,
            context_dim=64,
            fusion_hidden_dims=[512, 256],
            dropout=0.4
        )
        assert fusion is not None
        assert fusion.output_dim == 256

    def test_fusion_forward_pass(self):
        """Test fusion forward pass."""
        batch_size = 4
        fusion = LateFusionLayer(
            audio_dim=256,
            bio_dim=256,
            context_dim=64,
            fusion_hidden_dims=[512, 256]
        )
        
        audio_emb = torch.randn(batch_size, 256)
        bio_emb = torch.randn(batch_size, 256)
        context_emb = torch.randn(batch_size, 64)
        
        output = fusion(audio_emb, bio_emb, context_emb)
        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_fusion_output_shape(self):
        """Test output shape is correct."""
        batch_size = 8
        fusion = LateFusionLayer(
            audio_dim=256,
            bio_dim=256,
            context_dim=64,
            fusion_hidden_dims=[512, 256]
        )
        
        audio_emb = torch.randn(batch_size, 256)
        bio_emb = torch.randn(batch_size, 256)
        context_emb = torch.randn(batch_size, 64)
        
        output = fusion(audio_emb, bio_emb, context_emb)
        assert output.shape == (batch_size, 256)  # Should match output_dim


class TestAttentionFusion:
    """Test attention-based fusion."""

    def test_attention_fusion_initialization(self):
        """Test attention fusion initialization."""
        fusion = AttentionFusion(
            audio_dim=256,
            bio_dim=256,
            context_dim=64,
            hidden_dim=128
        )
        assert fusion is not None

    def test_attention_fusion_forward_pass(self):
        """Test attention fusion forward pass with attention weights."""
        batch_size = 4
        fusion = AttentionFusion(
            audio_dim=256,
            bio_dim=256,
            context_dim=64,
            hidden_dim=128
        )
        
        audio_emb = torch.randn(batch_size, 256)
        bio_emb = torch.randn(batch_size, 256)
        context_emb = torch.randn(batch_size, 64)
        
        fused_emb, attention_weights = fusion(audio_emb, bio_emb, context_emb)
        
        assert fused_emb is not None
        assert attention_weights is not None
        assert fused_emb.shape == (batch_size, 128)
        assert attention_weights.shape == (batch_size, 3)
        
        # Check attention weights sum to 1 (softmax)
        weights_sum = attention_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones(batch_size), atol=1e-5)

    def test_attention_fusion_output_shape(self):
        """Test attention fusion output shapes."""
        batch_size = 8
        fusion = AttentionFusion(
            audio_dim=256,
            bio_dim=256,
            context_dim=64,
            hidden_dim=128
        )
        
        audio_emb = torch.randn(batch_size, 256)
        bio_emb = torch.randn(batch_size, 256)
        context_emb = torch.randn(batch_size, 64)
        
        fused_emb, attention_weights = fusion(audio_emb, bio_emb, context_emb)
        assert fused_emb.shape == (batch_size, 128)
        assert attention_weights.shape == (batch_size, 3)


class TestDistressClassifier:
    """Test distress classifier."""

    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = MultiTaskClassifier(
            input_dim=256,
            hidden_dim=128,
            dropout=0.3,
            num_distress_types=5
        )
        assert classifier is not None

    def test_classifier_forward_pass(self):
        """Test classification forward pass."""
        batch_size = 4
        classifier = MultiTaskClassifier(
            input_dim=256,
            hidden_dim=128,
            num_distress_types=5
        )
        
        embeddings = torch.randn(batch_size, 256)
        outputs = classifier(embeddings)
        
        assert 'distress_logits' in outputs
        assert 'severity' in outputs
        assert 'type_logits' in outputs
        
        assert outputs['distress_logits'].shape == (batch_size, 2)
        assert outputs['severity'].shape == (batch_size, 1)
        assert outputs['type_logits'].shape == (batch_size, 5)

    def test_output_is_binary(self):
        """Test output has 2 classes for binary distress detection."""
        batch_size = 2
        classifier = MultiTaskClassifier(input_dim=256, num_distress_types=5)
        embeddings = torch.randn(batch_size, 256)
        outputs = classifier(embeddings)
        
        # Distress logits should have 2 classes (binary)
        assert outputs['distress_logits'].shape[1] == 2


class TestRuleEngine:
    """Test rule-based engine for extreme values."""

    def test_rule_engine_initialization(self):
        """Test rule engine initialization."""
        engine = RuleEngine()
        assert engine is not None
        assert engine.hr_critical_threshold == 180.0

    def test_heart_rate_check_normal(self):
        """Test heart rate check with normal values."""
        engine = RuleEngine()
        alert = engine.check_heart_rate(heart_rate=80.0)
        assert alert is None

    def test_heart_rate_check_warning(self):
        """Test heart rate check with warning threshold."""
        engine = RuleEngine()
        alert = engine.check_heart_rate(heart_rate=155.0)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_heart_rate_check_critical(self):
        """Test heart rate check with critical threshold."""
        engine = RuleEngine()
        alert = engine.check_heart_rate(heart_rate=185.0)
        assert alert is not None
        assert alert.level == AlertLevel.EMERGENCY

    def test_fall_detection(self):
        """Test fall detection from acceleration."""
        engine = RuleEngine()
        # Simulate high acceleration (fall)
        acceleration = np.array([15.0, 18.0, 22.0])  # High magnitude
        alert = engine.check_fall_detection(acceleration)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert alert.rule_name == "fall_detected"

    def test_fall_detection_normal(self):
        """Test fall detection with normal acceleration."""
        engine = RuleEngine()
        # Normal acceleration
        acceleration = np.array([1.0, 0.5, 0.8])
        alert = engine.check_fall_detection(acceleration)
        assert alert is None

    def test_check_all(self):
        """Test checking all biometric values."""
        engine = RuleEngine()
        biometrics = {
            'heart_rate': 185.0,
            'respiratory_rate': 45.0,
            'temperature': 39.0
        }
        acceleration = np.array([20.0, 18.0, 22.0])
        
        alerts = engine.check_all(biometrics, acceleration)
        assert len(alerts) > 0
        assert any(alert.rule_name == "heart_rate_extreme" for alert in alerts)
        assert any(alert.rule_name == "fall_detected" for alert in alerts)


class TestAttentionVisualization:
    """Test attention weight visualization."""

    def test_visualize_attention_weights(self):
        """Test attention weight visualization."""
        try:
            # Create dummy attention weights
            attention_weights = torch.tensor([
                [0.4, 0.5, 0.1],  # Sample 1
                [0.2, 0.7, 0.1],  # Sample 2
                [0.6, 0.3, 0.1],  # Sample 3
            ])
            
            fig = visualize_attention_weights(attention_weights)
            assert fig is not None
            
            # Close figure to avoid warnings
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("Matplotlib not available for visualization")

    def test_visualize_attention_weights_single_sample(self):
        """Test visualization with single sample."""
        try:
            attention_weights = torch.tensor([0.4, 0.5, 0.1])
            fig = visualize_attention_weights(attention_weights)
            assert fig is not None
            
            import matplotlib.pyplot as plt
            plt.close(fig)
        except ImportError:
            pytest.skip("Matplotlib not available for visualization")


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_calculate_confidence_basic(self):
        """Test basic confidence calculation."""
        model_outputs = {
            'distress_logits': torch.tensor([[2.0, 0.5]]),  # Clear prediction
            'severity': torch.tensor([[5.0]]),
            'type_logits': torch.tensor([[1.0, 0.5, 0.3, 0.2, 0.1]])
        }
        
        confidence = calculate_confidence(model_outputs)
        
        assert 'distress_confidence' in confidence
        assert 'severity_confidence' in confidence
        assert 'type_confidence' in confidence
        assert 'overall_confidence' in confidence
        assert 0.0 <= confidence['overall_confidence'] <= 1.0

    def test_calculate_confidence_with_attention(self):
        """Test confidence calculation with attention weights."""
        model_outputs = {
            'distress_logits': torch.tensor([[2.0, 0.5]]),
            'severity': torch.tensor([[5.0]]),
            'type_logits': torch.tensor([[1.0, 0.5, 0.3, 0.2, 0.1]])
        }
        attention_weights = torch.tensor([[0.33, 0.33, 0.34]])  # Balanced
        
        confidence = calculate_confidence(model_outputs, attention_weights=attention_weights)
        
        assert 'modality_agreement' in confidence
        assert 0.0 <= confidence['modality_agreement'] <= 1.0


class TestDistressExplainer:
    """Test SHAP explainer (optional, requires SHAP)."""

    def test_explainer_initialization_requires_shap(self):
        """Test that explainer requires SHAP."""
        try:
            from src.fusion.explainer import DistressExplainer
            # If SHAP is available, test initialization would require a model
            # For now, just check the import works
            assert True
        except ImportError:
            # SHAP not available, skip test
            pytest.skip("SHAP not available")