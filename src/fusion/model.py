"""End-to-end distress detection model."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.fusion.late_fusion import LateFusionLayer
from src.fusion.attention_fusion import AttentionFusion
from src.fusion.context_encoder import ContextEncoder
from src.fusion.classifier import MultiTaskClassifier
from src.audio.encoder import AudioEncoder
from src.biometric.encoder import BiometricEncoder
from src.audio.preprocessing import AudioPreprocessor

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DistressDetectionModel(nn.Module):
    """
    Complete end-to-end distress detection model.
    
    Architecture:
    1. Audio Encoder (256-dim)
    2. Biometric Encoder (256-dim)
    3. Context Encoder (64-dim)
    4. Late Fusion (concatenation) or Attention Fusion
    5. Multi-task Classifier (3 heads: binary, severity, type)
    """
    
    def __init__(
        self,
        audio_encoder: Optional[nn.Module] = None,
        biometric_encoder: Optional[nn.Module] = None,
        context_encoder: Optional[nn.Module] = None,
        use_attention_fusion: bool = False,
        audio_dim: int = 256,
        bio_dim: int = 256,
        context_dim: int = 64,
        fusion_hidden_dims: list = [512, 256],
        classifier_hidden_dim: int = 128,
        dropout: float = 0.4,
        num_distress_types: int = 5
    ):
        """
        Initialize distress detection model.
        
        Args:
            audio_encoder: Pre-initialized audio encoder (or None to create default)
            biometric_encoder: Pre-initialized biometric encoder (or None to create default)
            context_encoder: Pre-initialized context encoder (or None to create default)
            use_attention_fusion: Whether to use attention fusion instead of simple concatenation
            audio_dim: Audio embedding dimension
            bio_dim: Biometric embedding dimension
            context_dim: Context embedding dimension
            fusion_hidden_dims: Hidden layer dimensions for fusion
            classifier_hidden_dim: Hidden dimension for classifier
            dropout: Dropout probability
            num_distress_types: Number of distress type classes
        """
        super().__init__()
        
        # Encoders
        if audio_encoder is not None:
            self.audio_encoder = audio_encoder
        else:
            # Default audio encoder (will need audio features, not raw audio)
            # For now, create a placeholder - in practice, use AudioEncoder from src.audio.encoder
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_dim, audio_dim),  # Placeholder
                nn.ReLU()
            )
            logger.warning("Using placeholder audio encoder. Provide proper AudioEncoder for production.")
        
        if biometric_encoder is not None:
            self.biometric_encoder = biometric_encoder
        else:
            # Default biometric encoder (will need biometric features, not raw data)
            # In practice, use BiometricEncoder from src.biometric.encoder
            self.biometric_encoder = nn.Sequential(
                nn.Linear(bio_dim, bio_dim),  # Placeholder
                nn.ReLU()
            )
            logger.warning("Using placeholder biometric encoder. Provide proper BiometricEncoder for production.")
        
        if context_encoder is not None:
            self.context_encoder = context_encoder
        else:
            self.context_encoder = ContextEncoder(embedding_dim=context_dim)
        
        # Fusion
        self.use_attention_fusion = use_attention_fusion

        # Ablation flags (default = full model)
        self.use_audio = True
        self.use_biometric = True
        self.use_context = True

        if use_attention_fusion:
            self.fusion = AttentionFusion(
                audio_dim=audio_dim,
                bio_dim=bio_dim,
                context_dim=context_dim,
                hidden_dim=fusion_hidden_dims[-1] if fusion_hidden_dims else 256
            )
            fusion_output_dim = fusion_hidden_dims[-1] if fusion_hidden_dims else 256
        else:
            self.fusion = LateFusionLayer(
                audio_dim=audio_dim,
                bio_dim=bio_dim,
                context_dim=context_dim,
                fusion_hidden_dims=fusion_hidden_dims,
                dropout=dropout
            )
            fusion_output_dim = self.fusion.output_dim
        
        # Classifier
        self.classifier = MultiTaskClassifier(
            input_dim=fusion_output_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
            num_distress_types=num_distress_types
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        biometric_features: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            audio_features: Audio features (batch_size, audio_feature_dim) or raw audio (batch_size, seq_len)
            biometric_features: Biometric features (batch_size, bio_feature_dim) or raw biometric (batch_size, seq_len, feature_dim)
            context_features: Context features (batch_size, context_feature_dim) or None
            return_attention: Whether to return attention weights (only if using attention fusion)
            
        Returns:
            Dictionary with keys:
            - distress_logits: (batch_size, 2) - binary classification logits
            - severity: (batch_size, 1) - severity score (0-10)
            - type_logits: (batch_size, num_distress_types) - distress type logits
            - attention_weights: (batch_size, 3) - optional, only if return_attention=True and use_attention_fusion=True
        """
        # Encode modalities
        audio_emb = self.audio_encoder(audio_features)  # (B, audio_dim)
        bio_emb = self.biometric_encoder(biometric_features)  # (B, bio_dim)
        
        # ---- ABLATION GATING ----
        if not self.use_audio:
            audio_emb = torch.zeros_like(audio_emb)

        if not self.use_biometric:
            bio_emb = torch.zeros_like(bio_emb)
            
        # Encode context
        if context_features is not None:
            # If context_features is a tensor, pass through context encoder
            # For now, assume it's already encoded or use simple projection
            if context_features.dim() == 1:
                context_features = context_features.unsqueeze(0)
            if context_features.shape[-1] != 64:
                # Simple projection to context_dim
                context_emb = nn.Linear(context_features.shape[-1], 64).to(context_features.device)(context_features)
            else:
                context_emb = context_features
        else:
            # Use default context encoding
            batch_size = audio_emb.shape[0]
            context_emb = torch.zeros(batch_size, 64, device=audio_emb.device)

            if not self.use_context:
                context_emb = torch.zeros_like(context_emb)

        # Fuse modalities
        if self.use_attention_fusion:
            fused_emb, attention_weights = self.fusion(audio_emb, bio_emb, context_emb)
            if return_attention:
                results = self.classifier(fused_emb)
                results['attention_weights'] = attention_weights
                return results
            else:
                return self.classifier(fused_emb)
        else:
            fused_emb = self.fusion(audio_emb, bio_emb, context_emb)
            return self.classifier(fused_emb)
    
    def predict(
        self,
        audio_features: torch.Tensor,
        biometric_features: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
        return_probs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions (with model in eval mode).
        
        Args:
            audio_features: Audio features
            biometric_features: Biometric features
            context_features: Context features (optional)
            return_probs: Whether to return probabilities instead of logits
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio_features, biometric_features, context_features)
            
            if return_probs:
                outputs['distress_probs'] = torch.softmax(outputs['distress_logits'], dim=-1)
                outputs['type_probs'] = torch.softmax(outputs['type_logits'], dim=-1)
            
            # Add predicted classes
            outputs['distress_pred'] = torch.argmax(outputs['distress_logits'], dim=-1)
            outputs['type_pred'] = torch.argmax(outputs['type_logits'], dim=-1)
            
            return outputs

