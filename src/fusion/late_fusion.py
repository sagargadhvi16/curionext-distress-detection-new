"""Late fusion layer for multi-modal integration."""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LateFusionLayer(nn.Module):
    """
    Concatenation-based late fusion of audio, biometric, and context embeddings.

    Architecture:
    1. Audio Encoder (256-dim) ───┐
    2. Biometric Encoder (256-dim)├─→ Concat (576-dim) → Fusion Layers → Embedding
    3. Context Encoder (64-dim) ───┘
    """

    def __init__(
        self,
        audio_dim: int = 256,
        bio_dim: int = 256,
        context_dim: int = 64,
        fusion_hidden_dims: list = [512, 256],
        dropout: float = 0.4
    ):
        """
        Initialize late fusion layer.

        Args:
            audio_dim: Audio embedding dimension
            bio_dim: Biometric embedding dimension
            context_dim: Context embedding dimension
            fusion_hidden_dims: Hidden layer dimensions for fusion
            dropout: Dropout probability
        """
        super().__init__()
        
        # Input dimension after concatenation
        input_dim = audio_dim + bio_dim + context_dim
        
        # Build fusion layers with batch normalization and dropout
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in fusion_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.fusion_layers = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        bio_emb: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse audio, biometric, and context embeddings.

        Args:
            audio_emb: Audio embeddings (batch_size, audio_dim)
            bio_emb: Biometric embeddings (batch_size, bio_dim)
            context_emb: Context embeddings (batch_size, context_dim), optional

        Returns:
            Fused embeddings (batch_size, fusion_output_dim)
        """
        # Concatenate embeddings
        if context_emb is not None:
            fused = torch.cat([audio_emb, bio_emb, context_emb], dim=-1)
        else:
            # Use zeros if context not provided
            batch_size = audio_emb.shape[0]
            context_emb = torch.zeros(batch_size, 64, device=audio_emb.device)
            fused = torch.cat([audio_emb, bio_emb, context_emb], dim=-1)
        
        # Pass through fusion layers
        output = self.fusion_layers(fused)
        
        return output


# Keep old class name for backward compatibility
LateFusionModel = LateFusionLayer
