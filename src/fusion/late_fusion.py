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

    def fuse_audio_tail(
        self,
        audio_tail_emb: torch.Tensor,
        context_emb: torch.Tensor,
        bio_emb_last: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse an audio-only tail segment when biometric data is shorter.

        This handles cases where audio duration exceeds biometric duration
        (e.g., audio=60s, biometric=30s). The remaining audio tail can still
        contain distress information. We use the last available biometric
        embedding (if provided) or zeros, and perform fusion to produce
        a tail response.

        Args:
            audio_tail_emb: Tail audio embeddings (batch_size, audio_dim)
            context_emb: Context embeddings (batch_size, context_dim)
            bio_emb_last: Optional last available biometric embedding
                          (batch_size, bio_dim). If None, uses zeros.

        Returns:
            Fused tail embeddings (batch_size, fusion_output_dim)

        Notes:
            - This function is additive and does not alter existing code paths.
            - Callers can process the returned fused tail embedding to ensure
              the audio tail is not missed.
        """
        batch_size = audio_tail_emb.shape[0]

        if bio_emb_last is not None:
            # Use the last available biometric embedding for continuity
            bio_emb = bio_emb_last
        else:
            # If no biometric is available for the tail, use zeros
            # Infer bio_dim from the fusion layers
            bio_dim = 256  # Default bio_dim
            bio_emb = torch.zeros(batch_size, bio_dim, device=audio_tail_emb.device, dtype=audio_tail_emb.dtype)

        # Use the standard forward pass with the tail audio and bio embedding
        return self.forward(audio_tail_emb, bio_emb, context_emb)


# Keep old class name for backward compatibility
LateFusionModel = LateFusionLayer
