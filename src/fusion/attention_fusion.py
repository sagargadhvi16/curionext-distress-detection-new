"""Attention-based fusion for adaptive modality weighting."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionFusion(nn.Module):
    """
    Cross-modal attention for adaptive fusion weighting.
    
    Dynamically weights audio, biometric, and context embeddings
    based on their relevance to the task.
    """
    
    def __init__(
        self,
        audio_dim: int = 256,
        bio_dim: int = 256,
        context_dim: int = 64,
        hidden_dim: int = 128
    ):
        """
        Initialize attention fusion module.
        
        Args:
            audio_dim: Audio embedding dimension
            bio_dim: Biometric embedding dimension
            context_dim: Context embedding dimension
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()
        
        # Project each modality to common dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.bio_proj = nn.Linear(bio_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # Attention weights computation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 modalities: audio, bio, context
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        bio_emb: torch.Tensor,
        context_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention-based fusion.
        
        Args:
            audio_emb: Audio embeddings (batch_size, audio_dim)
            bio_emb: Biometric embeddings (batch_size, bio_dim)
            context_emb: Context embeddings (batch_size, context_dim)
            
        Returns:
            Tuple of (fused_embeddings, attention_weights)
            - fused_embeddings: (batch_size, hidden_dim)
            - attention_weights: (batch_size, 3) - weights for [audio, bio, context]
        """
        batch_size = audio_emb.shape[0]
        
        # Project to common dimension
        audio_proj = self.audio_proj(audio_emb)  # (B, hidden_dim)
        bio_proj = self.bio_proj(bio_emb)  # (B, hidden_dim)
        context_proj = self.context_proj(context_emb)  # (B, hidden_dim)
        
        # Concatenate for attention computation
        combined = torch.cat([audio_proj, bio_proj, context_proj], dim=-1)  # (B, hidden_dim * 3)
        
        # Compute attention weights
        attention_weights = self.attention(combined)  # (B, 3)
        
        # Apply attention weights
        weighted_audio = audio_proj * attention_weights[:, 0:1]
        weighted_bio = bio_proj * attention_weights[:, 1:2]
        weighted_context = context_proj * attention_weights[:, 2:3]
        
        # Concatenate weighted embeddings
        fused = torch.cat([weighted_audio, weighted_bio, weighted_context], dim=-1)  # (B, hidden_dim * 3)
        
        # Final projection
        output = self.output_proj(fused)  # (B, hidden_dim)
        
        return output, attention_weights

    def fuse_audio_tail(
        self,
        audio_tail_emb: torch.Tensor,
        context_emb: torch.Tensor,
        bio_emb_last: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse an audio-only tail segment when biometric data is shorter.

        This handles cases where audio duration exceeds biometric duration
        (e.g., audio=60s, biometric=30s). The remaining audio tail can still
        contain distress information. We construct a synthetic biometric input
        using the last available biometric embedding (if provided) or zeros,
        and perform attention-based fusion to produce a tail response.

        Args:
            audio_tail_emb: Tail audio embeddings (batch_size, audio_dim)
            context_emb: Context embeddings (batch_size, context_dim)
            bio_emb_last: Optional last available biometric embedding
                          (batch_size, bio_dim). If None, uses zeros.

        Returns:
            Tuple of (fused_tail_embeddings, attention_weights)
            - fused_tail_embeddings: (batch_size, hidden_dim)
            - attention_weights: (batch_size, 3) - weights for [audio, bio, context]

        Notes:
            - This function is additive and does not alter existing code paths.
            - Callers can process the returned fused tail embedding like other
              segments to ensure the tail is not missed.
        """
        batch_size = audio_tail_emb.shape[0]

        # Project modalities to common dimension
        audio_proj = self.audio_proj(audio_tail_emb)  # (B, hidden_dim)

        if bio_emb_last is not None:
            # Use the last available biometric embedding for continuity
            bio_proj = self.bio_proj(bio_emb_last)
        else:
            # If no biometric is available for the tail, use zeros
            bio_proj = torch.zeros(batch_size, audio_proj.shape[-1], device=audio_tail_emb.device, dtype=audio_tail_emb.dtype)

        context_proj = self.context_proj(context_emb)  # (B, hidden_dim)

        # Concatenate for attention computation
        combined = torch.cat([audio_proj, bio_proj, context_proj], dim=-1)  # (B, hidden_dim * 3)

        # Compute attention weights
        attention_weights = self.attention(combined)  # (B, 3)

        # Apply attention weights
        weighted_audio = audio_proj * attention_weights[:, 0:1]
        weighted_bio = bio_proj * attention_weights[:, 1:2]
        weighted_context = context_proj * attention_weights[:, 2:3]

        # Concatenate weighted embeddings and project to output
        fused = torch.cat([weighted_audio, weighted_bio, weighted_context], dim=-1)  # (B, hidden_dim * 3)
        output = self.output_proj(fused)  # (B, hidden_dim)

        return output, attention_weights

