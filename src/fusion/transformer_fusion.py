"""Transformer-based fusion layer for multimodal integration - adapted from WER-SSL."""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


class TransformerBatchNormEncoderLayer(nn.Module):
    """
    Transformer encoder layer with BatchNorm instead of LayerNorm.
    
    Adapted from WER-SSL: https://github.com/...
    
    This differs from torch's TransformerEncoderLayer in that it uses BatchNorm
    which normalizes across batch samples and time steps rather than features.
    This is more suitable for temporal wearable sensor data.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Args:
            src: (seq_len, batch_size, d_model)
            src_mask: optional mask
            src_key_padding_mask: optional padding mask
            is_causal: whether attention is causal (for PyTorch 2.0+ compatibility)
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        
        # BatchNorm (requires (batch, features, seq_len) format)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        # BatchNorm
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        
        return src


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding using sine and cosine functions.
    
    Adapted from WER-SSL.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerFusion(nn.Module):
    """
    Transformer-based multimodal fusion layer.
    
    Replaces simple concatenation with attention-based fusion that allows
    audio and biometric modalities to interact dynamically.
    
    Architecture:
    1. Project audio, biometric, context to common dimension (d_model)
    2. Add positional encoding
    3. Apply multi-head transformer encoder
    4. Project shared embedding to output dimension
    
    Adapted from WER-SSL's TCN_TRANS model.
    """

    def __init__(
        self,
        audio_dim: int = 256,
        bio_dim: int = 256,
        context_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        output_dim: int = 256,
        max_seq_len: int = 1024,
        positional_encoding: str = "fixed"
    ):
        """
        Initialize transformer fusion layer.
        
        Args:
            audio_dim: Audio embedding dimension
            bio_dim: Biometric embedding dimension
            context_dim: Context embedding dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension in transformer
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
            use_batch_norm: Whether to use BatchNorm in transformer layers
            output_dim: Final output dimension
            max_seq_len: Maximum sequence length for positional encoding
            positional_encoding: Type of positional encoding ('fixed' or 'learnable')
        """
        super().__init__()
        
        # Project embeddings to common dimension
        self.audio_project = nn.Linear(audio_dim, d_model)
        self.bio_project = nn.Linear(bio_dim, d_model)
        self.context_project = nn.Linear(context_dim, d_model)
        
        # Layer normalization for projections
        self.audio_norm = nn.LayerNorm(d_model)
        self.bio_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.positional_encoding = FixedPositionalEncoding(
            d_model, 
            dropout=dropout, 
            max_len=max_seq_len
        )
        
        # Transformer encoder
        if use_batch_norm:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        else:
            # Use standard PyTorch transformer with LayerNorm
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False,
                norm_first=False
            )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,  # Disable nested tensor for custom layer
            mask_check=False  # Disable mask checking for compatibility
        )
        
        # Output projection
        self.output_project = nn.Linear(d_model * 3, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        self.d_model = d_model
        self.output_dim = output_dim

    def forward(
        self,
        audio_emb: torch.Tensor,
        bio_emb: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse multimodal embeddings using transformer attention.
        
        Args:
            audio_emb: Audio embeddings (batch_size, audio_dim)
            bio_emb: Biometric embeddings (batch_size, bio_dim)
            context_emb: Context embeddings (batch_size, context_dim), optional
            
        Returns:
            Fused embeddings (batch_size, output_dim)
        """
        batch_size = audio_emb.shape[0]
        device = audio_emb.device
        
        # Handle missing context
        if context_emb is None:
            context_emb = torch.zeros(batch_size, 64, device=device, dtype=audio_emb.dtype)
        
        # Project to common dimension
        audio_proj = self.audio_norm(self.audio_project(audio_emb))  # (B, d_model)
        bio_proj = self.bio_norm(self.bio_project(bio_emb))          # (B, d_model)
        context_proj = self.context_norm(self.context_project(context_emb))  # (B, d_model)
        
        # Stack into sequence: (3, batch_size, d_model)
        # Order: [audio, bio, context] for interpretability
        src = torch.stack([audio_proj, bio_proj, context_proj], dim=0)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        
        # Apply transformer encoder
        # src: (seq_len=3, batch_size, d_model)
        output = self.transformer_encoder(src)
        
        # Aggregate: (batch_size, d_model*3)
        output = output.permute(1, 0, 2)  # (batch_size, 3, d_model)
        output = output.reshape(batch_size, -1)  # (batch_size, d_model*3)
        
        # Project to output dimension
        fused = self.output_project(output)  # (batch_size, output_dim)
        fused = self.output_norm(fused)
        
        return fused

    def get_attention_weights(
        self,
        audio_emb: torch.Tensor,
        bio_emb: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get attention weights showing how modalities interact.
        
        Returns weights showing the interaction strength between modalities.
        Higher weight indicates stronger attention/interaction.
        """
        batch_size = audio_emb.shape[0]
        device = audio_emb.device
        
        if context_emb is None:
            context_emb = torch.zeros(batch_size, 64, device=device, dtype=audio_emb.dtype)
        
        # Project to common dimension
        audio_proj = self.audio_norm(self.audio_project(audio_emb))
        bio_proj = self.bio_norm(self.bio_project(bio_emb))
        context_proj = self.context_norm(self.context_project(context_emb))
        
        # Stack into sequence
        src = torch.stack([audio_proj, bio_proj, context_proj], dim=0)
        src = self.positional_encoding(src)
        
        # Get attention from first transformer layer
        with torch.no_grad():
            encoder_layer = self.transformer_encoder.layers[0]
            # Self-attention returns (output, attention_weights)
            _, attn_weights = encoder_layer.self_attn(
                src, src, src, need_weights=True
            )
        
        return attn_weights  # (batch_size, seq_len, seq_len)


def _get_activation_fn(activation: str):
    """Get activation function."""
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    else:
        raise ValueError(f"Activation should be 'relu' or 'gelu', not {activation}")
