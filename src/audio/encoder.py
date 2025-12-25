"""Audio encoders for CurioNext distress detection."""

import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------
def init_weights(module):
    """
    Initialize model weights for stable training.
    Uses He initialization for Conv and Linear layers (ReLU-based).
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


# ---------------------------------------------------------------------
# YAMNet Extractor (external pretrained model)
# ---------------------------------------------------------------------
class YAMNetExtractor:
    """
    YAMNet-based audio encoder for extracting embeddings.
    Outputs 1024-dimensional embeddings.
    """

    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract(self, audio: np.ndarray, sr: int = 16000,pool:bool=True) -> np.ndarray:
        if audio.size == 0:
            raise ValueError("Cannot extract YAMNet embeddings from empty audio")
        if audio.ndim != 1:
            raise ValueError("YAMNet expects mono audio")
        if sr != 16000:
            raise ValueError("YAMNet requires 16 kHz audio")

        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = self.model(audio_tensor) #(T,1024)

        if pool:
            return tf.reduce_mean(embeddings, axis=0).numpy()  # (1024,)
        else:
            return embeddings.numpy()  # (T, 1024)

# ---------------------------------------------------------------------
# CNN backbone (RETURNS TEMPORAL FEATURES)
# ---------------------------------------------------------------------
class AudioCNNEncoder(nn.Module):
    """
    CNN-based audio encoder.
    Preserves temporal dimension for attention modeling.
    """

    def __init__(self, out_channels: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            # Block 3
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, freq_bins, time_steps)

        Returns:
            Tensor of shape (batch, time_steps, channels)
        """
        x = self.cnn(x)              # (B, C, F', T')
        x = x.mean(dim=2)            # Average over frequency → (B, C, T')
        x = x.transpose(1, 2)        # (B, T', C)
        return x


# ---------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over temporal dimension.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time_steps, embed_dim)

        Returns:
            (batch, time_steps, embed_dim)
        """
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


# ---------------------------------------------------------------------
# Final Audio Encoder (CNN + Temporal Attention)
# ---------------------------------------------------------------------
class AudioEncoder(nn.Module):
    """
    Complete audio encoder with CNN backbone and temporal attention.

    Outputs fixed 256-dimensional embeddings.
    """

    def __init__(
        self,
        cnn_channels: int = 128,
        output_dim: int = 256,
        num_heads: int = 4
    ):
        super().__init__()

        self.cnn = AudioCNNEncoder(out_channels=cnn_channels)
        self.temporal_attention = TemporalAttention(
            embed_dim=cnn_channels,
            num_heads=num_heads
        )

        self.fc = nn.Linear(cnn_channels, output_dim)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, freq_bins, time_steps)

        Returns:
            (batch, 256)
        """
        # CNN feature extraction (keeps time)
        features = self.cnn(x)               # (B, T, C)

        # Temporal attention
        features = self.temporal_attention(features)

        # Temporal pooling AFTER attention to preserve sequence modeling
        features = features.mean(dim=1)     # (B, C)

        # Final projection
        return self.fc(features)
