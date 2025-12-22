"""Audio encoder using YAMNet."""
import tensorflow_hub as hub
import numpy as np
from typing import Optional
import tensorflow as tf
import torch
import torch.nn as nn

def init_weights(module):
    """
    Initialize model weights for stable training.
    Uses He initialization for Conv and Linear layers. bcoz these layers have trainable weights
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu") #He (Kaiming) initialization.
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class YAMNetExtractor:
    """
    YAMNet-based audio encoder for extracting embeddings.

    YAMNet outputs 1024-dimensional embeddings from audio.
    """

    def __init__(self, freeze_layers: int = 5):
        """
        Load pretrained YAMNet model from TensorFlow Hub.
        """
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract audio embeddings using YAMNet.

        Args:
            audio: Mono audio waveform (1D numpy array)
            sr: Sample rate (must be 16000 Hz)

        Returns:
            1024-dimensional embedding vector
        """
        if audio.size == 0:
            raise ValueError("Cannot extract YAMNet embeddings from empty audio")

        if audio.ndim != 1:
            raise ValueError("YAMNet expects mono audio input")

        if sr != 16000:
            raise ValueError("YAMNet requires 16 kHz audio")

         # Convert audio to TensorFlow tensor
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

        # Run YAMNet
        scores, embeddings, spectrogram = self.model(audio_tensor)

        # embeddings shape: (num_frames, 1024)
        # Mean pooling across time to get fixed-size vector
        embedding = tf.reduce_mean(embeddings, axis=0)

        return embedding.numpy()
    

class AudioCNNEncoder(nn.Module):
    """
    CNN-based audio encoder for feature extraction.

    Designed for log-mel or MFCC inputs.
    Outputs a fixed-size embedding.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )    
        self.fc = nn.Linear(128, embedding_dim)

#kept a higher-capacity CNN variant commented for future scaling once dataset size increases and representation bottlenecks appear
        '''
        self.cnn = nn.Sequential(
                      # Block 1
                      nn.Conv2d(1, 64, kernel_size=3, padding=1),
                      nn.BatchNorm2d(64),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
         
                      # Block 2
                      nn.Conv2d(64, 128, kernel_size=3, padding=1),
                      nn.BatchNorm2d(128),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
         
                      # Block 3
                      nn.Conv2d(128, 256, kernel_size=3, padding=1),
                      nn.BatchNorm2d(256),
                      nn.ReLU(),
                      nn.AdaptiveAvgPool2d((1, 1))
                  )
         
                  self.fc = nn.Linear(256, embedding_dim)
        '''
        #appying weight initialisation
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, freq_bins, time_steps)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over temporal dimension.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time_steps, embed_dim)

        Returns:
            Tensor of same shape with temporal attention applied
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x) #query=key=value 

        # Residual connection + normalization
        out = self.norm(x + attn_out)

        return out

class AudioEncoder(nn.Module):
    """
    Complete audio encoder with CNN backbone and temporal attention.

    Outputs fixed 256-dimensional embeddings.
    """

    def __init__(
        self,
        cnn_embedding_dim: int = 128, #Size of feature vector produced by CNN
        output_dim: int = 256, #Final embedding size
        num_heads: int = 4 #no of attention heads
    ):
        super().__init__()

        # CNN backbone
        self.cnn_encoder = AudioCNNEncoder(
            embedding_dim=cnn_embedding_dim
        )

        # Temporal attention over time steps
        self.temporal_attention = TemporalAttention(
            embed_dim=cnn_embedding_dim,
            num_heads=num_heads
        )

        # Final projection
        self.fc = nn.Linear(cnn_embedding_dim, output_dim)

        # Weight initialization
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, freq_bins, time_steps)

        Returns:
            (batch, 256) embedding
        """
        # CNN feature extraction
        # Output: (B, C, 1, 1) because AudioCNNEncoder uses AdaptiveAvgPool
        features = self.cnn_encoder(x)

        # Flatten spatial dimensions → (B, C)
        features = features.view(features.size(0), -1)

        # Expand to fake temporal dimension (length=1)
        # Shape: (B, T=1, C)
        features = features.unsqueeze(1)

        # Temporal attention (safe even for T=1)
        features = self.temporal_attention(features)

        # Pool over time
        features = features.mean(dim=1)

        # Final embedding
        return self.fc(features)

