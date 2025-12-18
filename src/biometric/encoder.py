"""
Biometric signal encoder using BiLSTM.

Encodes temporal biometric feature sequences (HRV + accelerometer)
into a fixed-size embedding.
"""

import torch
import torch.nn as nn


class BiometricEncoder(nn.Module):
    """
    BiLSTM-based temporal encoder for biometric features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Number of biometric features per time step
            hidden_dim: LSTM hidden size
            embedding_dim: Output embedding dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # BiLSTM outputs hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Tensor of shape (batch_size, time_steps, input_dim)

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        lstm_out, _ = self.lstm(features)

        # Take last time-step (context-aware representation)
        last_step = lstm_out[:, -1, :]

        embedding = self.fc(last_step)
        return embedding
