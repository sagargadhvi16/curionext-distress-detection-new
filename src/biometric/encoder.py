"""Biometric signal encoder with BiLSTM + Attention."""
import torch
import torch.nn as nn


class BiometricEncoder(nn.Module):
    """
    BiLSTM-based encoder with attention for biometric time-series.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Output projection
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tencsor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, time_steps, input_dim)

        Returns:
            embedding: (batch_size, embedding_dim)
        """

        # BiLSTM output
        lstm_out, _ = self.bilstm(x)
        # lstm_out: (B, T, 2H)

        # Attention scores
        attn_scores = self.attention(lstm_out)
        # (B, T, 1)

        attn_weights = torch.softmax(attn_scores, dim=1)
        # (B, T, 1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)
        # (B, 2H)

        context = self.dropout(context)

        embedding = self.fc(context)
        # (B, embedding_dim)

        return embedding
