"""
Biometric Encoder
-----------------
Encodes temporal biometric features using:
- BiLSTM
- Attention mechanism

Input:
- Biometric feature sequence (B, T, F)

Output:
- Fixed-length embedding (B, embedding_dim)
"""

import torch
import torch.nn as nn
import numpy as np


class BiometricEncoder(nn.Module):
    """
    BiLSTM-based encoder with attention for biometric time-series.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        # -----------------------------
        # BiLSTM layer
        # -----------------------------
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # -----------------------------
        # Attention layer
        # -----------------------------
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # -----------------------------
        # Final projection
        # -----------------------------
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor or NumPy array of shape (B, T, F)

        Returns:
            embedding: Tensor of shape (B, embedding_dim)
        """

        # -------------------------------------------------
        # ✅ FIX: Convert NumPy → Torch if needed
        # -------------------------------------------------
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if not torch.is_tensor(x):
            raise TypeError("Input must be torch.Tensor or np.ndarray")

        # -----------------------------
        # BiLSTM
        # -----------------------------
        lstm_out, _ = self.bilstm(x)
        # (B, T, 2H)

        # -----------------------------
        # Attention
        # -----------------------------
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)

        # -----------------------------
        # Final embedding
        # -----------------------------
        embedding = self.fc(context)

        return embedding


# -------------------------------------------------
# 🧪 Quick sanity test
# -------------------------------------------------
if __name__ == "__main__":
    encoder = BiometricEncoder(input_dim=10)

    dummy_np = np.random.randn(1, 5, 10)
    dummy_torch = torch.randn(1, 5, 10)

    print("From NumPy:", encoder(dummy_np).shape)
    print("From Torch:", encoder(dummy_torch).shape)
