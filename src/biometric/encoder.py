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


class BiometricEncoder(nn.Module):
    """
    BiLSTM-based encoder with attention for biometric time-series.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, time_steps, input_dim)

        Returns:
            embedding: Tensor of shape (batch_size, embedding_dim)
        """

        # -----------------------------
        # BiLSTM output
        # -----------------------------
        lstm_out, _ = self.bilstm(x)
        # lstm_out → (B, T, 2H)

        # -----------------------------
        # Attention weights
        # -----------------------------
        attn_scores = self.attention(lstm_out)
        # (B, T, 1)

        attn_weights = torch.softmax(attn_scores, dim=1)
        # (B, T, 1)

        # -----------------------------
        # Weighted sum (context vector)
        # -----------------------------
        context = torch.sum(attn_weights * lstm_out, dim=1)
        # (B, 2H)

        context = self.dropout(context)

        # -----------------------------
        # Final embedding
        # -----------------------------
        embedding = self.fc(context)
        # (B, embedding_dim)

        return embedding


# -------------------------------------------------
# 🧪 Quick sanity test
# -------------------------------------------------
if __name__ == "__main__":
    encoder = BiometricEncoder(input_dim=10)

    dummy_input = torch.randn(1, 5, 10)  # (B=1, T=5, F=10)
    output = encoder(dummy_input)

    print("Output shape:", output.shape)
