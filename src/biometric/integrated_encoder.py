"""
Integrated Biometric Encoder

Combines:
- Per-child baseline normalization
- BiLSTM temporal modeling
- Attention mechanism

Input:
- Sequence of biometric feature dictionaries

Output:
- Fixed-length biometric embedding
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from src.biometric.baseline import ChildBaselineModel


class IntegratedBiometricEncoder(nn.Module):
    """
    Complete biometric encoder with:
    - Per-child baseline normalization
    - BiLSTM
    - Attention
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        feature_keys: Optional[List[str]] = None
    ):
        super().__init__()

        # ---------------------------------
        # Baseline model (non-trainable)
        # ---------------------------------
        self.baseline_model = ChildBaselineModel()

        # ---------------------------------
        # Feature ordering (VERY IMPORTANT)
        # ---------------------------------
        if feature_keys is None:
            self.feature_keys = [
                "RMSSD",
                "LF_HF",
                "ACC_MEAN_MAG",
                "f1", "f2", "f3", "f4", "f5", "f6", "f7",
            ]
        else:
            self.feature_keys = feature_keys

        assert len(self.feature_keys) == input_dim, (
            f"input_dim={input_dim} but {len(self.feature_keys)} feature_keys provided"
        )

        # ---------------------------------
        # BiLSTM
        # ---------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # ---------------------------------
        # Attention layer
        # ---------------------------------
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # ---------------------------------
        # Final projection
        # ---------------------------------
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------
    def forward(
        self,
        child_id: str,
        raw_feature_dicts: Optional[List[Dict[str, float]]] = None,
        feature_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode biometric sequence.

        Args:
            child_id: Child identifier
            raw_feature_dicts: List of biometric feature dicts
            feature_sequence: Optional tensor (B, T, F)

        Returns:
            embedding: Tensor (B, embedding_dim)
        """

        # ---------------------------------
        # Convert dicts → tensor
        # ---------------------------------
        if feature_sequence is None:
            if raw_feature_dicts is None:
                raise ValueError("Either raw_feature_dicts or feature_sequence required")

            # Baseline normalization
            normalized = [
                self.baseline_model.normalize(child_id, f)
                for f in raw_feature_dicts
            ]

            # Fixed feature order
            matrix = [
                [float(f.get(k, 0.0)) for k in self.feature_keys]
                for f in normalized
            ]

            feature_sequence = torch.tensor(
                matrix, dtype=torch.float32
            ).unsqueeze(0)  # (1, T, F)

        # ---------------------------------
        # BiLSTM
        # ---------------------------------
        lstm_out, _ = self.lstm(feature_sequence)  # (B, T, 2H)

        # ---------------------------------
        # Attention
        # ---------------------------------
        scores = self.attention(lstm_out)          # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, 2H)

        # ---------------------------------
        # Final embedding
        # ---------------------------------
        embedding = self.projection(context)       # (B, embedding_dim)

        return embedding


# -------------------------------------------------
# 🧪 Local test
# -------------------------------------------------
if __name__ == "__main__":
    encoder = IntegratedBiometricEncoder(input_dim=10)

    baseline_data = [
        {"RMSSD": 60, "LF_HF": 1.2, "ACC_MEAN_MAG": 1.0,
         "f1": 1, "f2": 1, "f3": 1, "f4": 1, "f5": 1, "f6": 1, "f7": 1},
        {"RMSSD": 62, "LF_HF": 1.1, "ACC_MEAN_MAG": 1.1,
         "f1": 1, "f2": 1, "f3": 1, "f4": 1, "f5": 1, "f6": 1, "f7": 1},
    ]

    encoder.baseline_model.fit("child_01", baseline_data)

    features = [
        {"RMSSD": 30, "LF_HF": 3.0, "ACC_MEAN_MAG": 1.4,
         "f1": 1, "f2": 1, "f3": 1, "f4": 1, "f5": 1, "f6": 1, "f7": 1},
        {"RMSSD": 28, "LF_HF": 3.5, "ACC_MEAN_MAG": 1.6,
         "f1": 1, "f2": 1, "f3": 1, "f4": 1, "f5": 1, "f6": 1, "f7": 1},
    ]

    emb = encoder("child_01", raw_feature_dicts=features)
    print("Embedding shape:", emb.shape)
